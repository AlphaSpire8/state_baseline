"""
cd /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/scGen
CUDA_VISIBLE_DEVICES=7 python scripts/scgen_pipeline.py
"""

from pathlib import Path
import gc
import logging
import re
import shutil
import warnings
import hashlib

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scgen
import torch

import scvi.data._manager as _scvi_manager


# =========================
# scvi / scGen 基础设置
# =========================
# 这里保留一个最必要的兼容补丁：
# 允许把训练中未出现过的新 cell_name 从测试集传入预测流程。
# 你的测试 cell_name 和训练 cell_name 不重合，因此这一步是需要的。
_orig_transfer_fields = _scvi_manager.AnnDataManager.transfer_fields


def _patched_transfer_fields(self, adata_target, **kwargs):
    kwargs.setdefault("extend_categories", True)
    return _orig_transfer_fields(self, adata_target, **kwargs)


_scvi_manager.AnnDataManager.transfer_fields = _patched_transfer_fields

logging.getLogger("scvi").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


# =========================
# 基础路径配置
# =========================
BASE_DIR = Path("/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/scGen")
DATA_DIR = Path("/data1/fanpeishan/STATE/for_state/data/nips")

PREDICT_DIR = BASE_DIR / "predict_result"
TEMP_PRED_DIR = BASE_DIR / "tmp_predictions"

TRAIN_FILES = [
    DATA_DIR / "train_b_cells.h5ad",
    DATA_DIR / "train_nk_cells.h5ad",
    DATA_DIR / "train_t_cells_cd4.h5ad",
]

TARGET_FILES = [
    DATA_DIR / "test_myeloid_cells.h5ad",
    DATA_DIR / "test_t_cells.h5ad",
    DATA_DIR / "val_t_cells_cd8.h5ad",
]

CONTROL_DRUG = "DMSO_TF"


# =========================
# 训练参数
# =========================
# 数据集较小，因此这里使用相对简洁、稳妥的参数。
MAX_EPOCHS = 64
BATCH_SIZE = 8192
EARLY_STOPPING_PATIENCE = 8
RANDOM_SEED = 16


# =========================
# 工具函数
# =========================
def cleanup_memory():
    """简单清理内存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unique_in_order(values):
    """保持首次出现顺序地去重。"""
    return pd.Index(values).drop_duplicates().tolist()


def safe_name(text):
    """把字符串转成适合文件名的形式。"""
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "empty"


def stable_int(text):
    """把字符串稳定映射成整数，用于随机种子和文件名去冲突。"""
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def get_temp_pred_path(sample_name, cell_name, drug_name):
    """
    返回某个临时预测块的路径。
    这里按 sample / cell_name / drug 分层保存，便于后续 merge。
    """
    cell_dir = TEMP_PRED_DIR / sample_name / f"{safe_name(cell_name)}__{stable_int(cell_name)}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{safe_name(drug_name)}__{stable_int(drug_name)}_pred.h5ad"
    return cell_dir / file_name


def adata_to_numpy(x):
    """把表达矩阵统一转成 float32 numpy。"""
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def collect_train_drugs(train_adatas):
    """
    只从训练集收集非 control 药物列表。
    按首次出现顺序去重。
    """
    drug_list = []
    seen = set()

    for adata_train in train_adatas.values():
        for drug_name in unique_in_order(adata_train.obs["drug"]):
            if drug_name == CONTROL_DRUG:
                continue
            if drug_name in seen:
                continue
            seen.add(drug_name)
            drug_list.append(drug_name)

    return drug_list


def build_training_adata_for_one_drug(drug_name, train_adatas):
    """
    为当前药物构造 scGen 的训练 AnnData。

    逻辑：
    - 遍历每个训练文件
    - 在每个文件中，再按 cell_name 分组
    - 对每个训练 cell_name，只保留：
        1) DMSO_TF
        2) 当前药物
    - 最后把所有训练块拼起来
    """
    train_blocks = []

    for sample_name, adata_train in train_adatas.items():
        cell_name_list = unique_in_order(adata_train.obs["cell_name"])

        for cell_name in cell_name_list:
            ctrl_block = adata_train[
                (adata_train.obs["cell_name"] == cell_name) &
                (adata_train.obs["drug"] == CONTROL_DRUG)
            ].copy()

            stim_block = adata_train[
                (adata_train.obs["cell_name"] == cell_name) &
                (adata_train.obs["drug"] == drug_name)
            ].copy()

            # 如果该训练 context 下没有当前药物，就跳过。
            if ctrl_block.n_obs == 0 or stim_block.n_obs == 0:
                del ctrl_block, stim_block
                continue

            adata_block = ad.concat(
                [ctrl_block, stim_block],
                join="inner",
                merge="same",
                index_unique=None,
            )
            adata_block.obs_names_make_unique()
            train_blocks.append(adata_block)

            del ctrl_block, stim_block

    if len(train_blocks) == 0:
        raise ValueError(f"药物 '{drug_name}' 在训练集中没有可用样本。")

    adata_train_one_drug = ad.concat(
        train_blocks,
        join="inner",
        merge="same",
        index_unique=None,
    )
    adata_train_one_drug.obs_names_make_unique()

    del train_blocks
    cleanup_memory()

    return adata_train_one_drug


def sample_control_to_target(ctrl_test, n_target, random_seed):
    """
    先从目标 cell_name 自身的 control 中采样到和真实药物块相同的细胞数，
    再送入 scGen.predict()。
    """
    rng = np.random.default_rng(random_seed)
    sample_index = rng.choice(
        ctrl_test.n_obs,
        size=n_target,
        replace=(n_target > ctrl_test.n_obs),
    )
    return ctrl_test[sample_index, :].copy()


def train_one_scgen_model(adata_train):
    """
    训练单药物的 scGen 模型。
    """
    scgen.SCGEN.setup_anndata(
        adata_train,
        batch_key="drug",
        labels_key="cell_name",
    )

    model = scgen.SCGEN(adata_train)
    model.train(
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )
    return model


def predict_one_block(model, drug_name, ctrl_sampled, stim_skeleton):
    """
    直接调用 scGen.predict() 做预测。

    这里不手动计算 delta，也不手动做 latent arithmetic。
    预测结果只借用真实药物块的 obs / var 作为骨架。
    """
    pred_adata, _ = model.predict(
        ctrl_key=CONTROL_DRUG,
        stim_key=drug_name,
        adata_to_predict=ctrl_sampled,
    )

    pred_X = adata_to_numpy(pred_adata.X)
    pred_X = np.clip(pred_X, a_min=0.0, a_max=None).astype(np.float32, copy=False)

    adata_pred = ad.AnnData(
        X=pred_X,
        obs=stim_skeleton.obs.copy(),
        var=stim_skeleton.var.copy(),
    )
    adata_pred.obs["pred_or_real"] = "pred"

    return adata_pred


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    PREDICT_DIR.mkdir(parents=True, exist_ok=True)

    # 这版脚本按“从头完整跑一遍”的方式设计，
    # 因此每次重跑前先清理旧的临时预测块。
    if TEMP_PRED_DIR.exists():
        shutil.rmtree(TEMP_PRED_DIR)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("开始执行精简版 scGen 流程")
    print(f"control drug: {CONTROL_DRUG}")
    print(f"训练文件数: {len(TRAIN_FILES)}")
    print(f"目标文件数: {len(TARGET_FILES)}")
    print(f"训练 batch size: {BATCH_SIZE}")
    print(f"最大 epoch: {MAX_EPOCHS}")
    print("=" * 100)

    # ------------------------------------------------------------
    # 一次性读入训练文件和目标文件。
    # ------------------------------------------------------------
    train_adatas = {}
    for file_path in TRAIN_FILES:
        train_adatas[file_path.stem] = sc.read_h5ad(file_path)

    target_adatas = {}
    for file_path in TARGET_FILES:
        target_adatas[file_path.stem] = sc.read_h5ad(file_path)

    # 只从训练集收集药物集合。
    train_drug_list = collect_train_drugs(train_adatas)

    print(f"训练集中非 control 药物数量: {len(train_drug_list)}")
    print("训练集 cell_name:")
    for sample_name, adata_train in train_adatas.items():
        for cell_name in unique_in_order(adata_train.obs["cell_name"]):
            print(f"  - {sample_name}: {cell_name}")

    print("目标文件 cell_name:")
    for sample_name, adata_target in target_adatas.items():
        for cell_name in unique_in_order(adata_target.obs["cell_name"]):
            print(f"  - {sample_name}: {cell_name}")

    # ============================================================
    # 第一阶段：按药物训练并生成临时预测块
    # ============================================================
    for loop_id, drug_name in enumerate(train_drug_list, start=1):
        print("\n" + "-" * 100)
        print(f"[{loop_id}/{len(train_drug_list)}] 当前药物: {drug_name}")

        print("构造当前药物的训练 AnnData ...")
        adata_train_one_drug = build_training_adata_for_one_drug(
            drug_name=drug_name,
            train_adatas=train_adatas,
        )
        print(f"训练数据 shape: {adata_train_one_drug.shape}")

        print("开始训练 scGen 模型 ...")
        model = train_one_scgen_model(adata_train_one_drug)
        print("模型训练完成")

        # 当前药物训练完成后，对所有目标文件逐个生成预测块。
        for sample_name, adata_target in target_adatas.items():
            print(f"  -> 开始处理目标文件: {sample_name}")

            file_cell_names = unique_in_order(adata_target.obs["cell_name"])

            for cell_name in file_cell_names:
                adata_cell = adata_target[adata_target.obs["cell_name"] == cell_name].copy()
                drug_order = unique_in_order(adata_cell.obs["drug"])

                # 如果当前目标 cell_name 根本没有这个药物，则无需生成预测块。
                if drug_name not in drug_order:
                    del adata_cell
                    continue

                ctrl_test = adata_cell[adata_cell.obs["drug"] == CONTROL_DRUG].copy()
                if ctrl_test.n_obs == 0:
                    raise ValueError(
                        f"目标文件 '{sample_name}' 的 cell_name '{cell_name}' 中找不到 control 细胞。"
                    )

                stim_skeleton = adata_cell[adata_cell.obs["drug"] == drug_name].copy()
                if stim_skeleton.n_obs == 0:
                    del adata_cell, ctrl_test, stim_skeleton
                    continue

                ctrl_sampled = sample_control_to_target(
                    ctrl_test=ctrl_test,
                    n_target=stim_skeleton.n_obs,
                    random_seed=RANDOM_SEED + stable_int(f"{sample_name}::{cell_name}::{drug_name}"),
                )

                adata_pred = predict_one_block(
                    model=model,
                    drug_name=drug_name,
                    ctrl_sampled=ctrl_sampled,
                    stim_skeleton=stim_skeleton,
                )

                save_path = get_temp_pred_path(sample_name, cell_name, drug_name)
                adata_pred.write_h5ad(save_path)

                print(
                    f"    已保存: {save_path} | "
                    f"cell_name={cell_name} | "
                    f"control={ctrl_test.n_obs} | "
                    f"skeleton={stim_skeleton.n_obs}"
                )

                del adata_cell, ctrl_test, stim_skeleton, ctrl_sampled, adata_pred
                cleanup_memory()

        del model, adata_train_one_drug
        cleanup_memory()

    # ============================================================
    # 第二阶段：按目标文件合并最终输出
    # ============================================================
    print("\n" + "=" * 100)
    print("开始合并最终预测文件")
    print("=" * 100)

    for sample_name, adata_target in target_adatas.items():
        print(f"合并目标文件: {sample_name}")

        output_blocks = []
        file_cell_names = unique_in_order(adata_target.obs["cell_name"])

        # 按文件中 cell_name 的实际出现顺序处理。
        for cell_name in file_cell_names:
            adata_cell = adata_target[adata_target.obs["cell_name"] == cell_name].copy()
            drug_order = unique_in_order(adata_cell.obs["drug"])

            if CONTROL_DRUG not in drug_order:
                raise ValueError(
                    f"目标文件 '{sample_name}' 的 cell_name '{cell_name}' 中缺少 control 药物 '{CONTROL_DRUG}'。"
                )

            # 真实 control block 直接保留。
            ctrl_real = adata_cell[adata_cell.obs["drug"] == CONTROL_DRUG].copy()
            ctrl_real.obs = ctrl_real.obs.copy()
            ctrl_real.obs["pred_or_real"] = "real_ctrl"
            output_blocks.append(ctrl_real)

            # 其他药物按该 cell_name 自己的实际出现顺序读取预测块。
            for drug_name in drug_order:
                if drug_name == CONTROL_DRUG:
                    continue

                pred_file = get_temp_pred_path(sample_name, cell_name, drug_name)
                if not pred_file.exists():
                    raise FileNotFoundError(f"缺少预测块，无法 merge: {pred_file}")

                output_blocks.append(sc.read_h5ad(pred_file))

            del adata_cell
            cleanup_memory()

        final_adata = ad.concat(
            output_blocks,
            join="inner",
            merge="same",
            index_unique=None,
        )
        final_adata.obs_names_make_unique()

        final_path = PREDICT_DIR / f"{sample_name}_pred.h5ad"
        final_adata.write_h5ad(final_path)

        print(f"已生成: {final_path}")
        print(f"最终 shape: {final_adata.shape}")

        del output_blocks, final_adata
        cleanup_memory()

    print("\n" + "=" * 100)
    print("全部流程完成")
    print("最终输出文件:")
    for file_path in TARGET_FILES:
        print(PREDICT_DIR / f"{file_path.stem}_pred.h5ad")
    print("=" * 100)