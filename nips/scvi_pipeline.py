"""
cd /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/scVI/scripts
CUDA_VISIBLE_DEVICES=5 python scvi_pipeline.py
"""

from pathlib import Path
import gc
import hashlib
import logging
import re
import shutil
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch


# =========================
# 基础设置
# =========================
logging.getLogger("scvi").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Observation names are not unique")
warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_float32_matmul_precision("medium")
scvi.settings.seed = 42


# =========================
# 路径配置
# =========================
BASE_DIR = Path("/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/scVI")
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
LIBRARY_SIZE = 4000.0
RANDOM_SEED = 42


# =========================
# 模型参数
# =========================
# 这里沿用你参考脚本的主体结构，但去掉了复杂的续跑与内存工程逻辑。
REF_N_LAYERS = 3
REF_N_HIDDEN = 512
REF_N_LATENT = 64
REF_DISPERSION = "gene-batch"
REF_GENE_LIKELIHOOD = "nb"

REF_MAX_EPOCHS = 16
QUERY_MAX_EPOCHS = 8

REF_BATCH_SIZE = 8192
QUERY_BATCH_SIZE = 4096
PRED_BATCH_SIZE = 4096

REF_CHECK_VAL_EVERY = 2
QUERY_CHECK_VAL_EVERY = 2

TRAIN_NUM_WORKERS = 0
QUERY_NUM_WORKERS = 0


# =========================
# 工具函数
# =========================
def cleanup_memory():
    """简单清理 Python / CUDA 内存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unique_in_order(values):
    """保持首次出现顺序去重。"""
    return pd.Index(values).drop_duplicates().tolist()


def safe_name(text):
    """把字符串转成适合文件名的形式。"""
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "empty"


def stable_int(text):
    """把字符串稳定映射为整数，便于文件名去冲突与随机种子构造。"""
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def get_temp_pred_path(sample_name, cell_name, drug_name):
    """
    返回某个临时预测块的保存路径。
    路径按 sample / cell_name / drug 分层组织，后续 merge 更直接。
    """
    cell_dir = TEMP_PRED_DIR / sample_name / f"{safe_name(cell_name)}__{stable_int(cell_name)}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{safe_name(drug_name)}__{stable_int(drug_name)}_pred.h5ad"
    return cell_dir / file_name


def to_template_matrix(matrix, template_x):
    """
    让预测矩阵和 skeleton 的存储风格保持一致：
    - 如果 skeleton.X 是稀疏矩阵，则转成 csr_matrix
    - 否则保留为 dense numpy
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if sp.issparse(template_x):
        return sp.csr_matrix(matrix)
    return matrix


def collect_train_drugs(train_adatas):
    """
    只从训练集统计非 control 药物。
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


def sample_query_indices(n_source, n_target, random_seed):
    """
    先采样 query indices，再做 decode。
    这样输出行数天然和真实药物块 skeleton 完全一致。
    """
    if n_source <= 0:
        raise RuntimeError("query source 为空，无法采样")

    rng = np.random.default_rng(random_seed)
    sampled_idx = rng.choice(
        n_source,
        size=n_target,
        replace=(n_target > n_source),
    )
    return sampled_idx


def build_training_adata_for_one_drug(drug_name, train_adatas):
    """
    为当前药物构造 reference model 的训练 AnnData。

    逻辑：
    - 遍历每个训练文件
    - 在每个文件内部按 cell_name 分组
    - 对每个训练 cell_name，只保留：
        1) DMSO_TF
        2) 当前药物
    - 最后把所有训练块 concat 成当前药物的训练数据
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

            # 如果该训练 context 下没有当前药物，则跳过。
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


def train_reference_model(adata_train, accelerator, devices):
    """
    训练当前药物对应的 scVI reference model。
    """
    scvi.model.SCVI.setup_anndata(
        adata_train,
        batch_key="drug",
    )

    ref_model = scvi.model.SCVI(
        adata_train,
        n_layers=REF_N_LAYERS,
        n_hidden=REF_N_HIDDEN,
        n_latent=REF_N_LATENT,
        dispersion=REF_DISPERSION,
        gene_likelihood=REF_GENE_LIKELIHOOD,
        use_observed_lib_size=True,
    )

    ref_model.train(
        max_epochs=REF_MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        batch_size=REF_BATCH_SIZE,
        early_stopping=True,
        check_val_every_n_epoch=REF_CHECK_VAL_EVERY,
        datasplitter_kwargs={"num_workers": TRAIN_NUM_WORKERS},
    )

    return ref_model


def predict_one_block(
    ref_model,
    ctrl_test,
    stim_skeleton,
    drug_name,
    random_seed,
    accelerator,
    devices,
):
    """
    对单个目标药物块做 query adaptation + 预测。

    流程：
    1) 先把目标 control 子集准备成 query adata
    2) 用 load_query_data 构造 query model
    3) 训练 query model
    4) 先采样 indices，再调用 get_normalized_expression(transform_batch=drug_name)
    5) 用真实药物块 skeleton 的 obs/var 组装预测块
    """
    ctrl_test = ctrl_test.copy()
    ctrl_test.obs_names_make_unique()
    stim_skeleton = stim_skeleton.copy()
    stim_skeleton.obs_names_make_unique()

    scvi.model.SCVI.prepare_query_anndata(ctrl_test, ref_model, inplace=True)

    query_model = scvi.model.SCVI.load_query_data(
        adata=ctrl_test,
        reference_model=ref_model,
        freeze_expression=True,
        freeze_decoder_first_layer=True,
        accelerator=accelerator,
        device="auto",
    )

    query_model.train(
        max_epochs=QUERY_MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        batch_size=QUERY_BATCH_SIZE,
        early_stopping=True,
        check_val_every_n_epoch=QUERY_CHECK_VAL_EVERY,
        datasplitter_kwargs={"num_workers": QUERY_NUM_WORKERS},
    )

    n_d = stim_skeleton.n_obs
    sampled_idx = sample_query_indices(
        n_source=ctrl_test.n_obs,
        n_target=n_d,
        random_seed=random_seed,
    )

    pred_sample = query_model.get_normalized_expression(
        adata=query_model.adata,
        indices=sampled_idx,
        transform_batch=drug_name,
        library_size=LIBRARY_SIZE,
        batch_size=PRED_BATCH_SIZE,
        return_numpy=True,
    ).astype(np.float32, copy=False)

    pred_sample = np.clip(pred_sample, a_min=0.0, a_max=None)

    pred_block = ad.AnnData(
        X=to_template_matrix(pred_sample, stim_skeleton.X),
        obs=stim_skeleton.obs.copy(),
        var=stim_skeleton.var.copy(),
    )
    pred_block.obs["pred_or_real"] = "pred"

    del query_model, sampled_idx, pred_sample
    cleanup_memory()

    return pred_block


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    PREDICT_DIR.mkdir(parents=True, exist_ok=True)

    # 这版脚本按“从头完整跑一遍”的方式设计，
    # 因此每次重跑前直接清理旧的临时预测块。
    if TEMP_PRED_DIR.exists():
        shutil.rmtree(TEMP_PRED_DIR)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else "auto"

    print("=" * 100)
    print("开始执行精简版 scVI pipeline")
    print(f"control drug: {CONTROL_DRUG}")
    print(f"library_size: {LIBRARY_SIZE}")
    print(f"训练文件数: {len(TRAIN_FILES)}")
    print(f"目标文件数: {len(TARGET_FILES)}")
    print(f"REF_BATCH_SIZE: {REF_BATCH_SIZE}")
    print(f"QUERY_BATCH_SIZE: {QUERY_BATCH_SIZE}")
    print(f"PRED_BATCH_SIZE: {PRED_BATCH_SIZE}")
    print("=" * 100)

    # ------------------------------------------------------------
    # 一次性读取训练文件和目标文件。
    # ------------------------------------------------------------
    train_adatas = {}
    for file_path in TRAIN_FILES:
        train_adatas[file_path.stem] = sc.read_h5ad(file_path)

    target_adatas = {}
    for file_path in TARGET_FILES:
        target_adatas[file_path.stem] = sc.read_h5ad(file_path)

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
    # 第一阶段：按药物训练 reference model，并生成临时预测块
    # ============================================================
    for loop_id, drug_name in enumerate(train_drug_list, start=1):
        print("\n" + "-" * 100)
        print(f"[{loop_id}/{len(train_drug_list)}] 当前药物: {drug_name}")

        print("构造当前药物的训练数据 ...")
        adata_train_one_drug = build_training_adata_for_one_drug(
            drug_name=drug_name,
            train_adatas=train_adatas,
        )
        print(f"训练数据 shape: {adata_train_one_drug.shape}")

        print("开始训练当前药物的 reference model ...")
        ref_model = train_reference_model(
            adata_train=adata_train_one_drug,
            accelerator=accelerator,
            devices=devices,
        )
        print("reference model 训练完成")

        # 当前药物训练完成后，对每个目标文件逐个生成预测块。
        for sample_name, adata_target in target_adatas.items():
            print(f"  -> 开始处理目标文件: {sample_name}")

            file_cell_names = unique_in_order(adata_target.obs["cell_name"])

            for cell_name in file_cell_names:
                adata_cell = adata_target[adata_target.obs["cell_name"] == cell_name].copy()
                drug_order = unique_in_order(adata_cell.obs["drug"])

                # 如果当前目标 cell_name 根本没有这个药物，就跳过。
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

                pred_block = predict_one_block(
                    ref_model=ref_model,
                    ctrl_test=ctrl_test,
                    stim_skeleton=stim_skeleton,
                    drug_name=drug_name,
                    random_seed=RANDOM_SEED + stable_int(f"{sample_name}::{cell_name}::{drug_name}"),
                    accelerator=accelerator,
                    devices=devices,
                )

                save_path = get_temp_pred_path(sample_name, cell_name, drug_name)
                pred_block.write_h5ad(save_path)

                print(
                    f"    已保存: {save_path} | "
                    f"cell_name={cell_name} | "
                    f"control={ctrl_test.n_obs} | "
                    f"skeleton={stim_skeleton.n_obs}"
                )

                del adata_cell, ctrl_test, stim_skeleton, pred_block
                cleanup_memory()

        del ref_model, adata_train_one_drug
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

        # 按目标文件中 cell_name 的实际出现顺序处理。
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

            # 其余药物按该 cell_name 自己的实际出现顺序读取预测块。
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