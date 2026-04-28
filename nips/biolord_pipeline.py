"""
NIPS biolord baseline，按 scPerturBench OOD 思路重写。

运行示例：
CUDA_VISIBLE_DEVICES=5 python /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/biolord/scripts/biolord_pipeline.py

核心协议：
1. 每个目标文件、每个目标 cell_name 单独训练一个 biolord 模型。
2. 当前目标 cell_name 的非 control 药物细胞作为 OOD。
3. 其余所有非 OOD 细胞按 9:1 随机划分 train/valid，和参考代码一致。
4. NIPS 数据没有 layers['logNor']，也没有 obs['dose']，因此只注册 categorical attributes。
"""

from pathlib import Path
import gc
import hashlib
import logging
import re
import warnings

import anndata as ad
import biolord
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch


# ============================================================
# 路径配置：保持原脚本的数据路径和输出路径
# ============================================================
DATA_DIR = Path("/data1/fanpeishan/STATE/for_state/data/nips")
BASE_DIR = Path(
    "/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/biolord"
)
PREDICT_DIR = BASE_DIR / "predict_result"
TRAIN_LOG_DIR = BASE_DIR / "train_logs"

TRAIN_FILES = [
    "train_b_cells.h5ad",
    "train_nk_cells.h5ad",
    "train_t_cells_cd4.h5ad",
]

# val_t_cells_cd8 也作为目标任务输出，不再固定当 validation 文件。
TARGET_FILES = [
    "test_myeloid_cells.h5ad",
    "test_t_cells.h5ad",
    "val_t_cells_cd8.h5ad",
]

CONTROL_DRUG = "DMSO_TF"


# ============================================================
# 训练参数：除去 NIPS 不存在的 dose/logNor 外，按参考代码设置
# ============================================================
SPLIT_RANDOM_SEED = 1116
TORCH_RANDOM_SEED = 42

N_LATENT = 256
BATCH_SIZE = 128
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 20
CHECK_VAL_EVERY_N_EPOCH = 5

MODULE_PARAMS = {
    "decoder_width": 1024,
    "decoder_depth": 4,
    "attribute_nn_width": 512,
    "attribute_nn_depth": 2,
    "n_latent_attribute_categorical": 4,
    "gene_likelihood": "normal",
    "reconstruction_penalty": 1e2,
    "unknown_attribute_penalty": 1e1,
    "unknown_attribute_noise_param": 1e-1,
    "attribute_dropout_rate": 0.1,
    "use_batch_norm": False,
    "use_layer_norm": False,
    "seed": 42,
}

TRAINER_PARAMS = {
    "n_epochs_warmup": 0,
    "latent_lr": 1e-4,
    "latent_wd": 1e-4,
    "decoder_lr": 1e-4,
    "decoder_wd": 1e-4,
    "attribute_nn_lr": 1e-2,
    "attribute_nn_wd": 4e-8,
    "step_size_lr": 45,
    "cosine_scheduler": True,
    "scheduler_final_lr": 1e-5,
}


# ============================================================
# 运行设置
# ============================================================
logging.getLogger("scvi").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_name(text):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "empty"


def stable_int(text):
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def unique_in_order(values):
    return pd.Index(values).drop_duplicates().tolist()


def adata_to_numpy(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def to_template_matrix(matrix, template_x):
    matrix = np.asarray(matrix, dtype=np.float32)
    if sp.issparse(template_x):
        return sp.csr_matrix(matrix)
    return matrix


def resize_prediction_to_skeleton(pred_x, stim_skeleton, random_seed):
    """把 biolord 预测块采样到真实 stimulated block 的细胞数，便于 cell-eval 对齐。"""
    n_target = stim_skeleton.n_obs
    if pred_x.shape[0] == n_target:
        return pred_x.astype(np.float32, copy=False)
    if pred_x.shape[0] == 0:
        raise RuntimeError("biolord 返回了空预测矩阵。")

    rng = np.random.default_rng(random_seed)
    sampled_idx = rng.choice(
        pred_x.shape[0],
        size=n_target,
        replace=(n_target > pred_x.shape[0]),
    )
    return pred_x[sampled_idx].astype(np.float32, copy=False)


def read_one_file(file_name):
    """读取 h5ad，并补齐参考代码使用的 condition1/condition2 命名。"""
    file_path = DATA_DIR / file_name
    adata = sc.read_h5ad(file_path).copy()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if "cell_name" not in adata.obs.columns:
        raise KeyError(f"缺少 obs['cell_name']: {file_path}")
    if "drug" not in adata.obs.columns:
        raise KeyError(f"缺少 obs['drug']: {file_path}")

    adata.obs["cell_name"] = adata.obs["cell_name"].astype(str)
    adata.obs["drug"] = adata.obs["drug"].astype(str)
    adata.obs["condition1"] = adata.obs["cell_name"].astype(str)
    adata.obs["condition2"] = adata.obs["drug"].astype(str)
    adata.obs["source_file"] = file_name
    return adata


def clear_biolord_manager(model, adata_task):
    """长脚本连续训练多个模型时，清理 scvi/biolord 的 AnnDataManager。"""
    if model is not None:
        try:
            model.deregister_manager()
        except Exception:
            pass
    if model is not None and adata_task is not None:
        try:
            model.deregister_manager(adata_task)
        except Exception:
            pass
    if adata_task is not None:
        try:
            biolord.Biolord.deregister_manager(adata_task)
        except Exception:
            pass


if __name__ == "__main__":
    np.random.seed(SPLIT_RANDOM_SEED)
    torch.manual_seed(TORCH_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_RANDOM_SEED)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("开始执行 NIPS biolord OOD pipeline")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {PREDICT_DIR}")
    print(f"训练文件: {TRAIN_FILES}")
    print(f"目标文件: {TARGET_FILES}")
    print(f"control drug: {CONTROL_DRUG}")
    print(f"max epochs: {MAX_EPOCHS}")
    print(f"batch size: {BATCH_SIZE}")
    print(f"early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print("=" * 100)

    # 训练文件只读一次；每个目标任务都会和当前目标文件重新拼接。
    train_adatas = []
    for train_file in TRAIN_FILES:
        adata_train = read_one_file(train_file)
        train_adatas.append(adata_train)
        print(f"已读取训练文件: {train_file}, shape={adata_train.shape}")

    for target_file in TARGET_FILES:
        print("\n" + "#" * 100)
        print(f"开始处理目标文件: {target_file}")
        print("#" * 100)

        target_raw = read_one_file(target_file)
        target_cell_names = unique_in_order(target_raw.obs["condition1"])
        output_blocks = []

        for target_cell in target_cell_names:
            print("\n" + "-" * 100)
            print(f"目标 cell_name: {target_cell}")

            model = None
            adata_task = None

            try:
                # 当前目标 cell 的真实 control 会保留在最终输出里，也会作为 biolord 预测源。
                target_cell_raw = target_raw[target_raw.obs["condition1"] == target_cell].copy()
                real_ctrl = target_cell_raw[target_cell_raw.obs["condition2"] == CONTROL_DRUG].copy()
                if real_ctrl.n_obs == 0:
                    raise RuntimeError(f"{target_file}/{target_cell} 没有 control={CONTROL_DRUG} 细胞。")
                real_ctrl.obs["pred_or_real"] = "real_ctrl"
                output_blocks.append(real_ctrl)

                # 参考代码是在完整数据里标记当前 outSample 的 stimulated 为 OOD。
                # 这里用三个 train 文件 + 当前目标文件构造同样的任务数据。
                adata_task = ad.concat(
                    train_adatas + [target_raw.copy()],
                    join="inner",
                    merge="same",
                    index_unique=None,
                )
                adata_task.obs_names_make_unique()
                adata_task.var_names_make_unique()

                ood_mask = (
                    (adata_task.obs["condition1"].astype(str).to_numpy() == str(target_cell))
                    & (adata_task.obs["condition2"].astype(str).to_numpy() != CONTROL_DRUG)
                )
                ood_count = int(ood_mask.sum())
                if ood_count == 0:
                    raise RuntimeError(f"{target_file}/{target_cell} 没有可作为 OOD 的 stimulated 细胞。")

                # 和参考代码一致：非 OOD 细胞按 0.9/0.1 随机切 train/valid。
                np.random.seed(SPLIT_RANDOM_SEED)
                adata_task.obs["split"] = None
                adata_task.obs.loc[ood_mask, "split"] = "ood"
                non_ood_count = adata_task.n_obs - ood_count
                adata_task.obs.loc[~ood_mask, "split"] = np.random.choice(
                    ["train", "valid"],
                    size=non_ood_count,
                    replace=True,
                    p=[0.9, 0.1],
                )

                adata_task.obs["condition1"] = pd.Categorical(adata_task.obs["condition1"].astype(str))
                adata_task.obs["condition2"] = pd.Categorical(adata_task.obs["condition2"].astype(str))
                adata_task.obs["split"] = pd.Categorical(
                    adata_task.obs["split"],
                    categories=["train", "valid", "ood"],
                )

                print(f"任务数据 shape: {adata_task.shape}")
                print(f"train 细胞数: {(adata_task.obs['split'] == 'train').sum()}")
                print(f"valid 细胞数: {(adata_task.obs['split'] == 'valid').sum()}")
                print(f"ood   细胞数: {(adata_task.obs['split'] == 'ood').sum()}")

                # NIPS 没有 dose/logNor，所以这里只注册两个 categorical attributes。
                biolord.Biolord.setup_anndata(
                    adata=adata_task,
                    categorical_attributes_keys=["condition1", "condition2"],
                    layer=None,
                )

                model = biolord.Biolord(
                    adata=adata_task,
                    n_latent=N_LATENT,
                    model_name=f"{Path(target_file).stem}_{safe_name(target_cell)}",
                    module_params=MODULE_PARAMS,
                    train_classifiers=False,
                    split_key="split",
                    train_split="train",
                    valid_split="valid",
                    test_split="ood",
                )

                print("开始训练 biolord 模型")
                model.train(
                    max_epochs=MAX_EPOCHS,
                    batch_size=BATCH_SIZE,
                    plan_kwargs=TRAINER_PARAMS,
                    early_stopping=True,
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                    check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
                    enable_checkpointing=False,
                )
                print("模型训练完成")

                source_idx = np.where(
                    (adata_task.obs["condition1"].astype(str).to_numpy() == str(target_cell))
                    & (adata_task.obs["condition2"].astype(str).to_numpy() == CONTROL_DRUG)
                )[0]
                if len(source_idx) == 0:
                    raise RuntimeError(f"{target_file}/{target_cell} 没有可用于预测的 control source。")

                adata_source = adata_task[source_idx].copy()
                adata_preds = model.compute_prediction_adata(
                    adata_task,
                    adata_source,
                    target_attributes=["condition2"],
                )
                adata_preds = adata_preds[
                    adata_preds.obs["condition2"].astype(str).to_numpy() != CONTROL_DRUG
                ].copy()

                target_drugs = [
                    drug
                    for drug in unique_in_order(target_cell_raw.obs["condition2"].astype(str))
                    if drug != CONTROL_DRUG
                ]

                for drug_rank, target_drug in enumerate(target_drugs, start=1):
                    print(f"开始生成预测: cell_name={target_cell}, drug={target_drug}")
                    stim_skeleton = target_cell_raw[
                        target_cell_raw.obs["condition2"].astype(str).to_numpy() == str(target_drug)
                    ].copy()
                    if stim_skeleton.n_obs == 0:
                        continue

                    pred_for_drug = adata_preds[
                        adata_preds.obs["condition2"].astype(str).to_numpy() == str(target_drug)
                    ].copy()
                    if pred_for_drug.n_obs == 0:
                        raise RuntimeError(f"biolord 没有生成 drug={target_drug} 的预测。")

                    pred_x = adata_to_numpy(pred_for_drug.X)
                    pred_x = resize_prediction_to_skeleton(
                        pred_x,
                        stim_skeleton,
                        random_seed=SPLIT_RANDOM_SEED
                        + stable_int(f"{target_file}::{target_cell}::{target_drug}"),
                    )
                    pred_x = np.clip(pred_x, a_min=0.0, a_max=None)

                    pred_block = ad.AnnData(
                        X=to_template_matrix(pred_x, stim_skeleton.X),
                        obs=stim_skeleton.obs.copy(),
                        var=stim_skeleton.var.copy(),
                    )
                    pred_block.obs["pred_or_real"] = "pred"
                    output_blocks.append(pred_block)

                    del stim_skeleton, pred_for_drug, pred_x, pred_block
                    cleanup_memory()

                del adata_source, adata_preds, target_cell_raw, real_ctrl

            finally:
                clear_biolord_manager(model, adata_task)
                del model, adata_task
                cleanup_memory()

        final_adata = ad.concat(
            output_blocks,
            join="inner",
            merge="same",
            index_unique=None,
        )
        final_adata.obs_names_make_unique()

        if final_adata.n_obs != target_raw.n_obs:
            raise RuntimeError(
                "最终预测文件细胞数与目标文件不一致："
                f"pred_n_obs={final_adata.n_obs}, target_n_obs={target_raw.n_obs}, target_file={target_file}"
            )

        output_path = PREDICT_DIR / f"{Path(target_file).stem}_pred.h5ad"
        final_adata.write_h5ad(output_path)

        print("-" * 100)
        print(f"目标文件完成: {target_file}")
        print(f"输出文件: {output_path}")
        print(f"输出 shape: {final_adata.shape}")

        del target_raw, output_blocks, final_adata
        cleanup_memory()

    print("\n" + "=" * 100)
    print("NIPS biolord OOD pipeline 全部完成")
    print("=" * 100)
