# CUDA_VISIBLE_DEVICES=5 python /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/biolord/scripts/biolord_pipeline.py
from pathlib import Path
import gc
import logging
import warnings

import anndata as ad
import biolord
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch


# ============================================================
# 基础配置
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
TARGET_FILES = [
    "test_myeloid_cells.h5ad",
    "test_t_cells.h5ad",
    "val_t_cells_cd8.h5ad",
]

CONTROL_DRUG = "DMSO_TF"
RANDOM_SEED = 16
SPLIT_RANDOM_SEED = 1116

N_LATENT = 256
MAX_EPOCHS = 500
TRAIN_BATCH_SIZE = 128
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

TRAIN_PLAN_KWARGS = {
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
scvi.settings.dl_num_workers = 0
scvi.settings.dl_persistent_workers = False
scvi.settings.seed = RANDOM_SEED

logging.getLogger("scvi").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Observation names are not unique")

torch.set_float32_matmul_precision("medium")


def cleanup_memory() -> None:
    """清理长流程中累积的内存和显存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_nips_h5ad(file_name: str) -> ad.AnnData:
    """读取 NIPS 文件，并补齐 biolord 参考实现需要的列名。"""
    file_path = DATA_DIR / file_name
    adata = sc.read_h5ad(file_path).copy()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    for column in ["drug", "cell_name", "dose_uM"]:
        if column not in adata.obs.columns:
            raise KeyError(f"文件缺少 obs['{column}']: {file_path}")

    adata.obs["drug"] = adata.obs["drug"].astype(str)
    adata.obs["cell_name"] = adata.obs["cell_name"].astype(str)
    adata.obs["dose_uM"] = pd.to_numeric(adata.obs["dose_uM"], errors="raise").astype(float)

    # 与 scPerturBench 的 condition1/condition2/dose 命名对齐。
    adata.obs["condition1"] = adata.obs["cell_name"].astype(str)
    adata.obs["condition2"] = adata.obs["drug"].astype(str)
    adata.obs["dose"] = adata.obs["dose_uM"].astype(float)
    adata.obs["source_file"] = file_name
    return adata


def matrix_to_numpy(matrix) -> np.ndarray:
    """把稀疏或稠密矩阵统一转成 float32 dense 矩阵。"""
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def match_prediction_rows(pred_x: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    """按固定随机种子采样或重采样，使预测行数严格等于真实目标行数。"""
    if pred_x.shape[0] == n_target:
        return pred_x.astype(np.float32, copy=False)
    if pred_x.shape[0] == 0:
        raise RuntimeError("Biolord 返回了空预测块，无法对齐真实目标细胞。")

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        pred_x.shape[0],
        size=n_target,
        replace=(n_target > pred_x.shape[0]),
    )
    return pred_x[indices].astype(np.float32, copy=False)


def clear_biolord_managers(model, adata_task) -> None:
    """尽量释放 scvi/biolord 注册的 AnnData manager。"""
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
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("开始执行 NIPS biolord pipeline")
    print(f"数据目录: {DATA_DIR}")
    print(f"预测输出目录: {PREDICT_DIR}")
    print(f"训练文件: {TRAIN_FILES}")
    print(f"目标文件: {TARGET_FILES}")
    print(f"control drug: {CONTROL_DRUG}")
    print(f"split seed: {SPLIT_RANDOM_SEED}")
    print("=" * 100)

    for target_file in TARGET_FILES:
        task_name = Path(target_file).stem
        output_path = PREDICT_DIR / f"{task_name}_pred.h5ad"
        model = None
        adata_task = None

        print("\n" + "#" * 100)
        print(f"开始处理目标任务: {target_file}")
        print("#" * 100)

        try:
            # ------------------------------------------------------------
            # 1. 读取训练集和当前目标集。validation 不再来自固定文件。
            # ------------------------------------------------------------
            all_adatas = []
            all_file_names = TRAIN_FILES + [target_file]

            for train_file in TRAIN_FILES:
                adata_train = read_nips_h5ad(train_file)
                all_adatas.append(adata_train)
                print(f"已读取训练集 {train_file}, shape={adata_train.shape}")

            adata_target_raw = read_nips_h5ad(target_file)
            all_adatas.append(adata_target_raw.copy())
            print(f"已读取目标集 {target_file}, shape={adata_target_raw.shape}")

            base_var_names = all_adatas[0].var_names
            for check_file, check_adata in zip(all_file_names[1:], all_adatas[1:]):
                if not base_var_names.equals(check_adata.var_names):
                    raise RuntimeError(
                        "检测到不同 h5ad 的基因顺序不一致，不能安全拼接。\n"
                        f"不一致文件: {check_file}"
                    )

            adata_task = ad.concat(
                all_adatas,
                join="inner",
                merge="same",
                index_unique=None,
            )
            adata_task.obs_names_make_unique()
            adata_task.var_names_make_unique()

            # ------------------------------------------------------------
            # 2. 按参考实现划分 OOD 和随机 train/valid。
            # 当前目标文件的非 control 是 OOD，其余全部非 OOD 随机 9:1 划分。
            # ------------------------------------------------------------
            meta = adata_task.obs.copy()
            ood_mask = (
                (meta["source_file"].astype(str).to_numpy() == target_file)
                & (meta["condition2"].astype(str).to_numpy() != CONTROL_DRUG)
            )
            meta["split"] = None
            meta.loc[ood_mask, "split"] = "ood"

            non_ood_mask = ~ood_mask
            split_rng = np.random.RandomState(SPLIT_RANDOM_SEED)
            meta.loc[non_ood_mask, "split"] = split_rng.choice(
                ["train", "valid"],
                size=int(non_ood_mask.sum()),
                replace=True,
                p=[0.9, 0.1],
            )

            adata_task.obs = meta
            adata_task.obs["condition1"] = pd.Categorical(adata_task.obs["condition1"])
            adata_task.obs["condition2"] = pd.Categorical(adata_task.obs["condition2"])
            adata_task.obs["split"] = pd.Categorical(
                adata_task.obs["split"],
                categories=["train", "valid", "ood"],
            )
            adata_task.obs["source_file"] = pd.Categorical(adata_task.obs["source_file"])
            adata_task.obs["dose"] = pd.to_numeric(
                adata_task.obs["dose"], errors="raise"
            ).astype(float)

            print("任务数据构建完成")
            print(f"总 shape: {adata_task.shape}")
            print(f"train 细胞数: {(adata_task.obs['split'] == 'train').sum()}")
            print(f"valid 细胞数: {(adata_task.obs['split'] == 'valid').sum()}")
            print(f"ood   细胞数: {(adata_task.obs['split'] == 'ood').sum()}")

            # ------------------------------------------------------------
            # 3. 按 scPerturBench 参考配置注册并训练 biolord。
            # NIPS 没有 logNor layer，因此 layer=None，直接使用 X。
            # ------------------------------------------------------------
            biolord.Biolord.setup_anndata(
                adata=adata_task,
                ordered_attributes_keys="dose",
                categorical_attributes_keys=["condition1", "condition2"],
                layer=None,
            )

            model = biolord.Biolord(
                adata=adata_task,
                n_latent=N_LATENT,
                model_name=task_name,
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
                batch_size=TRAIN_BATCH_SIZE,
                plan_kwargs=TRAIN_PLAN_KWARGS,
                early_stopping=True,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
                enable_checkpointing=False,
            )
            print("模型训练完成")

            # ------------------------------------------------------------
            # 4. 使用参考实现的 compute_prediction_adata 生成预测。
            # 每个目标 cell_name 使用自身真实 control 作为 source。
            # ------------------------------------------------------------
            output_blocks = []

            real_ctrl = adata_target_raw[
                adata_target_raw.obs["condition2"].astype(str) == CONTROL_DRUG
            ].copy()
            if real_ctrl.n_obs == 0:
                raise RuntimeError(f"{target_file} 中找不到 control={CONTROL_DRUG} 细胞。")
            real_ctrl.obs["pred_or_real"] = "real_ctrl"
            output_blocks.append(real_ctrl)

            target_cell_names = pd.Index(
                adata_target_raw.obs["condition1"].astype(str)
            ).drop_duplicates()

            for cell_name in target_cell_names:
                print("-" * 80)
                print(f"开始预测 cell_name: {cell_name}")

                target_cell = adata_target_raw[
                    adata_target_raw.obs["condition1"].astype(str) == str(cell_name)
                ].copy()
                source_mask = (
                    (adata_task.obs["source_file"].astype(str).to_numpy() == target_file)
                    & (adata_task.obs["condition1"].astype(str).to_numpy() == str(cell_name))
                    & (adata_task.obs["condition2"].astype(str).to_numpy() == CONTROL_DRUG)
                )
                source_indices = np.flatnonzero(source_mask)
                if len(source_indices) == 0:
                    raise RuntimeError(f"{target_file}/{cell_name} 没有可用 control source。")

                adata_source = adata_task[source_indices].copy()
                adata_preds = model.compute_prediction_adata(
                    adata_task,
                    adata_source,
                    target_attributes=["condition2", "dose"],
                )

                target_pairs = (
                    target_cell.obs[["condition2", "dose"]]
                    .assign(
                        condition2=lambda df: df["condition2"].astype(str),
                        dose=lambda df: pd.to_numeric(df["dose"], errors="raise").astype(float),
                    )
                    .query("condition2 != @CONTROL_DRUG")
                    .drop_duplicates()
                )

                for pair_rank, row in enumerate(target_pairs.itertuples(index=False), start=1):
                    target_drug = str(row.condition2)
                    target_dose = float(row.dose)

                    stim_mask = (
                        (target_cell.obs["condition2"].astype(str).to_numpy() == target_drug)
                        & np.isclose(
                            pd.to_numeric(target_cell.obs["dose"], errors="raise").to_numpy(dtype=float),
                            target_dose,
                        )
                    )
                    stim_skeleton = target_cell[stim_mask].copy()
                    if stim_skeleton.n_obs == 0:
                        continue

                    pred_obs = adata_preds.obs
                    pred_mask = pred_obs["condition2"].astype(str).to_numpy() == target_drug
                    if "condition1" in pred_obs.columns:
                        pred_mask &= pred_obs["condition1"].astype(str).to_numpy() == str(cell_name)

                    pred_dose = pd.to_numeric(
                        pred_obs["dose"], errors="raise"
                    ).to_numpy(dtype=float)
                    pred_mask &= np.isclose(pred_dose, target_dose)

                    pred_block = adata_preds[pred_mask].copy()
                    if pred_block.n_obs == 0:
                        raise RuntimeError(
                            "缺少预测块: "
                            f"target_file={target_file}, cell_name={cell_name}, "
                            f"drug={target_drug}, dose={target_dose}"
                        )

                    pred_x = matrix_to_numpy(pred_block.X)
                    pred_x = match_prediction_rows(
                        pred_x,
                        n_target=stim_skeleton.n_obs,
                        seed=RANDOM_SEED + pair_rank,
                    )
                    pred_x = np.clip(pred_x, a_min=0.0, a_max=None)

                    if pred_x.shape[1] != stim_skeleton.n_vars:
                        raise RuntimeError(
                            "预测基因数与真实骨架不一致: "
                            f"pred={pred_x.shape[1]}, target={stim_skeleton.n_vars}"
                        )

                    if sp.issparse(stim_skeleton.X):
                        pred_x_out = sp.csr_matrix(pred_x)
                    else:
                        pred_x_out = pred_x

                    adata_pred_block = ad.AnnData(
                        X=pred_x_out,
                        obs=stim_skeleton.obs.copy(),
                        var=stim_skeleton.var.copy(),
                    )
                    adata_pred_block.obs["pred_or_real"] = "pred"
                    output_blocks.append(adata_pred_block)

                    print(
                        f"已生成预测块: drug={target_drug}, dose={target_dose}, "
                        f"n_obs={adata_pred_block.n_obs}"
                    )

                    del stim_skeleton, pred_block, pred_x, adata_pred_block
                    cleanup_memory()

                del target_cell, adata_source, adata_preds
                cleanup_memory()

            # ------------------------------------------------------------
            # 5. 拼接最终 cell-eval 输入文件：真实 control + 预测 stimulated。
            # ------------------------------------------------------------
            adata_output = ad.concat(
                output_blocks,
                join="inner",
                merge="same",
                index_unique=None,
            )
            adata_output.obs_names_make_unique()

            if adata_output.n_obs != adata_target_raw.n_obs:
                raise RuntimeError(
                    "最终输出细胞数与真实目标文件不一致: "
                    f"pred_n_obs={adata_output.n_obs}, real_n_obs={adata_target_raw.n_obs}"
                )
            if adata_output.n_vars != adata_target_raw.n_vars:
                raise RuntimeError(
                    "最终输出基因数与真实目标文件不一致: "
                    f"pred_n_vars={adata_output.n_vars}, real_n_vars={adata_target_raw.n_vars}"
                )

            pred_labels = set(adata_output.obs["pred_or_real"].astype(str))
            if pred_labels != {"real_ctrl", "pred"}:
                raise RuntimeError(f"pred_or_real 标签异常: {sorted(pred_labels)}")

            adata_output.write_h5ad(output_path)
            print("-" * 80)
            print(f"任务完成: {target_file}")
            print(f"输出文件: {output_path}")
            print(f"输出 shape: {adata_output.shape}")

            del output_blocks, adata_output, adata_target_raw
            cleanup_memory()

        finally:
            clear_biolord_managers(model, adata_task)
            del model, adata_task
            cleanup_memory()

    print("\n" + "=" * 100)
    print("全部 NIPS biolord 目标任务完成")
    print("=" * 100)
