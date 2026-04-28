"""
Tahoe biolord baseline（按 scPerturBench OOD 口径重写）。

运行代码的命令：
1. 在本文件中设置：
   RUN_TRAINING = True
   RUN_FINAL_MERGE = True 或 False
   START_TARGET_INDEX / END_TARGET_INDEX 指向要运行的目标 cell_type 范围
2. 执行：
   CUDA_VISIBLE_DEVICES=7 python /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_tahoe/biolord/biolord_pipeline.py

续跑命令：
1. 保持 SKIP_EXISTING_PRED_FILES = True
2. 修改 START_TARGET_INDEX / END_TARGET_INDEX，只覆盖还没跑完的目标 cell_type
3. 执行同一个命令：
   CUDA_VISIBLE_DEVICES=7 python /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_tahoe/biolord/biolord_pipeline.py

执行最终 merge 的命令：
1. 设置：
   RUN_TRAINING = False
   RUN_FINAL_MERGE = True
   START_TARGET_INDEX = 0
   END_TARGET_INDEX = len(TARGET_CELL_TYPES) - 1
2. 执行同一个命令：
   CUDA_VISIBLE_DEVICES=7 python /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_tahoe/biolord/biolord_pipeline.py

核心协议：
1. 每个目标 cell_type 单独训练一个 biolord 模型；
2. 训练池为全量 50 个 cell_type × 全量 drug；
3. 只把当前目标 cell_type 的非 control 细胞设为 OOD；
4. 其余所有细胞，包括当前目标 cell_type 的 control，随机 9:1 划分为 train/valid；
5. 预测调用 biolord 原生 model.compute_prediction_adata()；
6. 输出时把所有小于 2 的预测值置为 0，并按真实 stimulated block 的 sparse/dense 格式保存。
"""

from pathlib import Path
import ctypes
import gc
import logging
import os
import warnings

import anndata as ad
import biolord
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None


# =========================
# 运行控制
# =========================
# TARGET_CELL_TYPES 的下标范围。建议一次只跑一个 target，降低单进程内存峰值。
START_TARGET_INDEX = 0
END_TARGET_INDEX = 6

# 是否执行训练和临时预测生成。
RUN_TRAINING = True

# 是否把 tmp_predictions 合并成 cell-eval 使用的最终 h5ad。
RUN_FINAL_MERGE = True

# 续跑时保持 True，已存在的 drug 预测块会跳过。
SKIP_EXISTING_PRED_FILES = True

# 为避免误删已生成结果，这里只保留防呆开关，不在脚本里自动清空输出目录。
RESET_OUTPUT_DIRS = False

# 是否打印 RSS 内存日志。
ENABLE_RSS_LOG = True

# Linux/glibc 下尝试释放空闲堆内存。
TRY_MALLOC_TRIM = True


# =========================
# 基础路径配置
# =========================
# 路径保持当前脚本中的设置；如服务器路径不同，请手动修改这里。
SPLIT_BY_DRUG_DIR = Path("/data1/fanpeishan/STATE/for_state/data/split_by_drug")
DRUG_NAME_CSV = Path("/data1/fanpeishan/STATE/for_state/about_baseline/docs/drug_name_list.csv")

OUTPUT_DIR = Path("/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_tahoe/biolord/outputs")
TEMP_PRED_DIR = OUTPUT_DIR / "tmp_predictions"
TRAIN_LOG_DIR = OUTPUT_DIR / "biolord_train_logs"

CONTROL_DRUG = "DMSO_TF"


# =========================
# scPerturBench biolord 训练参数
# =========================
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


# =========================
# Tahoe cell_type 设置
# =========================
def cells(start, end):
    return [f"c{i}" for i in range(start, end + 1)]


ALL_CELL_TYPES = cells(0, 49)
TARGET_CELL_TYPES = ["c4", "c17", "c18", "c19", "c20", "c22", "c23", "c24", "c31", "c38"]


# =========================
# 日志、内存和通用工具
# =========================
logging.getLogger("scvi").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Observation names are not unique")
warnings.filterwarnings("ignore", message=".*is a view of an AnnData object.*")
torch.set_float32_matmul_precision("medium")


def log_rss(tag):
    """打印当前进程 RSS，方便定位内存峰值。"""
    if not ENABLE_RSS_LOG:
        return
    if psutil is None:
        print(f"[MEM] {tag}: psutil 未安装")
        return
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[MEM] {tag}: RSS = {rss_gb:.2f} GB")


def try_malloc_trim():
    """Linux/glibc 下尝试把已释放内存还给系统。"""
    if not TRY_MALLOC_TRIM or os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def cleanup_memory():
    """集中清理 Python 和 CUDA 缓存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try_malloc_trim()


def read_drug_name_list():
    """读取 drug_index 到 drug name 的映射，并保持原始顺序。"""
    return pd.read_csv(DRUG_NAME_CSV)["drug_name"].astype(str).tolist()


def get_split_file(cell_type, drug_index):
    """返回已经按 cell_type 和 drug_index 拆分后的 h5ad 路径。"""
    return (
        SPLIT_BY_DRUG_DIR
        / f"celltype_{cell_type}"
        / f"celltype_{cell_type}_drugindex_{drug_index}.h5ad"
    )


def get_temp_pred_path(target_cell_type, drug_index):
    """返回单个 target cell_type / drug_index 的临时预测文件路径。"""
    pred_dir = TEMP_PRED_DIR / target_cell_type
    pred_dir.mkdir(parents=True, exist_ok=True)
    return pred_dir / f"{target_cell_type}_drugindex_{drug_index}_pred.h5ad"


def read_one_block(cell_type, drug_index):
    """读取一个 cell_type + drug_index block，并统一关键 obs 类型。"""
    path = get_split_file(cell_type, drug_index)
    if not path.exists():
        raise FileNotFoundError(f"找不到拆分数据文件: {path}")

    adata = sc.read_h5ad(path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs = adata.obs.copy()

    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = cell_type
    if "drug_index" not in adata.obs.columns:
        adata.obs["drug_index"] = drug_index

    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.obs["drug_index"] = adata.obs["drug_index"].astype(int)
    if "drug" in adata.obs.columns:
        adata.obs["drug"] = adata.obs["drug"].astype(str)

    return adata


def adata_to_numpy(x):
    """把 sparse/dense 表达矩阵统一转成 float32 dense array。"""
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def to_template_matrix(matrix, template_x):
    """按真实 stimulated block 的 sparse/dense 格式保存预测矩阵。"""
    matrix = np.asarray(matrix, dtype=np.float32)
    if sp.issparse(template_x):
        return sp.csr_matrix(matrix)
    return matrix


def clear_biolord_managers(model, adata_task=None):
    """best-effort 清理 biolord/scvi 注册的 AnnDataManager。"""
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


def selected_target_items():
    """根据 START/END 配置返回本轮要处理的 target cell_type。"""
    if START_TARGET_INDEX < 0:
        raise ValueError("START_TARGET_INDEX 不能小于 0")

    end_index = len(TARGET_CELL_TYPES) - 1 if END_TARGET_INDEX is None else END_TARGET_INDEX
    if end_index >= len(TARGET_CELL_TYPES):
        raise ValueError(
            f"END_TARGET_INDEX 超出范围: {end_index}, "
            f"最大允许值为 {len(TARGET_CELL_TYPES) - 1}"
        )
    if end_index < START_TARGET_INDEX:
        return []

    return [
        (idx, cell_type)
        for idx, cell_type in enumerate(TARGET_CELL_TYPES)
        if START_TARGET_INDEX <= idx <= end_index
    ]


# =========================
# 数据构造
# =========================
def build_target_cell_adata(target_cell_type, control_drug_index, n_drugs):
    """
    构造当前 target cell_type 的 biolord 任务数据。

    scPerturBench biolord 口径：
    - 当前 target cell_type 的非 control 为 ood；
    - 其余所有细胞，包括 target cell_type 的 control，随机 9:1 划为 train/valid。
    """
    all_blocks = []

    print("读取全量 Tahoe block 并构建当前 target 任务 AnnData ...")
    print(f"target cell_type: {target_cell_type}")
    log_rss(f"{target_cell_type}:build:start")

    total_blocks = len(ALL_CELL_TYPES) * n_drugs
    block_count = 0

    for cell_type in ALL_CELL_TYPES:
        print(f"  读取 cell_type={cell_type}")
        for drug_index in range(n_drugs):
            block_count += 1
            block = read_one_block(cell_type, drug_index)
            all_blocks.append(block)

            if block_count % 200 == 0:
                print(f"    已读取 block: {block_count}/{total_blocks}")
                log_rss(f"{target_cell_type}:after_{block_count}_blocks")

    adata = ad.concat(
        all_blocks,
        join="inner",
        merge="same",
        index_unique=None,
    )
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    del all_blocks
    cleanup_memory()

    cell_values = adata.obs["cell_type"].astype(str).to_numpy()
    drug_values = adata.obs["drug_index"].astype(int).to_numpy()
    ood_mask = (cell_values == target_cell_type) & (drug_values != control_drug_index)
    ood_count = int(ood_mask.sum())
    if ood_count == 0:
        raise RuntimeError(f"{target_cell_type} 没有可作为 OOD 的非 control 细胞")

    np.random.seed(SPLIT_RANDOM_SEED)
    split_values = np.empty(adata.n_obs, dtype=object)
    split_values[ood_mask] = "ood"
    split_values[~ood_mask] = np.random.choice(
        ["train", "valid"],
        size=int((~ood_mask).sum()),
        replace=True,
        p=[0.9, 0.1],
    )

    adata.obs["split"] = pd.Categorical(
        split_values,
        categories=["train", "valid", "ood"],
    )
    adata.obs["cell_type"] = pd.Categorical(adata.obs["cell_type"].astype(str))
    adata.obs["drug_index"] = pd.Categorical(adata.obs["drug_index"].astype(int))

    target_control_count = int(
        (
            (adata.obs["cell_type"].astype(str).to_numpy() == target_cell_type)
            & (adata.obs["drug_index"].astype(int).to_numpy() == control_drug_index)
        ).sum()
    )
    if target_control_count == 0:
        raise RuntimeError(f"{target_cell_type} 没有 control source 细胞")

    print("当前 target 任务 AnnData 构建完成")
    print(f"adata shape: {adata.shape}")
    print(f"train cells: {(adata.obs['split'] == 'train').sum()}")
    print(f"valid cells: {(adata.obs['split'] == 'valid').sum()}")
    print(f"ood cells  : {(adata.obs['split'] == 'ood').sum()}")
    print(f"target control cells: {target_control_count}")
    log_rss(f"{target_cell_type}:build:end")

    return adata


# =========================
# 预测与输出
# =========================
def resize_prediction_to_skeleton(pred_x, stim_skeleton, random_seed):
    """把 biolord 预测块采样到真实 stimulated block 的细胞数。"""
    n_target = stim_skeleton.n_obs
    if pred_x.shape[0] == n_target:
        return pred_x.astype(np.float32, copy=False)

    if pred_x.shape[0] == 0:
        raise RuntimeError("biolord 返回了空预测矩阵")

    rng = np.random.default_rng(random_seed)
    sampled_idx = rng.choice(
        pred_x.shape[0],
        size=n_target,
        replace=(n_target > pred_x.shape[0]),
    )
    return pred_x[sampled_idx].astype(np.float32, copy=False)


def build_prediction_from_skeleton(pred_x, stim_skeleton):
    """用真实 stimulated block 的 obs/var/矩阵格式组装预测 AnnData。"""
    pred_x = np.asarray(pred_x, dtype=np.float32)
    pred_x[pred_x < 2.0] = 0.0

    adata_pred = ad.AnnData(
        X=to_template_matrix(pred_x, stim_skeleton.X),
        obs=stim_skeleton.obs.copy(),
        var=stim_skeleton.var.copy(),
    )
    adata_pred.obs["pred_or_real"] = "pred"
    return adata_pred


def validate_prediction_block(adata_pred, stim_skeleton, target_cell_type, drug_index):
    """保存前做基本结构和数值检查。"""
    if adata_pred.n_obs != stim_skeleton.n_obs:
        raise RuntimeError(
            f"{target_cell_type}/drug_index={drug_index} 行数不一致: "
            f"pred={adata_pred.n_obs}, target={stim_skeleton.n_obs}"
        )
    if adata_pred.n_vars != stim_skeleton.n_vars:
        raise RuntimeError(
            f"{target_cell_type}/drug_index={drug_index} 基因数不一致: "
            f"pred={adata_pred.n_vars}, target={stim_skeleton.n_vars}"
        )

    pred_x = adata_to_numpy(adata_pred.X)
    if np.isnan(pred_x).any():
        raise RuntimeError(f"{target_cell_type}/drug_index={drug_index} 预测矩阵包含 NaN")
    if ((pred_x > 0.0) & (pred_x < 2.0)).any():
        raise RuntimeError(f"{target_cell_type}/drug_index={drug_index} 预测矩阵仍包含 0 到 2 之间的非零值")


def all_prediction_files_exist_for_target(target_cell_type, n_drugs, control_drug_index):
    """检查当前 target cell_type 的所有非 control 临时预测是否已经存在。"""
    for drug_index in range(n_drugs):
        if drug_index == control_drug_index:
            continue
        if not get_temp_pred_path(target_cell_type, drug_index).exists():
            return False
    return True


def train_and_predict_one_target(target_rank, target_cell_type, drug_name_list, control_drug_index):
    """训练一个 target cell_type 的 biolord，并保存全部 drug 的临时预测。"""
    print("\n" + "#" * 100)
    print(f"开始处理 target cell_type: {target_cell_type} (target_index={target_rank})")
    print("#" * 100)
    log_rss(f"{target_cell_type}:before_start")

    n_drugs = len(drug_name_list)
    if SKIP_EXISTING_PRED_FILES and all_prediction_files_exist_for_target(
        target_cell_type,
        n_drugs,
        control_drug_index,
    ):
        print(f"{target_cell_type} 的全部非 control 临时预测都已存在，跳过训练")
        return

    model = None
    adata_task = None

    try:
        adata_task = build_target_cell_adata(
            target_cell_type=target_cell_type,
            control_drug_index=control_drug_index,
            n_drugs=n_drugs,
        )

        print("开始 setup_anndata ...")
        biolord.Biolord.setup_anndata(
            adata=adata_task,
            categorical_attributes_keys=["cell_type", "drug_index"],
            layer=None,
        )
        log_rss(f"{target_cell_type}:after_setup_anndata")

        print("开始构建 biolord 模型 ...")
        model = biolord.Biolord(
            adata=adata_task,
            n_latent=N_LATENT,
            model_name=f"tahoe_biolord_{target_cell_type}",
            module_params=MODULE_PARAMS,
            train_classifiers=False,
            split_key="split",
            train_split="train",
            valid_split="valid",
            test_split="ood",
        )
        log_rss(f"{target_cell_type}:after_model_init")

        print("开始训练 biolord 模型 ...")
        model.train(
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            plan_kwargs=TRAIN_PLAN_KWARGS,
            early_stopping=True,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
            enable_checkpointing=False,
        )
        print("biolord 模型训练完成")
        log_rss(f"{target_cell_type}:after_train")

        source_mask = (
            (adata_task.obs["cell_type"].astype(str).to_numpy() == target_cell_type)
            & (adata_task.obs["drug_index"].astype(int).to_numpy() == control_drug_index)
        )
        source_indices = np.flatnonzero(source_mask)
        if len(source_indices) == 0:
            raise RuntimeError(f"{target_cell_type} 没有可用于预测的 control source")

        adata_source = adata_task[source_indices].copy()
        print(f"source control cells: {adata_source.n_obs}")
        print("开始调用 model.compute_prediction_adata() ...")
        adata_preds = model.compute_prediction_adata(
            adata_task,
            adata_source,
            target_attributes=["drug_index"],
        )
        adata_preds.obs_names_make_unique()
        log_rss(f"{target_cell_type}:after_compute_prediction")

        pred_drug_values = adata_preds.obs["drug_index"].astype(int).to_numpy()

        for drug_index, drug_name in tqdm(
            list(enumerate(drug_name_list)),
            desc=f"Saving predictions for {target_cell_type}",
        ):
            if drug_index == control_drug_index:
                continue

            save_path = get_temp_pred_path(target_cell_type, drug_index)
            if SKIP_EXISTING_PRED_FILES and save_path.exists():
                continue

            stim_skeleton = read_one_block(target_cell_type, drug_index)
            pred_for_drug = adata_preds[pred_drug_values == drug_index].copy()
            if pred_for_drug.n_obs == 0:
                raise RuntimeError(
                    f"biolord 没有生成 target={target_cell_type}, "
                    f"drug={drug_name}, drug_index={drug_index} 的预测"
                )

            pred_x = adata_to_numpy(pred_for_drug.X)
            pred_x = resize_prediction_to_skeleton(
                pred_x,
                stim_skeleton=stim_skeleton,
                random_seed=SPLIT_RANDOM_SEED + target_rank * 1000 + drug_index,
            )
            adata_pred = build_prediction_from_skeleton(pred_x, stim_skeleton)
            validate_prediction_block(
                adata_pred=adata_pred,
                stim_skeleton=stim_skeleton,
                target_cell_type=target_cell_type,
                drug_index=drug_index,
            )
            adata_pred.write_h5ad(save_path)

            del stim_skeleton, pred_for_drug, pred_x, adata_pred
            cleanup_memory()

        del adata_source, adata_preds
        cleanup_memory()
        log_rss(f"{target_cell_type}:after_save_all_predictions")

    finally:
        clear_biolord_managers(model, adata_task)
        del model, adata_task
        cleanup_memory()
        log_rss(f"{target_cell_type}:after_cleanup")


def merge_one_target(target_cell_type, drug_name_list, control_drug_index):
    """把当前 target cell_type 的临时预测合并成 cell-eval 使用的最终 h5ad。"""
    print("\n" + "=" * 100)
    print(f"开始 merge target cell_type: {target_cell_type}")
    print("=" * 100)
    log_rss(f"{target_cell_type}:merge:start")

    output_blocks = []

    ctrl_real = read_one_block(target_cell_type, control_drug_index)
    ctrl_real.obs = ctrl_real.obs.copy()
    ctrl_real.obs["pred_or_real"] = "real_ctrl"
    output_blocks.append(ctrl_real)

    for drug_index, _ in tqdm(
        list(enumerate(drug_name_list)),
        desc=f"Merging {target_cell_type}",
    ):
        if drug_index == control_drug_index:
            continue

        pred_path = get_temp_pred_path(target_cell_type, drug_index)
        if not pred_path.exists():
            raise FileNotFoundError(
                f"缺少临时预测文件，不能 merge: {pred_path}\n"
                "请先跑完该 target cell_type 的全部非 control drug 预测。"
            )
        output_blocks.append(sc.read_h5ad(pred_path))

    final_adata = ad.concat(
        output_blocks,
        join="inner",
        merge="same",
        index_unique=None,
    )
    final_adata.obs_names_make_unique()

    final_path = OUTPUT_DIR / f"{target_cell_type}_predicted.h5ad"
    final_adata.write_h5ad(final_path)

    print(f"已生成: {final_path}")
    print(f"最终 shape: {final_adata.shape}")
    log_rss(f"{target_cell_type}:merge:after_write")

    del output_blocks, ctrl_real, final_adata
    cleanup_memory()
    log_rss(f"{target_cell_type}:merge:after_cleanup")


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    if RESET_OUTPUT_DIRS:
        raise RuntimeError(
            "当前脚本不自动清空输出目录。请手动确认后再删除历史 tmp_predictions 或最终输出。"
        )

    np.random.seed(SPLIT_RANDOM_SEED)
    torch.manual_seed(TORCH_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    drug_name_list = read_drug_name_list()
    drug_to_index = {drug_name: idx for idx, drug_name in enumerate(drug_name_list)}
    if CONTROL_DRUG not in drug_to_index:
        raise KeyError(f"drug_name_list 中找不到 control drug: {CONTROL_DRUG}")
    control_drug_index = drug_to_index[CONTROL_DRUG]

    target_items = selected_target_items()
    if not target_items:
        raise RuntimeError("当前 START/END 配置下，没有选中任何 target cell_type")

    print("=" * 100)
    print("开始执行 Tahoe biolord scPerturBench OOD 流程")
    print(f"control drug: {CONTROL_DRUG} (drug_index={control_drug_index})")
    print(f"全部 cell_type 数量: {len(ALL_CELL_TYPES)}")
    print(f"目标 cell_type 数量: {len(TARGET_CELL_TYPES)}")
    print(f"本轮 target cell_type: {target_items}")
    print(f"drug 数量: {len(drug_name_list)}")
    print(f"RUN_TRAINING: {RUN_TRAINING}")
    print(f"RUN_FINAL_MERGE: {RUN_FINAL_MERGE}")
    print(f"max_epochs: {MAX_EPOCHS}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"early_stopping_patience: {EARLY_STOPPING_PATIENCE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 100)
    log_rss("script_start")

    if RUN_TRAINING:
        for target_rank, target_cell_type in target_items:
            train_and_predict_one_target(
                target_rank=target_rank,
                target_cell_type=target_cell_type,
                drug_name_list=drug_name_list,
                control_drug_index=control_drug_index,
            )

    if RUN_FINAL_MERGE:
        for _, target_cell_type in target_items:
            merge_one_target(
                target_cell_type=target_cell_type,
                drug_name_list=drug_name_list,
                control_drug_index=control_drug_index,
            )

    print("\n" + "=" * 100)
    print("Tahoe biolord 流程完成")
    print("=" * 100)
    log_rss("script_end")
