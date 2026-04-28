"""
Tahoe biolord baseline aligned with scPerturBench OOD biolord.

运行方式示例：
  cd /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_tahoe/biolord
  CUDA_VISIBLE_DEVICES=7 python scripts/biolord_pipeline.py

核心协议：
1. 采用 Tahoe no-validation 口径：
   - 40 个 train cell_type 用作训练上下文；
   - 10 个 holdout cell_type 逐个作为预测目标。
2. 每个 target cell_type 单独训练一个 biolord 模型。
3. 当前 target cell_type 的非 control 细胞标记为 OOD。
4. 其它所有细胞，包括当前 target cell_type 的 control，按 9:1 随机划分 train/valid。
5. 用当前 target cell_type 的 control 作为 source，逐 drug 做低内存属性替换预测。
6. biolord 为每个 observation 学习一个 latent 参数，Tahoe 全量细胞无法直接进模型；
   因此训练任务按 cell_type × drug block 做固定随机下采样。
7. 最终只需要 outputs/{cell_type}_predicted.h5ad；tmp_predictions 只是断点续跑和低内存 merge 的中间文件。
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


# =========================
# 运行控制
# =========================
# TARGET_CELL_TYPES 的下标范围。Tahoe 全量任务较重，建议一次只跑一个或少数 target。
START_TARGET_INDEX = 0
END_TARGET_INDEX = 9

# 是否执行训练和临时预测生成。
RUN_TRAINING = True

# 是否把 tmp_predictions 合并成 cell-eval 使用的最终 h5ad。
RUN_FINAL_MERGE = True

# 续跑时保持 True，已存在的 drug 预测块会跳过。
SKIP_EXISTING_PRED_FILES = True

# 是否打印 RSS 内存日志。
ENABLE_RSS_LOG = True

# Linux/glibc 下尝试释放空闲堆内存。
TRY_MALLOC_TRIM = True


# =========================
# 基础路径配置
# =========================
# 路径保持 Tahoe scVIDR 成功脚本的服务器风格；如服务器路径不同，请只改这里。
SPLIT_BY_DRUG_DIR = Path("/data1/fanpeishan/STATE/for_state/data/split_by_drug")
DRUG_NAME_CSV = Path("/data1/fanpeishan/STATE/for_state/about_baseline/docs/drug_name_list.csv")

OUTPUT_DIR = Path(
    "/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_tahoe/biolord/outputs"
)
TEMP_PRED_DIR = OUTPUT_DIR / "tmp_predictions"

CONTROL_DRUG = "DMSO_TF"


# =========================
# scPerturBench biolord 参数
# =========================
SPLIT_RANDOM_SEED = 1116
TORCH_RANDOM_SEED = 42
PREDICTION_ZERO_THRESHOLD = 2.0

N_LATENT = 256
BATCH_SIZE = 128
PREDICT_CELL_BATCH_SIZE = 4096
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 20
CHECK_VAL_EVERY_N_EPOCH = 5

# biolord 的 latent 参数规模约为 n_obs × N_LATENT。
# Tahoe 当前 target 任务全量约 6800 万细胞，会在 model.to(cuda) 时单 latent 参数就申请约 65GB。
# 这里保留所有 cell_type/drug 的覆盖，但限制每个 block 进入训练任务的细胞数。
MAX_CELLS_PER_BLOCK = 64
MAX_TARGET_CONTROL_CELLS = 4096

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


TRAIN_CELL_TYPES = (
    cells(0, 3)
    + cells(5, 16)
    + ["c21"]
    + cells(25, 30)
    + cells(32, 37)
    + cells(39, 49)
)

TARGET_CELL_TYPES = ["c4", "c17", "c18", "c23", "c38", "c19", "c20", "c22", "c24", "c31"]


# =========================
# 日志、内存和通用工具
# =========================
logging.getLogger("scvi").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Observation names are not unique")
warnings.filterwarnings("ignore", message=".*is a view of an AnnData object.*")
warnings.filterwarnings("ignore", category=FutureWarning)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")

try:
    import psutil
except ImportError:
    psutil = None


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
    return pd.read_csv(DRUG_NAME_CSV)["drug_name"].astype(str).str.strip().tolist()


def get_split_file(cell_type, drug_index):
    """返回按 cell_type 和 drug_index 拆分后的 h5ad 路径。"""
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


def read_one_block(cell_type, drug_index, expected_drug_name=None):
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

    if "drug" not in adata.obs.columns:
        if expected_drug_name is None:
            raise KeyError(f"{path} 缺少 obs['drug']，且没有 expected_drug_name 可补充")
        adata.obs["drug"] = expected_drug_name

    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.obs["drug_index"] = adata.obs["drug_index"].astype(int)
    adata.obs["drug"] = adata.obs["drug"].astype(str).str.strip()

    if expected_drug_name is not None:
        observed_drugs = set(adata.obs["drug"].unique().tolist())
        if observed_drugs != {expected_drug_name}:
            raise ValueError(
                f"{path} 的 obs['drug'] 与 drug_name_list 不一致: "
                f"observed={sorted(observed_drugs)}, expected={expected_drug_name}"
            )

    return adata


def stable_block_seed(cell_type, drug_index):
    """为每个 cell_type/drug block 生成稳定随机种子。"""
    cell_id = int(str(cell_type).lstrip("c"))
    return SPLIT_RANDOM_SEED + cell_id * 1000 + int(drug_index)


def downsample_block(adata, max_cells, random_seed):
    """按固定 seed 对单个 block 下采样，避免 biolord per-observation latent 爆显存。"""
    if max_cells is None or adata.n_obs <= max_cells:
        return adata

    rng = np.random.default_rng(random_seed)
    sampled_idx = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    return adata[sampled_idx].copy()


def adata_to_numpy(x):
    """把 sparse/dense 表达矩阵统一转成 float32 dense array。"""
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def to_template_matrix(matrix, template_x):
    """按真实 skeleton 的 sparse/dense 格式保存预测矩阵。"""
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


def clear_biolord_temp_manager(model, adata_temp=None):
    """best-effort 清理预测阶段临时 AnnDataManager，不清理主训练 manager。"""
    if adata_temp is None:
        return

    if model is not None:
        try:
            model.deregister_manager(adata_temp)
        except Exception:
            pass

    try:
        biolord.Biolord.deregister_manager(adata_temp)
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


def all_prediction_files_exist_for_target(target_cell_type, drug_name_list, control_drug_index):
    """检查当前 target cell_type 的全部非 control 临时预测是否已经存在。"""
    for drug_index, _ in enumerate(drug_name_list):
        if drug_index == control_drug_index:
            continue
        if not get_temp_pred_path(target_cell_type, drug_index).exists():
            return False
    return True


# =========================
# 数据构造
# =========================
def build_target_cell_adata(target_cell_type, drug_name_list, control_drug_index):
    """
    构造当前 target cell_type 的 biolord 任务数据。

    数据池为 40 个 train cell_type 的全 drug block + 当前 target cell_type 的全 drug block。
    当前 target cell_type 的非 control block 为 OOD，其余全部随机 9:1 划为 train/valid。
    """
    all_blocks = []
    source_cell_types = TRAIN_CELL_TYPES + [target_cell_type]
    n_drugs = len(drug_name_list)

    print("读取 Tahoe block 并构建当前 target 任务 AnnData ...")
    print(f"target cell_type: {target_cell_type}")
    print(f"source cell_type 数量: {len(source_cell_types)}")
    log_rss(f"{target_cell_type}:build:start")

    total_blocks = len(source_cell_types) * n_drugs
    block_count = 0

    for cell_type in source_cell_types:
        print(f"  读取 cell_type={cell_type}")
        for drug_index, drug_name in enumerate(drug_name_list):
            block_count += 1
            block = read_one_block(
                cell_type=cell_type,
                drug_index=drug_index,
                expected_drug_name=drug_name,
            )
            max_cells = MAX_CELLS_PER_BLOCK
            if cell_type == target_cell_type and drug_index == control_drug_index:
                max_cells = MAX_TARGET_CONTROL_CELLS
            block = downsample_block(
                adata=block,
                max_cells=max_cells,
                random_seed=stable_block_seed(cell_type, drug_index),
            )
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
    adata.obs = adata.obs.copy()

    del all_blocks
    cleanup_memory()

    adata.obs["condition1"] = adata.obs["cell_type"].astype(str)
    adata.obs["condition2"] = adata.obs["drug"].astype(str)

    cell_values = adata.obs["condition1"].astype(str).to_numpy()
    drug_values = adata.obs["condition2"].astype(str).to_numpy()
    ood_mask = (cell_values == target_cell_type) & (drug_values != CONTROL_DRUG)
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
    adata.obs["condition1"] = pd.Categorical(adata.obs["condition1"].astype(str))
    adata.obs["condition2"] = pd.Categorical(adata.obs["condition2"].astype(str))

    target_control_count = int(
        ((cell_values == target_cell_type) & (drug_values == CONTROL_DRUG)).sum()
    )
    if target_control_count == 0:
        raise RuntimeError(f"{target_cell_type} 没有 control source 细胞")

    print("当前 target 任务 AnnData 构建完成")
    print(f"adata shape: {adata.shape}")
    print(f"max cells per block: {MAX_CELLS_PER_BLOCK}")
    print(f"max target control cells: {MAX_TARGET_CONTROL_CELLS}")
    print(f"train cells: {(adata.obs['split'] == 'train').sum()}")
    print(f"valid cells: {(adata.obs['split'] == 'valid').sum()}")
    print(f"ood cells  : {(adata.obs['split'] == 'ood').sum()}")
    print(f"target control cells: {target_control_count}")
    log_rss(f"{target_cell_type}:build:end")

    return adata


# =========================
# 预测与输出
# =========================
def sample_source_to_skeleton(adata_source, stim_skeleton, random_seed):
    """先把 target control source 采样到真实 stimulated block 的细胞数。"""
    n_target = stim_skeleton.n_obs
    if adata_source.n_obs == n_target:
        return adata_source.copy()

    if adata_source.n_obs == 0:
        raise RuntimeError("target control source 为空，无法预测")

    rng = np.random.default_rng(random_seed)
    sampled_idx = rng.choice(
        adata_source.n_obs,
        size=n_target,
        replace=(n_target > adata_source.n_obs),
    )
    return adata_source[sampled_idx].copy()


def repeat_attribute_value(value, n_obs):
    """把单个 categorical attribute 张量扩展到 n_obs 行。"""
    if value.dim() == 0:
        value = value.view(1)
    if value.dim() == 1:
        return value.unsqueeze(0).repeat(n_obs, 1)

    repeat_shape = [n_obs] + [1] * value.dim()
    return value.unsqueeze(0).repeat(*repeat_shape)


@torch.no_grad()
def predict_one_drug_low_memory(model, adata_task, adata_source, drug_name):
    """
    只预测一个目标 drug，避免 biolord.compute_prediction_adata() 展开全部 drug 类别。

    biolord 原生 compute_prediction_adata() 会遍历训练时注册的全部
    categorical_attributes_map["condition2"]，Tahoe 有 380 个 drug，会一次性物化
    380 个预测矩阵。这里复用其内部思路，但只把 source 的 condition2 张量替换成
    当前 drug 的张量，然后直接调用 module.get_expression()。
    """
    drug_mask = adata_task.obs["condition2"].astype(str).to_numpy() == drug_name
    drug_indices = np.flatnonzero(drug_mask)
    if len(drug_indices) == 0:
        raise RuntimeError(f"训练任务 AnnData 中找不到 drug={drug_name}")

    adata_drug_template = adata_task[drug_indices[:1]].copy()
    pred_blocks = []
    try:
        dataset_drug = model.get_dataset(adata_drug_template)
        target_value = dataset_drug["condition2"][0, :].to(model.device)

        for start in range(0, adata_source.n_obs, PREDICT_CELL_BATCH_SIZE):
            end = min(start + PREDICT_CELL_BATCH_SIZE, adata_source.n_obs)
            adata_source_chunk = adata_source[start:end].copy()

            try:
                dataset_source = model.get_dataset(adata_source_chunk)
                layer_key = "X" if "X" in dataset_source else "layers"
                n_obs = dataset_source[layer_key].size(0)

                dataset_comb = {}
                for key_dataset in dataset_source:
                    dataset_comb[key_dataset] = dataset_source[key_dataset].to(model.device)

                dataset_comb["condition2"] = repeat_attribute_value(target_value, n_obs).to(model.device)

                model.module.eval()
                pred, _ = model.module.get_expression(dataset_comb)
                pred_blocks.append(pred.detach().cpu().numpy().astype(np.float32))

            finally:
                clear_biolord_temp_manager(model, adata_source_chunk)
                del adata_source_chunk
                cleanup_memory()

        return np.concatenate(pred_blocks, axis=0).astype(np.float32)

    finally:
        clear_biolord_temp_manager(model, adata_drug_template)
        del adata_drug_template, pred_blocks
        cleanup_memory()


def build_prediction_from_skeleton(pred_x, stim_skeleton):
    """用真实 stimulated block 的 obs/var/矩阵格式组装预测 AnnData。"""
    pred_x = np.asarray(pred_x, dtype=np.float32)
    pred_x[pred_x < PREDICTION_ZERO_THRESHOLD] = 0.0

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
    if ((pred_x > 0.0) & (pred_x < PREDICTION_ZERO_THRESHOLD)).any():
        raise RuntimeError(
            f"{target_cell_type}/drug_index={drug_index} 预测矩阵仍包含 0 到 "
            f"{PREDICTION_ZERO_THRESHOLD:g} 之间的非零值"
        )


def train_and_predict_one_target(target_rank, target_cell_type, drug_name_list, control_drug_index):
    """训练一个 target cell_type 的 biolord，并保存全部非 control drug 的临时预测。"""
    print("\n" + "#" * 100)
    print(f"开始处理 target cell_type: {target_cell_type} (target_index={target_rank})")
    print("#" * 100)
    log_rss(f"{target_cell_type}:before_start")

    if SKIP_EXISTING_PRED_FILES and all_prediction_files_exist_for_target(
        target_cell_type=target_cell_type,
        drug_name_list=drug_name_list,
        control_drug_index=control_drug_index,
    ):
        print(f"{target_cell_type} 的全部非 control 临时预测都已存在，跳过训练")
        return

    model = None
    adata_task = None
    adata_source = None
    adata_source_drug = None

    try:
        adata_task = build_target_cell_adata(
            target_cell_type=target_cell_type,
            drug_name_list=drug_name_list,
            control_drug_index=control_drug_index,
        )

        print("开始 setup_anndata ...")
        biolord.Biolord.setup_anndata(
            adata=adata_task,
            categorical_attributes_keys=["condition1", "condition2"],
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
            (adata_task.obs["condition1"].astype(str).to_numpy() == target_cell_type)
            & (adata_task.obs["condition2"].astype(str).to_numpy() == CONTROL_DRUG)
        )
        source_indices = np.flatnonzero(source_mask)
        if len(source_indices) == 0:
            raise RuntimeError(f"{target_cell_type} 没有可用于预测的 control source")

        adata_source = adata_task[source_indices].copy()
        print(f"source control cells: {adata_source.n_obs}")

        for drug_index, drug_name in tqdm(
            list(enumerate(drug_name_list)),
            desc=f"Saving predictions for {target_cell_type}",
        ):
            if drug_index == control_drug_index:
                continue

            save_path = get_temp_pred_path(target_cell_type, drug_index)
            if SKIP_EXISTING_PRED_FILES and save_path.exists():
                continue

            stim_skeleton = read_one_block(
                cell_type=target_cell_type,
                drug_index=drug_index,
                expected_drug_name=drug_name,
            )

            adata_source_drug = sample_source_to_skeleton(
                adata_source=adata_source,
                stim_skeleton=stim_skeleton,
                random_seed=SPLIT_RANDOM_SEED + target_rank * 1000 + drug_index,
            )
            pred_x = predict_one_drug_low_memory(
                model=model,
                adata_task=adata_task,
                adata_source=adata_source_drug,
                drug_name=drug_name,
            )
            adata_pred = build_prediction_from_skeleton(pred_x, stim_skeleton)
            validate_prediction_block(
                adata_pred=adata_pred,
                stim_skeleton=stim_skeleton,
                target_cell_type=target_cell_type,
                drug_index=drug_index,
            )
            adata_pred.write_h5ad(save_path)

            clear_biolord_temp_manager(model, adata_source_drug)
            del (
                stim_skeleton,
                adata_source_drug,
                pred_x,
                adata_pred,
            )
            adata_source_drug = None
            cleanup_memory()
            log_rss(f"{target_cell_type}:drug_{drug_index}:after_save")

        del adata_source
        adata_source = None
        cleanup_memory()
        log_rss(f"{target_cell_type}:after_save_all_predictions")

    finally:
        clear_biolord_managers(model, adata_task)
        del model, adata_task, adata_source, adata_source_drug
        cleanup_memory()
        log_rss(f"{target_cell_type}:after_cleanup")


def merge_one_target(target_cell_type, drug_name_list, control_drug_index):
    """把当前 target cell_type 的临时预测合并成 cell-eval 使用的最终 h5ad。"""
    print("\n" + "=" * 100)
    print(f"开始 merge target cell_type: {target_cell_type}")
    print("=" * 100)
    log_rss(f"{target_cell_type}:merge:start")

    output_blocks = []

    for drug_index, drug_name in tqdm(
        list(enumerate(drug_name_list)),
        desc=f"Merging {target_cell_type}",
    ):
        if drug_index == control_drug_index:
            ctrl_real = read_one_block(
                cell_type=target_cell_type,
                drug_index=control_drug_index,
                expected_drug_name=CONTROL_DRUG,
            )
            ctrl_real.obs = ctrl_real.obs.copy()
            ctrl_real.obs["pred_or_real"] = "real_ctrl"
            output_blocks.append(ctrl_real)
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

    del output_blocks, final_adata
    cleanup_memory()
    log_rss(f"{target_cell_type}:merge:after_cleanup")


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    np.random.seed(SPLIT_RANDOM_SEED)
    torch.manual_seed(TORCH_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"训练 cell_type 数量: {len(TRAIN_CELL_TYPES)}")
    print(f"目标 cell_type 数量: {len(TARGET_CELL_TYPES)}")
    print(f"本轮 target cell_type: {target_items}")
    print(f"drug 数量: {len(drug_name_list)}")
    print(f"RUN_TRAINING: {RUN_TRAINING}")
    print(f"RUN_FINAL_MERGE: {RUN_FINAL_MERGE}")
    print(f"max_epochs: {MAX_EPOCHS}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"predict cell batch size: {PREDICT_CELL_BATCH_SIZE}")
    print(f"max cells per block: {MAX_CELLS_PER_BLOCK}")
    print(f"max target control cells: {MAX_TARGET_CONTROL_CELLS}")
    print(f"early_stopping_patience: {EARLY_STOPPING_PATIENCE}")
    print(f"prediction zero threshold: < {PREDICTION_ZERO_THRESHOLD:g} -> 0")
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
