"""
Tahoe scVIDR baseline aligned with scPerturBench OOD scVIDR.

运行方式示例：
  cd /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_tahoe/scVIDR
  CUDA_VISIBLE_DEVICES=7 python scripts/scVIDR_pipeline.py

核心协议：
1. 采用 Tahoe no-validation 口径：
   - 40 个 train cell_type 用于训练；
   - 10 个 holdout cell_type 逐个作为预测目标。
2. 对每个 target cell_type 和每个非 control drug 单独训练一个 scVIDR 模型。
3. 每个任务传给 scVIDR prepare_data() 的 AnnData 包含：
   - 40 个 train cell_type 的 control + 当前 drug；
   - 当前 target cell_type 的 control + 当前 drug。
4. 让 prepare_data() 按 scPerturBench OOD 逻辑排除 target cell_type 的真实 stimulated block。
5. 输出时沿用真实 stimulated block 的 obs/var/n_obs 作为 skeleton，只替换 X。
6. 预测矩阵按 biolord Tahoe 脚本的规则处理：所有小于 2 的预测值置为 0。
"""

from pathlib import Path
import ctypes
import gc
import hashlib
import logging
import os
import re
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from tqdm import tqdm


# =========================
# scVIDR import path
# =========================
SCVIDR_ROOT = Path(os.environ.get("SCVIDR_ROOT", "/data1/fanpeishan/scVIDR"))
sys.path.insert(1, str(SCVIDR_ROOT / "vidr"))
sys.path.insert(1, str(SCVIDR_ROOT))

from vidr.vidr import VIDR  # noqa: E402
from vidr.utils import prepare_data  # noqa: E402


# =========================
# 运行控制
# =========================
# TARGET_CELL_TYPES 的下标范围。Tahoe 全量任务较重，建议一次只跑一个 target 或一个 drug 区间。
START_TARGET_INDEX = 0
END_TARGET_INDEX = 9

# drug_index 范围。None 表示跑到 drug_name_list 的最后一个 drug。
START_DRUG_INDEX = 0
END_DRUG_INDEX = None

# 是否执行训练和临时预测生成。
RUN_TRAINING = True

# 是否把 tmp_predictions 合并成 cell-eval 使用的最终 h5ad。
RUN_FINAL_MERGE = True

# 续跑时保持 True，已存在的 drug 预测块会跳过。
SKIP_EXISTING_PRED_FILES = True

# Tahoe 全量会产生大量模型文件。若只需要评估结果，可保持 False。
SAVE_MODELS = False

# 是否打印 RSS 内存日志。
ENABLE_RSS_LOG = True

# Linux/glibc 下尝试释放空闲堆内存。
TRY_MALLOC_TRIM = True


# =========================
# 基础路径配置
# =========================
# 路径保持服务器脚本风格；如服务器路径不同，请只改这里。
SPLIT_BY_DRUG_DIR = Path("/data1/fanpeishan/STATE/for_state/data/split_by_drug")
DRUG_NAME_CSV = Path("/data1/fanpeishan/STATE/for_state/about_baseline/docs/drug_name_list.csv")

OUTPUT_DIR = Path(
    "/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_tahoe/scVIDR/outputs"
)
TEMP_PRED_DIR = OUTPUT_DIR / "tmp_predictions"
MODEL_DIR = OUTPUT_DIR / "models"

CONTROL_DRUG = "DMSO_TF"


# =========================
# scPerturBench scVIDR 参数
# =========================
MAX_EPOCHS = 100
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 25
RANDOM_SEED = 16
PREDICTION_ZERO_THRESHOLD = 2.0


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


def safe_name(text):
    """把任意药物名转换成适合文件名的短字符串。"""
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "empty"


def stable_int(text):
    """为文件名生成稳定短 hash，避免不同药物名清洗后重名。"""
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


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


def get_model_path(target_cell_type, drug_index, drug_name):
    """返回单个 scVIDR 模型 checkpoint 路径。"""
    model_dir = MODEL_DIR / target_cell_type
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{target_cell_type}_drugindex_{drug_index}_{safe_name(drug_name)}__{stable_int(drug_name)}.pt"


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


def selected_drug_items(drug_name_list, control_drug_index):
    """根据 START/END 配置返回本轮要处理的非 control drug。"""
    if START_DRUG_INDEX < 0:
        raise ValueError("START_DRUG_INDEX 不能小于 0")

    end_index = len(drug_name_list) - 1 if END_DRUG_INDEX is None else END_DRUG_INDEX
    if end_index >= len(drug_name_list):
        raise ValueError(
            f"END_DRUG_INDEX 超出范围: {end_index}, "
            f"最大允许值为 {len(drug_name_list) - 1}"
        )
    if end_index < START_DRUG_INDEX:
        return []

    return [
        (idx, drug_name)
        for idx, drug_name in enumerate(drug_name_list)
        if START_DRUG_INDEX <= idx <= end_index and idx != control_drug_index
    ]


def all_prediction_files_exist_for_target(target_cell_type, selected_drugs):
    """检查当前 target cell_type 的本轮非 control 临时预测是否已经存在。"""
    for drug_index, _ in selected_drugs:
        if not get_temp_pred_path(target_cell_type, drug_index).exists():
            return False
    return True


# =========================
# 数据构造
# =========================
def build_pair_adata_for_target_drug(target_cell_type, drug_index, drug_name, control_drug_index):
    """
    构造当前 target cell_type / drug 的 scVIDR pair AnnData。

    这一步刻意保留 target cell_type 的真实 stimulated block，
    由 prepare_data(..., target_cell_type, drug_name) 负责按 scPerturBench OOD 逻辑排除。
    """
    blocks = []

    for train_cell_type in TRAIN_CELL_TYPES:
        ctrl_block = read_one_block(
            train_cell_type,
            control_drug_index,
            expected_drug_name=CONTROL_DRUG,
        )
        drug_block = read_one_block(
            train_cell_type,
            drug_index,
            expected_drug_name=drug_name,
        )
        blocks.append(ctrl_block)
        blocks.append(drug_block)

    target_ctrl = read_one_block(
        target_cell_type,
        control_drug_index,
        expected_drug_name=CONTROL_DRUG,
    )
    stim_skeleton = read_one_block(
        target_cell_type,
        drug_index,
        expected_drug_name=drug_name,
    )
    blocks.append(target_ctrl)
    blocks.append(stim_skeleton)

    if target_ctrl.n_obs == 0:
        raise RuntimeError(f"{target_cell_type} 没有 control cells")
    if stim_skeleton.n_obs == 0:
        raise RuntimeError(f"{target_cell_type}/drug_index={drug_index} 没有 stimulated cells")

    pair_adata = ad.concat(
        blocks,
        join="inner",
        merge="same",
        index_unique=None,
    )
    pair_adata.obs_names_make_unique()
    pair_adata.var_names_make_unique()
    pair_adata.obs = pair_adata.obs.copy()
    pair_adata.obs["condition1"] = pair_adata.obs["cell_type"].astype(str)
    pair_adata.obs["condition2"] = pair_adata.obs["drug"].astype(str)

    if CONTROL_DRUG not in set(pair_adata.obs["condition2"]):
        raise RuntimeError("pair_adata 中缺少 control drug")
    if drug_name not in set(pair_adata.obs["condition2"]):
        raise RuntimeError(f"pair_adata 中缺少目标 drug: {drug_name}")

    del blocks, target_ctrl
    cleanup_memory()
    return pair_adata, stim_skeleton


def validate_prepare_data_result(train_adata, target_cell_type, drug_name):
    """确认 prepare_data 按 OOD 口径保留 target control、排除 target stimulated。"""
    if "condition1" not in train_adata.obs.columns or "condition2" not in train_adata.obs.columns:
        raise KeyError("prepare_data 返回的 train_adata 缺少 condition1/condition2")

    cell_values = train_adata.obs["condition1"].astype(str).to_numpy()
    drug_values = train_adata.obs["condition2"].astype(str).to_numpy()

    target_ctrl_count = int(((cell_values == target_cell_type) & (drug_values == CONTROL_DRUG)).sum())
    target_stim_count = int(((cell_values == target_cell_type) & (drug_values == drug_name)).sum())

    if target_ctrl_count == 0:
        raise RuntimeError(
            f"prepare_data 后 train_adata 中没有 {target_cell_type} 的 control，无法预测"
        )
    if target_stim_count != 0:
        raise RuntimeError(
            f"prepare_data 后 train_adata 仍包含 {target_cell_type}/{drug_name} stimulated，"
            "这会泄漏测试目标"
        )


# =========================
# 训练、预测与输出
# =========================
def resize_prediction_to_skeleton(pred_x, stim_skeleton, random_seed):
    """把 scVIDR 预测块采样到真实 stimulated block 的细胞数。"""
    n_target = stim_skeleton.n_obs
    if pred_x.shape[0] == n_target:
        return pred_x.astype(np.float32, copy=False)

    if pred_x.shape[0] == 0:
        raise RuntimeError("scVIDR 返回了空预测矩阵")

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


def train_and_predict_one_drug(target_rank, target_cell_type, drug_index, drug_name, control_drug_index):
    """训练一个 target cell_type / drug 的 scVIDR，并保存临时预测。"""
    save_path = get_temp_pred_path(target_cell_type, drug_index)
    if SKIP_EXISTING_PRED_FILES and save_path.exists():
        print(f"  -> 跳过 drug_index={drug_index}，临时预测已存在: {save_path}")
        return

    print("-" * 100)
    print(f"target={target_cell_type} | drug={drug_name} | drug_index={drug_index}")
    log_rss(f"{target_cell_type}:drug_{drug_index}:start")

    model = None
    pair_adata = None
    train_adata = None
    stim_skeleton = None
    pred_adata = None

    try:
        print("  构造 pair AnnData ...")
        pair_adata, stim_skeleton = build_pair_adata_for_target_drug(
            target_cell_type=target_cell_type,
            drug_index=drug_index,
            drug_name=drug_name,
            control_drug_index=control_drug_index,
        )
        print(f"  pair shape: {pair_adata.shape}")
        log_rss(f"{target_cell_type}:drug_{drug_index}:after_pair")

        print("  调用 scVIDR prepare_data() ...")
        train_adata, test_adata = prepare_data(
            pair_adata,
            "condition1",
            "condition2",
            target_cell_type,
            drug_name,
            normalized=True,
        )
        validate_prepare_data_result(
            train_adata=train_adata,
            target_cell_type=target_cell_type,
            drug_name=drug_name,
        )
        del pair_adata, test_adata
        pair_adata = None
        cleanup_memory()
        print(f"  train shape after prepare_data: {train_adata.shape}")
        log_rss(f"{target_cell_type}:drug_{drug_index}:after_prepare")

        print("  训练 scVIDR 模型 ...")
        model = VIDR(train_adata, linear_decoder=False)
        model.train(
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            early_stopping=True,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )
        print("  模型训练完成")
        log_rss(f"{target_cell_type}:drug_{drug_index}:after_train")

        if SAVE_MODELS:
            model_path = get_model_path(target_cell_type, drug_index, drug_name)
            model.save(str(model_path))
            print(f"  模型已保存: {model_path}")

        print("  调用 model.predict() ...")
        pred_adata, _ = model.predict(
            ctrl_key=CONTROL_DRUG,
            treat_key=drug_name,
            cell_type_to_predict=target_cell_type,
            regression=False,
        )
        log_rss(f"{target_cell_type}:drug_{drug_index}:after_predict")

        pred_x = adata_to_numpy(pred_adata.X)
        pred_x = resize_prediction_to_skeleton(
            pred_x=pred_x,
            stim_skeleton=stim_skeleton,
            random_seed=RANDOM_SEED + target_rank * 1000 + drug_index,
        )

        adata_pred = build_prediction_from_skeleton(pred_x, stim_skeleton)
        validate_prediction_block(
            adata_pred=adata_pred,
            stim_skeleton=stim_skeleton,
            target_cell_type=target_cell_type,
            drug_index=drug_index,
        )
        adata_pred.write_h5ad(save_path)

        print(f"  已保存预测: {save_path}")
        print(f"  skeleton cells: {stim_skeleton.n_obs}")
        log_rss(f"{target_cell_type}:drug_{drug_index}:after_save")

        del pred_x, adata_pred

    finally:
        del model, pair_adata, train_adata, stim_skeleton, pred_adata
        cleanup_memory()
        log_rss(f"{target_cell_type}:drug_{drug_index}:after_cleanup")


def train_and_predict_one_target(target_rank, target_cell_type, selected_drugs, control_drug_index):
    """处理一个 target cell_type 的全部选中 drug。"""
    print("\n" + "#" * 100)
    print(f"开始处理 target cell_type: {target_cell_type} (target_index={target_rank})")
    print("#" * 100)
    log_rss(f"{target_cell_type}:target_start")

    if SKIP_EXISTING_PRED_FILES and all_prediction_files_exist_for_target(
        target_cell_type,
        selected_drugs,
    ):
        print(f"{target_cell_type} 的本轮非 control 临时预测都已存在，跳过训练")
        return

    for drug_rank, (drug_index, drug_name) in enumerate(selected_drugs, start=1):
        print(f"\n[{drug_rank}/{len(selected_drugs)}] 准备处理 drug_index={drug_index}")
        train_and_predict_one_drug(
            target_rank=target_rank,
            target_cell_type=target_cell_type,
            drug_index=drug_index,
            drug_name=drug_name,
            control_drug_index=control_drug_index,
        )

    log_rss(f"{target_cell_type}:target_end")


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
                target_cell_type,
                control_drug_index,
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
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_MODELS:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    drug_name_list = read_drug_name_list()
    drug_to_index = {drug_name: idx for idx, drug_name in enumerate(drug_name_list)}
    if CONTROL_DRUG not in drug_to_index:
        raise KeyError(f"drug_name_list 中找不到 control drug: {CONTROL_DRUG}")
    control_drug_index = drug_to_index[CONTROL_DRUG]

    target_items = selected_target_items()
    selected_drugs = selected_drug_items(
        drug_name_list=drug_name_list,
        control_drug_index=control_drug_index,
    )

    if not target_items:
        raise RuntimeError("当前 START/END target 配置下，没有选中任何 target cell_type")
    if RUN_TRAINING and not selected_drugs:
        raise RuntimeError("当前 START/END drug 配置下，没有选中任何非 control drug")

    print("=" * 100)
    print("开始执行 Tahoe scVIDR scPerturBench OOD 流程")
    print(f"SCVIDR_ROOT: {SCVIDR_ROOT}")
    print(f"control drug: {CONTROL_DRUG} (drug_index={control_drug_index})")
    print(f"训练 cell_type 数量: {len(TRAIN_CELL_TYPES)}")
    print(f"目标 cell_type 数量: {len(TARGET_CELL_TYPES)}")
    print(f"本轮 target cell_type: {target_items}")
    print(f"drug 总数: {len(drug_name_list)}")
    print(f"本轮非 control drug 数量: {len(selected_drugs)}")
    print(f"RUN_TRAINING: {RUN_TRAINING}")
    print(f"RUN_FINAL_MERGE: {RUN_FINAL_MERGE}")
    print(f"SAVE_MODELS: {SAVE_MODELS}")
    print(f"max_epochs: {MAX_EPOCHS}")
    print(f"batch_size: {BATCH_SIZE}")
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
                selected_drugs=selected_drugs,
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
    print("Tahoe scVIDR 流程完成")
    print("=" * 100)
    log_rss("script_end")
