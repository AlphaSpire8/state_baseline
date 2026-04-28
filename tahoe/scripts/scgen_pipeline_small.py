"""
Tahoe scGen 小规模复现脚本。

目标：
1. 严格按 scPerturBench OOD scGen 思路构造任务；
2. 每个 test cell_type 和 drug 单独训练一个 scGen 模型；
3. 预测时调用 model.predict，不手写 delta 或 decoder；
4. 默认只跑小规模配置，便于先验证逻辑和内存占用。

运行示例：
cd /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/complex_models/scGen
CUDA_VISIBLE_DEVICES=0 python scripts/scgen_pipeline_small.py
"""

from pathlib import Path
import ctypes
import gc
import hashlib
import logging
import os
import re
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scgen
import torch
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None


# =========================
# 基础配置
# =========================
SPLIT_BY_DRUG_DIR = Path("/data1/fanpeishan/STATE/for_state/data/split_by_drug")
OUTPUT_DIR = Path(
    "/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/"
    "complex_models/scGen/outputs_small_scperturbench"
)
TEMP_PRED_DIR = OUTPUT_DIR / "tmp_predictions"
EVAL_REF_DIR = OUTPUT_DIR / "eval_reference"
DRUG_NAME_CSV = Path("/data1/fanpeishan/STATE/for_state/about_baseline/docs/drug_name_list.csv")

CONTROL_DRUG = "DMSO_TF"
TRAIN_CELL_TYPES = ["c0", "c1", "c2"]
TEST_CELL_TYPES = ["c3", "c4", "c5"]

# 小规模默认只跑前 3 个非 control 药物。
NUM_TEST_DRUGS = 3

# 如果需要 smoke test，可以临时改成 ["c3"] 和 NUM_TEST_DRUGS = 1。
RANDOM_SEED = 16


# =========================
# scPerturBench 参考训练参数
# =========================
MAX_EPOCHS = 200
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 25


# =========================
# 内存与日志
# =========================
ENABLE_RSS_LOG = True
TRY_MALLOC_TRIM = True

logging.getLogger("scvi").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Observation names are not unique")
warnings.filterwarnings("ignore", message=".*is a view of an AnnData object.*")
torch.set_float32_matmul_precision("medium")

# 兼容部分环境下 PyTorch 2.6 读取 scvi/scgen 对象的行为。
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


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


def clear_scvi_managers(model, adata_train=None):
    """best-effort 清理 scvi/scgen 注册的 AnnDataManager。"""
    if model is None:
        return

    try:
        model.deregister_manager()
    except Exception:
        pass

    if adata_train is None:
        return

    try:
        model.deregister_manager(adata_train)
    except Exception:
        pass

    try:
        type(model).deregister_manager(adata_train)
    except Exception:
        pass


# =========================
# 数据工具
# =========================
def read_drug_name_list():
    """读取 drug_index 到 drug name 的映射，并保持原始顺序。"""
    return pd.read_csv(DRUG_NAME_CSV)["drug_name"].astype(str).tolist()


def get_split_file(cell_type, drug_index):
    """返回已经按 cell type 和 drug_index 拆分后的 h5ad 路径。"""
    return (
        SPLIT_BY_DRUG_DIR
        / f"celltype_{cell_type}"
        / f"celltype_{cell_type}_drugindex_{drug_index}.h5ad"
    )


def unique_in_order(values):
    """按出现顺序去重。"""
    return pd.Index(values).drop_duplicates().tolist()


def safe_name(text):
    """把药物名转换成安全文件名，同时保留 hash 避免重名。"""
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "empty"


def stable_int(text):
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def get_temp_pred_path(cell_type, drug_name):
    cell_dir = TEMP_PRED_DIR / cell_type
    cell_dir.mkdir(parents=True, exist_ok=True)
    return cell_dir / f"{safe_name(drug_name)}__{stable_int(drug_name)}_pred.h5ad"


def prepare_adata(adata):
    """统一关键 obs 列类型，避免 scgen 注册类别时出现混合类型。"""
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs["drug"] = adata.obs["drug"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    return adata


def adata_to_numpy(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def to_template_matrix(matrix, template_x):
    matrix = np.asarray(matrix, dtype=np.float32)
    if sp.issparse(template_x):
        return sp.csr_matrix(matrix)
    return matrix


def collect_small_drugs(drug_name_list):
    """选取小规模实验要跑的前几个非 control 药物。"""
    selected = []
    for drug_index, drug_name in enumerate(drug_name_list):
        if drug_name == CONTROL_DRUG:
            continue
        selected.append((drug_index, drug_name))
        if len(selected) >= NUM_TEST_DRUGS:
            break
    return selected


def read_one_block(cell_type, drug_index):
    path = get_split_file(cell_type, drug_index)
    if not path.exists():
        raise FileNotFoundError(f"找不到拆分数据文件: {path}")
    return prepare_adata(sc.read_h5ad(path))


def build_training_adata_for_task(test_cell_type, drug_index, drug_name, control_drug_index):
    """
    构造一个 scPerturBench OOD 任务的训练集。

    参考逻辑：
    adata = control + 当前 drug
    adata_train = adata 中排除目标 cell_type 的当前 drug stimulated block

    由于 Tahoe 已经按 cell_type/drug 拆分，这里等价于：
    1. 加入训练 cell type 的 control + 当前 drug；
    2. 加入目标 test cell type 的 control；
    3. 不加入目标 test cell type 的当前 drug。
    """
    train_blocks = []

    for cell_type in TRAIN_CELL_TYPES:
        ctrl_block = read_one_block(cell_type, control_drug_index)
        drug_block = read_one_block(cell_type, drug_index)
        train_blocks.extend([ctrl_block, drug_block])

    target_control = read_one_block(test_cell_type, control_drug_index)
    train_blocks.append(target_control)

    adata_train = ad.concat(
        train_blocks,
        join="inner",
        merge="same",
        index_unique=None,
    )
    prepare_adata(adata_train)

    if CONTROL_DRUG not in set(adata_train.obs["drug"]):
        raise ValueError("训练集中缺少 control 细胞")

    if drug_name not in set(adata_train.obs["drug"]):
        raise ValueError(f"训练集中缺少当前 drug 的 stimulated 细胞: {drug_name}")

    del train_blocks, target_control
    cleanup_memory()
    return adata_train


def train_one_model(adata_train):
    """按参考脚本参数训练一个 scGen 模型。"""
    scgen.SCGEN.setup_anndata(
        adata_train,
        batch_key="drug",
        labels_key="cell_type",
    )

    model = scgen.SCGEN(adata_train)
    model.train(
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )
    return model


def resize_prediction_to_skeleton(pred_x, stim_skeleton, random_seed):
    """把 scGen 预测行数采样到真实 stimulated block 的行数。"""
    n_target = stim_skeleton.n_obs
    if pred_x.shape[0] == n_target:
        return pred_x.astype(np.float32, copy=False)

    if pred_x.shape[0] == 0:
        raise RuntimeError("scGen 返回了空预测矩阵")

    rng = np.random.default_rng(random_seed)
    sampled_idx = rng.choice(
        pred_x.shape[0],
        size=n_target,
        replace=(n_target > pred_x.shape[0]),
    )
    return pred_x[sampled_idx].astype(np.float32, copy=False)


def predict_one_task(model, test_cell_type, drug_name, stim_skeleton, random_seed):
    """调用 scGen 原生 model.predict 预测一个目标 cell_type/drug。"""
    pred_adata, _ = model.predict(
        ctrl_key=CONTROL_DRUG,
        stim_key=drug_name,
        celltype_to_predict=test_cell_type,
    )

    pred_x = adata_to_numpy(pred_adata.X)
    pred_x = resize_prediction_to_skeleton(
        pred_x=pred_x,
        stim_skeleton=stim_skeleton,
        random_seed=random_seed,
    )
    pred_x = np.clip(pred_x, a_min=0.0, a_max=None)

    adata_pred = ad.AnnData(
        X=to_template_matrix(pred_x, stim_skeleton.X),
        obs=stim_skeleton.obs.copy(),
        var=stim_skeleton.var.copy(),
    )
    adata_pred.obs["pred_or_real"] = "pred"

    del pred_adata, pred_x
    cleanup_memory()
    return adata_pred


def validate_prediction_block(adata_pred, stim_skeleton, test_cell_type, drug_name):
    """保存前做最基本的结构和数值检查。"""
    if adata_pred.n_obs != stim_skeleton.n_obs:
        raise RuntimeError(
            f"{test_cell_type}/{drug_name} 行数不一致: "
            f"pred={adata_pred.n_obs}, target={stim_skeleton.n_obs}"
        )
    if adata_pred.n_vars != stim_skeleton.n_vars:
        raise RuntimeError(
            f"{test_cell_type}/{drug_name} 基因数不一致: "
            f"pred={adata_pred.n_vars}, target={stim_skeleton.n_vars}"
        )

    pred_x = adata_to_numpy(adata_pred.X)
    if np.isnan(pred_x).any():
        raise RuntimeError(f"{test_cell_type}/{drug_name} 预测矩阵包含 NaN")
    if (pred_x < 0).any():
        raise RuntimeError(f"{test_cell_type}/{drug_name} 预测矩阵包含负值")


def build_real_reference_for_eval(test_cell_type, selected_drugs, control_drug_index):
    """
    为 cell-eval 生成与小规模 prediction 药物集合完全一致的 real h5ad。

    cell-eval 要求 adata_pred 和 adata_real 的 pert_col 取值集合一致；
    小规模脚本只预测少量 drug，因此不能直接拿全量 tahoe_filtered/{cell}.h5ad 评估。
    """
    real_blocks = [read_one_block(test_cell_type, control_drug_index)]

    for drug_index, _ in selected_drugs:
        real_blocks.append(read_one_block(test_cell_type, drug_index))

    real_ref = ad.concat(
        real_blocks,
        join="inner",
        merge="same",
        index_unique=None,
    )
    real_ref.obs_names_make_unique()

    EVAL_REF_DIR.mkdir(parents=True, exist_ok=True)
    real_ref_path = EVAL_REF_DIR / f"{test_cell_type}_real_subset.h5ad"
    real_ref.write_h5ad(real_ref_path)

    print(f"已生成匹配 cell-eval 的真实子集: {real_ref_path}")
    print(f"真实子集 shape: {real_ref.shape}")

    del real_blocks, real_ref
    cleanup_memory()


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_REF_DIR.mkdir(parents=True, exist_ok=True)

    drug_name_list = read_drug_name_list()
    drug_to_index = {drug_name: idx for idx, drug_name in enumerate(drug_name_list)}
    control_drug_index = drug_to_index[CONTROL_DRUG]
    selected_drugs = collect_small_drugs(drug_name_list)

    print("=" * 100)
    print("开始执行 Tahoe scGen 小规模 scPerturBench OOD 复现脚本")
    print(f"control drug: {CONTROL_DRUG} (drug_index={control_drug_index})")
    print(f"训练 cell type: {TRAIN_CELL_TYPES}")
    print(f"测试 cell type: {TEST_CELL_TYPES}")
    print(f"非 control 药物数量: {len(selected_drugs)}")
    print(f"max_epochs: {MAX_EPOCHS}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"early_stopping_patience: {EARLY_STOPPING_PATIENCE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 100)
    log_rss("script_start")

    for test_cell_type in TEST_CELL_TYPES:
        print("\n" + "#" * 100)
        print(f"处理测试 cell_type: {test_cell_type}")
        print("#" * 100)

        for drug_rank, (drug_index, drug_name) in enumerate(selected_drugs, start=1):
            print("-" * 100)
            print(
                f"[{drug_rank}/{len(selected_drugs)}] "
                f"test_cell_type={test_cell_type}, drug={drug_name}, drug_index={drug_index}"
            )
            log_rss(f"{test_cell_type}/{drug_name}:before_task")

            model = None
            adata_train = None
            stim_skeleton = None
            adata_pred = None

            try:
                adata_train = build_training_adata_for_task(
                    test_cell_type=test_cell_type,
                    drug_index=drug_index,
                    drug_name=drug_name,
                    control_drug_index=control_drug_index,
                )
                print(f"  训练数据 shape: {adata_train.shape}")
                print(
                    "  训练数据 cell_type: "
                    f"{unique_in_order(adata_train.obs['cell_type'])}"
                )
                log_rss(f"{test_cell_type}/{drug_name}:after_build_train")

                model = train_one_model(adata_train)
                print("  模型训练完成")
                log_rss(f"{test_cell_type}/{drug_name}:after_train")

                stim_skeleton = read_one_block(test_cell_type, drug_index)
                print(f"  目标 stimulated block shape: {stim_skeleton.shape}")

                adata_pred = predict_one_task(
                    model=model,
                    test_cell_type=test_cell_type,
                    drug_name=drug_name,
                    stim_skeleton=stim_skeleton,
                    random_seed=RANDOM_SEED + stable_int(f"{test_cell_type}::{drug_name}"),
                )
                validate_prediction_block(
                    adata_pred=adata_pred,
                    stim_skeleton=stim_skeleton,
                    test_cell_type=test_cell_type,
                    drug_name=drug_name,
                )

                save_path = get_temp_pred_path(test_cell_type, drug_name)
                adata_pred.write_h5ad(save_path)
                print(f"  已保存临时预测: {save_path}")
                log_rss(f"{test_cell_type}/{drug_name}:after_save_pred")

            finally:
                clear_scvi_managers(model, adata_train)
                del model, adata_train, stim_skeleton, adata_pred
                cleanup_memory()
                log_rss(f"{test_cell_type}/{drug_name}:after_cleanup")

    print("\n" + "=" * 100)
    print("开始合并每个测试 cell_type 的 cell-eval 格式输出")
    print("=" * 100)

    for test_cell_type in tqdm(TEST_CELL_TYPES, desc="Merging"):
        output_blocks = []

        ctrl_real = read_one_block(test_cell_type, control_drug_index)
        ctrl_real.obs = ctrl_real.obs.copy()
        ctrl_real.obs["pred_or_real"] = "real_ctrl"
        output_blocks.append(ctrl_real)

        for _, drug_name in selected_drugs:
            pred_path = get_temp_pred_path(test_cell_type, drug_name)
            if not pred_path.exists():
                raise FileNotFoundError(f"缺少临时预测文件: {pred_path}")
            output_blocks.append(sc.read_h5ad(pred_path))

        final_adata = ad.concat(
            output_blocks,
            join="inner",
            merge="same",
            index_unique=None,
        )
        final_adata.obs_names_make_unique()

        final_path = OUTPUT_DIR / f"{test_cell_type}_predicted.h5ad"
        final_adata.write_h5ad(final_path)

        print(f"已生成: {final_path}")
        print(f"最终 shape: {final_adata.shape}")
        log_rss(f"{test_cell_type}:after_merge")

        del output_blocks, ctrl_real, final_adata
        cleanup_memory()

        build_real_reference_for_eval(
            test_cell_type=test_cell_type,
            selected_drugs=selected_drugs,
            control_drug_index=control_drug_index,
        )

    print("\n" + "=" * 100)
    print("Tahoe scGen 小规模复现流程完成")
    print("=" * 100)
    log_rss("script_end")
