"""
续跑版 scGen baseline 脚本（低内存预测路径）

主要修改：
1. 从指定 drug_index 开始续跑，不删除已有 tmp_predictions；
2. 默认一次只跑一个 drug（当前配置为 drug_index=18）；
3. 不再对每个 test_cell_type 重复调用重型 model.predict()；
   改为：
   - 每个 drug 只在训练集上计算一次 delta；
   - 每个 test_cell_type 先按目标块大小采样 control；
   - 再对采样后的 control 做 latent 编码 + delta 平移 + 分 batch 解码。
4. 每个 test_cell_type 预测后清理临时 AnnDataManager；
5. 每个 drug 完成后清理主训练 adata 对应的 AnnDataManager；
6. 下调训练 batch size，减少单轮峰值内存。

运行示例：
cd set_baseline/context_generalization/complex_models/scGen
CUDA_VISIBLE_DEVICES=7 python scgen_pipeline_resume.py
"""

from pathlib import Path
import ctypes
import gc
import logging
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scgen
import torch
from tqdm import tqdm

import scvi
from scvi import REGISTRY_KEYS
import scvi.data._manager as _scvi_manager
from scgen._utils import balancer


# =========================
# 续跑控制
# =========================
# 从哪个 drug_index 开始跑。
START_DRUG_INDEX = 999

# 跑到哪个 drug_index 结束（包含）。
# 为了避免单进程 RSS 长期累积，建议一次只跑一个 drug。
END_DRUG_INDEX = 999

# 是否跳过已存在的临时预测文件。
SKIP_EXISTING_PRED_FILES = True

# 是否执行最终 merge。
# 建议等所有 drug 都补齐后，再单独设为 True 跑 merge。
RUN_FINAL_MERGE = True

# 是否重置输出目录。
# 续跑时必须 False，否则会删除之前已经生成的 tmp_predictions。
RESET_OUTPUT_DIRS = False

# 是否输出 RSS 内存日志。
ENABLE_RSS_LOG = True

# 是否尝试 malloc_trim（Linux/glibc 下 best effort）。
TRY_MALLOC_TRIM = True


# =========================
# scvi / pytorch 设置
# =========================
scvi.settings.dl_num_workers = 4
scvi.settings.dl_persistent_workers = False

logging.getLogger("scvi").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Observation names are not unique")

# 修复部分环境下 PyTorch 2.6 加载 scvi/scgen 模型时的兼容性问题。
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
torch.set_float32_matmul_precision("medium")

# 允许推理阶段向模型传入训练时未出现过的新 cell_type。
_orig_transfer_fields = _scvi_manager.AnnDataManager.transfer_fields


def _patched_transfer_fields(self, adata_target, **kwargs):
    kwargs.setdefault("extend_categories", True)
    return _orig_transfer_fields(self, adata_target, **kwargs)


_scvi_manager.AnnDataManager.transfer_fields = _patched_transfer_fields

try:
    import psutil
except ImportError:
    psutil = None


# =========================
# 基础路径配置
# =========================
SPLIT_BY_DRUG_DIR = Path("/data1/fanpeishan/STATE/for_state/data/split_by_drug")
OUTPUT_DIR = Path("/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/complex_models/scGen/outputs")
TEMP_PRED_DIR = OUTPUT_DIR / "tmp_predictions"
DRUG_NAME_CSV = Path("/data1/fanpeishan/STATE/for_state/about_baseline/docs/drug_name_list.csv")
CONTROL_DRUG = "DMSO_TF"


# =========================
# 训练 / 推理参数
# =========================
MAX_EPOCHS = 16

# 方案 E：下调训练 batch size
BATCH_SIZE = 32768

EARLY_STOPPING_PATIENCE = 4
RANDOM_SEED = 16

# latent 编码和解码时的 batch size
LATENT_BATCH_SIZE = 8192
DECODE_BATCH_SIZE = 8192


# =========================
# 数据划分
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

TEST_CELL_TYPES = ["c4", "c17", "c18", "c19", "c20", "c22", "c23", "c24", "c31", "c38"]


# =========================
# 工具函数
# =========================
def read_drug_name_list():
    return pd.read_csv(DRUG_NAME_CSV)["drug_name"].tolist()


def get_split_file(cell_type, drug_index):
    return (
        SPLIT_BY_DRUG_DIR
        / f"celltype_{cell_type}"
        / f"celltype_{cell_type}_drugindex_{drug_index}.h5ad"
    )


def log_rss(tag):
    if not ENABLE_RSS_LOG:
        return
    if psutil is None:
        print(f"[MEM] {tag}: psutil 未安装，跳过 RSS 记录")
        return

    rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[MEM] {tag}: RSS = {rss_gb:.2f} GB")


def try_malloc_trim():
    if not TRY_MALLOC_TRIM:
        return
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try_malloc_trim()


def clear_temp_managers(model):
    """
    清理预测阶段产生的 instance-specific AnnDataManager。
    """
    try:
        model.deregister_manager()
        print("已清理临时 AnnDataManager")
    except Exception as e:
        print(f"警告：清理临时 AnnDataManager 失败: {repr(e)}")


def clear_main_manager(model, adata):
    """
    清理主训练 adata 对应的 AnnDataManager。
    """
    success = False

    try:
        model.deregister_manager(adata)
        success = True
    except Exception:
        pass

    if not success:
        try:
            type(model).deregister_manager(adata)
            success = True
        except Exception:
            pass

    if success:
        print("已清理主训练 adata 的 AnnDataManager")
    else:
        print("警告：主训练 adata 的 AnnDataManager 清理失败")


def collect_selected_drug_indices(drug_name_list):
    all_target = [
        idx for idx, drug_name in enumerate(drug_name_list)
        if drug_name != CONTROL_DRUG
    ]
    selected = [
        idx for idx in all_target
        if idx >= START_DRUG_INDEX and (END_DRUG_INDEX is None or idx <= END_DRUG_INDEX)
    ]
    return selected


def all_prediction_files_exist_for_drug(drug_index):
    for test_cell_type in TEST_CELL_TYPES:
        pred_file = TEMP_PRED_DIR / test_cell_type / f"{test_cell_type}_drugindex_{drug_index}_pred.h5ad"
        if not pred_file.exists():
            return False
    return True


def build_training_adata(drug_index, control_drug_index):
    """
    为当前 drug 构建训练集。

    与原脚本相比，改为“每个 cell_type 先拼 control + 当前 drug，再汇总”，
    减少列表里同时悬挂的 AnnData 对象数量。
    """
    train_blocks = []

    print("读取并构建当前 drug 的训练 AnnData ...")
    log_rss("build_training_adata:start")

    for idx, cell_type in enumerate(TRAIN_CELL_TYPES, start=1):
        print(f"  [{idx}/{len(TRAIN_CELL_TYPES)}] 当前训练 cell_type: {cell_type}")

        ctrl_block = sc.read_h5ad(get_split_file(cell_type, control_drug_index))
        drug_block = sc.read_h5ad(get_split_file(cell_type, drug_index))

        adata_cell = ad.concat(
            [ctrl_block, drug_block],
            join="inner",
            merge="same",
            index_unique=None,
        )
        adata_cell.obs_names_make_unique()
        train_blocks.append(adata_cell)

        print(f"    shape: {adata_cell.shape}")
        log_rss(f"after loading train {cell_type}")

        del ctrl_block, drug_block
        cleanup_memory()

    adata_train = ad.concat(
        train_blocks,
        join="inner",
        merge="same",
        index_unique=None,
    )
    adata_train.obs_names_make_unique()

    print("训练 AnnData 构建完成")
    print(f"训练数据 shape: {adata_train.shape}")
    print(f"训练数据中的 drug: {sorted(adata_train.obs['drug'].unique().tolist())}")
    print(f"训练数据中的 cell_type 数量: {adata_train.obs['cell_type'].nunique()}")
    log_rss("build_training_adata:end")

    del train_blocks
    cleanup_memory()

    return adata_train


def compute_delta_once(model, ctrl_key, stim_key, random_seed):
    """
    方案 B：每个 drug 只在训练集上计算一次 delta。

    这里尽量复用 scGen.predict() 的核心逻辑：
    1. 按 condition_key 切 control / stim；
    2. 用 balancer 做 cell_type 平衡；
    3. 随机抽到相同数量；
    4. 计算平均 latent 向量差 delta。
    """
    cell_type_key = model.adata_manager.get_state_registry(
        REGISTRY_KEYS.LABELS_KEY
    ).original_key
    condition_key = model.adata_manager.get_state_registry(
        REGISTRY_KEYS.BATCH_KEY
    ).original_key

    ctrl_x = model.adata[model.adata.obs[condition_key] == ctrl_key, :]
    stim_x = model.adata[model.adata.obs[condition_key] == stim_key, :]

    ctrl_x = balancer(ctrl_x, cell_type_key)
    stim_x = balancer(stim_x, cell_type_key)

    eq = min(ctrl_x.n_obs, stim_x.n_obs)
    if eq < 1:
        raise RuntimeError(
            f"无法计算 delta：control/stim 中至少有一边为空。ctrl={ctrl_x.n_obs}, stim={stim_x.n_obs}"
        )

    rng = np.random.default_rng(random_seed)
    ctrl_ind = rng.choice(ctrl_x.n_obs, size=eq, replace=False)
    stim_ind = rng.choice(stim_x.n_obs, size=eq, replace=False)

    ctrl_adata = ctrl_x[ctrl_ind, :].copy()
    stim_adata = stim_x[stim_ind, :].copy()

    log_rss("delta:before_latent_ctrl")
    latent_ctrl = model.get_latent_representation(
        ctrl_adata,
        batch_size=LATENT_BATCH_SIZE,
    )
    log_rss("delta:after_latent_ctrl")

    latent_stim = model.get_latent_representation(
        stim_adata,
        batch_size=LATENT_BATCH_SIZE,
    )
    log_rss("delta:after_latent_stim")

    delta = np.mean(latent_stim, axis=0) - np.mean(latent_ctrl, axis=0)
    delta = np.asarray(delta, dtype=np.float32)

    del ctrl_x, stim_x, ctrl_adata, stim_adata, latent_ctrl, latent_stim
    cleanup_memory()
    clear_temp_managers(model)
    cleanup_memory()
    log_rss("delta:after_cleanup")

    return delta


def sample_control_to_target(ctrl_test, n_target, random_seed):
    """
    方案 F：先采样后预测。
    """
    rng = np.random.default_rng(random_seed)
    sample_index = rng.choice(
        ctrl_test.n_obs,
        size=n_target,
        replace=(n_target > ctrl_test.n_obs),
    )
    return ctrl_test[sample_index, :].copy()


@torch.no_grad()
def decode_latent_batched(model, latent_array, batch_size):
    """
    把 latent 分 batch 解码成表达矩阵，避免一次性把全部 latent 丢给 decoder。
    """
    model.module.eval()

    decoded = []
    n = latent_array.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z_chunk = torch.tensor(
            latent_array[start:end],
            dtype=torch.float32,
            device=model.device,
        )
        px = model.module.generative(z_chunk)["px"].detach().cpu().numpy()
        decoded.append(px.astype(np.float32, copy=False))

        del z_chunk, px
        if (start // batch_size) % 4 == 0:
            cleanup_memory()

    X = np.concatenate(decoded, axis=0)
    del decoded
    cleanup_memory()
    return X


def predict_for_test_celltype(
    model,
    delta,
    ctrl_test,
    stim_skeleton,
    random_seed,
):
    """
    用“先采样 control -> latent 编码 -> + delta -> 分 batch 解码”的方式生成预测。
    """
    n_target = stim_skeleton.n_obs

    ctrl_sampled = sample_control_to_target(
        ctrl_test=ctrl_test,
        n_target=n_target,
        random_seed=random_seed,
    )
    log_rss("predict:after_ctrl_sampling")

    latent_ctrl = model.get_latent_representation(
        ctrl_sampled,
        batch_size=LATENT_BATCH_SIZE,
    )
    log_rss("predict:after_latent_ctrl")

    latent_pred = latent_ctrl + delta[None, :]
    latent_pred = np.asarray(latent_pred, dtype=np.float32)

    X_pred = decode_latent_batched(
        model=model,
        latent_array=latent_pred,
        batch_size=DECODE_BATCH_SIZE,
    )
    log_rss("predict:after_decode")

    X_pred = np.clip(X_pred, a_min=0.0, a_max=None)

    adata_pred = ad.AnnData(
        X=X_pred,
        obs=stim_skeleton.obs.copy(),
        var=stim_skeleton.var.copy(),
    )
    adata_pred.obs["pred_or_real"] = "pred"

    del ctrl_sampled, latent_ctrl, latent_pred, X_pred
    cleanup_memory()
    return adata_pred


def run_one_drug(drug_index, drug_name_list, control_drug_index):
    drug_name = drug_name_list[drug_index]

    print("\n" + "-" * 100)
    print(f"当前药物: {drug_name} (drug_index={drug_index})")
    log_rss(f"drug_{drug_index}:before_build")

    if SKIP_EXISTING_PRED_FILES and all_prediction_files_exist_for_drug(drug_index):
        print(f"drug_index={drug_index} 的所有测试预测文件都已存在，跳过")
        return

    model = None
    adata_train = None

    try:
        adata_train = build_training_adata(
            drug_index=drug_index,
            control_drug_index=control_drug_index,
        )
        log_rss(f"drug_{drug_index}:after_build")

        scgen.SCGEN.setup_anndata(
            adata_train,
            batch_key="drug",
            labels_key="cell_type",
        )
        log_rss(f"drug_{drug_index}:after_setup_anndata")

        print("开始训练 scGen 模型 ...")
        model = scgen.SCGEN(adata_train)
        model.train(
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            early_stopping=True,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )
        print("模型训练完成")
        log_rss(f"drug_{drug_index}:after_train")

        print("开始为当前 drug 计算一次 delta ...")
        delta = compute_delta_once(
            model=model,
            ctrl_key=CONTROL_DRUG,
            stim_key=drug_name,
            random_seed=RANDOM_SEED + 1000 * drug_index,
        )
        print(f"delta shape: {delta.shape}")
        log_rss(f"drug_{drug_index}:after_delta")

        for test_cell_type in TEST_CELL_TYPES:
            save_dir = TEMP_PRED_DIR / test_cell_type
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{test_cell_type}_drugindex_{drug_index}_pred.h5ad"

            if SKIP_EXISTING_PRED_FILES and save_path.exists():
                print(f"  -> 跳过 {test_cell_type}，文件已存在: {save_path}")
                continue

            print(f"  -> 开始预测 {test_cell_type}")

            ctrl_test = sc.read_h5ad(get_split_file(test_cell_type, control_drug_index))
            stim_skeleton = sc.read_h5ad(get_split_file(test_cell_type, drug_index))

            print(f"  -> control 模板细胞数: {ctrl_test.n_obs}")
            print(f"  -> 骨架细胞数: {stim_skeleton.n_obs}")
            log_rss(f"drug_{drug_index}:{test_cell_type}:after_read_inputs")

            adata_pred = predict_for_test_celltype(
                model=model,
                delta=delta,
                ctrl_test=ctrl_test,
                stim_skeleton=stim_skeleton,
                random_seed=RANDOM_SEED + 1000 * drug_index + int(test_cell_type[1:]),
            )

            adata_pred.write_h5ad(save_path)

            print(f"  -> 已保存: {save_path}")
            log_rss(f"drug_{drug_index}:{test_cell_type}:after_save_pred")

            del ctrl_test, stim_skeleton, adata_pred
            cleanup_memory()

            # 方案 C：每个 test_cell_type 预测后清理临时 managers
            clear_temp_managers(model)
            cleanup_memory()
            log_rss(f"drug_{drug_index}:{test_cell_type}:after_cleanup")

        del delta
        cleanup_memory()

    finally:
        if model is not None:
            clear_temp_managers(model)

        if model is not None and adata_train is not None:
            clear_main_manager(model, adata_train)

        del model, adata_train
        cleanup_memory()
        log_rss(f"drug_{drug_index}:after_full_cleanup")


def merge_final_outputs(drug_name_list, control_drug_index):
    print("\n" + "=" * 100)
    print("开始合并最终预测文件")
    print("=" * 100)
    log_rss("before_final_merge")

    for test_cell_type in TEST_CELL_TYPES:
        print(f"合并 {test_cell_type} 的最终输出文件 ...")
        output_blocks = []

        for drug_index in tqdm(range(len(drug_name_list)), desc=f"Merging {test_cell_type}"):
            if drug_index == control_drug_index:
                ctrl_real = sc.read_h5ad(get_split_file(test_cell_type, control_drug_index))
                ctrl_real.obs = ctrl_real.obs.copy()
                ctrl_real.obs["pred_or_real"] = "real_ctrl"
                output_blocks.append(ctrl_real)
            else:
                pred_file = TEMP_PRED_DIR / test_cell_type / f"{test_cell_type}_drugindex_{drug_index}_pred.h5ad"
                if not pred_file.exists():
                    raise FileNotFoundError(
                        f"缺少预测文件，无法 merge: {pred_file}"
                    )
                output_blocks.append(sc.read_h5ad(pred_file))

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
        log_rss(f"merge_{test_cell_type}:after_write")

        del output_blocks, final_adata
        cleanup_memory()
        log_rss(f"merge_{test_cell_type}:after_cleanup")


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)

    if RESET_OUTPUT_DIRS:
        raise RuntimeError(
            "当前是续跑场景，请不要把 RESET_OUTPUT_DIRS 设为 True，"
            "否则会删除之前已经生成的 tmp_predictions。"
        )

    drug_name_list = read_drug_name_list()
    drug_to_index = {drug_name: idx for idx, drug_name in enumerate(drug_name_list)}

    control_drug_index = drug_to_index[CONTROL_DRUG]
    selected_drug_indices = collect_selected_drug_indices(drug_name_list)

    print("=" * 100)
    print("开始执行 scGen 续跑/低内存版流程")
    print(f"control drug: {CONTROL_DRUG} (drug_index={control_drug_index})")
    print(f"训练 cell type 数量: {len(TRAIN_CELL_TYPES)}")
    print(f"测试 cell type 数量: {len(TEST_CELL_TYPES)}")
    print(f"训练 batch size: {BATCH_SIZE}")
    print(f"latent batch size: {LATENT_BATCH_SIZE}")
    print(f"decode batch size: {DECODE_BATCH_SIZE}")
    print(f"START_DRUG_INDEX: {START_DRUG_INDEX}")
    print(f"END_DRUG_INDEX  : {END_DRUG_INDEX}")
    print(f"本次选中的 drug_index: {selected_drug_indices}")
    print(f"RUN_FINAL_MERGE : {RUN_FINAL_MERGE}")
    print("=" * 100)
    log_rss("script_start")

    if len(selected_drug_indices) == 0 and not RUN_FINAL_MERGE:
        raise RuntimeError("当前 START/END 配置下，没有选中任何目标 drug_index")

    for loop_id, drug_index in enumerate(selected_drug_indices, start=1):
        print(f"\n[{loop_id}/{len(selected_drug_indices)}] 准备处理 drug_index={drug_index}")
        run_one_drug(
            drug_index=drug_index,
            drug_name_list=drug_name_list,
            control_drug_index=control_drug_index,
        )

    if RUN_FINAL_MERGE:
        merge_final_outputs(
            drug_name_list=drug_name_list,
            control_drug_index=control_drug_index,
        )

    print("\n" + "=" * 100)
    print("本轮流程完成")
    print("=" * 100)
    log_rss("script_end")