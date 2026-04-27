# -*- coding: utf-8 -*-
"""
cd /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/complex_models/scVI
CUDA_VISIBLE_DEVICES=6 python scvi_pipeline.py

这版相对原版的核心修改：

A. 修复 train_blocks 长时间占内存：
   - ad.concat(train_blocks) 后立刻 del train_blocks

B. 下调训练 / query / decode batch size：
   - REF_BATCH_SIZE   : 131072 -> 32768
   - QUERY_BATCH_SIZE : 32768  -> 16384
   - PRED_BATCH_SIZE  : 新增   -> 8192

C. 不再“先全量 decode 再采样”：
   - 先按 stim_skeleton.n_obs 采样 query indices
   - 再调用 get_normalized_expression(indices=sampled_idx)

D. 显式清理 scVI manager：
   - query_model 每个 test cell 结束后清理
   - ref_model 每个 drug 结束后清理

E. 改成 one-drug-per-process 风格：
   - 默认 START_DRUG_INDEX = 136
   - 默认 END_DRUG_INDEX   = 136
   - 不再在一个长进程里连续跑很多 drug

F. merge 独立成第二阶段：
   - 默认 RUN_FINAL_MERGE = False
   - 等所有 drug 的 tmp_predictions 都准备好后，再单独执行 merge
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
import scvi
import torch
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None


# =========================
# 运行控制
# =========================
# 从哪个 drug_index 开始续跑
START_DRUG_INDEX = 999

# 跑到哪个 drug_index 结束（包含）
# 为了避免单进程 RSS 累积，建议一次只跑一个 drug
END_DRUG_INDEX = 999

# 若某个 test cell 的预测文件已经存在，是否跳过
SKIP_EXISTING_PRED_FILES = True

# 是否重置输出目录
# 续跑时必须 False，否则会删掉历史结果
RESET_OUTPUT_DIRS = False

# 是否执行最终 merge
# 建议等全部 drug 的 tmp_predictions 都准备好后，再单独开启
RUN_FINAL_MERGE = True

# 是否尝试 malloc_trim()，尽力把部分空闲 heap 还给 OS
TRY_MALLOC_TRIM = True


# =========================
# 日志与基础设置
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
SPLIT_BY_DRUG_DIR = Path("/data1/fanpeishan/STATE/for_state/data/split_by_drug")
DRUG_NAME_CSV = Path("/data1/fanpeishan/STATE/for_state/about_baseline/docs/drug_name_list.csv")

OUTPUT_ROOT = Path(
    "/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/complex_models/scVI/outputs"
)
TEMP_PRED_DIR = OUTPUT_ROOT / "tmp_predictions"

CONTROL_DRUG = "DMSO_TF"
LIBRARY_SIZE = 4000.0
RANDOM_SEED = 42


# =========================
# 模型参数
# =========================
REF_N_LAYERS = 3
REF_N_HIDDEN = 512
REF_N_LATENT = 64
REF_DISPERSION = "gene-batch"
REF_GENE_LIKELIHOOD = "nb"
REF_MAX_EPOCHS = 16

# 修改 B：降 batch size
REF_BATCH_SIZE = 32768
QUERY_MAX_EPOCHS = 8
QUERY_BATCH_SIZE = 16384
PRED_BATCH_SIZE = 8192

REF_CHECK_VAL_EVERY = 2
QUERY_CHECK_VAL_EVERY = 2

# 显式指定 dataloader worker 数
TRAIN_NUM_WORKERS = 0
QUERY_NUM_WORKERS = 0


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

TEST_CELL_TYPES = [
    "c4", "c17", "c18", "c19", "c20",
    "c22", "c23", "c24", "c31", "c38"
]


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


def to_template_matrix(matrix, template_x):
    matrix = np.asarray(matrix, dtype=np.float32)
    if sp.issparse(template_x):
        return sp.csr_matrix(matrix)
    return matrix


def log_rss(tag):
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


def clear_scvi_managers(model, adata=None, tag=""):
    """
    显式清理 scVI 的 AnnDataManager。
    """
    if model is None:
        return

    if adata is not None:
        try:
            model.deregister_manager(adata)
            print(f"[MANAGER] {tag}: 已清理指定 adata 的 manager")
        except Exception as e:
            print(f"[MANAGER] {tag}: 清理指定 adata 的 manager 失败: {repr(e)}")

    try:
        model.deregister_manager()
        print(f"[MANAGER] {tag}: 已清理 instance-specific managers")
    except Exception as e:
        print(f"[MANAGER] {tag}: 清理 instance-specific managers 失败: {repr(e)}")


def sample_query_indices(n_source, n_target, random_seed):
    """
    修改 C：先采样，再 decode。
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


def collect_selected_drug_indices(drug_name_list, control_drug_index):
    all_target_drug_indices = [
        idx for idx, drug_name in enumerate(drug_name_list)
        if drug_name != CONTROL_DRUG
    ]
    selected = [
        idx for idx in all_target_drug_indices
        if idx >= START_DRUG_INDEX and (END_DRUG_INDEX is None or idx <= END_DRUG_INDEX)
    ]
    return selected


def all_prediction_files_exist_for_drug(drug_index):
    for test_cell_type in TEST_CELL_TYPES:
        pred_file = TEMP_PRED_DIR / test_cell_type / f"{test_cell_type}_drugindex_{drug_index}_pred.h5ad"
        if not pred_file.exists():
            return False
    return True


def build_pred_block_from_skeleton(X_pred, stim_skeleton):
    X_pred = np.asarray(X_pred, dtype=np.float32)
    X_pred = np.clip(X_pred, a_min=0.0, a_max=None)

    if X_pred.shape[0] != stim_skeleton.n_obs:
        raise RuntimeError(
            f"预测矩阵行数与 skeleton 不一致: pred={X_pred.shape[0]}, skeleton={stim_skeleton.n_obs}"
        )

    pred_block = ad.AnnData(
        X=to_template_matrix(X_pred, stim_skeleton.X),
        obs=stim_skeleton.obs.copy(),
        var=stim_skeleton.var.copy(),
    )
    return pred_block


def run_one_drug(
    drug_index,
    drug_name_list,
    control_drug_index,
    accelerator,
    devices,
):
    drug_name = drug_name_list[drug_index]

    print("\n" + "-" * 100)
    print(f"当前药物: {drug_name} (drug_index={drug_index})")
    log_rss(f"drug_{drug_index}:before_build")

    if SKIP_EXISTING_PRED_FILES and all_prediction_files_exist_for_drug(drug_index):
        print(f"drug_index={drug_index} 的全部 test 预测文件都已存在，直接跳过")
        return

    ref_model = None
    adata_train = None

    try:
        # ============================================================
        # 1) 构建训练集
        # ============================================================
        train_blocks = []

        print("读取训练集 control 文件 ...")
        for cell_type in TRAIN_CELL_TYPES:
            train_blocks.append(sc.read_h5ad(get_split_file(cell_type, control_drug_index)))

        print("读取训练集当前药物文件 ...")
        for cell_type in TRAIN_CELL_TYPES:
            train_blocks.append(sc.read_h5ad(get_split_file(cell_type, drug_index)))

        adata_train = ad.concat(
            train_blocks,
            join="inner",
            merge="same",
            index_unique=None,
        )
        adata_train.obs_names_make_unique()

        print(f"训练数据 shape: {adata_train.shape}")
        print(f"训练数据中的药物条件: {sorted(adata_train.obs['drug'].unique().tolist())}")
        log_rss(f"drug_{drug_index}:after_concat_train")

        # 修改 A：concat 后立即释放 train_blocks
        del train_blocks
        cleanup_memory()
        log_rss(f"drug_{drug_index}:after_del_train_blocks")

        # ============================================================
        # 2) 训练 reference model
        # ============================================================
        scvi.model.SCVI.setup_anndata(
            adata_train,
            batch_key="drug",
        )
        log_rss(f"drug_{drug_index}:after_setup_anndata")

        ref_model = scvi.model.SCVI(
            adata_train,
            n_layers=REF_N_LAYERS,
            n_hidden=REF_N_HIDDEN,
            n_latent=REF_N_LATENT,
            dispersion=REF_DISPERSION,
            gene_likelihood=REF_GENE_LIKELIHOOD,
            use_observed_lib_size=True,
        )
        log_rss(f"drug_{drug_index}:after_ref_model_init")

        print("开始训练当前药物的 reference model ...")
        ref_model.train(
            max_epochs=REF_MAX_EPOCHS,
            accelerator=accelerator,
            devices=devices,
            batch_size=REF_BATCH_SIZE,
            early_stopping=True,
            check_val_every_n_epoch=REF_CHECK_VAL_EVERY,
            datasplitter_kwargs={"num_workers": TRAIN_NUM_WORKERS},
        )
        print("reference model 训练完成")
        log_rss(f"drug_{drug_index}:after_ref_train")

        # ============================================================
        # 3) 对 10 个测试 cell type 做 query adaptation + 预测
        # ============================================================
        for test_cell_type in TEST_CELL_TYPES:
            save_dir = TEMP_PRED_DIR / test_cell_type
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{test_cell_type}_drugindex_{drug_index}_pred.h5ad"

            if SKIP_EXISTING_PRED_FILES and save_path.exists():
                print(f"  -> 跳过 {test_cell_type}，文件已存在: {save_path}")
                continue

            print(f"  -> 开始预测 {test_cell_type}")

            ctrl_test = None
            stim_skeleton = None
            query_model = None

            try:
                ctrl_test = sc.read_h5ad(get_split_file(test_cell_type, control_drug_index))
                stim_skeleton = sc.read_h5ad(get_split_file(test_cell_type, drug_index))
                ctrl_test.obs_names_make_unique()
                stim_skeleton.obs_names_make_unique()
                log_rss(f"drug_{drug_index}:{test_cell_type}:after_read_query")

                scvi.model.SCVI.prepare_query_anndata(ctrl_test, ref_model, inplace=True)
                log_rss(f"drug_{drug_index}:{test_cell_type}:after_prepare_query")

                query_model = scvi.model.SCVI.load_query_data(
                    adata=ctrl_test,
                    reference_model=ref_model,
                    freeze_expression=True,
                    freeze_decoder_first_layer=True,
                    accelerator=accelerator,
                    device="auto",
                )
                log_rss(f"drug_{drug_index}:{test_cell_type}:after_load_query_model")

                query_model.train(
                    max_epochs=QUERY_MAX_EPOCHS,
                    accelerator=accelerator,
                    devices=devices,
                    batch_size=QUERY_BATCH_SIZE,
                    early_stopping=True,
                    check_val_every_n_epoch=QUERY_CHECK_VAL_EVERY,
                    datasplitter_kwargs={"num_workers": QUERY_NUM_WORKERS},
                )
                log_rss(f"drug_{drug_index}:{test_cell_type}:after_query_train")

                # 修改 C：先采样，再 decode
                n_d = stim_skeleton.n_obs
                sampled_idx = sample_query_indices(
                    n_source=ctrl_test.n_obs,
                    n_target=n_d,
                    random_seed=RANDOM_SEED + drug_index * 100 + int(test_cell_type[1:]),
                )
                log_rss(f"drug_{drug_index}:{test_cell_type}:after_sampling")

                pred_sample = query_model.get_normalized_expression(
                    adata=query_model.adata,
                    indices=sampled_idx,
                    transform_batch=drug_name,
                    library_size=LIBRARY_SIZE,
                    batch_size=PRED_BATCH_SIZE,
                    return_numpy=True,
                ).astype(np.float32, copy=False)
                log_rss(f"drug_{drug_index}:{test_cell_type}:after_decode")

                pred_block = build_pred_block_from_skeleton(
                    X_pred=pred_sample,
                    stim_skeleton=stim_skeleton,
                )
                pred_block.write_h5ad(save_path)

                print(f"  -> 已保存: {save_path}")
                print(f"  -> control 模板细胞数: {ctrl_test.n_obs}, 骨架细胞数: {n_d}")
                log_rss(f"drug_{drug_index}:{test_cell_type}:after_save_pred")

                del sampled_idx, pred_sample, pred_block

            finally:
                # 修改 D：显式清理 query_model managers
                if query_model is not None:
                    clear_scvi_managers(
                        model=query_model,
                        adata=getattr(query_model, "adata", None),
                        tag=f"query_drug_{drug_index}_{test_cell_type}",
                    )

                del query_model, ctrl_test, stim_skeleton
                cleanup_memory()
                log_rss(f"drug_{drug_index}:{test_cell_type}:after_query_cleanup")

    finally:
        # 修改 D/E：显式清理 ref_model managers
        if ref_model is not None:
            clear_scvi_managers(
                model=ref_model,
                adata=adata_train,
                tag=f"ref_drug_{drug_index}",
            )

        del ref_model, adata_train
        cleanup_memory()
        log_rss(f"drug_{drug_index}:after_ref_cleanup")


def merge_final_outputs(drug_name_list, control_drug_index):
    """
    修改 F：merge 独立成第二阶段。
    建议所有 tmp_predictions 都齐了以后，再单独执行这一阶段。
    """
    print("\n" + "=" * 100)
    print("开始合并最终预测文件")
    print("=" * 100)
    log_rss("merge:start")

    for test_cell_type in TEST_CELL_TYPES:
        print(f"合并 {test_cell_type} 的最终输出文件 ...")
        log_rss(f"merge_{test_cell_type}:start")

        output_blocks = []

        for drug_index in tqdm(range(len(drug_name_list)), desc=f"Merging {test_cell_type}"):
            if drug_index == control_drug_index:
                output_blocks.append(sc.read_h5ad(get_split_file(test_cell_type, control_drug_index)))
            else:
                pred_file = TEMP_PRED_DIR / test_cell_type / f"{test_cell_type}_drugindex_{drug_index}_pred.h5ad"
                if not pred_file.exists():
                    raise FileNotFoundError(
                        f"缺少预测文件，无法 merge: {pred_file}\n"
                        "请先确认所有 drug 的 tmp_predictions 都已生成。"
                    )
                output_blocks.append(sc.read_h5ad(pred_file))

        final_adata = ad.concat(
            output_blocks,
            join="inner",
            merge="same",
            index_unique=None,
        )
        final_adata.obs_names_make_unique()

        final_path = OUTPUT_ROOT / f"{test_cell_type}_predicted.h5ad"
        final_adata.write_h5ad(final_path)

        print(f"已生成: {final_path}")
        print(f"最终 shape: {final_adata.shape}")
        log_rss(f"merge_{test_cell_type}:after_write")

        del output_blocks, final_adata
        cleanup_memory()
        log_rss(f"merge_{test_cell_type}:after_cleanup")

    log_rss("merge:end")


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)

    if RESET_OUTPUT_DIRS:
        raise RuntimeError(
            "当前是续跑场景，请保持 RESET_OUTPUT_DIRS = False，"
            "否则会删除之前已经生成的结果。"
        )

    drug_name_list = read_drug_name_list()
    drug_to_index = {drug_name: idx for idx, drug_name in enumerate(drug_name_list)}
    control_drug_index = drug_to_index[CONTROL_DRUG]

    selected_drug_indices = collect_selected_drug_indices(
        drug_name_list=drug_name_list,
        control_drug_index=control_drug_index,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else "auto"

    print("=" * 100)
    print("开始执行 scVI baseline（续跑 / 低内存版）")
    print(f"control drug: {CONTROL_DRUG} (drug_index={control_drug_index})")
    print(f"训练 cell type 数量: {len(TRAIN_CELL_TYPES)}")
    print(f"测试 cell type 数量: {len(TEST_CELL_TYPES)}")
    print(f"START_DRUG_INDEX: {START_DRUG_INDEX}")
    print(f"END_DRUG_INDEX  : {END_DRUG_INDEX}")
    print(f"本次选中的 drug_index: {selected_drug_indices}")
    print(f"RESET_OUTPUT_DIRS     : {RESET_OUTPUT_DIRS}")
    print(f"SKIP_EXISTING_PRED_FILES: {SKIP_EXISTING_PRED_FILES}")
    print(f"RUN_FINAL_MERGE       : {RUN_FINAL_MERGE}")
    print(f"REF_BATCH_SIZE        : {REF_BATCH_SIZE}")
    print(f"QUERY_BATCH_SIZE      : {QUERY_BATCH_SIZE}")
    print(f"PRED_BATCH_SIZE       : {PRED_BATCH_SIZE}")
    print("当前输入是 normalize_total(target_sum=4000) 数据，不是 raw counts。")
    print("scVI 使用 gene_likelihood='nb' 作为 practical fallback。")
    print("=" * 100)
    log_rss("script_start")

    if not RUN_FINAL_MERGE and len(selected_drug_indices) == 0:
        raise RuntimeError("当前 START/END 配置下，没有选中任何目标 drug_index")

    # 第一阶段：训练 + 预测
    for loop_id, drug_index in enumerate(selected_drug_indices, start=1):
        print(f"\n[{loop_id}/{len(selected_drug_indices)}] 准备处理 drug_index={drug_index}")
        run_one_drug(
            drug_index=drug_index,
            drug_name_list=drug_name_list,
            control_drug_index=control_drug_index,
            accelerator=accelerator,
            devices=devices,
        )

    # 第二阶段：最终 merge（默认关闭）
    if RUN_FINAL_MERGE:
        merge_final_outputs(
            drug_name_list=drug_name_list,
            control_drug_index=control_drug_index,
        )

    print("\n" + "=" * 100)
    print("本轮流程完成")
    print("=" * 100)
    log_rss("script_end")