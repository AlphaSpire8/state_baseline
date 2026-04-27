# export CUDA_VISIBLE_DEVICES=2

"""
Tahoe 数据集上的 biolord baseline 训练与推理脚本（续跑/降内存版）。

这版相对上一版，新增了这些关键修改：

1. 支持从指定 drug_index 继续运行，不会删除已有 tmp_predictions；
2. 默认只跑一个 drug（当前配置为 drug_index=12），避免单进程长期驻留导致 RSS 一路抬高；
3. 不再调用 biolord.compute_prediction_adata()；
   改为：
   - 先按目标测试块大小，从 source control 中采样；
   - 再分 batch 做目标药物预测；
   - 直接写成最终骨架输出。
4. 每个 test cell_type 预测后，调用 model.deregister_manager()
   清掉预测阶段临时 AnnData 触发的 instance-specific managers；
5. 每个 drug 结束后：
   - 先清理临时 managers
   - 再清理主 adata_drug 对应 manager
6. cleanup 时增加 best-effort malloc_trim()（Linux/glibc），
   尝试把部分空闲堆内存还给 OS。
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
import scvi
import torch
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None


# =========================
# 运行控制（这次续跑的核心配置）
# =========================
# 从哪个 drug_index 开始跑。
START_DRUG_INDEX = 99999

# 跑到哪个 drug_index 结束（包含该 index）。
# 为了避免单进程 RSS 一路抬高，建议一次只跑一个 drug。
END_DRUG_INDEX = 99999

# 是否跳过已经存在的临时预测文件。
SKIP_EXISTING_PRED_FILES = True

# 本次是否做最终 merge。
# 现在建议先 False，等全部 drug 都预测完，再单独做 merge。
RUN_FINAL_MERGE = True

# 是否重置输出目录。
# 续跑时必须 False，否则会把之前已经跑好的结果删掉。
RESET_OUTPUT_DIRS = False

# 预测时的 batch size（不是训练 batch size）。
PRED_BATCH_SIZE = 8192

# 是否尝试 malloc_trim()。
TRY_MALLOC_TRIM = True


scvi.settings.dl_num_workers = 0
scvi.settings.dl_persistent_workers = False
scvi.settings.seed = 16

# =========================
# 日志、警告与数值设置
# =========================
logging.getLogger("scvi").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Observation names are not unique")

torch.set_float32_matmul_precision("medium")


# =========================
# 基础路径配置
# =========================
SPLIT_BY_DRUG_DIR = Path("/datasets/fanpeishan/data/tahoe_split_by_drug")
DRUG_NAME_CSV = Path("/datasets/fanpeishan/data/docs/drug_name_list.csv")

OUTPUT_DIR = Path("/datasets/fanpeishan/data/tahoe_for_biolord")
TEMP_PRED_DIR = OUTPUT_DIR / "tmp_predictions"
TRAIN_LOG_DIR = OUTPUT_DIR / "biolord_train_logs"

CONTROL_DRUG = "DMSO_TF"


# =========================
# biolord 模型与训练参数
# =========================
N_LATENT = 256

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

MAX_EPOCHS = 32
BATCH_SIZE = 32768
EARLY_STOPPING_PATIENCE = 8
CHECK_VAL_EVERY_N_EPOCH = 2
RANDOM_SEED = 16

# 训练阶段显式传给 biolord.train() 的 num_workers。
# biolord 自己的 train() 默认是 0，这里写明避免歧义。
TRAIN_NUM_WORKERS = 0


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

VALID_CELL_TYPES = ["c4", "c17", "c18", "c23", "c38"]
TEST_CELL_TYPES = ["c19", "c20", "c22", "c24", "c31"]

ALL_CELL_TYPES = TRAIN_CELL_TYPES + VALID_CELL_TYPES + TEST_CELL_TYPES


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
    if psutil is None:
        print(f"[MEM] {tag}: psutil 未安装，跳过 RSS 记录")
        return

    rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[MEM] {tag}: RSS = {rss_gb:.2f} GB")


def try_malloc_trim():
    """
    Linux/glibc 下 best-effort 释放部分空闲 heap 回 OS。
    不保证一定有效，但有时能把 RSS 拉下来一点。
    """
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
    清理当前 model 在预测阶段对临时 AnnData 建立的 instance-specific managers。
    这一步非常关键。
    """
    try:
        model.deregister_manager()
        print("已清理 model 的临时 AnnDataManager")
    except Exception as e:
        print(f"警告：清理临时 AnnDataManager 失败: {repr(e)}")


def clear_main_manager(model, adata):
    """
    清理主训练 adata 对应的 manager。
    """
    success = False

    try:
        model.deregister_manager(adata)
        success = True
    except Exception:
        pass

    if not success:
        try:
            biolord.Biolord.deregister_manager(adata)
            success = True
        except Exception:
            pass

    if success:
        print("已清理主训练 adata 的 AnnDataManager")
    else:
        print("警告：主训练 adata 的 AnnDataManager 清理失败")


def build_drug_specific_adata(drug_index, control_drug_index):
    """
    构建“当前药物专属”的 biolord 训练数据。
    """
    all_celltype_adatas = []

    print("读取并构建当前药物专属 AnnData ...")
    log_rss("build_drug_specific_adata:start")

    for idx, cell_type in enumerate(ALL_CELL_TYPES, start=1):
        print(f"  [{idx}/{len(ALL_CELL_TYPES)}] 当前 cell_type: {cell_type}")

        ctrl_block = sc.read_h5ad(get_split_file(cell_type, control_drug_index))
        drug_block = sc.read_h5ad(get_split_file(cell_type, drug_index))

        adata_cell = ad.concat(
            [ctrl_block, drug_block],
            join="inner",
            merge="same",
            index_unique=None,
        )
        adata_cell.obs_names_make_unique()

        if cell_type in TRAIN_CELL_TYPES:
            adata_cell.obs["split"] = "train"
        elif cell_type in VALID_CELL_TYPES:
            adata_cell.obs["split"] = "valid"
        else:
            adata_cell.obs["split"] = "ood"

        all_celltype_adatas.append(adata_cell)

        print(f"    shape: {adata_cell.shape}")
        log_rss(f"after loading {cell_type}")

        del ctrl_block, drug_block
        cleanup_memory()

    adata = ad.concat(
        all_celltype_adatas,
        join="inner",
        merge="same",
        index_unique=None,
    )
    adata.obs_names_make_unique()

    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obs["drug_index"] = pd.Categorical(adata.obs["drug_index"].astype(int))
    adata.obs["split"] = pd.Categorical(
        adata.obs["split"],
        categories=["train", "valid", "ood"],
    )

    print("当前药物专属 AnnData 构建完成")
    print(f"adata shape: {adata.shape}")
    print(f"train cells: {(adata.obs['split'] == 'train').sum()}")
    print(f"valid cells: {(adata.obs['split'] == 'valid').sum()}")
    print(f"ood cells  : {(adata.obs['split'] == 'ood').sum()}")
    print(f"drug_index set: {sorted(adata.obs['drug_index'].astype(int).unique().tolist())}")
    log_rss("build_drug_specific_adata:end")

    return adata


def sample_source_indices(source_indices, n_target, random_seed):
    """
    先在 source control 上采样到目标大小。
    这样就不需要先预测全部 source，再做下采样。
    """
    if len(source_indices) == 0:
        raise RuntimeError("source_indices 为空，无法做预测")

    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(
        source_indices,
        size=n_target,
        replace=(n_target > len(source_indices)),
    )
    return sampled


def build_prediction_from_skeleton(X_pred, stim_skeleton):
    """
    用预测表达直接替换骨架的 X。
    这里要求 X_pred 的行数已经等于 stim_skeleton.n_obs。
    """
    X_pred = np.asarray(X_pred, dtype=np.float32)
    X_pred = np.clip(X_pred, a_min=0.0, a_max=None)

    if X_pred.shape[0] != stim_skeleton.n_obs:
        raise RuntimeError(
            f"预测行数与骨架行数不一致: pred={X_pred.shape[0]}, target={stim_skeleton.n_obs}"
        )

    adata_pred = ad.AnnData(
        X=X_pred,
        obs=stim_skeleton.obs.copy(),
        var=stim_skeleton.var.copy(),
    )
    adata_pred.obs["pred_or_real"] = "pred"
    return adata_pred


@torch.no_grad()
def predict_single_target_batched(
    model,
    adata_ref,
    sampled_source_indices,
    target_attribute,
    target_value,
    pred_batch_size,
):
    """
    用更轻量的方式替代 model.compute_prediction_adata()。

    思路：
    1. 只对已经采样后的 source cells 做预测；
    2. 目标属性（这里是 drug_index）只构造一个 1-row template；
    3. 按 batch 前向，不再一次性把整个 source 物化成一个大 batch；
    4. 不再枚举所有 target 类别，只预测当前一个 target。

    修复点：
    - 当最后一个 batch 的 batch_size=1 时，biolord 内部可能把输出 squeeze 成 1D；
    - 这里显式把 1D 预测恢复成 shape=(1, n_genes)，保证所有 batch 输出都是 2D；
    - 在 concatenate 前增加维度检查，便于更早发现异常。
    """
    # 采样后的 source adata（其行数已经等于最终目标块大小）。
    adata_source = adata_ref[sampled_source_indices].copy()
    if adata_source.n_obs == 0:
        raise RuntimeError("adata_source 为空，无法预测")

    # 找到当前 target_value 在 adata_ref 中的一个代表样本，用来抽取正确编码后的 target attribute tensor。
    target_mask = adata_ref.obs[target_attribute].astype(int).to_numpy() == int(target_value)
    target_indices = np.flatnonzero(target_mask)
    if len(target_indices) == 0:
        raise RuntimeError(f"在 adata_ref 中找不到 target {target_attribute}={target_value}")

    adata_target_one = adata_ref[target_indices[:1]].copy()

    # 通过 get_dataset() 拿到 target attribute 的“编码后张量模板”。
    target_dataset = model.get_dataset(adata_target_one)
    target_template = target_dataset[target_attribute][0, :].detach().cpu()

    preds = []

    # 用 batch dataloader，而不是 get_dataset(adata_source) 的整块物化。
    scdl = model._make_data_loader(
        adata=adata_source,
        indices=np.arange(adata_source.n_obs),
        batch_size=pred_batch_size,
        shuffle=False,
    )

    model.module.eval()

    for batch_id, tensors in enumerate(scdl, start=1):
        batch_size_cur = next(iter(tensors.values())).shape[0]

        batch_tensors = {}
        for key, val in tensors.items():
            if torch.is_tensor(val):
                batch_tensors[key] = val.to(model.device)
            else:
                batch_tensors[key] = val

        batch_tensors[target_attribute] = (
            target_template.unsqueeze(0).repeat(batch_size_cur, 1).to(model.device)
        )

        pred_mean, _ = model.module.get_expression(batch_tensors)

        pred_np = pred_mean.detach().cpu().numpy().astype(np.float32, copy=False)

        # 修复 singleton batch 被 squeeze 成 1D 的问题
        if pred_np.ndim == 1:
            pred_np = pred_np[None, :]
        elif pred_np.ndim != 2:
            raise RuntimeError(
                f"pred_mean 维度异常: ndim={pred_np.ndim}, shape={pred_np.shape}, "
                f"batch_id={batch_id}, batch_size_cur={batch_size_cur}"
            )

        # 额外做一个行数一致性检查
        if pred_np.shape[0] != batch_size_cur:
            raise RuntimeError(
                f"预测输出行数与当前 batch size 不一致: "
                f"pred_np.shape={pred_np.shape}, batch_size_cur={batch_size_cur}, batch_id={batch_id}"
            )

        preds.append(pred_np)

        del batch_tensors, pred_mean, pred_np, tensors
        if batch_id % 4 == 0:
            cleanup_memory()

    # concatenate 前做统一检查，便于排查未来其他异常
    for i, arr in enumerate(preds):
        if arr.ndim != 2:
            raise RuntimeError(f"preds[{i}] 维度异常: ndim={arr.ndim}, shape={arr.shape}")

    X_pred = np.concatenate(preds, axis=0)

    del preds, adata_source, adata_target_one, target_dataset, target_template, scdl
    cleanup_memory()

    return X_pred

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


def run_training_and_prediction_for_one_drug(
    drug_index,
    drug_name_list,
    control_drug_index,
):
    drug_name = drug_name_list[drug_index]

    print("\n" + "-" * 100)
    print(f"当前药物: {drug_name} (drug_index={drug_index})")
    log_rss(f"drug_{drug_index}:before_build")

    if SKIP_EXISTING_PRED_FILES and all_prediction_files_exist_for_drug(drug_index):
        print(f"drug_index={drug_index} 的 5 个测试 cell_type 预测文件都已存在，跳过")
        return

    model = None
    adata_drug = None

    try:
        adata_drug = build_drug_specific_adata(
            drug_index=drug_index,
            control_drug_index=control_drug_index,
        )
        log_rss(f"drug_{drug_index}:after_build")

        print("开始 setup_anndata ...")
        biolord.Biolord.setup_anndata(
            adata=adata_drug,
            categorical_attributes_keys=["cell_type", "drug_index"],
            layer=None,
        )
        log_rss(f"drug_{drug_index}:after_setup_anndata")

        print("开始构建 biolord 模型 ...")
        model = biolord.Biolord(
            adata=adata_drug,
            n_latent=N_LATENT,
            model_name=f"tahoe_biolord_drugindex_{drug_index}",
            module_params=MODULE_PARAMS,
            train_classifiers=False,
            split_key="split",
            train_split="train",
            valid_split="valid",
            test_split="ood",
        )
        log_rss(f"drug_{drug_index}:after_model_init")

        print("开始训练 biolord 模型 ...")
        model.train(
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            plan_kwargs=TRAIN_PLAN_KWARGS,
            early_stopping=True,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
            enable_checkpointing=False,
            default_root_dir=str(TRAIN_LOG_DIR / f"drugindex_{drug_index}"),
            num_workers=TRAIN_NUM_WORKERS,
        )
        print("当前药物模型训练完成")
        log_rss(f"drug_{drug_index}:after_train")

        for test_cell_type in TEST_CELL_TYPES:
            save_dir = TEMP_PRED_DIR / test_cell_type
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{test_cell_type}_drugindex_{drug_index}_pred.h5ad"

            if SKIP_EXISTING_PRED_FILES and save_path.exists():
                print(f"  -> 跳过 {test_cell_type}，文件已存在: {save_path}")
                continue

            print(f"  -> 开始预测 {test_cell_type}")

            stim_skeleton = sc.read_h5ad(get_split_file(test_cell_type, drug_index))
            n_target = stim_skeleton.n_obs

            source_mask = (
                (adata_drug.obs["cell_type"].astype(str).to_numpy() == test_cell_type)
                & (adata_drug.obs["drug_index"].astype(int).to_numpy() == control_drug_index)
            )
            source_indices = np.flatnonzero(source_mask)

            print(f"  -> source control 细胞数: {len(source_indices)}")
            print(f"  -> target 骨架细胞数: {n_target}")
            log_rss(f"drug_{drug_index}:{test_cell_type}:before_sampling")

            sampled_source_indices = sample_source_indices(
                source_indices=source_indices,
                n_target=n_target,
                random_seed=RANDOM_SEED + 1000 * drug_index + int(test_cell_type[1:]),
            )

            log_rss(f"drug_{drug_index}:{test_cell_type}:after_sampling")

            X_pred = predict_single_target_batched(
                model=model,
                adata_ref=adata_drug,
                sampled_source_indices=sampled_source_indices,
                target_attribute="drug_index",
                target_value=drug_index,
                pred_batch_size=PRED_BATCH_SIZE,
            )

            log_rss(f"drug_{drug_index}:{test_cell_type}:after_predict")

            adata_pred = build_prediction_from_skeleton(
                X_pred=X_pred,
                stim_skeleton=stim_skeleton,
            )

            adata_pred.write_h5ad(save_path)

            print(f"  -> 已保存: {save_path}")
            log_rss(f"drug_{drug_index}:{test_cell_type}:after_save_pred")

            del stim_skeleton, sampled_source_indices, X_pred, adata_pred
            cleanup_memory()

            # 关键：清掉预测时 adata_source / adata_target_one 等临时 AnnData 可能注册出来的 managers
            clear_temp_managers(model)
            cleanup_memory()
            log_rss(f"drug_{drug_index}:{test_cell_type}:after_cleanup")

    finally:
        if model is not None:
            # 先清临时 managers，再清主 adata manager
            clear_temp_managers(model)

        if model is not None and adata_drug is not None:
            clear_main_manager(model, adata_drug)

        del model, adata_drug
        cleanup_memory()
        log_rss(f"drug_{drug_index}:after_full_cleanup")


def merge_final_outputs(drug_name_list, control_drug_index):
    print("\n" + "=" * 100)
    print("开始合并最终预测文件")
    print("=" * 100)
    log_rss("before_final_merge")

    for test_cell_type in TEST_CELL_TYPES:
        print(f"合并 {test_cell_type} 的最终输出文件 ...")
        log_rss(f"merge_{test_cell_type}:start")

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
                        f"缺少预测文件，不能 merge: {pred_file}\n"
                        "请先把所有 drug 的临时预测都跑完，再执行 merge。"
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
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

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
    print("开始执行 biolord baseline（续跑/降内存版）")
    print(f"control drug: {CONTROL_DRUG} (drug_index={control_drug_index})")
    print(f"训练 cell type 数量: {len(TRAIN_CELL_TYPES)}")
    print(f"验证 cell type 数量: {len(VALID_CELL_TYPES)}")
    print(f"测试 cell type 数量: {len(TEST_CELL_TYPES)}")
    print(f"训练 batch size: {BATCH_SIZE}")
    print(f"预测 batch size: {PRED_BATCH_SIZE}")
    print(f"START_DRUG_INDEX: {START_DRUG_INDEX}")
    print(f"END_DRUG_INDEX  : {END_DRUG_INDEX}")
    print(f"本次选中的 drug_index: {selected_drug_indices}")
    print(f"RUN_FINAL_MERGE : {RUN_FINAL_MERGE}")
    print("=" * 100)
    log_rss("script_start")

if len(selected_drug_indices) == 0:
    if RUN_FINAL_MERGE:
        print("当前未选中任何 drug_index，进入 merge-only 模式")
    else:
        raise RuntimeError(
            "当前 START/END 配置下，没有选中任何目标 drug_index，"
            "且 RUN_FINAL_MERGE=False，因此没有任何可执行任务。"
        )
else:
    for loop_id, drug_index in enumerate(selected_drug_indices, start=1):
        print(f"\n[{loop_id}/{len(selected_drug_indices)}] 准备处理 drug_index={drug_index}")
        run_training_and_prediction_for_one_drug(
            drug_index=drug_index,
            drug_name_list=drug_name_list,
            control_drug_index=control_drug_index,
        )

if RUN_FINAL_MERGE:
    print("开始执行最终 merge，请确认所有 tmp_predictions 已经生成完成")
    merge_final_outputs(
        drug_name_list=drug_name_list,
        control_drug_index=control_drug_index,
    )

    print("\n" + "=" * 100)
    print("本轮流程完成")
    print("=" * 100)
    log_rss("script_end")