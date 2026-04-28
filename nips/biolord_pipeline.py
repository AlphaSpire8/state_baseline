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
import scvi
import torch


# ============================================================
# 基础配置
# ============================================================
# 数据目录：存放 3 个训练集、1 个验证集、2 个测试集
DATA_DIR = Path("/data1/fanpeishan/STATE/for_state/data/nips")

# 输出目录：存放训练日志与最终预测结果
BASE_DIR = Path(
    "/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/biolord"
)
PREDICT_DIR = BASE_DIR / "predict_result"
TRAIN_LOG_DIR = BASE_DIR / "train_logs"

# 文件划分
TRAIN_FILES = [
    "train_b_cells.h5ad",
    "train_nk_cells.h5ad",
    "train_t_cells_cd4.h5ad",
]
VALID_FILE = "val_t_cells_cd8.h5ad"
TEST_FILES = [
    "test_myeloid_cells.h5ad",
    "test_t_cells.h5ad",
]

# 对照药物名称
CONTROL_DRUG = "DMSO_TF"

# 随机种子：同时固定 numpy / torch / scvi
RANDOM_SEED = 16

# 训练与预测 batch size：按用户要求固定
TRAIN_BATCH_SIZE = 128
PRED_BATCH_SIZE = 4096

# 模型结构参数：主体参考 biolord 官方/参考实现的常用配置
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

# 训练计划参数：与参考实现保持一致
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

MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 8
CHECK_VAL_EVERY_N_EPOCH = 5
TRAIN_NUM_WORKERS = 0


# ============================================================
# 全局运行设置
# ============================================================
scvi.settings.dl_num_workers = 0
scvi.settings.dl_persistent_workers = False
scvi.settings.seed = RANDOM_SEED

logging.getLogger("scvi").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Observation names are not unique")

torch.set_float32_matmul_precision("medium")


# ============================================================
# 工具函数
# ============================================================
def set_random_seed(seed: int) -> None:
    """固定随机种子，保证训练与采样可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory() -> None:
    """做一次轻量内存清理，避免长脚本运行时显存/内存堆积。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_and_prepare_h5ad(file_name: str, split_name: str) -> ad.AnnData:
    """
    读取单个 h5ad，并补齐后续建模需要的基础列。

    这里只依赖两个关键 obs 列：
    1. obs['drug']
    2. obs['cell_name']
    """
    file_path = DATA_DIR / file_name
    adata = sc.read_h5ad(file_path)
    adata = adata.copy()

    # 保证名字唯一，减少后续 concat / 写文件时的歧义
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # 统一成字符串，避免 concat 后 category 编码出现隐式类型问题
    if "drug" not in adata.obs.columns:
        raise KeyError(f"文件缺少 obs['drug']: {file_path}")
    if "cell_name" not in adata.obs.columns:
        raise KeyError(f"文件缺少 obs['cell_name']: {file_path}")

    adata.obs["drug"] = adata.obs["drug"].astype(str)
    adata.obs["cell_name"] = adata.obs["cell_name"].astype(str)

    # 额外记录来源文件，便于后续精准筛出“当前测试集 control”作为预测源
    adata.obs["source_file"] = split_name

    return adata


def assert_same_genes(adatas: list[ad.AnnData], file_names: list[str]) -> None:
    """
    检查所有输入文件的基因顺序是否完全一致。

    之所以显式检查，而不是悄悄做 inner join，
    是因为您要求预测文件与真实文件的细胞数严格对齐，
    并且后续要直接送到 cell-eval。
    为了避免基因集合被静默截断，这里直接要求所有文件的 var_names 完全一致。
    """
    base_var_names = adatas[0].var_names
    base_file = file_names[0]

    for current_adata, current_file in zip(adatas[1:], file_names[1:]):
        if not base_var_names.equals(current_adata.var_names):
            raise RuntimeError(
                "检测到不同文件的基因集合或基因顺序不一致，无法安全直接拼接。\n"
                f"参考文件: {base_file}\n"
                f"不一致文件: {current_file}\n"
                "请先统一各文件的 var_names 与顺序后再运行。"
            )


def build_task_adata(test_file: str) -> tuple[ad.AnnData, ad.AnnData]:
    """
    为单个测试任务构建训练用 AnnData。

    设计规则如下：
    1. 三个 train 文件的全部细胞 -> split='train'
    2. val_t_cells_cd8.h5ad 的全部细胞 -> split='valid'
    3. 当前 test 文件中：
       - drug == DMSO_TF -> split='train'
       - drug != DMSO_TF -> split='ood'

    返回：
    - adata_task：供 biolord 训练与预测使用的总 AnnData
    - adata_test_raw：当前测试文件的原始副本，后续用于构造最终输出骨架
    """
    file_names = TRAIN_FILES + [VALID_FILE, test_file]
    all_adatas: list[ad.AnnData] = []

    print("=" * 100)
    print(f"开始构建任务数据：{test_file}")

    # 先读取训练集
    for file_name in TRAIN_FILES:
        adata_train = read_and_prepare_h5ad(file_name, split_name=file_name)
        adata_train.obs["split"] = "train"
        all_adatas.append(adata_train)
        print(f"已读取训练集 {file_name}，shape={adata_train.shape}")

    # 再读取固定验证集
    adata_valid = read_and_prepare_h5ad(VALID_FILE, split_name=VALID_FILE)
    adata_valid.obs["split"] = "valid"
    all_adatas.append(adata_valid)
    print(f"已读取验证集 {VALID_FILE}，shape={adata_valid.shape}")

    # 最后读取当前测试集
    adata_test_raw = read_and_prepare_h5ad(test_file, split_name=test_file)
    adata_test_task = adata_test_raw.copy()
    adata_test_task.obs["split"] = np.where(
        adata_test_task.obs["drug"].to_numpy() == CONTROL_DRUG,
        "train",
        "ood",
    )
    all_adatas.append(adata_test_task)
    print(f"已读取测试集 {test_file}，shape={adata_test_task.shape}")

    # 显式检查基因集合与顺序完全一致
    assert_same_genes(all_adatas, file_names)

    # 所有文件基因一致时，直接拼接即可，不做任何静默裁剪
    adata_task = ad.concat(
        all_adatas,
        join="inner",
        merge="same",
        index_unique=None,
    )
    adata_task.obs_names_make_unique()
    adata_task.var_names_make_unique()

    # 统一转成 category，便于 scvi/biolord 注册 categorical attributes
    adata_task.obs["cell_name"] = pd.Categorical(adata_task.obs["cell_name"])
    adata_task.obs["drug"] = pd.Categorical(adata_task.obs["drug"])
    adata_task.obs["source_file"] = pd.Categorical(adata_task.obs["source_file"])
    adata_task.obs["split"] = pd.Categorical(
        adata_task.obs["split"],
        categories=["train", "valid", "ood"],
    )

    print("任务数据构建完成")
    print(f"总 shape: {adata_task.shape}")
    print(f"train 细胞数: {(adata_task.obs['split'] == 'train').sum()}")
    print(f"valid 细胞数: {(adata_task.obs['split'] == 'valid').sum()}")
    print(f"ood   细胞数: {(adata_task.obs['split'] == 'ood').sum()}")

    return adata_task, adata_test_raw


def train_biolord_model(adata_task: ad.AnnData, task_name: str) -> biolord.Biolord:
    """
    训练单个测试任务对应的 biolord 模型。

    这里只使用两个 categorical attributes：
    - cell_name
    - drug
    """
    print("开始 setup_anndata ...")
    biolord.Biolord.setup_anndata(
        adata=adata_task,
        categorical_attributes_keys=["cell_name", "drug"],
        layer=None,
    )

    print("开始初始化 biolord 模型 ...")
    model = biolord.Biolord(
        adata=adata_task,
        n_latent=256,
        model_name=f"biolord_{task_name}",
        module_params=MODULE_PARAMS,
        train_classifiers=False,
        split_key="split",
        train_split="train",
        valid_split="valid",
        test_split="ood",
    )

    print("开始训练 biolord 模型 ...")
    model.train(
        max_epochs=MAX_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        plan_kwargs=TRAIN_PLAN_KWARGS,
        early_stopping=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
        enable_checkpointing=False,
        default_root_dir=str(TRAIN_LOG_DIR / task_name),
        # num_workers=TRAIN_NUM_WORKERS,
    )

    print("模型训练完成")
    return model


def sample_source_indices(
    source_indices: np.ndarray,
    n_target: int,
    random_seed: int,
) -> np.ndarray:
    """
    从目标测试 cell type 的真实 control 中采样 source cells。

    采样规则：
    - 若 control 数量足够，则无放回采样
    - 若 control 数量不足，则有放回采样

    这样可以保证每个目标药物的预测细胞数，
    与真实该药物块的细胞数完全一致。
    """
    if len(source_indices) == 0:
        raise RuntimeError("当前测试任务没有可用的 control 细胞，无法进行预测。")

    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.choice(
        source_indices,
        size=n_target,
        replace=(n_target > len(source_indices)),
    )
    return sampled_indices


@torch.no_grad()
def predict_single_drug_batched(
    model: biolord.Biolord,
    adata_ref: ad.AnnData,
    sampled_source_indices: np.ndarray,
    target_drug: str,
    pred_batch_size: int,
) -> np.ndarray:
    """
    用 batched forward 的方式预测单个目标药物的表达。

    这里不直接调用 compute_prediction_adata()，
    而是手动把 source batch 的 'drug' 属性替换成目标药物的编码模板，
    再分批前向，原因有两个：
    1. 输出行数更容易严格对齐到真实 target block 的细胞数；
    2. 更接近您要求的“真实骨架 + 预测表达”的组织方式。
    """
    # 先取采样后的 source control 细胞
    adata_source = adata_ref[sampled_source_indices].copy()
    if adata_source.n_obs == 0:
        raise RuntimeError("采样后的 source AnnData 为空，无法预测。")

    # 为了拿到目标药物的“编码后张量模板”，
    # 从参考数据中找任意一个该药物对应的样本即可。
    target_mask = adata_ref.obs["drug"].astype(str).to_numpy() == str(target_drug)
    target_indices = np.flatnonzero(target_mask)
    if len(target_indices) == 0:
        raise RuntimeError(f"在参考数据中找不到目标药物: {target_drug}")

    adata_target_one = adata_ref[target_indices[:1]].copy()
    target_dataset = model.get_dataset(adata_target_one)
    target_drug_template = target_dataset["drug"][0, :].detach().cpu()

    preds: list[np.ndarray] = []

    # 用内部 dataloader 分批前向，避免一次性物化太大的 source batch
    scdl = model._make_data_loader(
        adata=adata_source,
        indices=np.arange(adata_source.n_obs),
        batch_size=pred_batch_size,
        shuffle=False,
    )

    model.module.eval()

    for batch_id, tensors in enumerate(scdl, start=1):
        # 读取当前 batch 的真实 batch size
        batch_size_cur = next(iter(tensors.values())).shape[0]

        # 把 tensor 移到模型所在设备上
        batch_tensors = {}
        for key, val in tensors.items():
            if torch.is_tensor(val):
                batch_tensors[key] = val.to(model.device)
            else:
                batch_tensors[key] = val

        # 关键操作：只替换 drug 属性，cell_name 保持 source cell 本身不变
        batch_tensors["drug"] = (
            target_drug_template.unsqueeze(0).repeat(batch_size_cur, 1).to(model.device)
        )

        pred_mean, _ = model.module.get_expression(batch_tensors)
        pred_np = pred_mean.detach().cpu().numpy().astype(np.float32, copy=False)

        # 兼容 batch_size=1 时可能被内部 squeeze 成 1 维的情况
        if pred_np.ndim == 1:
            pred_np = pred_np[None, :]
        elif pred_np.ndim != 2:
            raise RuntimeError(
                f"预测输出维度异常: ndim={pred_np.ndim}, shape={pred_np.shape}, batch_id={batch_id}"
            )

        if pred_np.shape[0] != batch_size_cur:
            raise RuntimeError(
                "预测输出行数与当前 batch 大小不一致："
                f"pred_rows={pred_np.shape[0]}, batch_size_cur={batch_size_cur}, batch_id={batch_id}"
            )

        preds.append(pred_np)

        del batch_tensors, pred_mean, pred_np, tensors
        if batch_id % 4 == 0:
            cleanup_memory()

    # 拼接所有 batch 的预测结果
    x_pred = np.concatenate(preds, axis=0)

    del preds, adata_source, adata_target_one, target_dataset, target_drug_template, scdl
    cleanup_memory()

    return x_pred


def build_prediction_block(x_pred: np.ndarray, stim_skeleton: ad.AnnData) -> ad.AnnData:
    """
    用预测表达替换真实 stimulated block 的 X，构造最终预测块。

    这里直接复用真实 stimulated block 的 obs 与 var，
    这样最终输出文件与真实文件在细胞组织上天然一一对应。
    """
    x_pred = np.asarray(x_pred, dtype=np.float32)
    x_pred = np.clip(x_pred, a_min=0.0, a_max=None)

    if x_pred.shape[0] != stim_skeleton.n_obs:
        raise RuntimeError(
            "预测矩阵行数与目标骨架细胞数不一致："
            f"pred_rows={x_pred.shape[0]}, target_rows={stim_skeleton.n_obs}"
        )
    if x_pred.shape[1] != stim_skeleton.n_vars:
        raise RuntimeError(
            "预测矩阵列数与目标骨架基因数不一致："
            f"pred_cols={x_pred.shape[1]}, target_cols={stim_skeleton.n_vars}"
        )

    adata_pred = ad.AnnData(
        X=x_pred,
        obs=stim_skeleton.obs.copy(),
        var=stim_skeleton.var.copy(),
    )
    adata_pred.obs["pred_or_real"] = "pred"
    return adata_pred


def safe_clear_managers(model: biolord.Biolord | None, adata_task: ad.AnnData | None) -> None:
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


def run_one_task(test_file: str) -> None:
    """
    运行单个测试任务。

    输出文件只包含两部分：
    1. 真实 control
    2. 预测 stimulated

    因此最终预测文件与真实测试文件的细胞总数严格一致。
    """
    task_name = Path(test_file).stem
    output_path = PREDICT_DIR / f"{task_name}_pred.h5ad"

    print("\n" + "#" * 100)
    print(f"开始处理任务：{task_name}")
    print("#" * 100)

    model = None
    adata_task = None

    try:
        # 1) 构建该任务对应的训练/验证/测试总数据
        adata_task, adata_test_raw = build_task_adata(test_file)

        # 2) 训练 biolord 模型
        model = train_biolord_model(adata_task, task_name)

        # 3) 取当前测试集中的真实 control，作为最终输出的一部分
        real_ctrl = adata_test_raw[adata_test_raw.obs["drug"].astype(str) == CONTROL_DRUG].copy()
        if real_ctrl.n_obs == 0:
            raise RuntimeError(f"测试文件 {test_file} 中没有找到 control={CONTROL_DRUG} 细胞。")
        real_ctrl.obs["pred_or_real"] = "real_ctrl"

        # 4) 在 adata_task 中找到“当前测试文件的 control 细胞”，它们将作为预测 source
        source_mask = (
            (adata_task.obs["source_file"].astype(str).to_numpy() == test_file)
            & (adata_task.obs["drug"].astype(str).to_numpy() == CONTROL_DRUG)
        )
        source_indices = np.flatnonzero(source_mask)
        if len(source_indices) == 0:
            raise RuntimeError(f"任务 {task_name} 没有可用的测试集 control source cells。")

        print(f"可用 source control 细胞数: {len(source_indices)}")

        # 5) 按真实测试文件中的药物出现顺序，逐药物生成预测块
        drug_order = pd.unique(adata_test_raw.obs["drug"].astype(str))
        target_drugs = [drug for drug in drug_order if drug != CONTROL_DRUG]

        output_blocks = [real_ctrl]

        for drug_rank, target_drug in enumerate(target_drugs, start=1):
            stim_skeleton = adata_test_raw[adata_test_raw.obs["drug"].astype(str) == target_drug].copy()
            n_target = stim_skeleton.n_obs
            if n_target == 0:
                continue

            print("-" * 80)
            print(f"开始预测药物: {target_drug}")
            print(f"目标细胞数: {n_target}")

            # 这里用固定随机种子 + 药物序号，保证每个药物的采样可复现
            sampled_source_indices = sample_source_indices(
                source_indices=source_indices,
                n_target=n_target,
                random_seed=RANDOM_SEED + drug_rank,
            )

            x_pred = predict_single_drug_batched(
                model=model,
                adata_ref=adata_task,
                sampled_source_indices=sampled_source_indices,
                target_drug=target_drug,
                pred_batch_size=PRED_BATCH_SIZE,
            )

            adata_pred_block = build_prediction_block(x_pred, stim_skeleton)
            output_blocks.append(adata_pred_block)

            del stim_skeleton, sampled_source_indices, x_pred, adata_pred_block
            cleanup_memory()

        # 6) 拼接成最终预测文件
        adata_output = ad.concat(
            output_blocks,
            join="inner",
            merge="same",
            index_unique=None,
        )
        adata_output.obs_names_make_unique()

        # 再做一次总细胞数校验，确保与真实测试文件完全一致
        if adata_output.n_obs != adata_test_raw.n_obs:
            raise RuntimeError(
                "最终预测文件的细胞数与真实测试文件不一致："
                f"pred_n_obs={adata_output.n_obs}, real_n_obs={adata_test_raw.n_obs}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata_output.write_h5ad(output_path)

        print("-" * 80)
        print(f"任务 {task_name} 已完成")
        print(f"输出文件: {output_path}")
        print(f"输出 shape: {adata_output.shape}")

        del real_ctrl, output_blocks, adata_output, adata_test_raw
        cleanup_memory()

    finally:
        safe_clear_managers(model, adata_task)
        del model, adata_task
        cleanup_memory()


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    set_random_seed(RANDOM_SEED)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("开始执行 biolord pipeline")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {PREDICT_DIR}")
    print(f"训练日志目录: {TRAIN_LOG_DIR}")
    print(f"control drug: {CONTROL_DRUG}")
    print(f"训练文件: {TRAIN_FILES}")
    print(f"验证文件: {VALID_FILE}")
    print(f"测试文件: {TEST_FILES}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print(f"TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE}")
    print(f"PRED_BATCH_SIZE: {PRED_BATCH_SIZE}")
    print("=" * 100)

    for test_file in TEST_FILES:
        run_one_task(test_file)

    print("\n" + "=" * 100)
    print("全部任务完成")
    print("=" * 100)
