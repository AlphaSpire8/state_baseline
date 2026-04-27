"""
cd /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/baseMLP
CUDA_VISIBLE_DEVICES=5 python scripts/baseMLP.py
"""

from pathlib import Path
import copy
import gc
import hashlib
import re
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


torch.set_float32_matmul_precision("medium")


# =========================
# 基础路径配置
# =========================
BASE_DIR = Path("/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/baseMLP")
DATA_DIR = Path("/data1/fanpeishan/STATE/for_state/data/nips")

# 最终预测结果目录。
PREDICT_DIR = BASE_DIR / "predict_result"

# 临时预测块目录。
# 第一阶段先按“sample + cell_name + drug”保存临时预测块；
# 第二阶段再合并成最终的 {sample}_pred.h5ad。
TEMP_PRED_DIR = BASE_DIR / "tmp_predictions"

# 训练文件。
TRAIN_FILES = [
    DATA_DIR / "train_b_cells.h5ad",
    DATA_DIR / "train_nk_cells.h5ad",
    DATA_DIR / "train_t_cells_cd4.h5ad",
]

# 目标预测文件。
# 这里与后续 eval.sh 完全对齐。
TARGET_FILES = [
    DATA_DIR / "test_myeloid_cells.h5ad",
    DATA_DIR / "test_t_cells.h5ad",
    DATA_DIR / "val_t_cells_cd8.h5ad",
]

# 对照药物名称。
CONTROL_DRUG = "DMSO_TF"


# =========================
# 训练超参数
# =========================
# 这里保持和参考 baseMLP 尽量一致：
# - 单隐藏层
# - MSELoss
# - Adam(lr=1e-3)
# - 8:2 train/val
# - early stopping
HIDDEN_DIM = 1024
LEARNING_RATE = 1e-3
BATCH_SIZE = 131072
MAX_EPOCHS = 64
EARLY_STOPPING_PATIENCE = 8
MIN_DELTA = 1e-4
VAL_RATIO = 0.2
RANDOM_SEED = 16

# 运行设备。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 模型定义
# =========================
# 刻意保持参考实现的风格：
# 不在 fc1 后加激活，而是在 fc2 后统一过 ReLU。
class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


# =========================
# 工具函数
# =========================
def cleanup_memory():
    """主动清理 Python 和 CUDA 内存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def adata_to_numpy(adata):
    """
    将 AnnData.X 统一转成 float32 的 numpy 数组。
    这样后续可以直接送入 PyTorch。
    """
    if sp.issparse(adata.X):
        return adata.X.toarray().astype(np.float32)
    return np.asarray(adata.X, dtype=np.float32)


def unique_in_order(values):
    """
    保持首次出现顺序地去重。
    常用于读取 cell_name 顺序和 drug 顺序。
    """
    return pd.Index(values).drop_duplicates().tolist()


def safe_name(text):
    """
    将字符串转换成适合文件名的形式。
    这里只用于临时目录命名，不影响最终 h5ad 内容。
    """
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "empty"


def stable_int(text):
    """
    将字符串稳定映射成整数。
    不能直接用 Python 内置 hash，因为跨进程不稳定。
    """
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def get_temp_pred_path(sample_name, cell_name, drug_name):
    """
    返回某个临时预测块的保存路径。
    路径中同时保留可读名和稳定整数，避免重名冲突。
    """
    cell_dir = TEMP_PRED_DIR / sample_name / f"{safe_name(cell_name)}__{stable_int(cell_name)}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{safe_name(drug_name)}__{stable_int(drug_name)}_pred.h5ad"
    return cell_dir / file_name


def collect_train_drugs(train_adatas):
    """
    只从训练集统计非 control 药物集合。
    按首次出现顺序去重。
    """
    ordered_drugs = []
    seen = set()

    for adata in train_adatas.values():
        for drug_name in unique_in_order(adata.obs["drug"]):
            if drug_name == CONTROL_DRUG:
                continue
            if drug_name in seen:
                continue
            seen.add(drug_name)
            ordered_drugs.append(drug_name)

    return ordered_drugs


def build_training_arrays_for_one_drug(drug_name, train_adatas):
    """
    为“单个药物”构造监督训练数据。

    具体做法：
    - 遍历所有训练文件
    - 在每个训练文件内部，再按 cell_name 分组
    - 对每个训练 cell_name：
        x = 同一 cell_name 的 DMSO_TF
        y = 同一 cell_name 的当前 drug
    - 取两者细胞数最小值 N 做截断配对
    - 最后把所有训练 context 的配对样本拼接起来
    """
    x_blocks = []
    y_blocks = []

    for sample_name, adata_train in train_adatas.items():
        cell_name_list = unique_in_order(adata_train.obs["cell_name"])

        for cell_name in cell_name_list:
            mask_cell = adata_train.obs["cell_name"] == cell_name
            mask_ctrl = adata_train.obs["drug"] == CONTROL_DRUG
            mask_drug = adata_train.obs["drug"] == drug_name

            ctrl_adata = adata_train[mask_cell & mask_ctrl].copy()
            stim_adata = adata_train[mask_cell & mask_drug].copy()

            # 如果该训练 context 中没有当前药物，直接跳过。
            if ctrl_adata.n_obs == 0 or stim_adata.n_obs == 0:
                del ctrl_adata, stim_adata
                continue

            x_ctrl = adata_to_numpy(ctrl_adata)
            y_stim = adata_to_numpy(stim_adata)

            # 采用最简单直接的 paired-sample 协议：
            # 取前 N=min(n_ctrl, n_stim) 个细胞做一一配对。
            n_pair = min(x_ctrl.shape[0], y_stim.shape[0])

            x_blocks.append(x_ctrl[:n_pair])
            y_blocks.append(y_stim[:n_pair])

            del ctrl_adata, stim_adata, x_ctrl, y_stim

    if len(x_blocks) == 0:
        raise ValueError(f"药物 '{drug_name}' 在训练集中没有可用的配对样本。")

    x_train = np.concatenate(x_blocks, axis=0).astype(np.float32)
    y_train = np.concatenate(y_blocks, axis=0).astype(np.float32)

    del x_blocks, y_blocks
    cleanup_memory()

    return x_train, y_train


def train_model(x_train, y_train):
    """
    训练单药物 MLP。

    训练范式：
    - TensorDataset
    - random_split 做 8:2 train/val
    - MSELoss
    - Adam
    - Early stopping
    """
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train)

    dataset = TensorDataset(x_tensor, y_tensor)

    if len(dataset) < 2:
        raise ValueError("训练样本数不足，无法划分 train/val。")

    val_size = max(1, int(len(dataset) * VAL_RATIO))
    if val_size >= len(dataset):
        val_size = len(dataset) - 1
    train_size = len(dataset) - val_size

    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = OneHiddenLayerMLP(
        input_dim=x_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        output_dim=y_train.shape[1],
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 一开始就保存一份初始权重，确保 best_state_dict 永远不为 None。
    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # =========================
        # 训练阶段
        # =========================
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            batch_size_now = batch_x.size(0)
            train_loss_sum += loss.item() * batch_size_now
            train_sample_count += batch_size_now

        train_loss = train_loss_sum / train_sample_count

        # =========================
        # 验证阶段
        # =========================
        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)

                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                batch_size_now = batch_x.size(0)
                val_loss_sum += loss.item() * batch_size_now
                val_sample_count += batch_size_now

        val_loss = val_loss_sum / val_sample_count

        print(
            f"Epoch [{epoch:03d}/{MAX_EPOCHS}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        # =========================
        # Early stopping
        # =========================
        if best_val_loss - val_loss > MIN_DELTA:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"触发 early stopping，停止在 epoch={epoch}")
            break

    model.load_state_dict(best_state_dict)
    model.eval()

    del x_tensor, y_tensor, dataset, train_dataset, val_dataset, train_loader, val_loader
    cleanup_memory()

    return model


def predict_in_batches(model, x_input):
    """
    分 batch 推理，避免一次性占满显存。
    """
    dataset = TensorDataset(torch.from_numpy(x_input.astype(np.float32)))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    pred_blocks = []

    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_pred = model(batch_x).cpu().numpy().astype(np.float32)
            pred_blocks.append(batch_pred)

    pred = np.concatenate(pred_blocks, axis=0).astype(np.float32)

    del dataset, loader, pred_blocks
    cleanup_memory()

    return pred


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    PREDICT_DIR.mkdir(parents=True, exist_ok=True)

    # 每次重跑前，先清空旧的临时预测块，避免和上一轮混淆。
    if TEMP_PRED_DIR.exists():
        shutil.rmtree(TEMP_PRED_DIR)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("开始执行 baseMLP 训练与推理流程")
    print(f"运行设备: {DEVICE}")
    print(f"control drug: {CONTROL_DRUG}")
    print(f"训练文件数: {len(TRAIN_FILES)}")
    print(f"目标文件数: {len(TARGET_FILES)}")
    print(f"预测目录: {PREDICT_DIR}")
    print("=" * 100)

    # ------------------------------------------------------------
    # 一次性读取训练文件和目标文件。
    # 这样脚本结构更直接，后续按药物循环时不需要反复读盘。
    # ------------------------------------------------------------
    train_adatas = {}
    for file_path in TRAIN_FILES:
        sample_name = file_path.stem
        train_adatas[sample_name] = sc.read_h5ad(file_path)

    target_adatas = {}
    for file_path in TARGET_FILES:
        sample_name = file_path.stem
        target_adatas[sample_name] = sc.read_h5ad(file_path)

    # 只从训练集统计非 control 药物。
    train_drug_list = collect_train_drugs(train_adatas)

    print(f"训练药物数量（不含 {CONTROL_DRUG}）: {len(train_drug_list)}")
    print("训练 cell_name:")
    for sample_name, adata_train in train_adatas.items():
        for cell_name in unique_in_order(adata_train.obs["cell_name"]):
            print(f"  - {sample_name}: {cell_name}")

    print("目标文件中的 cell_name:")
    for sample_name, adata_target in target_adatas.items():
        for cell_name in unique_in_order(adata_target.obs["cell_name"]):
            print(f"  - {sample_name}: {cell_name}")

    # ============================================================
    # 第一阶段：按药物训练 + 生成临时预测块
    # ============================================================
    for loop_id, drug_name in enumerate(train_drug_list, start=1):
        print("\n" + "-" * 100)
        print(f"[{loop_id}/{len(train_drug_list)}] 当前药物: {drug_name}")

        # -------------------------
        # 构造当前药物的监督训练数据
        # -------------------------
        print("构造训练数组 ...")
        x_train, y_train = build_training_arrays_for_one_drug(
            drug_name=drug_name,
            train_adatas=train_adatas,
        )
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        # -------------------------
        # 训练当前药物模型
        # -------------------------
        print("开始训练模型 ...")
        model = train_model(x_train, y_train)
        print("模型训练完成")

        del x_train, y_train
        cleanup_memory()

        # -------------------------
        # 对每个目标文件做预测
        # -------------------------
        for sample_name, adata_target in target_adatas.items():
            print(f"  -> 开始处理目标文件: {sample_name}")

            file_cell_names = unique_in_order(adata_target.obs["cell_name"])

            for cell_name in file_cell_names:
                # 当前目标 cell_name 子集。
                adata_cell = adata_target[adata_target.obs["cell_name"] == cell_name].copy()

                # 如果这个目标 cell_name 根本不包含当前药物，则无需预测。
                drug_order = unique_in_order(adata_cell.obs["drug"])
                if drug_name not in drug_order:
                    del adata_cell
                    continue

                # 输入模板：当前目标 cell_name 自身的 control 细胞。
                ctrl_test = adata_cell[adata_cell.obs["drug"] == CONTROL_DRUG].copy()
                if ctrl_test.n_obs == 0:
                    raise ValueError(
                        f"目标文件 '{sample_name}' 的 cell_name '{cell_name}' 中找不到 control 细胞。"
                    )

                x_test = adata_to_numpy(ctrl_test)
                pred_full = predict_in_batches(model, x_test)

                # 真实药物块仅作为 skeleton，借用其细胞数 / obs / var。
                stim_skeleton = adata_cell[adata_cell.obs["drug"] == drug_name].copy()
                n_d = stim_skeleton.n_obs

                # 若预测模板细胞数与真实药物块细胞数不同，则按真实块大小重采样。
                # 采样策略和你之前确认的 baseControl 一致：
                # - n_d <= n_ctrl: 无放回采样
                # - n_d > n_ctrl : 有放回采样
                rng = np.random.default_rng(
                    RANDOM_SEED + stable_int(f"{sample_name}::{cell_name}::{drug_name}")
                )
                sample_index = rng.choice(
                    pred_full.shape[0],
                    size=n_d,
                    replace=(n_d > pred_full.shape[0]),
                )
                x_sample = pred_full[sample_index].astype(np.float32, copy=False)

                # 预测块沿用真实 skeleton 的 obs / var，只替换 X。
                adata_pred = ad.AnnData(
                    X=x_sample,
                    obs=stim_skeleton.obs.copy(),
                    var=stim_skeleton.var.copy(),
                )
                adata_pred.obs["pred_or_real"] = "pred"

                save_path = get_temp_pred_path(sample_name, cell_name, drug_name)
                adata_pred.write_h5ad(save_path)

                print(
                    f"    已保存: {save_path} | "
                    f"cell_name={cell_name} | "
                    f"control={ctrl_test.n_obs} | "
                    f"skeleton={n_d}"
                )

                del adata_cell, ctrl_test, x_test, pred_full, stim_skeleton, x_sample, adata_pred
                cleanup_memory()

        del model
        cleanup_memory()

    # ============================================================
    # 第二阶段：按目标文件合并最终输出
    # ============================================================
    print("\n" + "=" * 100)
    print("开始合并最终预测文件")
    print("=" * 100)

    for sample_name, adata_target in target_adatas.items():
        print(f"合并目标文件: {sample_name}")

        output_blocks = []
        file_cell_names = unique_in_order(adata_target.obs["cell_name"])

        # 这里保持与 baseControl 一致的组织方式：
        # 对每个 cell_name，
        # - 先放真实 control block
        # - 再按该 cell_name 自身实际出现的 drug 顺序放预测 block
        for cell_name in file_cell_names:
            adata_cell = adata_target[adata_target.obs["cell_name"] == cell_name].copy()
            drug_order = unique_in_order(adata_cell.obs["drug"])

            if CONTROL_DRUG not in drug_order:
                raise ValueError(
                    f"目标文件 '{sample_name}' 的 cell_name '{cell_name}' 中缺少 control 药物 '{CONTROL_DRUG}'。"
                )

            # 真实 control 直接保留。
            ctrl_real = adata_cell[adata_cell.obs["drug"] == CONTROL_DRUG].copy()
            ctrl_real.obs = ctrl_real.obs.copy()
            ctrl_real.obs["pred_or_real"] = "real_ctrl"
            output_blocks.append(ctrl_real)

            # 其余药物依次读取预测块。
            for drug_name in drug_order:
                if drug_name == CONTROL_DRUG:
                    continue

                pred_file = get_temp_pred_path(sample_name, cell_name, drug_name)

                # 按你的数据协议，这里理论上都应该存在。
                # 仍然保留防御式判断，便于发现异常。
                if not pred_file.exists():
                    print(f"    警告：缺少预测块，已跳过 -> {pred_file}")
                    continue

                output_blocks.append(sc.read_h5ad(pred_file))

            del adata_cell
            cleanup_memory()

        final_adata = ad.concat(
            output_blocks,
            join="inner",
            merge="same",
            index_unique=None,
        )
        final_adata.obs_names_make_unique()

        final_path = PREDICT_DIR / f"{sample_name}_pred.h5ad"
        final_adata.write_h5ad(final_path)

        print(f"已生成: {final_path}")
        print(f"最终 shape: {final_adata.shape}")

        del output_blocks, final_adata
        cleanup_memory()

    print("\n" + "=" * 100)
    print("全部流程完成")
    print("最终输出文件:")
    for file_path in TARGET_FILES:
        print(PREDICT_DIR / f"{file_path.stem}_pred.h5ad")
    print("=" * 100)