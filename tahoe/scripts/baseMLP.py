'''
cd /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/simple_models/baseMLP
CUDA_VISIBLE_DEVICES=5 python baseMLP.py
'''
from pathlib import Path
import copy
import gc
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

RESUME_FROM_DRUG_INDEX = 323  # 这个参数仅供调试时使用
# =========================
# 基础路径配置
# =========================
# 预处理后的按药物拆分数据目录。
# 每个文件对应“一个 cell_type + 一个 drug_index”的表达块。
SPLIT_BY_DRUG_DIR = Path("/data1/fanpeishan/STATE/for_state/data/split_by_drug")

# 最终输出目录。
# 第一阶段先写入按测试 cell type 分类的临时预测文件；
# 第二阶段再按完整 drug_index 顺序合并成最终输出。
OUTPUT_DIR = Path("/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/simple_models/baseMLP/outputs")
TEMP_PRED_DIR = OUTPUT_DIR / "tmp_predictions"

# 药物顺序文件。
# 该文件只有一列 drug_name，其行号就是 drug_index。
DRUG_NAME_CSV = Path("/data1/fanpeishan/STATE/for_state/about_baseline/docs/drug_name_list.csv")

# 对照药物名称。
CONTROL_DRUG = "DMSO_TF"


# =========================
# 训练参数
# =========================
# 这里严格沿用 scPerturBench/baseMLP.py 的核心设置：
# - 一个隐藏层，隐藏维度 1024
# - MSELoss
# - Adam(lr=0.001)
# - batch_size=128
# - max_epochs=100
# - patience=20
# - min_delta=1e-4
HIDDEN_DIM = 1024
LEARNING_RATE = 1e-3
BATCH_SIZE = 131072
MAX_EPOCHS = 64
EARLY_STOPPING_PATIENCE = 8
MIN_DELTA = 1e-4
VAL_RATIO = 0.2
RANDOM_SEED = 16

# 训练与推理设备。
DEVICE = torch.device("cuda")


# =========================
# 数据划分
# =========================
# 这里使用“不要验证集”的 Tahoe 划分：
# - 40 个训练 cell type
# - 10 个测试 cell type
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

TEST_CELL_TYPES = ["c4"] + cells(17, 20) + cells(22, 24) + ["c31", "c38"]


# =========================
# 模型定义
# =========================
# 这里刻意保持和 scPerturBench/baseMLP.py 一致：
# forward 中不在 fc1 后加激活，而是先 fc1、再 fc2、最后统一 ReLU。
# 因此模型输出天然非负。
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
def read_drug_name_list():
    # 按文件中的既定顺序读取药物名列表。
    return pd.read_csv(DRUG_NAME_CSV)["drug_name"].tolist()


def get_split_file(cell_type, drug_index):
    # 预处理文件的命名规则已经固定，因此直接按规则拼接路径。
    return (
        SPLIT_BY_DRUG_DIR
        / f"celltype_{cell_type}"
        / f"celltype_{cell_type}_drugindex_{drug_index}.h5ad"
    )


def cleanup_memory():
    # 每一轮训练 / 推理 / 合并后，都主动清理 Python 与 CUDA 显存。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def adata_to_numpy(adata):
    # h5ad 的 X 可能是稀疏矩阵，也可能已经是 dense。
    # 这里统一转成 float32 的 numpy 数组，便于后续送入 PyTorch。
    if sp.issparse(adata.X):
        return adata.X.toarray().astype(np.float32)
    return np.asarray(adata.X, dtype=np.float32)


def sort_pred_file(path):
    # 临时预测文件名中带有 drug_index。
    # 这里按数值排序，保证最终合并顺序稳定。
    return int(path.stem.split("_drugindex_")[1].split("_")[0])


def build_training_arrays_for_one_drug(drug_index, control_drug_index):
    # 这个函数负责为“当前药物”构造 baseMLP 的监督训练数据。
    #
    # 做法严格贴近 scPerturBench 的 paired-sample 思路：
    # - 对每个训练 cell type，分别读取 control block 与当前药物 block；
    # - 取两者细胞数的最小值 N；
    # - 直接截断到前 N 个细胞；
    # - 将 control 作为输入 X，将刺激后表达作为目标 y；
    # - 最后把所有训练 cell type 的配对样本拼接起来。
    x_blocks = []
    y_blocks = []

    for cell_type in TRAIN_CELL_TYPES:
        ctrl_adata = sc.read_h5ad(get_split_file(cell_type, control_drug_index))
        stim_adata = sc.read_h5ad(get_split_file(cell_type, drug_index))

        x_ctrl = adata_to_numpy(ctrl_adata)
        y_stim = adata_to_numpy(stim_adata)

        n_pair = min(x_ctrl.shape[0], y_stim.shape[0])

        x_blocks.append(x_ctrl[:n_pair])
        y_blocks.append(y_stim[:n_pair])

        del ctrl_adata, stim_adata, x_ctrl, y_stim
        cleanup_memory()

    x_train = np.concatenate(x_blocks, axis=0).astype(np.float32)
    y_train = np.concatenate(y_blocks, axis=0).astype(np.float32)

    del x_blocks, y_blocks
    cleanup_memory()

    return x_train, y_train


def train_model(x_train, y_train):
    # 训练函数沿用 scPerturBench/baseMLP.py 的训练范式：
    # - TensorDataset
    # - random_split 做 8:2 的 train/val 划分
    # - MSELoss
    # - Adam(lr=0.001)
    # - Early stopping
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train)

    dataset = TensorDataset(x_tensor, y_tensor)

    val_size = int(len(dataset) * VAL_RATIO)
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

    best_val_loss = float("inf")
    best_state_dict = None
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
        # 只有当验证损失至少改善了 MIN_DELTA，才认为真的变好了。
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
    # 推理阶段也按 batch 进行，避免一次性把所有测试细胞送入显存。
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 每次重跑脚本时，先清理旧的临时预测结果，避免和上一轮混在一起。
    # if TEMP_PRED_DIR.exists():
    #     shutil.rmtree(TEMP_PRED_DIR)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)

    drug_name_list = read_drug_name_list()
    drug_to_index = {drug_name: idx for idx, drug_name in enumerate(drug_name_list)}

    control_drug_index = drug_to_index[CONTROL_DRUG]
    target_drug_indices = [
        idx for idx, drug_name in enumerate(drug_name_list)
        if drug_name != CONTROL_DRUG
    ]

    print("=" * 100)
    print("开始执行 Tahoe 版本的 baseMLP 训练与推理流程")
    print(f"运行设备: {DEVICE}")
    print(f"control drug: {CONTROL_DRUG} (drug_index={control_drug_index})")
    print(f"训练 cell type 数量: {len(TRAIN_CELL_TYPES)}")
    print(f"测试 cell type 数量: {len(TEST_CELL_TYPES)}")
    print(f"目标药物数量: {len(target_drug_indices)}")
    print("=" * 100)

    # ============================================================
    # 第一阶段：按药物训练 + 对 10 个测试 cell type 推理。
    # 对每一个非 control 药物：
    # 1) 用 40 个训练 cell type 的 control/stim block 构造 paired sample；
    # 2) 训练一个 baseMLP；
    # 3) 对 10 个测试 cell type 的 control block 做推理；
    # 4) 用对应药物的测试 skeleton 替换 X，写出临时预测文件。
    # ============================================================
    for loop_id, drug_index in enumerate(target_drug_indices, start=1):
        
        if drug_index < RESUME_FROM_DRUG_INDEX:
            continue

        drug_name = drug_name_list[drug_index]

        print("\n" + "-" * 100)
        print(f"[{loop_id}/{len(target_drug_indices)}] 当前药物: {drug_name} (drug_index={drug_index})")

        print("构造当前药物的训练数组 ...")
        x_train, y_train = build_training_arrays_for_one_drug(
            drug_index=drug_index,
            control_drug_index=control_drug_index,
        )

        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        print("开始训练 baseMLP 模型 ...")
        model = train_model(x_train, y_train)
        print("模型训练完成")

        del x_train, y_train
        cleanup_memory()

        # 当前药物训练完成后，逐个测试 cell type 做推理。
        for test_cell_type in TEST_CELL_TYPES:
            print(f"  -> 开始预测 {test_cell_type}")

            # 输入只使用测试 cell type 的 control 细胞。
            ctrl_test = sc.read_h5ad(get_split_file(test_cell_type, control_drug_index))
            x_test = adata_to_numpy(ctrl_test)

            pred_full = predict_in_batches(model, x_test)

            # skeleton-replacement 协议：
            # 读取测试 cell type 在当前药物下的真实块，仅借用其 n_obs / obs / var。
            stim_skeleton = sc.read_h5ad(get_split_file(test_cell_type, drug_index))
            n_d = stim_skeleton.n_obs

            # 若预测细胞数与 skeleton 细胞数不同，则按照真实块大小重采样。
            # 这样最终输出文件中，每个药物块的行数和测试集真实结构完全一致。
            rng = np.random.default_rng(RANDOM_SEED + drug_index)
            sample_index = rng.choice(len(pred_full), size=n_d, replace=(n_d > len(pred_full)))
            x_sample = pred_full[sample_index].astype(np.float32)

            # 输出文件直接沿用 skeleton 的 obs / var，只替换 X。
            # pred_or_real 标记为 pred，便于后续区分。
            adata_pred = ad.AnnData(
                X=x_sample,
                obs=stim_skeleton.obs.copy(),
                var=stim_skeleton.var.copy(),
            )
            adata_pred.obs["pred_or_real"] = "pred"

            save_dir = TEMP_PRED_DIR / test_cell_type
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{test_cell_type}_drugindex_{drug_index}_pred.h5ad"
            adata_pred.write_h5ad(save_path)

            print(f"  -> 已保存: {save_path}")
            print(f"  -> control 模板细胞数: {ctrl_test.n_obs}, 骨架细胞数: {n_d}, 预测矩阵最小值: {x_sample.min():.6f}")

            del ctrl_test, x_test, pred_full, stim_skeleton, x_sample, adata_pred
            cleanup_memory()

        del model
        cleanup_memory()

    # ============================================================
    # 第二阶段：按测试 cell type 合并最终输出。
    # 最终输出遵循你当前 scGen baseline 的文件组织协议：
    # - control block：保留真实 control 表达
    # - 非 control block：使用第一阶段的预测结果
    # - 按完整 drug_index 顺序 concat
    # ============================================================
    print("\n" + "=" * 100)
    print("开始合并最终预测文件")
    print("=" * 100)

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

        del output_blocks, final_adata
        cleanup_memory()

    print("\n" + "=" * 100)
    print("全部流程完成")
    print("最终输出文件:")
    for test_cell_type in TEST_CELL_TYPES:
        print(OUTPUT_DIR / f"{test_cell_type}_predicted.h5ad")
    print("=" * 100)
