# export CUDA_VISIBLE_DEVICES=2
import os
import warnings

import biolord
import numpy as np
import scanpy as sc
import scvi
import torch

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("medium")
scvi.settings.seed = 42

# =========================
# 路径配置
# =========================
TRAIN_PATH = "/data1/fanpeishan/STATE/for_state/data/for_chemCPA/train_c33_c34_c35.h5ad"
OUTPUT_DIR = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/biolord/"
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
VAR_NAMES_PATH = os.path.join(OUTPUT_DIR, "train_var_names.txt")

CONTROL_DRUG = "DMSO_TF"
RANDOM_SEED = 42

# 训练参数
VALID_FRACTION = 0.15
MAX_EPOCHS = 30
BATCH_SIZE = 8192

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# 工具函数
# =========================
def prepare_obs(adata):
    adata.obs["drug"] = adata.obs["drug"].astype(str)
    adata.obs["dose"] = adata.obs["dose"].astype(np.float32)
    return adata


def build_split(adata, valid_fraction=0.02, seed=42):
    rng = np.random.default_rng(seed)
    split = np.full(adata.n_obs, "train", dtype=object)

    n_valid = max(1, int(round(adata.n_obs * valid_fraction)))
    valid_idx = rng.choice(adata.n_obs, size=n_valid, replace=False)
    split[valid_idx] = "test"

    adata.obs["split"] = split
    adata.obs["split"] = adata.obs["split"].astype("category")
    return adata


# =========================
# 读取训练数据
# =========================
print("Reading train data:", TRAIN_PATH)
adata = sc.read_h5ad(TRAIN_PATH)
adata.obs_names_make_unique()
adata = prepare_obs(adata)
adata = build_split(adata, valid_fraction=VALID_FRACTION, seed=RANDOM_SEED)

print("Train AnnData:", adata)
print("Total drugs:", adata.obs["drug"].nunique())
print("Total doses:", adata.obs["dose"].nunique())
print("Control cells:", int((adata.obs["drug"] == CONTROL_DRUG).sum()))
print("Warning: current input is normalize_total(target_sum=4000) data without raw counts.")
print("Using biolord with gene_likelihood='normal'.")

# 保存训练基因顺序，供预测阶段对齐
np.savetxt(VAR_NAMES_PATH, adata.var_names.to_numpy(), fmt="%s")

# =========================
# biolord 注册
# 只把 drug 和 dose 当作已知 perturbation 属性
# 不把 cell_type 注册成已知属性，避免 unseen NCI-H23 推理失败
# =========================
biolord.Biolord.setup_anndata(
    adata,
    ordered_attributes_keys=["dose"],
    categorical_attributes_keys=["drug"],
    retrieval_attribute_key=None,
    layer=None,
)

# =========================
# 构建模型
# =========================
model = biolord.Biolord(
    adata=adata,
    model_name="chemCPA_biolord",
    n_latent=128,
    split_key="split",
    module_params={
        "gene_likelihood": "normal",
        "n_latent_attribute_categorical": 64,
        "n_latent_attribute_ordered": 16,
        "decoder_width": 512,
        "decoder_depth": 4,
        "decoder_dropout_rate": 0.1,
        "unknown_attributes": True,
        "seed": RANDOM_SEED,
    },
)

print("模型架构：")
print(model)

# 注意：
# biolord 内置的 max_epochs 自动估计对超大数据集会偏小，
# 你的数据量很大，因此这里显式指定 MAX_EPOCHS。
print("Start training...")
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
device = "auto"

model.train(
    max_epochs=MAX_EPOCHS,
    accelerator=accelerator,
    device=device,
    batch_size=BATCH_SIZE,
    early_stopping=True,
    check_val_every_n_epoch=1,
    early_stopping_patience=10,
    enable_checkpointing=False,
)

print("Saving model to:", MODEL_DIR)
model.save(MODEL_DIR, overwrite=True, save_anndata=False)

print("Training finished.")
print("Model saved to:", MODEL_DIR)
print("Gene order saved to:", VAR_NAMES_PATH)

"""
(biolord) fanpeishan@ubun:/data1/fanpeishan$ CUDA_VISIBLE_DEVICES=1 python /data1/fanpeishan/STATE/for_state/set_baseline/scripts/deeplearning_baseline/biolord/train_biolord.py
Seed set to 42
Reading train data: /data1/fanpeishan/STATE/for_state/data/for_chemCPA/train_c33_c34_c35.h5ad
Train AnnData: AnnData object with n_obs × n_vars = 6131754 × 2000
    obs: 'drug', 'cell_type', 'drugname_drugconc', 'dose', 'split'
Total drugs: 380
Total doses: 4
Control cells: 144244
Warning: current input is normalize_total(target_sum=4000) data without raw counts.
Using biolord with gene_likelihood='normal'.
Seed set to 42
模型架构：
Biolord training status: Not trained

Start training...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [1]
/data3/fanpeishan/miniconda3/envs/biolord/lib/python3.12/site-packages/lightning/pytorch/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
Epoch 11/30:  37%|▎| 11/30 [22:40<39:09, 123.67s/it, v_num=1, val_generative_mean_accuracy=0.552, val_generative_var_accuracy=-4.91, val_biolord_metric=-2.18, val_LOSS_KEYS.RECONSTRUCTION=111, val
Monitored metric val_biolord_metric did not improve in the last 10 records. Best score: 0.418. Signaling Trainer to stop.
Saving model to: /data1/fanpeishan/STATE/for_state/set_baseline/outputs/biolord/model
Training finished.
Model saved to: /data1/fanpeishan/STATE/for_state/set_baseline/outputs/biolord/model
Gene order saved to: /data1/fanpeishan/STATE/for_state/set_baseline/outputs/biolord/train_var_names.txt
"""
