# export CUDA_VISIBLE_DEVICES=2
import os
import warnings

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
OUTPUT_DIR = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/scVI/"
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LIBRARY_SIZE = 4000.0

# =========================
# 工具函数
# =========================
def build_drug_dose(adata):
    adata.obs["drug_dose"] = (
        adata.obs["drug"].astype(str) + "_" + adata.obs["dose"].astype(str)
    )
    adata.obs["drug_dose"] = adata.obs["drug_dose"].astype("category")
    return adata


# =========================
# 读取数据
# =========================
print("Reading data from:", TRAIN_PATH)
adata = sc.read_h5ad(TRAIN_PATH)
adata.obs_names_make_unique()
adata = build_drug_dose(adata)

print("Data:", adata)
print("Total conditions:", adata.obs["drug_dose"].nunique())
print("Warning: current input is normalize_total(target_sum=4000) data without raw counts.")
print("Using a non-standard scVI setup with gene_likelihood='nb' as a practical fallback.")

# =========================
# 注册 SCVI 输入
# 当前没有 raw counts layer，只能直接使用 X
# =========================
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="drug_dose",
)

# =========================
# 构建模型
# 说明：
# 1. 在当前三种候选里选择 nb，作为最稳妥的折中
# 2. observed library size 在当前数据上基本恒定为 4000
# =========================
model = scvi.model.SCVI(
    adata,
    n_layers=3,
    n_hidden=512,
    n_latent=64,
    dispersion="gene-batch",
    gene_likelihood="nb",
    use_observed_lib_size=True,
)

print("模型架构：")
print(model)

# =========================
# 训练模型
# =========================
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = 1 if torch.cuda.is_available() else "auto"

print("Start training...")
model.train(
    max_epochs=200,
    accelerator=accelerator,
    devices=devices,
    batch_size=65536,
    early_stopping=True,
    check_val_every_n_epoch=15,
)

# =========================
# 保存模型
# =========================
model.save(MODEL_DIR, overwrite=True)
print("Model saved to:", MODEL_DIR)

"""
(scvi_baseline) fanpeishan@ubun:/data1/fanpeishan$ CUDA_VISIBLE_DEVICES=2 python /data1/fanpeishan/STATE/for_state/set_baseline/scripts/deeplearning_baseline/scVI/train_scVI.py
Seed set to 42
Reading data from: /data1/fanpeishan/STATE/for_state/data/for_chemCPA/train_c33_c34_c35.h5ad
Data: AnnData object with n_obs × n_vars = 6131754 × 2000
    obs: 'drug', 'cell_type', 'drugname_drugconc', 'dose', 'drug_dose'
Total conditions: 1138
Warning: current input is normalize_total(target_sum=4000) data without raw counts.
Using a non-standard scVI setup with gene_likelihood='nb' as a practical fallback.
模型架构：
SCVI model with the following parameters: 
n_hidden: 512, n_latent: 64, n_layers: 3, dropout_rate: 0.1, dispersion: gene-batch, gene_likelihood: nb, latent_distribution: normal.
Training status: Not Trained
Model's adata is minified?: False

Start training...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [2]
/data3/fanpeishan/miniconda3/envs/scvi_baseline/lib/python3.10/site-packages/lightning/pytorch/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
Epoch 200/200: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [2:27:14<00:00, 44.57s/it, v_num=1, train_loss_step=529, train_loss_epoch=528]`Trainer.fit` stopped: `max_epochs=200` reached.
Epoch 200/200: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [2:27:14<00:00, 44.17s/it, v_num=1, train_loss_step=529, train_loss_epoch=528]
Model saved to: /data1/fanpeishan/STATE/for_state/set_baseline/outputs/scVI/model
"""