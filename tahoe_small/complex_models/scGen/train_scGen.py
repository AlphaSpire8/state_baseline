"""
scGen 训练脚本
仅在 train_c33_c34_c35.h5ad (NCI-H1573/H1792/H2030) 上训练。

推断阶段若需预测新 cell_type，请在 predict 脚本中通过
adata_to_predict 传入该 cell_type 的对照细胞，而非在训练时混入测试数据。
"""
# export CUDA_VISIBLE_DEVICES=0
import scanpy as sc
import scgen
import torch
torch.set_float32_matmul_precision('high')

# ── 路径配置 ──────────────────────────────────────────────────────────────────
TRAIN_PATH  = "/data1/fanpeishan/STATE/for_state/data/for_chemCPA/train_c33_c34_c35.h5ad"
MODEL_PATH  = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/scgen/"

# ── 1. 加载训练数据 ──────────────────────────────────────────────────────────
print("Loading training data...")
adata = sc.read_h5ad(TRAIN_PATH)
print(adata)

# ── 2. 仅使用训练集训练 ───────────────────────────────────────────────────────
adata.obs_names_make_unique()
print(f"Training data: {adata.shape}")
print(f"Cell types : {adata.obs['cell_type'].unique().tolist()}")
print(f"Drugs      : {adata.obs['drug'].nunique()} unique (incl. ctrl)")

# ── 3. 注册 AnnData ──────────────────────────────────────────────────────────
# batch_key=扰动条件，labels_key=细胞类型（与官方教程一致）
scgen.SCGEN.setup_anndata(
    adata,
    batch_key="drug",        # 药物/对照条件
    labels_key="cell_type",  # 细胞系类型
)

# ── 4. 构建模型 ──────────────────────────────────────────────────────────────
model = scgen.SCGEN(adata)
print(model)

# ── 5. 训练 ──────────────────────────────────────────────────────────────────
model.train(
    max_epochs=150,
    batch_size=32768,
    early_stopping=True,
    early_stopping_patience=30,
)

# ── 6. 训练完成后保存模型 ────────────────────────────────────────────────────
model.save(MODEL_PATH, overwrite=True)
print(f"Model saved to: {MODEL_PATH}")

# 终端输出：
"""
(scGen) fanpeishan@ubun:/data1/fanpeishan$ export CUDA_VISIBLE_DEVICES=2
(scGen) fanpeishan@ubun:/data1/fanpeishan$ python /data1/fanpeishan/STATE/for_state/set_baseline/scripts/deeplearning_baseline/scGen/train_scGen.py
Loading training data...
AnnData object with n_obs × n_vars = 6131754 × 2000
    obs: 'drug', 'cell_type', 'drugname_drugconc', 'dose'
Training data: (6131754, 2000)
Cell types : ['NCI-H1573', 'NCI-H1792', 'NCI-H2030']
Drugs      : 380 unique (incl. ctrl)
SCGEN Model with the following params: 
n_hidden: 800, n_latent: 100, n_layers: 2, dropout_rate: 0.2
Training status: Not Trained

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [2]
Epoch 40/150:  26%|██████████████████████| 39/150 [30:14<1:32:26, 49.97s/it, v_num=1, train_loss_step=313, train_loss_epoch=314]
Monitored metric elbo_validation did not improve in the last 30 records. Best score: 1148.768. Signaling Trainer to stop.
Model saved to: /data1/fanpeishan/STATE/for_state/set_baseline/outputs/scgen/
"""