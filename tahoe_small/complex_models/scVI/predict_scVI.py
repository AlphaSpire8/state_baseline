# export CUDA_VISIBLE_DEVICES=2
import os
import warnings
import re

import anndata as ad
import numpy as np
import scanpy as sc
import scvi
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("medium")
scvi.settings.seed = 42

# =========================
# 路径配置
# =========================
TEST_PATH = "/data1/fanpeishan/STATE/for_state/data/for_chemCPA/test_c37.h5ad"
MODEL_DIR = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/scVI/model/"
QUERY_MODEL_DIR = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/scVI/query_model/"
OUTPUT_PATH = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/scVI/c37_pred.h5ad"

CONTROL_DRUG = "DMSO_TF"
RANDOM_SEED = 42
LIBRARY_SIZE = 4000.0

os.makedirs(QUERY_MODEL_DIR, exist_ok=True)

# =========================
# 工具函数
# =========================
def _normalize_condition(x):
    x = re.sub(r"\s+", " ", str(x)).strip()
    x = re.sub(r"\s*_\s*", "_", x)
    return x


def build_drug_dose(adata):
    drug = adata.obs["drug"].astype(str).map(_normalize_condition)
    dose = adata.obs["dose"].astype(str).map(_normalize_condition)
    adata.obs["drug"] = drug
    adata.obs["dose"] = dose
    adata.obs["drug_dose"] = (drug + "_" + dose).map(_normalize_condition).astype(
        "category"
    )
    return adata

def to_dense_float32(matrix):
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


# =========================
# 读取测试数据
# =========================
print("Reading test data:", TEST_PATH)
adata = sc.read_h5ad(TEST_PATH)
adata.obs_names_make_unique()
adata = build_drug_dose(adata)

print("Test AnnData:", adata)
print("Total conditions in test:", adata.obs["drug_dose"].nunique())
print("Warning: current input is normalize_total(target_sum=4000) data without raw counts.")
print("Prediction will be decoded on the same library size scale:", LIBRARY_SIZE)

# 以测试集原始 X 为输出骨架
pred_X = to_dense_float32(adata.X)

# 按 reference 模型对齐 var 顺序与注册信息
scvi.model.SCVI.prepare_query_anndata(adata, MODEL_DIR, inplace=True)

# =========================
# 目标域 control
# =========================
control_mask = (adata.obs["drug"].astype(str) == CONTROL_DRUG).values
pert_mask = ~control_mask

if control_mask.sum() == 0:
    raise ValueError("Test dataset has no control cells.")

print("Control cells:", int(control_mask.sum()))
print("Perturbation cells:", int(pert_mask.sum()))

ctrl_adata = adata[control_mask].copy()

# =========================
# 仅用目标域 control 做 query adaptation
# =========================
print("Loading reference model and adapting on target-domain controls only...")
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = 1 if torch.cuda.is_available() else "auto"

query_model = scvi.model.SCVI.load_query_data(
    adata=ctrl_adata,
    reference_model=MODEL_DIR,
    freeze_expression=True,
    freeze_decoder_first_layer=True,
    accelerator=accelerator,
    device="auto",
)

query_model.train(
    max_epochs=20,
    accelerator=accelerator,
    devices=devices,
    batch_size=8192,
    early_stopping=True,
    check_val_every_n_epoch=2,
)

query_model.save(QUERY_MODEL_DIR, overwrite=True)
query_model.to_device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 按条件批量预测
# 统一使用 library_size=4000，使输出与输入尺度一致
# =========================
rng = np.random.default_rng(RANDOM_SEED)
ctrl_pool = query_model.adata

batch_state = query_model.adata_manager.get_state_registry("batch")
batch_lookup = {
    _normalize_condition(batch): str(batch)
    for batch in np.asarray(batch_state.categorical_mapping).tolist()
}

adata.obs["drug_dose_norm"] = adata.obs["drug_dose"].map(_normalize_condition)

cond_order = (
    adata.obs.loc[pert_mask, "drug_dose_norm"]
    .value_counts()
    .index
    .tolist()
)

unknown_conditions = [cond for cond in cond_order if cond not in batch_lookup]
if unknown_conditions:
    preview = ", ".join(map(str, unknown_conditions[:10]))
    raise ValueError(
        "Found perturbation conditions absent from scVI batch registry: "
        f"{preview}"
    )
print("Unique perturbation conditions:", len(cond_order))
print("Starting prediction...")

with torch.no_grad():
    for i, cond_norm in enumerate(cond_order, start=1):
        batch_name = batch_lookup[cond_norm]
        idx = np.where(
            (adata.obs["drug_dose_norm"].astype(str).values == cond_norm) & pert_mask
        )[0]
        n_cells = len(idx)

        if n_cells == 0:
            continue

        sampled_ctrl_idx = rng.choice(
            query_model.adata.n_obs,
            size=n_cells,
            replace=(n_cells > query_model.adata.n_obs),
        )

        pred = query_model.get_normalized_expression(
            adata=query_model.adata,
            indices=sampled_ctrl_idx,
            transform_batch=batch_name,
            library_size=LIBRARY_SIZE,
            batch_size=8192,
            return_numpy=True,
        ).astype(np.float32)

        pred = np.clip(pred, a_min=0.0, a_max=None)
        pred_X[idx] = pred

        print(f"[{i}/{len(cond_order)}] condition={batch_name} cells={n_cells}")
        
# =========================
# 保存结果
# =========================
print("Writing output:", OUTPUT_PATH)

adata_pred = ad.AnnData(
    X=sp.csr_matrix(pred_X) if sp.issparse(adata.X) else pred_X,
    obs=adata.obs.copy(),
    var=adata.var.copy(),
)
adata_pred.obs["pred_or_real"] = "pred"
adata_pred.obs.loc[control_mask, "pred_or_real"] = "real_ctrl"

adata_pred.write_h5ad(OUTPUT_PATH)

print("Prediction finished.")
print("Saved to:", OUTPUT_PATH)

# 运行结果
"""
Loading reference model and adapting on target-domain controls only...
INFO     File /data1/fanpeishan/STATE/for_state/set_baseline/outputs/scVI/model/model.pt already downloaded                                                                                         
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
/data3/fanpeishan/miniconda3/envs/scvi_baseline/lib/python3.10/site-packages/lightning/pytorch/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
Epoch 20/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:08<00:00,  2.66it/s, v_num=1, train_loss_step=411, train_loss_epoch=410]
`Trainer.fit` stopped: `max_epochs=20` reached.
Epoch 20/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:08<00:00,  2.41it/s, v_num=1, train_loss_step=411, train_loss_epoch=410]
Unique perturbation conditions: 1137
Starting prediction...
[1/1137] condition=Adagrasib_0.05 cells=23449
[2/1137] condition=Afatinib_0.5 cells=6042
[3/1137] condition=Almonertinib (mesylate)_0.5 cells=5277
（省略大量输出）
[1134/1137] condition=SBI-0640756_0.5 cells=19
[1135/1137] condition=Belzutifan_0.05 cells=18
[1136/1137] condition=Erdafitinib _0.5 cells=3
Writing output: /data1/fanpeishan/STATE/for_state/set_baseline/outputs/scVI/c37_pred.h5ad
Prediction finished.
Saved to: /data1/fanpeishan/STATE/for_state/set_baseline/outputs/scVI/c37_pred.h5ad
"""