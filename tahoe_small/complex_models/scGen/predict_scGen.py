"""
scGen 推断脚本
以 test_c37.h5ad 为骨架，保持其细胞数、obs 元数据和 var 不变：
  - 对照组细胞（DMSO_TF）：X 保持原值不变
  - 非对照组细胞（drug-d）：从 ctrl_test 中随机采样等量细胞，施加 latent
    delta 后解码，将预测表达谱写入对应行，替换原始 X

预测逻辑（对每种药物 d，共 n_d 个处理细胞）：
  1. 从 ctrl_test 中随机采样 n_d 个细胞（有放回，若 ctrl 细胞数不足）
  2. Δ = mean_z(训练集 drug-d) - mean_z(训练集 DMSO)
  3. z_pred = z(采样的 ctrl 细胞) + Δ → decode → X_pred (n_d × n_genes)
  4. 将 X_pred 写入输出文件中对应行区间
"""

import logging
import warnings
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import scgen
import anndata as ad
import torch
from tqdm import tqdm

# 抑制 scvi-tools 冗余 INFO 日志
logging.getLogger("scvi").setLevel(logging.WARNING)
# 抑制 scgen 内部 balancer() 触发的重复 obs_names 警告（不影响结果正确性）
warnings.filterwarnings("ignore", message="Observation names are not unique")

# 修复 PyTorch 2.6 兼容性
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
torch.set_float32_matmul_precision('high')

# 允许推理时自动扩展未见过的类别（如新 cell_type）
import scvi.data._manager as _scvi_manager
_orig_tf = _scvi_manager.AnnDataManager.transfer_fields
def _patched_tf(self, adata_target, **kwargs):
    kwargs.setdefault('extend_categories', True)
    return _orig_tf(self, adata_target, **kwargs)
_scvi_manager.AnnDataManager.transfer_fields = _patched_tf


# ── 路径配置 ──────────────────────────────────────────────────────────────────
TRAIN_PATH  = "/data1/fanpeishan/STATE/for_state/data/for_chemCPA/train_c33_c34_c35.h5ad"
TEST_PATH   = "/data1/fanpeishan/STATE/for_state/data/for_chemCPA/test_c37.h5ad"
MODEL_PATH  = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/scgen/"
OUTPUT_PATH = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/scgen/c37_pred.h5ad"
CTRL_KEY    = "DMSO_TF"
RANDOM_SEED = 42

# ── 1. 加载数据 ───────────────────────────────────────────────────────────────
print("Loading training data...")
adata_train = sc.read_h5ad(TRAIN_PATH)

print("Loading test data...")
adata_test = sc.read_h5ad(TEST_PATH)
print(f"Test data shape: {adata_test.shape}")

# ── 2. 提取待推理 cell_type 的对照细胞（仅用于 adata_to_predict）──────────────
ctrl_test      = adata_test[adata_test.obs["drug"] == CTRL_KEY].copy()
print(f"number of ctrl cells in NCI-H23: {ctrl_test.n_obs}")

# ── 3. 加载已训练模型 ─────────────────────────────────────────────────────────
print("Loading model...")
model = scgen.SCGEN.load(dir_path=MODEL_PATH, adata=adata_train)

# ── 4. 确定待预测药物列表 ─────────────────────────────────────────────────────
drugs_all   = [d for d in adata_test.obs["drug"].unique() if d != CTRL_KEY]
valid_drugs = [d for d in drugs_all if (adata_train.obs["drug"] == d).sum() > 0]
skipped     = set(drugs_all) - set(valid_drugs)
if skipped:
    print(f"[SKIP] {len(skipped)} drugs not found in training set")
print(f"number of drugs to predict: {len(valid_drugs)}")

# ── 5. 构建输出 X 矩阵（以 test_c37 的行顺序为准）────────────────────────────
# 初始化为 float32 dense 矩阵，ctrl 行直接填入原始值
n_cells = adata_test.n_obs
n_genes = adata_test.n_vars
X_out   = np.empty((n_cells, n_genes), dtype='float32')

# 先将 ctrl 行的原始表达值写入
ctrl_mask = adata_test.obs["drug"] == CTRL_KEY
X_test    = adata_test.X.toarray() if sp.issparse(adata_test.X) else np.asarray(adata_test.X)
X_out[ctrl_mask.values] = X_test[ctrl_mask.values].astype('float32')

rng = np.random.default_rng(RANDOM_SEED)

# ── 6. 逐药物预测，按行写回 X_out ─────────────────────────────────────────────
for drug in tqdm(valid_drugs, desc="Predicting"):
    drug_mask = (adata_test.obs["drug"] == drug).values
    n_d       = int(drug_mask.sum())

    # 通过 adata_to_predict 传入目标 cell_type 的 ctrl 细胞，
    # 不要求该 cell_type 出现在训练集 labels 中
    pred, _ = model.predict(
        ctrl_key=CTRL_KEY,
        stim_key=drug,
        adata_to_predict=ctrl_test,
    )
    X_pred_full = pred.X.toarray() if sp.issparse(pred.X) else np.asarray(pred.X)

    # 从预测结果中随机采样 n_d 行，与测试集中实际的 n_d 个处理细胞对应
    idx      = rng.choice(len(X_pred_full), size=n_d, replace=(n_d > len(X_pred_full)))
    X_sample = X_pred_full[idx].astype('float32')
    # VAE 解码输出可能含负值，截断为非负（基因表达量不能为负）
    X_out[drug_mask] = np.clip(X_sample, a_min=0.0, a_max=None)

    del pred, X_pred_full   # 及时释放

# ── 7. 组装并写出结果 AnnData ─────────────────────────────────────────────────
print("Writing output...")
adata_pred       = ad.AnnData(
    X   = X_out,
    obs = adata_test.obs.copy(),
    var = adata_test.var.copy(),
)
# 标注哪些细胞是真实 ctrl，哪些是预测值
adata_pred.obs["pred_or_real"] = "pred"
adata_pred.obs.loc[ctrl_mask, "pred_or_real"] = "real_ctrl"

print(f"Output shape: {adata_pred.shape}")
adata_pred.write_h5ad(OUTPUT_PATH)
print(f"Saved to: {OUTPUT_PATH}")

# 终端输出：
"""
(scGen) fanpeishan@ubun:/data1/fanpeishan$ python /data1/fanpeishan/STATE/for_state/set_baseline/scripts/deeplearning_baseline/scGen/predict_scGen.py
Loading training data...
Loading test data...
Test data shape: (1835947, 2000)
number of ctrl cells in NCI-H23: 45150
Loading model...
number of drugs to predict: 379
Predicting: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 379/379 [59:33<00:00,  9.43s/it]
Writing output...
Output shape: (1835947, 2000)
Saved to: /data1/fanpeishan/STATE/for_state/set_baseline/outputs/scgen/c37_pred.h5ad
"""