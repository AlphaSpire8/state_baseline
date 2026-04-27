import gc
import numpy as np
import anndata as ad
from scipy.sparse import issparse, csr_matrix
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "/data1/fanpeishan/STATE/for_state/data/State-Tahoe-Filtered-processed"
OUT_PATH = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/pca_ridge/c37_pca_ridge.h5ad"
CONTROL = "DMSO_TF"
PCA_COMPONENTS = 64
RIDGE_ALPHA = 1.0
BATCH_SIZE = 10000

train_files = ["c33_prep.h5ad", "c34_prep.h5ad", "c35_prep.h5ad"]

# ── Step 1: Fit IncrementalPCA on control cells of training files ────────────
print("Step 1: Fitting IncrementalPCA on control cells...")
ipca = IncrementalPCA(n_components=PCA_COMPONENTS)

for f in train_files:
    print(f"  {f}")
    adata = ad.read_h5ad(f"{DATA_DIR}/{f}")
    X_ctrl = adata[adata.obs["drug"] == CONTROL].X
    X_ctrl = X_ctrl.toarray() if issparse(X_ctrl) else X_ctrl
    for start in range(0, X_ctrl.shape[0], BATCH_SIZE):
        ipca.partial_fit(X_ctrl[start:start + BATCH_SIZE])
    del adata, X_ctrl
    gc.collect()

# ── Step 2: Build training samples (one per file × drug) ────────────────────
print("Step 2: Building training samples...")

# Collect all drug names (excluding control) from first file for label encoding
adata0 = ad.read_h5ad(f"{DATA_DIR}/{train_files[0]}")
all_drugs = sorted([d for d in adata0.obs["drug"].unique() if d != CONTROL])
del adata0
gc.collect()

le = LabelEncoder().fit(all_drugs)
n_drugs = len(all_drugs)  # 379

X_train, y_train = [], []

for f in train_files:
    print(f"  {f}")
    adata = ad.read_h5ad(f"{DATA_DIR}/{f}")
    drugs = adata.obs["drug"]

    # PCA-transform control mean for this file
    X_ctrl = adata[drugs == CONTROL].X
    X_ctrl = X_ctrl.toarray() if issparse(X_ctrl) else X_ctrl
    ctrl_mean_pca = ipca.transform(X_ctrl.mean(axis=0, keepdims=True))  # (1, k)

    for drug in all_drugs:
        mask = (drugs == drug).values
        if mask.sum() == 0:
            continue
        X_drug = adata[mask].X
        X_drug = X_drug.toarray() if issparse(X_drug) else X_drug
        drug_mean = X_drug.mean(axis=0)  # (2000,)

        # One-hot for drug
        onehot = np.zeros(n_drugs, dtype=np.float32)
        onehot[le.transform([drug])[0]] = 1.0

        feat = np.concatenate([ctrl_mean_pca[0], onehot])  # (k + n_drugs,)
        X_train.append(feat)
        y_train.append(drug_mean)

    del adata, X_ctrl, X_drug
    gc.collect()

X_train = np.array(X_train, dtype=np.float32)  # (1137, k + 379)
y_train = np.array(y_train, dtype=np.float32)  # (1137, 2000)

# ── Step 3: Train Ridge Regression ──────────────────────────────────────────
print("Step 3: Training Ridge Regression...")
ridge = Ridge(alpha=RIDGE_ALPHA)
ridge.fit(X_train, y_train)
del X_train, y_train
gc.collect()

# ── Step 4: Predict and replace in test file ─────────────────────────────────
print("Step 4: Processing test file...")
adata_test = ad.read_h5ad(f"{DATA_DIR}/c37_prep.h5ad")
drugs_test = adata_test.obs["drug"]

# PCA-transform control mean of test file
X_ctrl_test = adata_test[drugs_test == CONTROL].X
X_ctrl_test = X_ctrl_test.toarray() if issparse(X_ctrl_test) else X_ctrl_test
ctrl_mean_pca_test = ipca.transform(X_ctrl_test.mean(axis=0, keepdims=True))  # (1, k)
del X_ctrl_test
gc.collect()

X_test_dense = adata_test.X.toarray() if issparse(adata_test.X) else adata_test.X.copy()

for drug in all_drugs:
    mask = (drugs_test == drug).values
    if mask.sum() == 0:
        continue
    onehot = np.zeros(n_drugs, dtype=np.float32)
    onehot[le.transform([drug])[0]] = 1.0
    feat = np.concatenate([ctrl_mean_pca_test[0], onehot])[None, :]  # (1, k + 379)
    pred = ridge.predict(feat)[0]             # (2000,)
    pred = np.clip(pred, 0, None)             # no negative expression
    X_test_dense[mask] = pred

adata_test.X = csr_matrix(X_test_dense)
del X_test_dense
gc.collect()

adata_test.write_h5ad(OUT_PATH)
print(f"Saved to {OUT_PATH}")