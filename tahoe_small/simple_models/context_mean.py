import gc
import numpy as np
import anndata as ad
from scipy.sparse import issparse, csr_matrix

DATA_DIR = "/data1/fanpeishan/STATE/for_state/data/State-Tahoe-Filtered-processed"
OUT_PATH = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/context_mean/c37_context_mean.h5ad"
CONTROL = "DMSO_TF"

# Incrementally accumulate sum and count to avoid holding multiple files in memory
train_files = ["c33_prep.h5ad", "c34_prep.h5ad", "c35_prep.h5ad"]
sum_expr = None
count = 0

for f in train_files:
    print(f"Processing {f}...")
    adata = ad.read_h5ad(f"{DATA_DIR}/{f}")
    mask = adata.obs["drug"] != CONTROL
    X = adata[mask].X
    X = X.toarray() if issparse(X) else X  # only non-control subset, much smaller

    if sum_expr is None:
        sum_expr = X.sum(axis=0)
    else:
        sum_expr += X.sum(axis=0)
    count += X.shape[0]

    del adata, X
    gc.collect()

train_mean = sum_expr / count  # (2000,)
del sum_expr
gc.collect()

# Replace non-control cells in test file with training mean
print("Processing test file...")
adata_test = ad.read_h5ad(f"{DATA_DIR}/c37_prep.h5ad")
mask_test = (adata_test.obs["drug"] != CONTROL).values

if issparse(adata_test.X):
    X_test = adata_test.X.toarray()
    X_test[mask_test] = train_mean
    adata_test.X = csr_matrix(X_test)
    del X_test
else:
    adata_test.X[mask_test] = train_mean

gc.collect()
adata_test.write_h5ad(OUT_PATH)
print(f"Saved to {OUT_PATH}")