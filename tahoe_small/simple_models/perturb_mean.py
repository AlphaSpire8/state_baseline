import gc
from collections import defaultdict
import numpy as np
import anndata as ad
from scipy.sparse import issparse, csr_matrix

DATA_DIR = "/data1/fanpeishan/STATE/for_state/data/State-Tahoe-Filtered-processed"
OUT_PATH = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/perturb_mean/c37_perturb_mean.h5ad"
CONTROL = "DMSO_TF"

# Incrementally accumulate per-drug sum and count across training files
train_files = ["c33_prep.h5ad", "c34_prep.h5ad", "c35_prep.h5ad"]
drug_sum = defaultdict(lambda: None)
drug_count = defaultdict(int)

for f in train_files:
    print(f"Processing {f}...")
    adata = ad.read_h5ad(f"{DATA_DIR}/{f}")
    drugs = adata.obs["drug"]

    for drug in drugs.unique():
        if drug == CONTROL:
            continue
        X = adata[drugs == drug].X
        X = X.toarray() if issparse(X) else X
        drug_sum[drug] = X.sum(axis=0) if drug_sum[drug] is None else drug_sum[drug] + X.sum(axis=0)
        drug_count[drug] += X.shape[0]

    del adata
    gc.collect()

# Compute per-drug mean
drug_mean = {drug: drug_sum[drug] / drug_count[drug] for drug in drug_sum}
del drug_sum
gc.collect()

# Replace each drug's cells in test file with the corresponding training mean
print("Processing test file...")
adata_test = ad.read_h5ad(f"{DATA_DIR}/c37_prep.h5ad")

if issparse(adata_test.X):
    X_test = adata_test.X.toarray()
    for drug, mean_vec in drug_mean.items():
        mask = (adata_test.obs["drug"] == drug).values
        X_test[mask] = mean_vec
    adata_test.X = csr_matrix(X_test)
    del X_test
else:
    for drug, mean_vec in drug_mean.items():
        mask = (adata_test.obs["drug"] == drug).values
        adata_test.X[mask] = mean_vec

gc.collect()
adata_test.write_h5ad(OUT_PATH)
print(f"Saved to {OUT_PATH}")