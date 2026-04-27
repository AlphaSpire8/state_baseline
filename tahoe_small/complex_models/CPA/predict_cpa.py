# export CUDA_VISIBLE_DEVICES=0
import scanpy as sc
import anndata as ad
from cpa import CPA
import torch
torch.set_float32_matmul_precision("medium")

# -----------------------------
# 1 load training data (for var)
# -----------------------------

print("Loading training data...")
train_adata = sc.read_h5ad(
    "/data1/fanpeishan/STATE/for_state/data/for_chemCPA/train_c33_c34_c35.h5ad"
)

# -----------------------------
# 2 load test data
# -----------------------------

print("Loading test data...")
test_adata = sc.read_h5ad(
    "/data1/fanpeishan/STATE/for_state/data/for_chemCPA/test_c37_modified.h5ad"
    # "/data1/fanpeishan/STATE/for_state/data/for_chemCPA/test_c37.h5ad"
)

# gene order必须一致
test_adata = test_adata[:, train_adata.var_names].copy()

# -----------------------------
# 3 register test_adata with setup_anndata
#   (must match the configuration used during training)
# -----------------------------

CPA.setup_anndata(
    test_adata,
    perturbation_key="drug",
    control_group="DMSO_TF",
    dosage_key="dose",
    categorical_covariate_keys=["cell_type"],
    max_comb_len=1
)

# -----------------------------
# 4 load trained model
# -----------------------------

model = CPA.load(
    dir_path="/data1/fanpeishan/STATE/for_state/set_baseline/outputs/cpa/cpa_model/",
    adata=test_adata,
    use_gpu=True
)

# -----------------------------
# 5 generate prediction
#   predict() stores results in test_adata.obsm['CPA_pred'] and returns None
# -----------------------------

model.predict(test_adata, batch_size=4096)
pred = test_adata.obsm['CPA_pred']
print("prediction shape:", pred.shape)

# -----------------------------
# 6 build predicted AnnData
# -----------------------------

pred_adata = ad.AnnData(
    X=pred,
    obs=test_adata.obs.copy(),
    var=test_adata.var.copy()
)

# -----------------------------
# 7 save prediction
# -----------------------------

print("Saving prediction...")
pred_adata.write(
    "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/cpa/c37_pred.h5ad"
)
print("Prediction finished")

# 终端输出：
"""
(cpa_env) fanpeishan@ubun:/data1/fanpeishan$ python /data1/fanpeishan/STATE/for_state/set_baseline/scripts/deeplearning_baseline/predict_cpa.py
Global seed set to 0
100%|| 1835947/1835947 [01:50<00:00, 16588.91it/s]
100%|| 1835947/1835947 [00:01<00:00, 1064913.11it/s]
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
INFO     Generating sequential column names                                                                                                                                                                                                                                                        
INFO     Generating sequential column names                                                                                                                                                                                                                                                        
INFO     File /data1/fanpeishan/STATE/for_state/set_baseline/outputs/cpa/cpa_model/model.pt already downloaded                                                                                                                                                                                     
100%|| 1835947/1835947 [01:38<00:00, 18721.37it/s]
100%|| 1835947/1835947 [00:01<00:00, 1107058.54it/s]
Global seed set to 42
100%|| 449/449 [00:18<00:00, 24.74it/s]
prediction shape: (1835947, 2000)
Prediction finished
"""