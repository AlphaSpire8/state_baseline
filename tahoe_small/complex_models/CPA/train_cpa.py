# export CUDA_VISIBLE_DEVICES=0
import scanpy as sc
from cpa import CPA
import torch
torch.set_float32_matmul_precision("medium")

# -----------------------------
# 1 load training data
# -----------------------------

adata = sc.read_h5ad("/data1/fanpeishan/STATE/for_state/data/for_chemCPA/train_c33_c34_c35.h5ad")

# -----------------------------
# 2 setup anndata for CPA
# -----------------------------

CPA.setup_anndata(
    adata,
    perturbation_key="drug",
    control_group="DMSO_TF",
    dosage_key="dose",
    categorical_covariate_keys=["cell_type"],
    max_comb_len=1
)

# -----------------------------
# 3 build model
# -----------------------------

model_params = {
    "n_latent": 128,
    "recon_loss": "nb",
    "doser_type": "linear",
    "n_hidden_encoder": 128,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": True,
    "dropout_rate_encoder": 0.0,
    "dropout_rate_decoder": 0.1,
    "variational": False,
    "seed": 42,
}
model = CPA(
    adata,
    **model_params,
)

# -----------------------------
# 4 train model
# -----------------------------

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 0,
    "mixup_alpha": 0.0,
    "adv_steps": None,
    "n_hidden_adv": 64,
    "n_layers_adv": 3,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 5.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,
    "step_size_lr": 10,
}
model.train(max_epochs=1024,
    use_gpu=True,
    batch_size=65536,
    plan_kwargs=trainer_params,
    early_stopping_patience=5,
    check_val_every_n_epoch=5,
    save_path='/data1/fanpeishan/STATE/for_state/set_baseline/outputs/cpa/cpa_model/',
)
print("Training finished")

# -----------------------------
# 5 trainning progress visualization
# -----------------------------

import matplotlib.pyplot as plt
import cpa
cpa.pl.plot_history(model)
plt.savefig(
    "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/cpa/training_history.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()