# export CUDA_VISIBLE_DEVICES=2
import os
import warnings

import anndata as ad
import biolord
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("medium")

# =========================
# 路径配置
# =========================
TEST_PATH = "/data1/fanpeishan/STATE/for_state/data/for_chemCPA/test_c37.h5ad"
MODEL_DIR = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/biolord/model"
VAR_NAMES_PATH = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/biolord/train_var_names.txt"
OUTPUT_PATH = "/data1/fanpeishan/STATE/for_state/set_baseline/outputs/biolord/c37_pred.h5ad"

CONTROL_DRUG = "DMSO_TF"
RANDOM_SEED = 42
BATCH_SIZE = 4096


# =========================
# 工具函数
# =========================
def prepare_obs(adata):
    adata.obs["drug"] = adata.obs["drug"].astype(str)
    adata.obs["dose"] = adata.obs["dose"].astype(np.float32)
    return adata


def to_dense_float32(matrix):
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def align_var_names(adata, train_var_names):
    test_vars = adata.var_names.to_numpy()

    if len(test_vars) != len(train_var_names):
        raise ValueError(
            f"Gene number mismatch: test={len(test_vars)} train={len(train_var_names)}"
        )

    if np.array_equal(test_vars, train_var_names):
        return adata

    if set(test_vars) != set(train_var_names):
        raise ValueError("Test genes do not match train genes.")

    return adata[:, train_var_names].copy()


def load_biolord_model_for_query(model_dir, query_adata):
    model_path = os.path.join(model_dir, "model.pt")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    attr_dict = checkpoint["attr_dict"]
    registry = attr_dict["registry_"]
    init_params = attr_dict["init_params_"]["non_kwargs"]

    # 复用训练时的 registry，确保 drug/dose 编码与训练阶段完全一致。
    biolord.Biolord.setup_anndata(
        query_adata,
        source_registry=registry,
        **registry["setup_args"],
    )

    model = biolord.Biolord(adata=query_adata, **init_params)
    model.module.on_load(model)

    model_state_dict = checkpoint["model_state_dict"]
    model_state_dict.pop("pyro_param_store", None)
    skipped_key = "latent_codes.embedding.weight"
    model_state_dict.pop(skipped_key, None)

    incompatible_keys = model.module.load_state_dict(model_state_dict, strict=False)
    unexpected_missing = set(incompatible_keys.missing_keys) - {skipped_key}
    if unexpected_missing or incompatible_keys.unexpected_keys:
        raise RuntimeError(
            "Failed to load biolord checkpoint for query data. "
            f"missing_keys={sorted(unexpected_missing)}, "
            f"unexpected_keys={sorted(incompatible_keys.unexpected_keys)}"
        )

    # query cells 不在训练样本中，没有可复用的 per-sample latent code，推理时置零。
    with torch.no_grad():
        model.module.latent_codes.embedding.weight.zero_()

    model.is_trained_ = True
    model.to_device("cuda" if torch.cuda.is_available() else "cpu")
    model.module.eval()

    del checkpoint
    return model


# =========================
# 读取训练基因顺序
# =========================
train_var_names = np.loadtxt(VAR_NAMES_PATH, dtype=str)

# =========================
# 读取测试数据
# =========================
print("Reading test data:", TEST_PATH)
adata = sc.read_h5ad(TEST_PATH)
adata.obs_names_make_unique()
adata = prepare_obs(adata)
adata = align_var_names(adata, train_var_names)

print("Test AnnData:", adata)

control_mask = (adata.obs["drug"].astype(str) == CONTROL_DRUG).values
pert_mask = ~control_mask

n_ctrl = int(control_mask.sum())
n_pert = int(pert_mask.sum())

print("Control cells:", n_ctrl)
print("Perturbation cells:", n_pert)

if n_ctrl == 0:
    raise ValueError("Test dataset has no control cells.")

# 先把原始测试集作为输出骨架，control 保持不变
pred_X = to_dense_float32(adata.X)

if n_pert == 0:
    print("No perturbation cells found. Writing original test data.")
    adata_pred = ad.AnnData(
        X=sp.csr_matrix(pred_X) if sp.issparse(adata.X) else pred_X,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    adata_pred.obs["pred_or_real"] = "real_ctrl"
    adata_pred.write_h5ad(OUTPUT_PATH)
    print("Saved to:", OUTPUT_PATH)
    raise SystemExit(0)

ctrl_adata = adata[control_mask].copy()
pert_adata = adata[pert_mask].copy()

# =========================
# 为每个 perturbation cell 采样一个目标域 control cell 作为 source
# source expression 来自 test control
# target drug / dose 来自对应 perturbation cell
# =========================
rng = np.random.default_rng(RANDOM_SEED)
sampled_ctrl_idx = rng.choice(
    ctrl_adata.n_obs,
    size=pert_adata.n_obs,
    replace=(pert_adata.n_obs > ctrl_adata.n_obs),
)

source_adata = ctrl_adata[sampled_ctrl_idx].copy()

# 强制用目标条件覆盖 source 的 perturbation 属性
source_adata.obs["drug"] = pert_adata.obs["drug"].astype(str).to_numpy()
source_adata.obs["dose"] = pert_adata.obs["dose"].astype(np.float32).to_numpy()

# 保留一些追踪信息，方便后续检查
source_adata.obs["target_obs_name"] = pert_adata.obs_names.to_numpy()
source_adata.obs["source_obs_name"] = source_adata.obs_names.to_numpy()

# 训练时模型使用了 split_key="split"，load 时要求 query adata 也包含该列
source_adata.obs["split"] = "train"
source_adata.obs["split"] = source_adata.obs["split"].astype("category")

# =========================
# 加载模型并预测
# =========================
print("Loading biolord model from:", MODEL_DIR)
model = load_biolord_model_for_query(MODEL_DIR, source_adata)

print("Starting prediction...")
pred_mean, pred_var = model.predict(
    adata=source_adata,
    batch_size=BATCH_SIZE,
)

pred_expr = np.asarray(pred_mean.X, dtype=np.float32)
pred_expr = np.clip(pred_expr, a_min=0.0, a_max=None)

pred_X[pert_mask] = pred_expr

# =========================
# 保存结果
# control 保持原值
# perturbation cells 替换为预测值
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