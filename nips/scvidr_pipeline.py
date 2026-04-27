"""
NIPS scVIDR baseline aligned with scPerturBench OOD scVIDR.

Server setup:
  The scVIDR repository is expected at /data1/fanpeishan/scVIDR.
  Override with SCVIDR_ROOT if needed.

Run example:
  cd /data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/scVIDR
  CUDA_VISIBLE_DEVICES=7 python scripts/scvidr_pipeline.py

Protocol:
  - For each target file, target cell_name, and drug, build a pair dataset
    containing control/current-drug cells from training files plus the target
    cell's own control/current-drug cells.
  - Let scVIDR's prepare_data exclude the target cell's true stimulated block
    from training while retaining target-cell control for prediction.
  - Save final cell-eval compatible h5ad files: real controls + predicted
    stimulated blocks using the real stimulated obs/var skeleton.
"""

from pathlib import Path
import gc
import hashlib
import logging
import os
import re
import shutil
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch


# =========================
# scVIDR import path
# =========================
SCVIDR_ROOT = Path(os.environ.get("SCVIDR_ROOT", "/data1/fanpeishan/scVIDR"))
sys.path.insert(1, str(SCVIDR_ROOT / "vidr"))
sys.path.insert(1, str(SCVIDR_ROOT))

from vidr.vidr import VIDR  # noqa: E402
from vidr.utils import prepare_data  # noqa: E402


logging.getLogger("scvi").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


# =========================
# Paths
# =========================
BASE_DIR = Path("/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/scVIDR")
DATA_DIR = Path("/data1/fanpeishan/STATE/for_state/data/nips")

PREDICT_DIR = BASE_DIR / "predict_result"
TEMP_PRED_DIR = BASE_DIR / "tmp_predictions"
MODEL_DIR = BASE_DIR / "models"

TRAIN_FILES = [
    DATA_DIR / "train_b_cells.h5ad",
    DATA_DIR / "train_nk_cells.h5ad",
    DATA_DIR / "train_t_cells_cd4.h5ad",
]

TARGET_FILES = [
    DATA_DIR / "test_myeloid_cells.h5ad",
    DATA_DIR / "test_t_cells.h5ad",
    DATA_DIR / "val_t_cells_cd8.h5ad",
]

CONTROL_DRUG = "DMSO_TF"


# =========================
# scPerturBench scVIDR parameters
# =========================
MAX_EPOCHS = 100
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 25
LINEAR_DECODER = False
REGRESSION = False
RANDOM_SEED = 16


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unique_in_order(values):
    return pd.Index(values).drop_duplicates().tolist()


def safe_name(text):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "empty"


def stable_int(text):
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def get_temp_pred_path(sample_name, cell_name, drug_name):
    cell_dir = TEMP_PRED_DIR / sample_name / f"{safe_name(cell_name)}__{stable_int(cell_name)}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{safe_name(drug_name)}__{stable_int(drug_name)}_pred.h5ad"
    return cell_dir / file_name


def get_model_path(sample_name, cell_name, drug_name):
    model_dir = MODEL_DIR / sample_name / f"{safe_name(cell_name)}__{stable_int(cell_name)}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{safe_name(drug_name)}__{stable_int(drug_name)}.pt"


def adata_to_numpy(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def to_template_matrix(matrix, template_x):
    matrix = np.asarray(matrix, dtype=np.float32)
    if sp.issparse(template_x):
        return sp.csr_matrix(matrix)
    return matrix


def use_log_layer_if_available(adata):
    if "logNor" in adata.layers:
        adata.X = adata.layers["logNor"].copy()
    return adata


def prepare_base_adata(adata):
    adata = adata.copy()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs["drug"] = adata.obs["drug"].astype(str)
    adata.obs["cell_name"] = adata.obs["cell_name"].astype(str)
    return use_log_layer_if_available(adata)


def collect_train_drugs(train_adatas):
    drug_list = []
    seen = set()

    for adata_train in train_adatas.values():
        for drug_name in unique_in_order(adata_train.obs["drug"]):
            if drug_name == CONTROL_DRUG or drug_name in seen:
                continue
            seen.add(drug_name)
            drug_list.append(drug_name)

    return drug_list


def build_pair_adata_for_target_drug(drug_name, train_adatas, target_cell_adata):
    """
    Build the AnnData passed to scVIDR prepare_data.

    It intentionally includes target-cell control and target-cell stimulated
    rows. prepare_data(..., cell_name, drug_name) is then responsible for
    holding out the target-cell stimulated rows from training, matching
    scPerturBench's myscVIDR.py.
    """
    blocks = []

    for adata_train in train_adatas.values():
        train_pair = adata_train[
            adata_train.obs["drug"].isin([CONTROL_DRUG, drug_name])
        ].copy()
        if train_pair.n_obs > 0:
            blocks.append(train_pair)

    target_pair = target_cell_adata[
        target_cell_adata.obs["drug"].isin([CONTROL_DRUG, drug_name])
    ].copy()
    if (target_pair.obs["drug"] == CONTROL_DRUG).sum() == 0:
        raise ValueError("Target cell has no control cells.")
    if (target_pair.obs["drug"] == drug_name).sum() == 0:
        raise ValueError(f"Target cell has no stimulated cells for drug '{drug_name}'.")
    blocks.append(target_pair)

    pair_adata = ad.concat(
        blocks,
        join="inner",
        merge="same",
        index_unique=None,
    )
    pair_adata.obs_names_make_unique()
    pair_adata.var_names_make_unique()
    pair_adata.obs["condition1"] = pair_adata.obs["cell_name"].astype(str)
    pair_adata.obs["condition2"] = pair_adata.obs["drug"].astype(str)

    if CONTROL_DRUG not in set(pair_adata.obs["condition2"]):
        raise ValueError(f"Pair data for drug '{drug_name}' has no control cells.")
    if drug_name not in set(pair_adata.obs["condition2"]):
        raise ValueError(f"Pair data for drug '{drug_name}' has no stimulated cells.")

    del blocks, target_pair
    cleanup_memory()
    return pair_adata


def resize_prediction_to_skeleton(pred_x, stim_skeleton, random_seed):
    n_target = stim_skeleton.n_obs
    if pred_x.shape[0] == n_target:
        return pred_x.astype(np.float32, copy=False)

    if pred_x.shape[0] == 0:
        raise RuntimeError("scVIDR returned an empty prediction matrix.")

    rng = np.random.default_rng(random_seed)
    sampled_idx = rng.choice(
        pred_x.shape[0],
        size=n_target,
        replace=(n_target > pred_x.shape[0]),
    )
    return pred_x[sampled_idx].astype(np.float32, copy=False)


def train_and_predict_one_block(sample_name, cell_name, drug_name, pair_adata, stim_skeleton):
    train_adata, _ = prepare_data(
        pair_adata,
        "condition1",
        "condition2",
        cell_name,
        drug_name,
        normalized=True,
    )

    model = VIDR(train_adata, linear_decoder=LINEAR_DECODER)
    model.train(
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    model_path = get_model_path(sample_name, cell_name, drug_name)
    model.save(str(model_path))

    pred_adata, _ = model.predict(
        ctrl_key=CONTROL_DRUG,
        treat_key=drug_name,
        cell_type_to_predict=cell_name,
        regression=REGRESSION,
    )

    pred_x = adata_to_numpy(pred_adata.X)
    pred_x = resize_prediction_to_skeleton(
        pred_x=pred_x,
        stim_skeleton=stim_skeleton,
        random_seed=RANDOM_SEED + stable_int(f"{sample_name}::{cell_name}::{drug_name}"),
    )
    pred_x = np.clip(pred_x, a_min=0.0, a_max=None)

    pred_block = ad.AnnData(
        X=to_template_matrix(pred_x, stim_skeleton.X),
        obs=stim_skeleton.obs.copy(),
        var=stim_skeleton.var.copy(),
    )
    pred_block.obs["pred_or_real"] = "pred"

    del model, train_adata, pred_adata, pred_x
    cleanup_memory()
    return pred_block


def validate_prediction_block(adata_pred, stim_skeleton, sample_name, cell_name, drug_name):
    if adata_pred.n_obs != stim_skeleton.n_obs:
        raise RuntimeError(
            f"Prediction row mismatch for {sample_name}/{cell_name}/{drug_name}: "
            f"pred={adata_pred.n_obs}, target={stim_skeleton.n_obs}"
        )
    if adata_pred.n_vars != stim_skeleton.n_vars:
        raise RuntimeError(
            f"Prediction gene mismatch for {sample_name}/{cell_name}/{drug_name}: "
            f"pred={adata_pred.n_vars}, target={stim_skeleton.n_vars}"
        )

    pred_x = adata_to_numpy(adata_pred.X)
    if np.isnan(pred_x).any():
        raise RuntimeError(f"Prediction contains NaN for {sample_name}/{cell_name}/{drug_name}.")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if TEMP_PRED_DIR.exists():
        shutil.rmtree(TEMP_PRED_DIR)
    TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Starting NIPS scVIDR pipeline aligned with scPerturBench OOD")
    print(f"SCVIDR_ROOT: {SCVIDR_ROOT}")
    print(f"control drug: {CONTROL_DRUG}")
    print(f"train files: {len(TRAIN_FILES)}")
    print(f"target files: {len(TARGET_FILES)}")
    print(f"batch size: {BATCH_SIZE}")
    print(f"max epochs: {MAX_EPOCHS}")
    print(f"early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print("=" * 100)

    train_adatas = {}
    for file_path in TRAIN_FILES:
        train_adatas[file_path.stem] = prepare_base_adata(sc.read_h5ad(file_path))

    target_adatas = {}
    for file_path in TARGET_FILES:
        target_adatas[file_path.stem] = prepare_base_adata(sc.read_h5ad(file_path))

    train_drug_list = collect_train_drugs(train_adatas)
    train_drug_set = set(train_drug_list)

    print(f"non-control drugs in training data: {len(train_drug_list)}")
    print("target cell_name values:")
    for sample_name, adata_target in target_adatas.items():
        for cell_name in unique_in_order(adata_target.obs["cell_name"]):
            print(f"  - {sample_name}: {cell_name}")

    # Stage 1: train and predict one target-file/cell/drug task at a time.
    for sample_name, adata_target in target_adatas.items():
        print("\n" + "#" * 100)
        print(f"Processing target file: {sample_name}")
        print("#" * 100)

        for cell_name in unique_in_order(adata_target.obs["cell_name"]):
            target_cell_adata = adata_target[
                adata_target.obs["cell_name"] == cell_name
            ].copy()
            drug_order = unique_in_order(target_cell_adata.obs["drug"])

            if CONTROL_DRUG not in drug_order:
                raise ValueError(
                    f"Target file '{sample_name}' cell_name '{cell_name}' has no control drug '{CONTROL_DRUG}'."
                )

            target_drugs = [drug for drug in drug_order if drug != CONTROL_DRUG]
            print("-" * 100)
            print(f"Target cell: {cell_name}")
            print(f"target drugs in file: {len(target_drugs)}")

            for drug_rank, drug_name in enumerate(target_drugs, start=1):
                if drug_name not in train_drug_set:
                    raise ValueError(
                        f"Target drug '{drug_name}' for {sample_name}/{cell_name} is absent from training data."
                    )

                print(f"[{drug_rank}/{len(target_drugs)}] drug={drug_name}")

                stim_skeleton = target_cell_adata[
                    target_cell_adata.obs["drug"] == drug_name
                ].copy()
                if stim_skeleton.n_obs == 0:
                    del stim_skeleton
                    continue

                pair_adata = build_pair_adata_for_target_drug(
                    drug_name=drug_name,
                    train_adatas=train_adatas,
                    target_cell_adata=target_cell_adata,
                )
                print(f"  pair shape: {pair_adata.shape}")

                pred_block = train_and_predict_one_block(
                    sample_name=sample_name,
                    cell_name=cell_name,
                    drug_name=drug_name,
                    pair_adata=pair_adata,
                    stim_skeleton=stim_skeleton,
                )
                validate_prediction_block(
                    adata_pred=pred_block,
                    stim_skeleton=stim_skeleton,
                    sample_name=sample_name,
                    cell_name=cell_name,
                    drug_name=drug_name,
                )

                save_path = get_temp_pred_path(sample_name, cell_name, drug_name)
                pred_block.write_h5ad(save_path)

                print(
                    f"  saved: {save_path} | "
                    f"cell_name={cell_name} | "
                    f"skeleton={stim_skeleton.n_obs}"
                )

                del pair_adata, stim_skeleton, pred_block
                cleanup_memory()

            del target_cell_adata
            cleanup_memory()

    # Stage 2: merge final prediction files by target file.
    print("\n" + "=" * 100)
    print("Merging final prediction files")
    print("=" * 100)

    for sample_name, adata_target in target_adatas.items():
        print(f"Merge target file: {sample_name}")

        output_blocks = []

        for cell_name in unique_in_order(adata_target.obs["cell_name"]):
            target_cell_adata = adata_target[
                adata_target.obs["cell_name"] == cell_name
            ].copy()
            drug_order = unique_in_order(target_cell_adata.obs["drug"])

            ctrl_real = target_cell_adata[
                target_cell_adata.obs["drug"] == CONTROL_DRUG
            ].copy()
            if ctrl_real.n_obs == 0:
                raise ValueError(
                    f"Target file '{sample_name}' cell_name '{cell_name}' has no control cells."
                )
            ctrl_real.obs = ctrl_real.obs.copy()
            ctrl_real.obs["pred_or_real"] = "real_ctrl"
            output_blocks.append(ctrl_real)

            for drug_name in drug_order:
                if drug_name == CONTROL_DRUG:
                    continue

                pred_file = get_temp_pred_path(sample_name, cell_name, drug_name)
                if not pred_file.exists():
                    raise FileNotFoundError(f"Missing prediction block for merge: {pred_file}")

                output_blocks.append(sc.read_h5ad(pred_file))

            del target_cell_adata
            cleanup_memory()

        final_adata = ad.concat(
            output_blocks,
            join="inner",
            merge="same",
            index_unique=None,
        )
        final_adata.obs_names_make_unique()

        final_path = PREDICT_DIR / f"{sample_name}_pred.h5ad"
        final_adata.write_h5ad(final_path)

        print(f"generated: {final_path}")
        print(f"final shape: {final_adata.shape}")

        del output_blocks, final_adata
        cleanup_memory()

    print("\n" + "=" * 100)
    print("NIPS scVIDR pipeline finished")
    print("final outputs:")
    for file_path in TARGET_FILES:
        print(PREDICT_DIR / f"{file_path.stem}_pred.h5ad")
    print("=" * 100)
