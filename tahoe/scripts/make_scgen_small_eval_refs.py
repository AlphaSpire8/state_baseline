"""
为 Tahoe scGen 小规模预测结果生成匹配的 cell-eval real reference 子集。

用途：
scgen_pipeline_small.py 已经生成 predicted h5ad 后，如果直接用全量
tahoe_filtered/{cell}.h5ad 做 cell-eval，real 里会包含未预测的 drug，
cell-eval 会报“找不到药物”。本脚本从 predicted h5ad 反推 drug 集合，
再把全量 real h5ad 过滤到完全相同的 drug 集合。

运行示例：
python make_scgen_small_eval_refs.py
"""

from pathlib import Path
import gc

import anndata as ad
import pandas as pd
import scanpy as sc


# =========================
# 路径配置
# =========================
PREDICT_DIR = Path(
    "/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/"
    "complex_models/scGen/outputs_small_scperturbench"
)
REAL_DIR = Path("/data1/fanpeishan/STATE/for_state/data/tahoe_filtered")
OUTPUT_DIR = PREDICT_DIR / "eval_reference"

TEST_CELL_TYPES = ["c3", "c4", "c5"]
CONTROL_DRUG = "DMSO_TF"
PERT_COL = "drug"


def cleanup_memory():
    gc.collect()


def unique_in_order(values):
    return pd.Index(values).drop_duplicates().astype(str).tolist()


def make_one_reference(test_cell_type):
    pred_path = PREDICT_DIR / f"{test_cell_type}_predicted.h5ad"
    real_path = REAL_DIR / f"{test_cell_type}.h5ad"
    out_path = OUTPUT_DIR / f"{test_cell_type}_real_subset.h5ad"

    if not pred_path.exists():
        raise FileNotFoundError(f"找不到 prediction 文件: {pred_path}")
    if not real_path.exists():
        raise FileNotFoundError(f"找不到 real 文件: {real_path}")

    print("-" * 100)
    print(f"处理 {test_cell_type}")
    print(f"prediction: {pred_path}")
    print(f"real      : {real_path}")

    adata_pred = sc.read_h5ad(pred_path)
    pred_drugs = unique_in_order(adata_pred.obs[PERT_COL])
    if CONTROL_DRUG not in pred_drugs:
        raise ValueError(f"prediction 中缺少 control drug: {CONTROL_DRUG}")

    adata_real = sc.read_h5ad(real_path)
    real_drug_set = set(adata_real.obs[PERT_COL].astype(str))
    missing_drugs = [drug for drug in pred_drugs if drug not in real_drug_set]
    if missing_drugs:
        raise ValueError(f"real 中缺少 prediction 里的 drug: {missing_drugs}")

    keep_mask = adata_real.obs[PERT_COL].astype(str).isin(pred_drugs).to_numpy()
    adata_ref = adata_real[keep_mask, :].copy()

    # 按 prediction 的 drug 顺序重排，方便人工检查；cell-eval 本身不依赖行顺序。
    blocks = []
    for drug in pred_drugs:
        block = adata_ref[adata_ref.obs[PERT_COL].astype(str) == drug, :].copy()
        if block.n_obs == 0:
            raise ValueError(f"过滤后 real 子集中缺少 drug: {drug}")
        blocks.append(block)

    adata_ref_ordered = ad.concat(
        blocks,
        join="inner",
        merge="same",
        index_unique=None,
    )
    adata_ref_ordered.obs_names_make_unique()
    adata_ref_ordered.write_h5ad(out_path)

    print(f"prediction drug 数: {len(pred_drugs)}")
    print(f"real subset shape: {adata_ref_ordered.shape}")
    print(f"已保存: {out_path}")

    del adata_pred, adata_real, adata_ref, blocks, adata_ref_ordered
    cleanup_memory()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for test_cell_type in TEST_CELL_TYPES:
        make_one_reference(test_cell_type)

    print("=" * 100)
    print("cell-eval real reference 子集生成完成")
    print("=" * 100)
