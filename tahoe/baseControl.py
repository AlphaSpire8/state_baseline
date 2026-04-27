# export CUDA_VISIBLE_DEVICES=4

from pathlib import Path
import gc

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm


# =========================
# 基础路径配置
# =========================
# 预处理后的按药物拆分数据目录。
# 每个文件对应“一个 cell_type + 一个 drug_index”。
SPLIT_BY_DRUG_DIR = Path("/datasets/fanpeishan/data/tahoe_for_scgen/split_by_drug")

# 最终输出目录。
# 这里直接输出 10 个最终预测文件，不额外保存中间结果。
OUTPUT_DIR = Path("/datasets/fanpeishan/data/tahoe_for_baseControl")

# 药物顺序文件。
# 该文件只有一列，列名为 drug_name；这一列从上到下的顺序就是 drug_index 的定义顺序。
DRUG_NAME_CSV = Path("/datasets/fanpeishan/data/docs/drug_name_list.csv")

# baseControl 中的对照药物名称。
CONTROL_DRUG = "DMSO_TF"

# 随机种子只用于从测试 cell type 的 control 细胞中采样，
# 使每个药物块的细胞数与真实测试药物块一致。
RANDOM_SEED = 16


# =========================
# 数据划分
# =========================
# 这里采用“不要验证集”的设定：
# - 40 个训练 cell type 仅用于和其他方法保持一致的评测协议；
# - 由于 baseControl 不训练任何参数，因此脚本本身不会实际读取训练集；
# - 原验证集 5 个 cell type 被并入测试集，所以最终测试集共 10 个 cell type。
def cells(start, end):
    return [f"c{i}" for i in range(start, end + 1)]


TRAIN_CELL_TYPES = (
    cells(0, 3)
    + cells(5, 16)
    + ["c21"]
    + cells(25, 30)
    + cells(32, 37)
    + cells(39, 49)
)

TEST_CELL_TYPES = ["c4", "c17", "c18", "c23", "c38", "c19", "c20", "c22", "c24", "c31"]


# =========================
# 工具函数
# =========================
def read_drug_name_list():
    # 直接读取 drug_name 列，每一行就是一个药物名。
    return pd.read_csv(DRUG_NAME_CSV)["drug_name"].tolist()



def get_split_file(cell_type, drug_index):
    # 预处理阶段的命名规则已经固定，因此这里直接按规则拼接路径。
    return (
        SPLIT_BY_DRUG_DIR
        / f"celltype_{cell_type}"
        / f"celltype_{cell_type}_drugindex_{drug_index}.h5ad"
    )



def cleanup_memory():
    # 每处理完一个测试 cell type，就主动回收一次内存。
    gc.collect()


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取药物顺序，并建立 drug -> drug_index 映射。
    drug_name_list = read_drug_name_list()
    drug_to_index = {drug_name: idx for idx, drug_name in enumerate(drug_name_list)}
    control_drug_index = drug_to_index[CONTROL_DRUG]

    print("=" * 100)
    print("开始执行 baseControl 推理流程")
    print("baseControl 不训练任何参数，预测结果直接来自测试 cell type 自身的 control 表达")
    print(f"control drug: {CONTROL_DRUG} (drug_index={control_drug_index})")
    print(f"训练 cell type 数量: {len(TRAIN_CELL_TYPES)}")
    print(f"测试 cell type 数量: {len(TEST_CELL_TYPES)}")
    print(f"药物数量: {len(drug_name_list)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 100)

    # ============================================================
    # 对每一个测试 cell type，直接重建一个最终输出文件。
    # 最终文件的组织方式与 scGen baseline 保持一致：
    # - control block：保留真实 control 的原始表达；
    # - 非 control block：使用同一测试 cell type 的 control 表达作为预测值；
    # - obs / var 完全沿用真实测试药物块的骨架；
    # - 最终按 drug_index 顺序 concat，生成 {cell_type}_pred.h5ad。
    # ============================================================
    for test_cell_type in TEST_CELL_TYPES:
        print("\n" + "-" * 100)
        print(f"开始处理测试 cell type: {test_cell_type}")

        # 先读取该测试 cell type 的真实 control block。
        # 这部分数据既会作为最终文件中的真实 control，也会作为所有非 control 药物的预测来源。
        ctrl_real = sc.read_h5ad(get_split_file(test_cell_type, control_drug_index))
        ctrl_real.obs = ctrl_real.obs.copy()
        ctrl_real.obs["pred_or_real"] = "real_ctrl"

        # baseControl 的核心是假设“加药后仍然等于 control”。
        # 因此这里先把 control 表达矩阵取出来，后面所有药物块都从这里采样复制。
        ctrl_X = ctrl_real.X.toarray() if sp.issparse(ctrl_real.X) else np.asarray(ctrl_real.X)

        print(f"control block shape: {ctrl_real.shape}")
        print("开始按 drug_index 顺序构建最终输出 ...")

        output_blocks = []

        for drug_index in tqdm(range(len(drug_name_list)), desc=f"Building {test_cell_type}"):
            if drug_index == control_drug_index:
                # control 药物块直接保留真实表达，不做任何替换。
                output_blocks.append(ctrl_real)
                continue

            # 读取当前药物在该测试 cell type 下的真实骨架文件。
            # 这个文件只提供三类信息：
            # 1) 真实细胞数 n_d；
            # 2) 原始 obs；
            # 3) 原始 var。
            stim_skeleton = sc.read_h5ad(get_split_file(test_cell_type, drug_index))
            n_d = stim_skeleton.n_obs

            # 为了让最终输出文件与真实测试集严格对齐，
            # 这里按照当前药物真实块的细胞数 n_d，从 control 细胞中采样。
            # 若 n_d 大于 control 细胞数，则允许有放回采样。
            test_cell_type_id = int(test_cell_type[1:])
            rng = np.random.default_rng(RANDOM_SEED + test_cell_type_id * 1000 + drug_index)
            sample_index = rng.choice(ctrl_X.shape[0], size=n_d, replace=(n_d > ctrl_X.shape[0]))
            X_sample = ctrl_X[sample_index].astype("float32")

            # 用真实测试药物块的 obs / var 做骨架，只替换 X。
            # 这样最终输出在文件结构上与 scGen baseline 完全一致。
            adata_pred = ad.AnnData(
                X=X_sample,
                obs=stim_skeleton.obs.copy(),
                var=stim_skeleton.var.copy(),
            )
            adata_pred.obs["pred_or_real"] = "pred"

            output_blocks.append(adata_pred)

            del stim_skeleton, X_sample, adata_pred

        # 将 380 个药物块按固定顺序拼接，生成该测试 cell type 的最终预测文件。
        final_adata = ad.concat(
            output_blocks,
            join="inner",
            merge="same",
            index_unique=None,
        )
        final_adata.obs_names_make_unique()

        final_path = OUTPUT_DIR / f"{test_cell_type}_pred.h5ad"
        final_adata.write_h5ad(final_path)

        print(f"已生成: {final_path}")
        print(f"最终 shape: {final_adata.shape}")

        del ctrl_real, ctrl_X, output_blocks, final_adata
        cleanup_memory()

    print("\n" + "=" * 100)
    print("全部流程完成")
    print("最终输出文件:")
    for test_cell_type in TEST_CELL_TYPES:
        print(OUTPUT_DIR / f"{test_cell_type}_pred.h5ad")
    print("=" * 100)
