from pathlib import Path
import gc
import re
import hashlib

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


# =========================
# 基础路径配置
# =========================
# 数据目录。
DATA_DIR = Path("/data1/fanpeishan/STATE/for_state/data/nips")

# 输出目录。
OUTPUT_DIR = Path("/data1/fanpeishan/STATE/for_state/about_baseline/context_generalization/data_nips/baseControl/predict_result")

# 训练文件。
# baseControl 不会用训练集学习参数，这里只用于打印训练 cell_name，方便检查数据协议。
TRAIN_FILES = [
    DATA_DIR / "train_b_cells.h5ad",
    DATA_DIR / "train_nk_cells.h5ad",
    DATA_DIR / "train_t_cells_cd4.h5ad",
]

# 测试文件。
# 这里把 test 和 val 一起视作最终需要生成预测结果的目标集合。
TEST_FILES = [
    DATA_DIR / "test_myeloid_cells.h5ad",
    DATA_DIR / "test_t_cells.h5ad",
    DATA_DIR / "val_t_cells_cd8.h5ad",
]

# 对照药物名称。
CONTROL_DRUG = "DMSO_TF"

# 随机种子。
# 只用于从测试 cell_name 自身的 control 细胞里采样。
RANDOM_SEED = 16


# =========================
# 工具函数
# =========================
def cleanup_memory():
    """主动回收内存。"""
    gc.collect()


def safe_name(text):
    """
    将 cell_name 转成适合文件名的形式。
    例如：
    - 'Myeloid cells' -> 'Myeloid_cells'
    - 'T cells CD8+' -> 'T_cells_CD8'
    """
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def stable_int(text):
    """
    把字符串稳定地映射成一个整数。
    不能直接用 Python 内置 hash，因为不同进程可能不稳定。
    这里用 md5 前 8 位转整数，保证可复现。
    """
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("开始执行 baseControl 推理流程")
    print("baseControl 不训练任何参数")
    print("预测表达直接来自同一测试 cell_name 自身的 control（DMSO_TF）细胞")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"control drug: {CONTROL_DRUG}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 100)

    # ------------------------------------------------------------
    # 先读取训练文件中的 cell_name，仅用于打印和核对数据划分。
    # 这里不参与任何模型训练，因为 baseControl 没有可学习参数。
    # ------------------------------------------------------------
    train_cell_names = []
    for train_file in TRAIN_FILES:
        adata_train = sc.read_h5ad(train_file)
        names = pd.Index(adata_train.obs["cell_name"]).drop_duplicates().tolist()
        train_cell_names.extend(names)
        del adata_train

    train_cell_names = pd.Index(train_cell_names).drop_duplicates().tolist()

    print("训练 cell_name:")
    for name in train_cell_names:
        print(f"  - {name}")

    print("-" * 100)

    # ------------------------------------------------------------
    # 逐个测试文件处理。
    # 不对文件名和 cell_name 做硬编码绑定，而是直接读取文件中的
    # obs['cell_name'].unique() 来识别目标测试 cell context。
    # ------------------------------------------------------------
    all_test_cell_names = []

    for test_file in TEST_FILES:
        print(f"\n读取测试文件: {test_file}")
        adata_test = sc.read_h5ad(test_file)

        # 取该文件里实际存在的 cell_name，保持首次出现顺序。
        file_cell_names = pd.Index(adata_test.obs["cell_name"]).drop_duplicates().tolist()

        print("该文件中的测试 cell_name:")
        for name in file_cell_names:
            print(f"  - {name}")

        all_test_cell_names.extend(file_cell_names)

        # --------------------------------------------------------
        # 对该文件中的每一个测试 cell_name，单独生成一个最终输出文件。
        # --------------------------------------------------------
        for test_cell_name in file_cell_names:
            print("\n" + "-" * 100)
            print(f"开始处理测试 cell_name: {test_cell_name}")

            # 只保留当前测试 cell_name 对应的细胞。
            adata_cell = adata_test[adata_test.obs["cell_name"] == test_cell_name].copy()

            # 按该测试 cell_name 自身在数据中实际出现的顺序，读取 drug 列表。
            # 这正是你确认的实现策略：每个测试 cell_name 用自己实际出现的 drug 顺序。
            drug_order = pd.Index(adata_cell.obs["drug"]).drop_duplicates().tolist()

            if CONTROL_DRUG not in drug_order:
                raise ValueError(
                    f"测试 cell_name '{test_cell_name}' 中找不到对照药物 '{CONTROL_DRUG}'。"
                )

            # 取真实 control block。
            # 这部分既会保留为最终文件中的真实 control，
            # 也会作为所有非 control 药物的预测来源。
            ctrl_real = adata_cell[adata_cell.obs["drug"] == CONTROL_DRUG].copy()
            ctrl_real.obs = ctrl_real.obs.copy()
            ctrl_real.obs["pred_or_real"] = "real_ctrl"

            if ctrl_real.n_obs == 0:
                raise ValueError(
                    f"测试 cell_name '{test_cell_name}' 的 control 细胞数量为 0。"
                )

            # 将 control 表达矩阵转成 numpy，后续所有预测块都从这里采样。
            ctrl_X = ctrl_real.X.toarray() if sp.issparse(ctrl_real.X) else np.asarray(ctrl_real.X)

            print(f"当前测试 cell_name 的总细胞数: {adata_cell.n_obs}")
            print(f"control 细胞数: {ctrl_real.n_obs}")
            print(f"drug 数量: {len(drug_order)}")
            print("drug 顺序:")
            for drug in drug_order:
                print(f"  - {drug}")

            # ----------------------------------------------------
            # 按该 cell_name 自己的 drug 顺序构建最终输出。
            # 规则与参考代码2一致：
            # - control block 保留真实表达；
            # - 非 control block 用 control 表达采样生成预测；
            # - obs / var 沿用真实药物块骨架；
            # - 最终 concat 成一个 {cell_name}_pred.h5ad。
            # ----------------------------------------------------
            output_blocks = []

            for drug_index, drug_name in enumerate(drug_order):
                # control 药物块：直接保留真实 control。
                if drug_name == CONTROL_DRUG:
                    output_blocks.append(ctrl_real.copy())
                    continue

                # 当前药物的真实骨架块。
                # 它只提供：
                # 1) 真实细胞数 n_d；
                # 2) 原始 obs；
                # 3) 原始 var。
                stim_skeleton = adata_cell[adata_cell.obs["drug"] == drug_name].copy()
                n_d = stim_skeleton.n_obs

                if n_d == 0:
                    # 理论上不会发生，因为 drug_order 就来自当前子集。
                    # 这里保留防御式判断，避免脏数据导致异常行为。
                    continue

                # 为了让结果严格对齐真实测试药物块，
                # 从同一测试 cell_name 的 control 细胞中采样 n_d 个表达向量。
                # 若 n_d 大于 control 细胞数，则允许有放回采样。
                rng = np.random.default_rng(
                    RANDOM_SEED + stable_int(test_cell_name) + drug_index
                )
                sample_index = rng.choice(
                    ctrl_X.shape[0],
                    size=n_d,
                    replace=(n_d > ctrl_X.shape[0]),
                )

                X_sample = ctrl_X[sample_index].astype("float32", copy=False)

                # 用真实测试药物块的 obs / var 做骨架，只替换 X。
                # 这样输出文件在结构上与真实块保持一致。
                adata_pred = ad.AnnData(
                    X=X_sample,
                    obs=stim_skeleton.obs.copy(),
                    var=stim_skeleton.var.copy(),
                )
                adata_pred.obs["pred_or_real"] = "pred"

                output_blocks.append(adata_pred)

                del stim_skeleton, X_sample, adata_pred

            # 将所有 block 按固定顺序拼接成最终输出。
            final_adata = ad.concat(
                output_blocks,
                join="inner",
                merge="same",
                index_unique=None,
            )
            final_adata.obs_names_make_unique()

            # 生成输出文件名。
            out_name = f"{safe_name(test_cell_name)}_pred.h5ad"
            final_path = OUTPUT_DIR / out_name

            final_adata.write_h5ad(final_path)

            print(f"已生成: {final_path}")
            print(f"最终 shape: {final_adata.shape}")

            del adata_cell, ctrl_real, ctrl_X, output_blocks, final_adata
            cleanup_memory()

        del adata_test
        cleanup_memory()

    # 去重后打印所有测试 cell_name。
    all_test_cell_names = pd.Index(all_test_cell_names).drop_duplicates().tolist()

    print("\n" + "=" * 100)
    print("全部流程完成")
    print("测试 cell_name:")
    for name in all_test_cell_names:
        print(f"  - {name}")

    print("最终输出文件:")
    for name in all_test_cell_names:
        print(OUTPUT_DIR / f"{safe_name(name)}_pred.h5ad")
    print("=" * 100)