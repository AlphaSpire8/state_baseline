from pathlib import Path
import pickle

import pandas as pd
import scanpy as sc


# =========================
# 路径配置
# =========================
# 原始 Tahoe 数据目录，目录下应为 c0.h5ad ~ c49.h5ad 这类文件
RAW_DATA_DIR = Path("~/data/tahoe")

# 参考高变基因列表所在的 pkl 文件
REF_GENE_PKL = Path("~/data/docs/ST_tahoe_var_dims.pkl")

# 药物名称顺序文件
DRUG_NAME_CSV = Path("~/data/docs/drug_name_list.csv")

# 输出目录根路径
OUTPUT_ROOT = Path("~/data")

# 按 drug_index 拆分后的输出目录
SPLIT_BY_DRUG_DIR = OUTPUT_ROOT / "split_by_drug"

# 仅保留这些obs 字段，其余全部丢弃
KEEP_OBS_COLUMNS = ["drugname_drugconc", "drug", "cell_name", "plate"]


# =========================
# 数据划分规则
# =========================
# 这部分来自“数据设置.md”，主要用于后续训练/验证/测试时保持固定划分。
# 本脚本的核心职责是做预处理并按 drug_index 拆分保存，因此这里先把规则固化为常量，
# 方便后续其他脚本直接 import 使用。

def cells(start, end):
    return [f"c{i}" for i in range(start, end + 1)]


SPLIT_WITH_VAL = {
    "train": cells(0, 3) + cells(5, 16) + ["c21"] + cells(25, 30) + cells(32, 37) + cells(39, 49),
    "val": ["c4", "c17", "c18", "c23", "c38"],
    "test": ["c19", "c20", "c22", "c24", "c31"],
}

SPLIT_NO_VAL = {
    "train": cells(0, 18) + ["c21", "c23"] + cells(25, 30) + cells(32, 49),
    "test": ["c19", "c20", "c22", "c24", "c31"],
}


# =========================
# 读取参考高变基因列表
# =========================
# pkl 文件中是一个字典，需要取出键名为 "gene_names" 的值。
with open(REF_GENE_PKL, "rb") as f:
    ref_var_dict = pickle.load(f)

ref_gene_names = list(ref_var_dict["gene_names"])


# =========================
# 读取药物名称顺序
# =========================
# 已知 csv 只有一列，且第一行是表头，名称为 "drug_name"。
# 这里直接读取这一列的全部内容，每一行就是一个药物名称。
# 最终得到 drug -> drug_index 的映射。
drug_name_list = pd.read_csv(DRUG_NAME_CSV)["drug_name"].tolist()
drug_to_index = {drug_name: idx for idx, drug_name in enumerate(drug_name_list)}


# =========================
# 搜集全部 h5ad 文件
# =========================
# 文件名形如 c10.h5ad，因此这里按数字部分排序，避免字符串排序时出现 c10 排在 c2 前面的情况。
h5ad_paths = sorted(
    RAW_DATA_DIR.glob("*.h5ad"),
    key=lambda path: int(path.stem[1:])
)

SPLIT_BY_DRUG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 逐个数据集做预处理
# =========================
for h5ad_path in h5ad_paths:
    cell_type = h5ad_path.stem
    print(f"\n{'=' * 80}")
    print(f"开始处理: {h5ad_path}")

    # 读取单个细胞类型数据
    adata = sc.read_h5ad(h5ad_path)

    # 1) 仅保留指定 obs 字段
    # 这里直接覆盖 adata.obs，只留下文档要求的 4 列。
    adata.obs = adata.obs[KEEP_OBS_COLUMNS].copy()

    # 2) 删除全部 obsm 信息
    adata.obsm.clear()

    # 3) 按参考高变基因列表严格对齐基因
    # 要求非常严格：
    # - pkl 里的基因如果当前 adata 里不存在，直接报错并中止程序；
    # - 最终保留下来的基因顺序必须与 pkl 中完全一致。
    current_gene_set = set(adata.var_names)
    missing_genes = [gene for gene in ref_gene_names if gene not in current_gene_set]
    if missing_genes:
        raise ValueError(
            f"{cell_type} 缺少 {len(missing_genes)} 个参考基因，程序中止。"
            f"前 10 个缺失基因: {missing_genes[:10]}"
        )

    # 由于前面已经检查过缺失基因，这里可以直接按参考顺序切片。
    # 这样既完成了基因筛选，也保证了顺序与 pkl 完全一致。
    adata = adata[:, ref_gene_names].copy()

    # 4) 增加 drug_index 和 cell_type 两列
    # drug_index 由 obs['drug'] 在 drug_name_list.csv 中的顺序决定。
    adata.obs["drug"] = adata.obs["drug"].astype(str).str.strip()
    adata.obs["drug_index"] = adata.obs["drug"].map(drug_to_index)

    # 如果出现 csv 里没有的药物名称，说明数据和字典不一致，直接报错。
    unknown_drugs = sorted(adata.obs.loc[adata.obs["drug_index"].isna(), "drug"].unique().tolist())
    if unknown_drugs:
        raise ValueError(
            f"{cell_type} 中存在 {len(unknown_drugs)} 个未出现在 drug_name_list.csv 的药物。"
            f"未知药物列表: {unknown_drugs[:10]}"
        )

    adata.obs["drug_index"] = adata.obs["drug_index"].astype(int)
    adata.obs["cell_type"] = cell_type

    # 5) 按 drug_index 拆分并保存
    # 输出目录结构要求如下：
    # /datasets/fanpeishan/data/tahoe_for_scgen/split_by_drug/
    #   └── celltype_c10/
    #         ├── celltype_c10_drugindex_0.h5ad
    #         ├── ...
    #         └── celltype_c10_drugindex_379.h5ad
    cell_output_dir = SPLIT_BY_DRUG_DIR / f"celltype_{cell_type}"
    cell_output_dir.mkdir(parents=True, exist_ok=True)

    drug_index_list = sorted(adata.obs["drug_index"].unique().tolist())
    for drug_index in drug_index_list:
        # 取出当前药物对应的子集
        adata_sub = adata[adata.obs["drug_index"] == drug_index].copy()

        # 组装输出文件名
        output_path = cell_output_dir / f"celltype_{cell_type}_drugindex_{drug_index}.h5ad"

        # 保存为 h5ad
        adata_sub.write_h5ad(output_path)

    print(f"处理完成: {cell_type}")
    print(f"细胞数: {adata.n_obs}, 基因数: {adata.n_vars}, 拆分后文件数: {len(drug_index_list)}")


print(f"\n{'=' * 80}")
print("全部数据处理完成。")
print(f"输出目录: {SPLIT_BY_DRUG_DIR}")
