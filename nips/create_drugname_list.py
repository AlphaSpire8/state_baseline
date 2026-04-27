import scanpy as sc
import os
import csv

# ========== 配置 ==========
data_dir = "/data1/fanpeishan/STATE/for_state/data/nips"

file_groups = {
    "train": [
        "train_b_cells.h5ad",
        "train_nk_cells.h5ad",
        "train_t_cells_cd4.h5ad",
    ],
    "validation": [
        "val_t_cells_cd8.h5ad",
    ],
    "test": [
        "test_myeloid_cells.h5ad",
        "test_t_cells.h5ad",
    ],
}

# ========== 1. 读取所有文件，收集全集 drug ==========
all_drugs = set()
for group_name, file_list in file_groups.items():
    for fname in file_list:
        fpath = os.path.join(data_dir, fname)
        adata = sc.read_h5ad(fpath)
        all_drugs.update(adata.obs['drug'].unique().tolist())

drug_name_list = sorted(all_drugs)

# ========== 2. 写入 CSV ==========
output_path = os.path.join(data_dir, "drug_name_list.csv")

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["drug_name"])       # 列名
    for drug in drug_name_list:
        writer.writerow([drug])

print(f"✅ 已写入 {len(drug_name_list)} 个药物名称至: {output_path}")
