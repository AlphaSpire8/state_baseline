import anndata as ad
import pandas as pd

# 1.读取h5ad文件
adata = ad.read_h5ad("/datasets/fanpeishan/data/tahoe/c39.h5ad")

# 2.获取obs['drug']列，去重并按字母顺序排序
drug_names = sorted(adata.obs['drug'].unique())

# 3.创建DataFrame
drug_df = pd.DataFrame({'drug_name': drug_names})

# 4.保存到CSV文件
drug_df.to_csv("/datasets/fanpeishan/data/docs/drug_name_list.csv", index=False)

print(f"共有 {len(drug_names)} 种不同的药物")
print("药物名称已按字母顺序排列")
print("文件已保存至 /datasets/fanpeishan/data/docs/drug_name_list.csv")

# 5.手动将drug_name_list.csv中的药物名称后面的空格去掉（有两个药物后面有空格）。
