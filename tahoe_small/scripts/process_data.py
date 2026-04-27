import anndata as ad
import ast

# 定义要处理的细胞编号列表（从 33 到 35）
cell_numbers = [33, 34, 35]

# 循环处理每个文件
for num in cell_numbers:
    # 构建输入和输出路径
    input_path = f"/data1/fanpeishan/STATE/for_state/data/State-Tahoe-Filtered-processed/c{num}_prep.h5ad"
    output_path = f"/data1/fanpeishan/STATE/for_state/data/for_chemCPA/c{num}_processed.h5ad"
    
    # 读取数据
    adata = ad.read_h5ad(input_path)
    
    # 只保留目标 obs 列，删除 obsm
    adata.obs = adata.obs[["drug", "cell_name", "drugname_drugconc"]]
    adata.obsm = {}
    
    # 重命名 cell_name -> cell_type
    adata.obs.rename(columns={"cell_name": "cell_type"}, inplace=True)
    
    # 提取剂量信息：[('DrugName', dose, 'unit')] -> dose
    adata.obs["dose"] = adata.obs["drugname_drugconc"].apply(
        lambda x: ast.literal_eval(x)[0][1]
    )
    
    # 写入处理后的文件
    adata.write_h5ad(output_path)
    print(f"Processed c{num}_prep.h5ad -> c{num}_processed.h5ad. Done.", adata)