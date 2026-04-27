import anndata as ad

base = "/data1/fanpeishan/STATE/for_state/data/for_chemCPA"

adatas = [ad.read_h5ad(f"{base}/c{i}_processed.h5ad") for i in [33, 34, 35]]

merged = ad.concat(adatas, join="inner")
merged.write_h5ad(f"{base}/train_c33_c34_c35.h5ad")
print("Done.", merged)