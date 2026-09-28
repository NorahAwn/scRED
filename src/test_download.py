import requests

API = "https://api.cellxgene.cziscience.com/curation/v1/datasets"
datasets = requests.get(API, timeout=60).json()
print(f"{len(datasets)} datasets in Census catalog\n")

hits = []
for d in datasets:
    diseases = [x["label"].lower() for x in d.get("disease", [])]
    tissues  = [x["label"].lower() for x in d.get("tissue", [])]
    if any("autism" in dz for dz in diseases):
        hits.append(d)
        print("TITLE :", d.get("title"))
        print("  id       :", d.get("dataset_id"))
        print("  disease  :", [x["label"] for x in d.get("disease", [])])
        print("  tissue   :", [x["label"] for x in d.get("tissue", [])])
        print("  assay    :", [x["label"] for x in d.get("assay", [])])
        print("  cells    :", d.get("cell_count"))
        # h5ad download link
        for a in d.get("assets", []):
            if a.get("filetype") == "H5AD":
                print("  H5AD     :", a.get("url"))
        print()

print(f"\n{len(hits)} autism dataset(s) found.")