#!/usr/bin/env python3
"""
Reduced-Feature A-E Validation Experiment
==========================================
Anchors on original A-E labels. Validates that they remain
clinically coherent in a denoised raw-binary feature space.

Configs tested:
  C1: prev>=200, raw binary, PCA 50, UMAP/graph PCs 15
  C2: prev>=200, raw binary, PCA 50, UMAP/graph PCs 50
  C3: prev>=300, raw binary, PCA 50, UMAP/graph PCs 15
  C4: prev>=300, raw binary, PCA 50, UMAP/graph PCs 50
"""

import json
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
H5AD = ROOT / "data/original_reference/data/cpdiag_adata_t_all.h5ad"
PIVOT = ROOT / "data/original_reference/data/cpphe_pivot_s.csv"
SUBCLUSTER = ROOT / "data/original_reference/cluster/subcluster.csv"
PHEDEFS = ROOT / "data/external/phecode/phecode_definitions1.2.csv"
OUT = ROOT / "outputs/reports/reduced_feature_validation"
OUT.mkdir(parents=True, exist_ok=True)

# ── styling ──────────────────────────────────────────────────────
CLUSTER_COLORS = {
    "A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c",
    "D": "#d62728", "E": "#9467bd",
}
GMFCS_COLORS = {
    "I": "#e41a1c", "II": "#377eb8", "III": "#4daf4a",
    "IV": "#984ea3", "V": "#ff7f00",
}

# ══════════════════════════════════════════════════════════════════
# PHASE 1: Load data
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: Loading data")
print("=" * 70)

adata = ad.read_h5ad(H5AD)
internal_subtypes = adata.obs["subtype"].astype(str)
gmfcs = adata.obs["GMFCS_M"].astype(str)
h5ad_ids = adata.obs_names.tolist()  # format: "123-ALL"

# The AnnData subtype labels are internal Leiden labels. The final paper A-E
# labels are encoded by the prefix in subcluster.csv. In this artifact, internal
# B and E are swapped relative to the paper labels:
#   internal B -> paper E, internal E -> paper B.
# Use paper labels for all paper-facing plots and A-vs-B comparisons.
subcluster = pd.read_csv(SUBCLUSTER, dtype={"PERSON_ID_h5ad": str})
subcluster["paper_cluster"] = subcluster["subcluster"].astype(str).str.split("_").str[0]
paper_label_by_h5ad = subcluster.set_index("PERSON_ID_h5ad")["paper_cluster"]
missing_labels = pd.Index(h5ad_ids).difference(paper_label_by_h5ad.index)
if len(missing_labels):
    raise RuntimeError(f"Missing paper labels for {len(missing_labels)} H5AD patients")
subtypes = paper_label_by_h5ad.loc[h5ad_ids]

# Extract numeric person_ids from h5ad obs_names
h5ad_numeric_ids = [int(x.split("-")[0]) for x in h5ad_ids]

# Load raw binary pivot
pivot_raw = pd.read_csv(PIVOT)
pivot_raw = pivot_raw.set_index("PERSON_ID")

# Align to h5ad patients
common_ids = [pid for pid in h5ad_numeric_ids if pid in pivot_raw.index]
print(f"  H5AD patients: {len(h5ad_ids)}")
print(f"  Pivot patients: {len(pivot_raw)}")
print(f"  Overlap: {len(common_ids)}")

# Build aligned binary matrix
pivot_aligned = pivot_raw.loc[common_ids]

# Also align labels
id_to_h5ad_idx = {int(x.split("-")[0]): i for i, x in enumerate(h5ad_ids)}
label_indices = [id_to_h5ad_idx[pid] for pid in common_ids]
subtypes_aligned = subtypes.iloc[label_indices].values
gmfcs_aligned = gmfcs.iloc[label_indices].values

binary_matrix = pivot_aligned.values.astype(np.float64)
feature_names = pivot_aligned.columns.tolist()

print(f"  Aligned matrix: {binary_matrix.shape[0]} patients × {binary_matrix.shape[1]} features")
print(f"  Final paper A-E distribution: { {s: int((subtypes_aligned == s).sum()) for s in 'ABCDE'} }")
print("  Internal subtype -> final paper label mapping:")
mapping_check = pd.crosstab(internal_subtypes.loc[h5ad_ids], subtypes.loc[h5ad_ids])
print(mapping_check.to_string())
print(f"  Value range: [{binary_matrix.min()}, {binary_matrix.max()}]")

# ── Load Phecode descriptions ────────────────────────────────────
desc_map = {}
if PHEDEFS.exists():
    defs = pd.read_csv(PHEDEFS)
    defs.columns = [c.strip().lower() for c in defs.columns]
    for _, row in defs.iterrows():
        desc_map[str(row["phecode"])] = str(row.get("phenotype", ""))

def get_desc(phe_str):
    d = desc_map.get(phe_str, "")
    if not d:
        try:
            d = desc_map.get(str(float(phe_str)), "")
        except:
            pass
    return d if d else "—"


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Prevalence filtering
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: Prevalence filtering")
print("=" * 70)

prevalence = binary_matrix.sum(axis=0)  # per-feature patient count

feature_sets = {}
for cutoff in [200, 300]:
    mask = prevalence >= cutoff
    n_feat = mask.sum()
    feature_sets[cutoff] = {
        "mask": mask,
        "n_features": int(n_feat),
        "matrix": binary_matrix[:, mask],
        "names": [feature_names[i] for i in range(len(feature_names)) if mask[i]],
    }
    print(f"  Prevalence >= {cutoff}: {n_feat} features retained")


# ══════════════════════════════════════════════════════════════════
# PHASE 3: PCA + UMAP + Leiden for all 4 configs
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: PCA, UMAP, Leiden across 4 configs")
print("=" * 70)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    print("  WARNING: umap-learn not installed, will use sklearn TSNE fallback")
    HAS_UMAP = False

try:
    import igraph
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    print("  WARNING: leidenalg not installed, skipping Leiden")
    HAS_LEIDEN = False

configs = []
for cutoff in [200, 300]:
    for graph_pcs in [15, 50]:
        configs.append({
            "name": f"prev{cutoff}_pc{graph_pcs}",
            "label": f"≥{cutoff} features, {graph_pcs} graph PCs",
            "cutoff": cutoff,
            "graph_pcs": graph_pcs,
        })

results = {}

for cfg in configs:
    name = cfg["name"]
    cutoff = cfg["cutoff"]
    gpc = cfg["graph_pcs"]
    fs = feature_sets[cutoff]
    mat = fs["matrix"]
    n_feat = fs["n_features"]

    print(f"\n  ── Config: {cfg['label']} ({n_feat} features) ──")

    # PCA
    n_pcs = min(50, n_feat, mat.shape[0])
    pca = PCA(n_components=n_pcs, random_state=42)
    pca_scores = pca.fit_transform(mat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    pc15_var = float(cumvar[14]) if n_pcs >= 15 else float(cumvar[-1])
    pc50_var = float(cumvar[min(49, n_pcs - 1)])
    pcs_80 = int(np.searchsorted(cumvar, 0.80) + 1) if cumvar[-1] >= 0.80 else None

    print(f"    PCA: PC15 cumvar={pc15_var:.4f} ({pc15_var*100:.1f}%)")
    print(f"    PCA: PC50 cumvar={pc50_var:.4f} ({pc50_var*100:.1f}%)")
    if pcs_80:
        print(f"    PCA: PCs to 80% = {pcs_80}")
    else:
        print(f"    PCA: 80% NOT reached within {n_pcs} PCs (max={cumvar[-1]*100:.1f}%)")

    # Select PCs for graph/UMAP
    use_pcs = min(gpc, n_pcs)
    scores_for_umap = pca_scores[:, :use_pcs]

    # UMAP — robust approach
    from sklearn.manifold import TSNE
    # Try umap-learn with explicit float64 and error handling
    umap_ok = False
    if HAS_UMAP:
        try:
            import umap.umap_ as umap_impl
            reducer = umap.UMAP(
                n_neighbors=30, metric="euclidean",
                random_state=0, n_components=2, min_dist=0.3,
            )
            # Force float64 input
            X_input = np.ascontiguousarray(scores_for_umap, dtype=np.float64)
            embedding = reducer.fit_transform(X_input)
            umap_ok = True
        except Exception as e:
            print(f"    umap-learn failed ({e}), falling back to TSNE")
    if not umap_ok:
        print("    Using TSNE fallback")
        embedding = TSNE(
            n_components=2, random_state=0, perplexity=30, init="pca"
        ).fit_transform(scores_for_umap)

    print(f"    UMAP computed: {embedding.shape}")

    # Leiden concordance
    leiden_results = {}
    if HAS_LEIDEN:
        # Build kNN graph
        nn = NearestNeighbors(n_neighbors=30, metric="euclidean")
        nn.fit(scores_for_umap)
        distances, indices = nn.kneighbors(scores_for_umap)

        # Build igraph
        n = scores_for_umap.shape[0]
        edges = []
        weights = []
        for i in range(n):
            for j_idx in range(1, 30):  # skip self
                j = indices[i, j_idx]
                if i < j:
                    edges.append((i, j))
                    weights.append(1.0 / (1.0 + distances[i, j_idx]))
        g = igraph.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights

        best_ari = -1
        best_res = None
        best_labels = None
        best_sizes = None

        for res in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                     1.0, 1.2, 1.5, 2.0]:
            part = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=res,
                weights="weight",
                seed=0,
                n_iterations=-1,
            )
            labels = np.array(part.membership)
            n_clust = len(set(labels))
            ari = adjusted_rand_score(subtypes_aligned, labels)

            if n_clust == 5 and ari > best_ari:
                best_ari = ari
                best_res = res
                best_labels = labels.copy()
                best_sizes = {str(c): int((labels == c).sum()) for c in sorted(set(labels))}

            leiden_results[res] = {
                "n_clusters": n_clust,
                "ari": round(ari, 4),
            }

        if best_labels is None:
            # pick best ARI overall
            for res in sorted(leiden_results, key=lambda r: leiden_results[r]["ari"], reverse=True):
                if leiden_results[res]["n_clusters"] >= 4 and leiden_results[res]["n_clusters"] <= 7:
                    best_ari = leiden_results[res]["ari"]
                    best_res = res
                    break

        print(f"    Leiden best 5-cluster: res={best_res}, ARI={best_ari:.4f}")
        if best_sizes:
            print(f"    Leiden cluster sizes: {best_sizes}")

    results[name] = {
        "config": cfg,
        "n_features": n_feat,
        "pc15_var": pc15_var,
        "pc50_var": pc50_var,
        "pcs_to_80": pcs_80,
        "embedding": embedding,
        "leiden_best_res": best_res if HAS_LEIDEN else None,
        "leiden_best_ari": best_ari if HAS_LEIDEN else None,
        "leiden_best_sizes": best_sizes if HAS_LEIDEN else None,
        "leiden_all": leiden_results if HAS_LEIDEN else {},
        "pca_scores": pca_scores,
        "cumvar": cumvar.tolist(),
    }


# ══════════════════════════════════════════════════════════════════
# PHASE 4: Generate UMAP visualizations
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Generating UMAP panels")
print("=" * 70)

for name, res in results.items():
    emb = res["embedding"]
    cfg_label = res["config"]["label"]

    fig = plt.figure(figsize=(24, 16))
    ari_text = f"Leiden ARI={res['leiden_best_ari']:.3f}" if res["leiden_best_ari"] is not None else "Leiden ARI=N/A"
    fig.suptitle(
        f"A-E Validation: {cfg_label} ({res['n_features']} features)\n"
        f"PC15 var={res['pc15_var']*100:.1f}%  PC50 var={res['pc50_var']*100:.1f}%  {ari_text}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.25, wspace=0.20)

    # Panel 1: A-E clusters
    ax = fig.add_subplot(gs[0, 0])
    for cluster in ["A", "B", "C", "D", "E"]:
        mask = subtypes_aligned == cluster
        n_c = mask.sum()
        ax.scatter(emb[mask, 0], emb[mask, 1], c=CLUSTER_COLORS[cluster],
                   s=4, alpha=0.5, label=f"{cluster} (n={n_c})", rasterized=True)
    ax.set_title("Clusters A–E (original labels)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panels 2-6: GMFCS I-V
    for idx, level in enumerate(["I", "II", "III", "IV", "V"]):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        ax = fig.add_subplot(gs[row, col])

        is_level = gmfcs_aligned == level
        not_level = ~is_level
        n_level = is_level.sum()

        ax.scatter(emb[not_level, 0], emb[not_level, 1],
                   c="#e0e0e0", s=3, alpha=0.3, rasterized=True)
        ax.scatter(emb[is_level, 0], emb[is_level, 1],
                   c=GMFCS_COLORS[level], s=6, alpha=0.7,
                   label=f"GMFCS {level} (n={n_level})", rasterized=True)
        ax.set_title(f"GMFCS {level} highlighted (n={n_level})", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, markerscale=3)
        ax.set_xticks([])
        ax.set_yticks([])

    outpath = OUT / f"umap_panels_{name}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath.name}")


# ══════════════════════════════════════════════════════════════════
# PHASE 5: A vs B Feature Comparison
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: A vs B feature comparison")
print("=" * 70)

mask_a = subtypes_aligned == "A"
mask_b = subtypes_aligned == "B"
n_a = mask_a.sum()
n_b = mask_b.sum()

rows = []
for i, feat in enumerate(feature_names):
    prev_a = binary_matrix[mask_a, i].mean()
    prev_b = binary_matrix[mask_b, i].mean()
    count_a = int(binary_matrix[mask_a, i].sum())
    count_b = int(binary_matrix[mask_b, i].sum())

    # Fisher's exact test
    table = np.array([
        [count_a, n_a - count_a],
        [count_b, n_b - count_b],
    ])
    try:
        _, pval = stats.fisher_exact(table)
    except:
        pval = 1.0

    ratio = prev_b / prev_a if prev_a > 0.01 else (99.0 if prev_b > 0 else 0.0)
    diff = prev_b - prev_a
    desc = get_desc(feat)

    # Keyword tagging
    keywords_found = []
    desc_lower = desc.lower() + " " + feat.lower()
    for kw in ["pain", "joint", "contracture", "hemiplegia", "speech",
               "language", "spasticity", "unilateral", "plegia", "paresis",
               "stiffness", "dysphagia", "aspiration"]:
        if kw in desc_lower:
            keywords_found.append(kw)

    rows.append({
        "phecode": feat,
        "description": desc,
        "prev_A": round(prev_a, 4),
        "prev_B": round(prev_b, 4),
        "prev_A_pct": f"{prev_a*100:.1f}%",
        "prev_B_pct": f"{prev_b*100:.1f}%",
        "diff": round(diff, 4),
        "ratio_B_over_A": round(ratio, 2),
        "p_value": pval,
        "keywords": "|".join(keywords_found) if keywords_found else "",
    })

ab_df = pd.DataFrame(rows)
ab_df = ab_df.sort_values("diff", ascending=False, key=abs)

# Save full table
ab_df.to_csv(OUT / "a_vs_b_full.csv", index=False)

# Save top differences + keyword-tagged
top_diff = ab_df.head(40)
keyword_rows = ab_df[ab_df["keywords"] != ""].head(30)
ab_highlight = pd.concat([top_diff, keyword_rows]).drop_duplicates("phecode")
ab_highlight = ab_highlight.sort_values("diff", ascending=False, key=abs)
ab_highlight.to_csv(OUT / "a_vs_b_highlights.csv", index=False)

print(f"  Total features compared: {len(ab_df)}")
print(f"  Keyword-tagged features: {len(ab_df[ab_df['keywords'] != ''])}")
print(f"  Saved: a_vs_b_full.csv, a_vs_b_highlights.csv")

# Print top highlights
print(f"\n  TOP 15 A vs B DIFFERENCES:")
print(f"  {'Phecode':<10s} {'Description':<40s} {'A':>6s} {'B':>6s} {'Ratio':>6s} {'p-val':>8s} {'Keywords'}")
print(f"  {'─'*10} {'─'*40} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*15}")
for _, r in ab_highlight.head(15).iterrows():
    d = r["description"][:38] if len(str(r["description"])) > 38 else r["description"]
    print(f"  {r['phecode']:<10s} {d:<40s} {r['prev_A_pct']:>6s} {r['prev_B_pct']:>6s} "
          f"{r['ratio_B_over_A']:>5.1f}x {r['p_value']:>8.1e} {r['keywords']}")

print(f"\n  KEYWORD-TAGGED FEATURES (pain, joint, hemiplegia, speech, etc.):")
print(f"  {'Phecode':<10s} {'Description':<40s} {'A':>6s} {'B':>6s} {'Ratio':>6s} {'Keywords'}")
print(f"  {'─'*10} {'─'*40} {'─'*6} {'─'*6} {'─'*6} {'─'*15}")
for _, r in ab_df[ab_df["keywords"] != ""].sort_values("diff", ascending=False, key=abs).head(20).iterrows():
    d = r["description"][:38] if len(str(r["description"])) > 38 else r["description"]
    print(f"  {r['phecode']:<10s} {d:<40s} {r['prev_A_pct']:>6s} {r['prev_B_pct']:>6s} "
          f"{r['ratio_B_over_A']:>5.1f}x {r['keywords']}")


# ══════════════════════════════════════════════════════════════════
# PHASE 6: Summary report
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: Summary")
print("=" * 70)

summary_rows = []
for name, res in results.items():
    summary_rows.append({
        "config": res["config"]["label"],
        "n_features": res["n_features"],
        "pc15_var_pct": f"{res['pc15_var']*100:.1f}%",
        "pc50_var_pct": f"{res['pc50_var']*100:.1f}%",
        "pcs_to_80": res["pcs_to_80"] if res["pcs_to_80"] else ">50",
        "leiden_best_res": res["leiden_best_res"],
        "leiden_ari": f"{res['leiden_best_ari']:.3f}" if res["leiden_best_ari"] else "N/A",
        "leiden_sizes": str(res["leiden_best_sizes"]) if res["leiden_best_sizes"] else "N/A",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "config_summary.csv", index=False)

print("\n  CONFIG COMPARISON:")
print(summary_df.to_string(index=False))

# Write markdown report
def markdown_table(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join(lines)

md_lines = [
    "# Reduced-Feature A-E Validation Results\n",
    f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
    f"**Patients:** {binary_matrix.shape[0]}\n",
    "**Label source:** final paper A-E labels from `data/original_reference/cluster/subcluster.csv`, "
    "not internal AnnData `obs['subtype']` labels.\n",
    f"**Reference A-E:** A={int((subtypes_aligned=='A').sum())}, "
    f"B={int((subtypes_aligned=='B').sum())}, C={int((subtypes_aligned=='C').sum())}, "
    f"D={int((subtypes_aligned=='D').sum())}, E={int((subtypes_aligned=='E').sum())}\n",
    "\n## Configuration Comparison\n",
    markdown_table(summary_df),
    "\n\n## UMAP Panels\n",
]
for name in results:
    md_lines.append(f"\n### {results[name]['config']['label']}\n")
    md_lines.append(f"![UMAP]({OUT / f'umap_panels_{name}.png'})\n")

md_lines.append("\n## A vs B Feature Highlights\n")
md_lines.append(markdown_table(ab_highlight.head(25)))

with open(OUT / "validation_report.md", "w") as f:
    f.write("\n".join(md_lines))

# Save JSON summary
json_summary = {
    "patients": int(binary_matrix.shape[0]),
    "total_features": int(binary_matrix.shape[1]),
    "label_source": "data/original_reference/cluster/subcluster.csv subcluster prefix",
    "internal_to_paper_label_crosstab": mapping_check.to_dict(),
    "configs": {name: {
        "label": res["config"]["label"],
        "n_features": res["n_features"],
        "pc15_var": res["pc15_var"],
        "pc50_var": res["pc50_var"],
        "pcs_to_80": res["pcs_to_80"],
        "leiden_best_res": res["leiden_best_res"],
        "leiden_best_ari": res["leiden_best_ari"],
        "leiden_best_sizes": res["leiden_best_sizes"],
    } for name, res in results.items()},
    "output_dir": str(OUT),
}
with open(OUT / "validation_summary.json", "w") as f:
    json.dump(json_summary, f, indent=2)

print(f"\n  All outputs in: {OUT}")
print("  DONE.")
