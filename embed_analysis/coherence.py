# Intra-object embedding coherence: multi-positive SupCon vs diagonal InfoNCE
#
# Uses proposal-level embeddings extracted by extract_pre_nms_embeddings.py
# which contain gt_objid for every foreground proposal
#
# For each checkpoint, we:
# 1. Load proposal embeddings with gt_objid
# 2. Group proposals by gt_objid
# 3. Compute intra-object cosine similarity (coherence) for objects with 2+ proposals
# 4. Compare coherence distributions across checkpoints

# If multi-positive SupCon fixes the diagonal InfoNCE false-negative problem,
# proposals sharing a gt_objid should cluster more tightly in embedding space.

from __future__ import annotations

from ctypes import alignment
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from itertools import combinations
# force line-buffering for immediate output (prevents logs from buffering until program end)
sys.stdout.reconfigure(line_buffering=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from sklearn.manifold import TSNE
from umap import UMAP
from pacmap import PaCMAP

# =============================================================================
# Config
# =============================================================================
seed = 42
np.random.seed(seed)
data_split = "test"
run_root = os.path.expanduser("~/lsst_runs")
coherence_root = os.path.join(run_root, "coherence_auto")

# Map of label -> run_name
RUNS = {
    # "og_50ep": "clip5_30k_4h200_bs64_ep50",
    "flatten_15ep": "clip5_flatten_30k_4h200_bs64_ep15",
    # "flatten_50ep": "clip5_flatten_30k_4h200_bs64_ep15_resume",
    "supCon_15ep": "clip5_30k_4h200_bs192_ep15",
    "supCon_15ep_lprj": "clip5_30k_4h200_bs192_ep15_lprj",
}
_MODE_SUBDIR = {
    "z_q":      "zq_mlp_lprj",
    "pooled_q": "pooledq_mlp_lprj",
    "z_k":      "zk_mlp_lprj",
    "pooled_k": "pooledk_mlp_lprj",
    "both":     "both",
}
# Embedding mode to analyze Options: "z_q", "pooled_q", "z_k", "pooled_k", or "both"
EMB_MODE = os.environ.get("EMB_MODE", "pooled_q")
if EMB_MODE not in {"z_q", "pooled_q", "z_k", "pooled_k", "both"}:
    raise ValueError(f"Invalid EMB_MODE: {EMB_MODE}. Use 'z_q', 'pooled_q', 'z_k', 'pooled_k', or 'both'.")
print(f"EMB_MODE: {EMB_MODE}")
out_dir = os.path.join(coherence_root, _MODE_SUBDIR[EMB_MODE])
os.makedirs(out_dir, exist_ok=True)

# Coherence computation
MIN_PROPOSALS_COHERENCE = 2   # min proposals per object to include in coherence calc
MIN_PROPOSALS_INTER = 2        # min proposals per object to include in inter-object similarity calc
MIN_PROPOSALS_PLOT = 3        # min proposals per object to include in reducer plots
INTER_N_OBJ_PAIRS = 5000    # object pairs to sample for inter-object similarity
MAX_CROSS_PAIRS_PER_OBJ_PAIR = None  # max cross-object proposal pairs to consider per object pair (None for no limit)
# Reducer / visualization
REDUCER_METHOD = "pacmap"     # "pacmap", "umap", or "tsne"
REDUCER_N_OBJECTS = 50        # top-N multi-proposal objects shown in reducer plots

# Plot output
SAVEFIG_DPI = 150
PANEL_WIDTH = 6               # inches per subplot panel
PANEL_HEIGHT = 5.5

def requested_emb_keys(mode):
    """Return representation keys requested by mode."""
    if mode == "z_q":
        return ["z_q"]
    if mode == "pooled_q":
        return ["pooled_q"]
    if mode == "z_k":
        return ["z_k"]
    if mode == "pooled_k":
        return ["pooled_k"]
    if mode == "both":
        return ["z_q", "pooled_q", "z_k", "pooled_k"]
    raise ValueError(f"Unsupported mode: {mode}")

def _available_rep_keys(emb_dict):
    """Return available representation keys for dense or manifest-backed data."""
    return emb_dict.get("_chunk_available_rep_keys", [k for k in ["z_q", "pooled_q", "z_k", "pooled_k"] if k in emb_dict])

def _emb_key_tag(emb_key):
    """Return a short label for plot titles and filenames."""
    return emb_key.replace("_", "-")

def candidate_embedding_filenames(mode):
    if mode in {"z_q", "z_k"}:
        return ["proposal_emb_z.pt", "proposal_emb.pt"]
    if mode in {"pooled_q", "pooled_k"}:
        return ["proposal_emb_pooled.pt", "proposal_emb_pooled_manifest.pt"]
    if mode == "both":
        return ["proposal_emb_both.pt", "proposal_emb_both_manifest.pt"]
    raise ValueError(f"Unsupported mode: {mode}")

# ======================
# Chunk manifest helpers
# ======================
def _is_chunk_manifest(d):
    """Return True if dict looks like a chunked embedding manifest."""
    return isinstance(d, dict) and d.get("chunked", False) and "rank_manifests" in d

def _resolve_chunk_path(chunk_path, emb_dir, manifest_dict):
    """Resolve a chunk file path from manifest metadata."""
    if os.path.isabs(chunk_path):
        return chunk_path
    chunk_dir = manifest_dict.get("chunk_dir", "")
    if chunk_dir:
        if os.path.isabs(chunk_dir):
            return os.path.join(chunk_dir, chunk_path)
        base = chunk_dir if os.path.isabs(chunk_dir) else os.path.join(emb_dir, chunk_dir)
        return os.path.join(base, chunk_path)
        # return os.path.join(emb_dir, chunk_dir, chunk_path)
    return os.path.join(emb_dir, chunk_path)

def _load_from_chunk_manifest(manifest_dict, emb_dir):
    """Load lightweight metadata from chunk manifest without reading chunks."""
    total_chunks = 0
    total_props = 0
    available_keys = set()
    mode = manifest_dict.get("mode")
    if mode in {"z_q", "z_k"}:
        available_keys.update(["z_q", "z_k"])
    elif mode in {"pooled_q", "pooled_k"}:
        available_keys.update(["pooled_q", "pooled_k"])
    elif mode == "both":
        available_keys.update(["z_q", "pooled_q", "z_k", "pooled_k"])

    for rank_manifest in manifest_dict.get("rank_manifests", []):
        for chunk_info in rank_manifest.get("chunk_meta", []):
            chunk_path = _resolve_chunk_path(chunk_info["path"], emb_dir, manifest_dict)
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Missing chunk file: {chunk_path}")
            total_chunks += 1
            total_props += int(chunk_info.get("num_props", 0))

    return {
        "_chunk_manifest": manifest_dict,
        "_chunk_emb_dir": emb_dir,
        "_chunk_total_chunks": total_chunks,
        "_chunk_num_props": total_props,
        "_chunk_available_rep_keys": sorted(available_keys),
    }, total_chunks

def _iter_manifest_chunks(manifest_dict, emb_dir):
    """Yield chunk dicts in manifest order."""
    for rank_manifest in manifest_dict.get("rank_manifests", []):
        for chunk_info in rank_manifest.get("chunk_meta", []):
            chunk_path = _resolve_chunk_path(chunk_info["path"], emb_dir, manifest_dict)
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Missing chunk file: {chunk_path}")
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
            yield chunk
            del chunk

def _load_chunk_labels_metadata(chunk_info, emb_dir, manifest_dict):
    """Load lightweight labels metadata for a chunk, preferring sidecar files."""
    meta_path = chunk_info.get("meta_path")
    if meta_path is not None:
        resolved_meta_path = _resolve_chunk_path(meta_path, emb_dir, manifest_dict)
        if os.path.exists(resolved_meta_path):
            return torch.load(resolved_meta_path, map_location="cpu", weights_only=False)

    # Backward-compatible fallback for older manifests without sidecar metadata.
    chunk_path = _resolve_chunk_path(chunk_info["path"], emb_dir, manifest_dict)
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Missing chunk file: {chunk_path}")
    chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
    return {
        "gt_objid": chunk["gt_objid"],
        "gt_classes": chunk["gt_classes"],
        "file_name": chunk["file_name"],
        "num_props": int(chunk["gt_objid"].shape[0]),
    }

def _assert_chunk_has_key(chunk, emb_key):
    if emb_key not in chunk:
        raise KeyError(f"Chunk is missing embedding key '{emb_key}'. Available keys: {list(chunk.keys())}")

# =============================================================================
# Load all checkpoints
# =============================================================================
def _load_first_existing(base_dir, filenames):
    """Load the first existing file from filenames under base_dir"""
    for fname in filenames:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=False), path
    return None, None

def load_proposal_emb(run_name, data_split, run_root, requested_keys):
    """Load proposal-level embeddings and validate requested representations"""
    emb_dir = os.path.join(run_root, run_name, "preds", data_split, "pre_nms_embeddings")
    primary, primary_path = _load_first_existing(emb_dir, candidate_embedding_filenames(EMB_MODE))
    if primary is None and EMB_MODE == "both":
        z_dict, z_path = _load_first_existing(emb_dir, ["proposal_emb_z.pt"])
        p_dict, p_path = _load_first_existing(emb_dir, ["proposal_emb_pooled.pt"])
        if z_dict is not None and p_dict is not None:
            if not torch.equal(z_dict["gt_objid"], p_dict["gt_objid"]):
                raise ValueError(f"Cannot merge {z_path} and {p_path}: gt_objid tensors differ")
            if list(z_dict["file_name"]) != list(p_dict["file_name"]):
                raise ValueError(f"Cannot merge {z_path} and {p_path}: file_name lists differ")
            primary = {
                "gt_objid": z_dict["gt_objid"],
                "gt_classes": z_dict["gt_classes"],
                "file_name": z_dict["file_name"],
                "z_q": z_dict.get("z_q"),
                "z_k": z_dict.get("z_k"),
                "pooled_q": p_dict.get("pooled_q"),
                "pooled_k": p_dict.get("pooled_k"),
            }
            primary_path = f"merged[{os.path.basename(z_path)}+{os.path.basename(p_path)}]"

    if primary is None:
        raise FileNotFoundError(f"Missing embedding file in {emb_dir}. Tried: {candidate_embedding_filenames(EMB_MODE)}")

    manifest_mode = _is_chunk_manifest(primary)
    if manifest_mode:
        d, n_chunks = _load_from_chunk_manifest(primary, emb_dir)
        primary_path = f"{primary_path} (chunk-manifest, chunks={n_chunks})"
    else:
        d = primary
    available_keys = _available_rep_keys(d)
    missing_keys = [k for k in requested_keys if k not in available_keys]
    if missing_keys:
        print(f"  [WARN] {run_name}: missing requested representations {missing_keys}")
    if not available_keys:
        raise KeyError(f"No supported representation keys in {primary_path}. Expected one of ['z_q', 'pooled_q', 'z_k', 'pooled_k']")

    if manifest_mode:
        n = d.get("_chunk_num_props", 0)
        print(
            f"  Loaded {run_name}: {n:,} proposals, unique objids=<deferred>, "
            f"available reps={available_keys}, file={primary_path}"
        )
    else:
        n = d[available_keys[0]].shape[0]
        n_unique = d["gt_objid"].unique().numel()
        print(
            f"  Loaded {run_name}: {n:,} proposals, {n_unique:,} unique objids, "
            f"available reps={available_keys}, file={primary_path}"
        )
    d["_available_rep_keys"] = available_keys
    return d

data = {}
requested_keys = requested_emb_keys(EMB_MODE)
for label, run_name in RUNS.items():
    print(f"\nLoading: {label} ({run_name})")
    try:
        d = load_proposal_emb(run_name, data_split, run_root, requested_keys)
        if EMB_MODE != "both" and EMB_MODE not in d["_available_rep_keys"]:
            print(f"  [SKIP] {run_name}: requested mode '{EMB_MODE}' not present in file")
            continue
        data[label] = d
    except (FileNotFoundError, KeyError) as e:
        print(f"  [SKIP] {e}")

print(f"\nSuccessfully loaded {len(data)} / {len(RUNS)} runs.")

# ======================
# Shared computation helpers
# ======================
def _pairwise_upper_cos_sim(z):
    """Upper-triangle pairwise cosine similarities for L2-normalized embeddings z (k, D)"""
    # b/c already L2-normalized, dot product = cosine sim
    # so we calculate all pairwise cosin sims for props of same obj
    #   sim[i, j] = cosine sim b/w prop i and prop j
    #   sim = [ [1,  a,  b,  ...],
    #           [a,  1,  c,  ...],
    #           [b,  c,  1,  ...],
    #           ...           ]
    # We only want unique off-diag pairs (i < j):
    #   mask = upper triangle of the matrix, excluding the diag
    sim = (z @ z.T).numpy() # (k, k) pairwise cosine similarity matrix
    # upper triangle, no diag
    return sim[np.triu(np.ones_like(sim, dtype=bool), k=1)] # 1D array of all unique proposal pairs

def _build_objid_to_idx(objids):
    objid_to_idx = defaultdict(list)
    for i, oid in enumerate(objids.tolist()):
        objid_to_idx[oid].append(i)
    return objid_to_idx

def _print_object_filter_stats(total_props, n_unique, n_valid, min_proposals, counts):
    print(f"  Total proposals: {total_props:,}")
    print(f"  Unique objects: {n_unique:,}")
    print(f"  Objects with >= {min_proposals} proposals: {n_valid:,}")
    # distribution of proposals per multi-proposal object
    if counts:
        print(f"  Proposals/object: mean={np.mean(counts):.1f}, "
              f"median={np.median(counts):.0f}, max={np.max(counts)}")

def _sample_object_pairs(multi_objids, n_object_pairs, seed):
    # (n * (n-1)) / 2 determines number of pairs of a given number of items (n)
    total_pairs = len(multi_objids) * (len(multi_objids) - 1) // 2
    if total_pairs == 0:
        return [], total_pairs
    rng = np.random.default_rng(seed)
    n_sample = min(n_object_pairs, total_pairs)
    print(f"  Total unordered object pairs: {total_pairs:,}") # 32,445,596,953
    print(f"  Sampled object pairs: {n_sample:,}")
    # random.choice bad bc it worked by first generating a complete permutation of all indices 
    # and then taking the first k samples (which is impossible for large n)
    # flat_indices = np.random.choice(total_pairs, size=n_sample, replace=False)
    flat_indices = rng.choice(total_pairs, size=n_sample, replace=False)
    # Convert flat index k -> (i, j) using cool formula for combinatorial number system 
    # i = floor((1 + sqrt(1 + 8k)) / 2), j = k - i*(i-1)//2 
    i = (np.floor((1 + np.sqrt(1 + 8 * flat_indices.astype(np.float64))) / 2)).astype(np.int64) 
    j = (flat_indices - i * (i - 1) // 2).astype(np.int64)
    pairs = [(multi_objids[ii], multi_objids[jj]) for ii, jj in zip(i, j)]
    # below is an alternative approach that generates all pairs and samples from them, 
    # but it's super big since there are many multi-proposal objects, 
    # so we use the above while loop sampling instead
    # all_pairs = list(combinations(multi_objids, 2))
    # pair_idx = np.random.choice(len(all_pairs), size=n_sample, replace=False)
    # return [all_pairs[i] for i in pair_idx], total_pairs
    return pairs, total_pairs

# =============================================================================
# Compute intra-object coherence
# =============================================================================
# take an obj, look at all proposals matched to that obj, compute pairwise sims among those proposals, then summarize them
def compute_intra_object_coherence(emb_dict, emb_key, min_proposals):
    """
    For each GT object with >= min_proposals foreground proposals, compute the
    mean pairwise cosine similarity among those proposals' selected embeddings

    Parameters
    ----------
    emb_dict : dict with keys emb_key (N, D), 'gt_objid' (N,), 'gt_classes' (N,), 'file_name' (N,)
    emb_key : str
        Representation key to use ('z_q' or 'pooled_q')
    min_proposals : int
        Minimum proposals per object to include

    Returns
    -------
    pd.DataFrame with columns:
        objid            : int64, GT object identifier
        n_proposals      : int, number of proposals matched to this object
        mean_cos_sim     : float, mean pairwise cosine similarity (intra-object)
        min_cos_sim      : float
        max_cos_sim      : float
        std_cos_sim      : float
        gt_class         : int, class label (0=galaxy, 1=star)
        file_name        : str, file name (from first proposal)
    """
    if emb_key not in emb_dict:
        if "_chunk_manifest" in emb_dict and emb_key in _available_rep_keys(emb_dict):
            return _compute_intra_object_coherence_chunked(emb_dict, emb_key=emb_key, min_proposals=min_proposals)
        raise KeyError(f"Missing embedding key '{emb_key}'. Available keys: {list(emb_dict.keys())}")

    z_any = emb_dict[emb_key].float()
    objids = emb_dict["gt_objid"]
    gt_cls = emb_dict["gt_classes"]
    file_names = emb_dict["file_name"]
    # group proposal indices by objid
    objid_to_idx = _build_objid_to_idx(objids)
    # filter to objects with enough proposals
    multi = {oid: prop_idxs for oid, prop_idxs in objid_to_idx.items() if len(prop_idxs) >= min_proposals}
    _print_object_filter_stats(len(z_any), len(objid_to_idx), len(multi), min_proposals, [len(v) for v in multi.values()])
    if not multi:
        print("  [WARNING] No multi-proposal objects found!")
        return pd.DataFrame()

    records = []
    for oid, idxs in multi.items():
        z = z_any[idxs]  # (k, D)
        pairwise = _pairwise_upper_cos_sim(z)
        records.append({
            "objid": oid,
            "n_proposals": len(idxs),
            "mean_cos_sim": pairwise.mean(),
            "min_cos_sim": pairwise.min(),
            "max_cos_sim": pairwise.max(),
            "std_cos_sim": pairwise.std(),
            "gt_class": gt_cls[idxs[0]].item(),
            "file_name": file_names[idxs[0]],
        })
    del z_any
    return pd.DataFrame(records)

def _compute_intra_object_coherence_chunked(emb_dict, emb_key, min_proposals):
    """Compute coherence by streaming embedding chunks instead of full materialization."""
    manifest = emb_dict["_chunk_manifest"]
    emb_dir = emb_dict["_chunk_emb_dir"]

    # Pass 1: count proposals per object and capture first class/file metadata.
    objid_to_count = defaultdict(int)
    objid_to_class = {}
    objid_to_file = {}
    total_props = 0
    for rank_manifest in manifest.get("rank_manifests", []):
        for chunk_info in rank_manifest.get("chunk_meta", []):
            meta = _load_chunk_labels_metadata(chunk_info, emb_dir, manifest)
            chunk_objids = meta["gt_objid"].tolist()
            chunk_classes = meta["gt_classes"].tolist()
            chunk_files = meta["file_name"]
            total_props += len(chunk_objids)
            for oid, cls, fn in zip(chunk_objids, chunk_classes, chunk_files):
                objid_to_count[oid] += 1
                if oid not in objid_to_class:
                    objid_to_class[oid] = int(cls)
                    objid_to_file[oid] = fn
            del meta

    multi_counts = {oid: c for oid, c in objid_to_count.items() if c >= min_proposals}
    _print_object_filter_stats(total_props, len(objid_to_count), len(multi_counts), min_proposals, list(multi_counts.values()))
    if not multi_counts:
        print("  [WARNING] No multi-proposal objects found!")
        return pd.DataFrame()

    target_objids = set(multi_counts.keys())
    print(f"  Streaming {sum(multi_counts.values()):,} target proposals across chunks...")

    # Pass 2: stream embeddings and finalize an object when all its proposals are seen.
    seen_per_obj = defaultdict(int)
    cache_per_obj = defaultdict(list)
    records = []
    for chunk in _iter_manifest_chunks(manifest, emb_dir):
        _assert_chunk_has_key(chunk, emb_key)
        grouped_local = defaultdict(list)
        for row_i, oid in enumerate(chunk["gt_objid"].tolist()):
            if oid in target_objids:
                grouped_local[oid].append(row_i)

        for oid, rows in grouped_local.items():
            row_idx = torch.as_tensor(rows, dtype=torch.long)
            cache_per_obj[oid].append(chunk[emb_key][row_idx].float().cpu())
            seen_per_obj[oid] += len(rows)
            if seen_per_obj[oid] == multi_counts[oid]:
                z = torch.cat(cache_per_obj[oid], dim=0)
                pairwise =  _pairwise_upper_cos_sim(z)
                records.append({
                    "objid": oid,
                    "n_proposals": multi_counts[oid],
                    "mean_cos_sim": pairwise.mean(),
                    "min_cos_sim": pairwise.min(),
                    "max_cos_sim": pairwise.max(),
                    "std_cos_sim": pairwise.std(),
                    "gt_class": objid_to_class.get(oid, -1),
                    "file_name": objid_to_file.get(oid, ""),
                })
                del cache_per_obj[oid]
        del chunk

    remaining = [oid for oid in target_objids if seen_per_obj[oid] != multi_counts[oid]]
    if remaining:
        print(f"  [WARN] {len(remaining)} objects were incomplete after chunk scan.")
    return pd.DataFrame(records)

def compute_coherence_all_runs(data, emb_key, min_proposals=MIN_PROPOSALS_COHERENCE):
    """Compute coherence dataframe per run for a selected embedding key."""
    coherence = {}
    for label, d in data.items():
        print(f"\n--- {label} [{emb_key}] ---")
        if emb_key not in _available_rep_keys(d):
            print(f"  [SKIP] Missing key '{emb_key}' for run {label}")
            coherence[label] = pd.DataFrame()
            continue
        df = compute_intra_object_coherence(d, emb_key=emb_key, min_proposals=min_proposals)
        coherence[label] = df
        if len(df) > 0:
            print(f"  Coherence: mean={df['mean_cos_sim'].mean():.4f}, "
                  f"median={df['mean_cos_sim'].median():.4f}, "
                  f"std={df['mean_cos_sim'].std():.4f}")
    return coherence

# =============================================================================
# Inter-object similarity 
# =============================================================================
def compute_inter_object_similarity(emb_dict, emb_key, n_object_pairs, min_proposals, 
                                    max_cross_pairs_per_obj_pair=None, seed=seed):
    """
    Return one row per sampled unordered object pair (objid_a, objid_b),
    summarizing cross-object proposal cosine sims
    (basically cosim b/w props matched to obj A and props matched to obj B)
    
    Parameters
    ----------
    emb_dict : dict
        Must contain emb_key, 'gt_objid', 'gt_classes', 'file_name'
    emb_key : str
        Representation key, e.g. 'z_q'/'z_k' or 'pooled_q'/'pooled_k'
    n_object_pairs : int
        Number of unordered object pairs to sample
    min_proposals : int
        Keep only objects with >= this many proposals
    max_cross_pairs_per_obj_pair : int or None
        If not None and n_props_a * n_props_b exceeds this, subsample proposals
        within the pair to keep cross-pair count bounded
    seed : int
        RNG seed
    Returns
    -------
    pd.DataFrame
    """
    if emb_key not in emb_dict:
        if "_chunk_manifest" in emb_dict and emb_key in _available_rep_keys(emb_dict):
            return _compute_inter_object_similarity_chunked(emb_dict, emb_key=emb_key, n_object_pairs=n_object_pairs,
                                                            min_proposals=min_proposals, 
                                                            max_cross_pairs_per_obj_pair=max_cross_pairs_per_obj_pair,
                                                            seed=seed)
        raise KeyError(f"Missing embedding key '{emb_key}'. Available keys: {list(emb_dict.keys())}")
    
    z_any = emb_dict[emb_key].float()
    objids = emb_dict["gt_objid"]
    gt_cls = emb_dict["gt_classes"]
    file_names = emb_dict["file_name"]
    # group proposal indices by objid
    objid_to_idx = _build_objid_to_idx(objids)
    # filter to objects with enough proposals
    multi = {oid: prop_idxs for oid, prop_idxs in objid_to_idx.items() if len(prop_idxs) >= min_proposals}
    _print_object_filter_stats(len(z_any), len(objid_to_idx), len(multi), min_proposals, [len(idx) for idx in multi.values()])
    multi_objids = sorted(multi.keys())
    selected_pairs, total_pairs = _sample_object_pairs(multi_objids, n_object_pairs, seed)
    if total_pairs == 0:
        print("  [WARNING] No multi-proposal object pairs found!")
        return pd.DataFrame()

    records = []
    for objid_a, objid_b in selected_pairs:
        idx_a = multi[objid_a]
        idx_b = multi[objid_b]
        n_props_a = len(idx_a)
        n_props_b = len(idx_b)
        use_idx_a = idx_a
        use_idx_b = idx_b
        # if the number of cross-object proposal pairs exceeds the max, subsample proposals within each object to keep it manageable
        if max_cross_pairs_per_obj_pair is not None:
            full_pairs = n_props_a * n_props_b
            if full_pairs > max_cross_pairs_per_obj_pair:
                max_props_each = max(1, int(np.sqrt(max_cross_pairs_per_obj_pair)))
                if n_props_a > max_props_each:
                    use_idx_a = np.random.choice(idx_a, size=max_props_each, replace=False).tolist()
                if n_props_b > max_props_each:
                    use_idx_b = np.random.choice(idx_b, size=max_props_each, replace=False).tolist()
        z_a = z_any[use_idx_a]
        z_b = z_any[use_idx_b]
        # we need to flatten since it's a 2D rectangular matrix, 
        # and we want to summarize over all entries as one 1D collection of sims
        sim = (z_a @ z_b.T).flatten().numpy()
        records.append({
            "objid_a": objid_a,
            "objid_b": objid_b,
            "n_props_a": n_props_a,
            "n_props_b": n_props_b,
            "n_pairs": int(sim.size),
            "mean_cos_sim": sim.mean(),
            "min_cos_sim": sim.min(),
            "max_cos_sim": sim.max(),
            "std_cos_sim": sim.std(),
            "gt_class_a": gt_cls[idx_a[0]].item(),
            "gt_class_b": gt_cls[idx_b[0]].item(),
            "file_name_a": file_names[idx_a[0]],
            "file_name_b": file_names[idx_b[0]],
        })
    del z_any
    return pd.DataFrame(records)

def _compute_inter_object_similarity_chunked(emb_dict, emb_key, n_object_pairs, min_proposals, 
                                             max_cross_pairs_per_obj_pair=None, seed=seed):
    """
    Chunked version: sample unordered object pairs, gather embeddings only for the
    selected objects by streaming chunks, then summarize cross-object similarities
    """
    manifest = emb_dict["_chunk_manifest"]
    emb_dir = emb_dict["_chunk_emb_dir"]
    # Pass 1: count proposals per object and capture first class/file metadata
    objid_to_count = defaultdict(int)
    objid_to_class = {}
    objid_to_file = {}
    total_props = 0
    for rank_manifest in manifest.get("rank_manifests", []):
        for chunk_info in rank_manifest.get("chunk_meta", []):
            meta = _load_chunk_labels_metadata(chunk_info, emb_dir, manifest)
            chunk_objids = meta["gt_objid"].tolist()
            chunk_classes = meta["gt_classes"].tolist()
            chunk_files = meta["file_name"]
            total_props += len(chunk_objids)
            for oid, cls, fn in zip(chunk_objids, chunk_classes, chunk_files):
                objid_to_count[oid] += 1
                if oid not in objid_to_class:
                    objid_to_class[oid] = int(cls)
                    objid_to_file[oid] = fn
            del meta
    multi_counts = {oid: c for oid, c in objid_to_count.items() if c >= min_proposals}
    _print_object_filter_stats(total_props, len(objid_to_count), len(multi_counts), min_proposals, list(multi_counts.values()))
    multi_objids = sorted(multi_counts.keys())
    selected_pairs, total_pairs = _sample_object_pairs(multi_objids, n_object_pairs, seed)
    if total_pairs == 0:
        print("  [WARNING] No multi-proposal object pairs found!")
        return pd.DataFrame()
    selected_objids = set()
    for a, b in selected_pairs:
        selected_objids.add(a)
        selected_objids.add(b)
    print(f"  Selected unique objects to stream: {len(selected_objids):,}")
    # Pass 2: stream embeddings and gather for selected objects
    cache_per_obj = defaultdict(list)
    for chunk in _iter_manifest_chunks(manifest, emb_dir):
        _assert_chunk_has_key(chunk, emb_key)
        grouped_local = defaultdict(list)
        for row_i, oid in enumerate(chunk["gt_objid"].tolist()):
            if oid in selected_objids:
                grouped_local[oid].append(row_i)
        for oid, rows in grouped_local.items():
            row_idx = torch.as_tensor(rows, dtype=torch.long)
            cache_per_obj[oid].append(chunk[emb_key][row_idx].float().cpu())
        del chunk
    obj_embs = {}
    for oid, parts in cache_per_obj.items():
        obj_embs[oid] = torch.cat(parts, dim=0)
    records = []
    for objid_a, objid_b in selected_pairs:
        z_a_full = obj_embs[objid_a]
        z_b_full = obj_embs[objid_b]
        n_props_a = z_a_full.shape[0]
        n_props_b = z_b_full.shape[0]
        z_a = z_a_full
        z_b = z_b_full
        if max_cross_pairs_per_obj_pair is not None:
            full_pairs = n_props_a * n_props_b
            if full_pairs > max_cross_pairs_per_obj_pair:
                max_props_each = max(1, int(np.sqrt(max_cross_pairs_per_obj_pair)))
                if n_props_a > max_props_each:
                    sel_a = torch.from_numpy(np.random.choice(n_props_a, size=max_props_each, replace=False)).long()
                    z_a = z_a_full[sel_a]
                if n_props_b > max_props_each:
                    sel_b = torch.from_numpy(np.random.choice(n_props_b, size=max_props_each, replace=False)).long()
                    z_b = z_b_full[sel_b]
        sim = (z_a @ z_b.T).flatten().numpy()
        records.append({
            "objid_a": objid_a,
            "objid_b": objid_b,
            "n_props_a": n_props_a,
            "n_props_b": n_props_b,
            "n_pairs": int(sim.size),
            "mean_cos_sim": sim.mean(),
            "min_cos_sim": sim.min(),
            "max_cos_sim": sim.max(),
            "std_cos_sim": sim.std(),
            "gt_class_a": objid_to_class.get(objid_a, -1),
            "gt_class_b": objid_to_class.get(objid_b, -1),
            "file_name_a": objid_to_file.get(objid_a, ""),
            "file_name_b": objid_to_file.get(objid_b, ""),
        })
    return pd.DataFrame(records)

def compute_inter_similarity_all_runs(data, emb_key, n_object_pairs=INTER_N_OBJ_PAIRS, 
                                      min_proposals=MIN_PROPOSALS_INTER,
                                      max_cross_pairs_per_obj_pair=MAX_CROSS_PAIRS_PER_OBJ_PAIR, 
                                      seed=seed):
    """Compute inter-object baseline similarity per run for a selected key"""
    inter_sim = {}
    for label, d in data.items():
        print(f"\n--- {label} [{emb_key}] ---")
        if emb_key not in _available_rep_keys(d):
            print(f"  {label}: [skip] missing embedding key {emb_key}")
            inter_sim[label] = pd.DataFrame()
            continue
        df = compute_inter_object_similarity(d, emb_key, n_object_pairs, min_proposals, 
                                             max_cross_pairs_per_obj_pair, seed)
        inter_sim[label] = df
        if len(df) > 0:
            print(f" Inter-object: rows={len(df):,}, "
                f"mean={df['mean_cos_sim'].mean():.4f}, "
                f"median={df['mean_cos_sim'].median():.4f}, "
                f"std={df['mean_cos_sim'].std():.4f}")
    return inter_sim

# ==========================================================
# Helper functions for coherence and inter-object similarity
# ==========================================================
# * just means everything after sim_dict and out_dir must be passed in as keywords (ylabel=..., title=..., etc)
# For violin/box plots of either coherence or inter-object similarity distributions across checkpoints
def _plot_similarity_distribution(sim_dict, out_dir, *, ylabel, title, fname, savefig):
    labels_with_data = [l for l in sim_dict if len(sim_dict[l]) > 0]
    if not labels_with_data:
        print("  [skip] No similarity data to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = [sim_dict[l]["mean_cos_sim"].values for l in labels_with_data]
    positions = range(1, len(labels_with_data) + 1)
    parts = ax.violinplot(plot_data, positions=positions, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.boxplot(plot_data, positions=positions, widths=0.15, patch_artist=False,
        showfliers=False, zorder=3, medianprops=dict(color="red", linewidth=1.5))
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels_with_data, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if savefig:
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=SAVEFIG_DPI)
        print(f"  Saved {path}")
        
# for coherence vs num props per obj and inter-obj sim vs prop count per obj pair
def _plot_similarity_vs_count(sim_dict, out_dir, *, x_builder, xlabel, ylabel, title,
                              fname, legend_count_label, savefig, min_count_per_bin):
# min_count_per_bin is important to avoid plotting very noisy means with few objects/pairs
    labels_with_data = [l for l in sim_dict if len(sim_dict[l]) > 0]
    if not labels_with_data:
        print("  [skip] No similarity data to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    for i, label in enumerate(labels_with_data):
        df = sim_dict[label].copy()
        df["plot_count"] = x_builder(df)
        grouped = df.groupby("plot_count")["mean_cos_sim"]
        means = grouped.mean()
        stds = grouped.std()
        counts = grouped.count()
        valid = counts >= min_count_per_bin
        if not valid.any():
            continue
        ax.errorbar(
            means.index[valid],
            means.values[valid],
            yerr=stds.values[valid],
            fmt="o-",
            color=colors[i % len(colors)],
            capsize=3,
            label=f"{label} ({legend_count_label}={len(df):,})",
            alpha=0.8,
        )
        # also annotate number of objects/pairs at each count bin
        for nd, cnt in counts[valid].items():
            ax.annotate(
                f"n={cnt}",
                (nd, means.loc[nd]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6,
                color=colors[i % len(colors)],
                alpha=0.7,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if savefig:
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=SAVEFIG_DPI)
        print(f"  Saved {path}")

# =============================================================================
# Violin/box of coherence across checkpoints
# =============================================================================
def plot_coherence_comparison(coherence, out_dir, emb_key, savefig=False):
    return _plot_similarity_distribution(coherence, out_dir,
        ylabel="Intra-object mean cosine similarity (proposal-level)",
        title=(
            f"Intra-object embedding coherence ({_emb_key_tag(emb_key)}, pre-NMS proposals)\n"
            "higher = same-object proposals cluster more tightly"
        ),
        fname="coherence_violin_proposals.png",
        savefig=savefig
    )
# =============================================================================
# Violin/box of Inter-object similarity across checkpoints
# =============================================================================
def plot_inter_similarity_distribution(inter_sim, out_dir, emb_key, savefig=False):
    return _plot_similarity_distribution(inter_sim, out_dir,
        ylabel="Inter-object mean cosine similarity (object-pair level)",
        title=(
            f"Inter-object similarity distribution ({_emb_key_tag(emb_key)}, pre-NMS proposals)\n"
            "lower = different objects are more separated in embedding space"
        ),
        fname="inter_similarity_distribution.png",
        savefig=savefig,
    )

# =============================================================================
# Coherence vs number of proposals per object
# =============================================================================
def plot_coherence_vs_nproposals(coherence, out_dir, emb_key, savefig=False):
    """
    Does the SupCon advantage grow for objects with more proposals?
    (More proposals per object = more false-negative pairs under diagonal InfoNCE.)
    """
    return _plot_similarity_vs_count(
        coherence,
        out_dir,
        x_builder=lambda df: df["n_proposals"],
        xlabel="Number of proposals per GT object",
        ylabel="Mean intra-object cosine similarity",
        title=(
            f"Coherence vs proposal count per object ({_emb_key_tag(emb_key)})\n"
            "(more proposals = stronger false-negative pressure under diagonal InfoNCE)"
        ),
        fname="coherence_vs_nproposals.png",
        legend_count_label="n_obj",
        savefig=savefig,
        min_count_per_bin=2
    )
# =============================================================================
# Inter-object similarity vs proposal count per object pair
# =============================================================================
def plot_inter_similarity_vs_nproposals(inter_sim, out_dir, emb_key, savefig=False, count_mode="min"):
    """
    Does inter-object similarity change with the number of proposals 
    in the objects being compared?

    count_mode:
        "min" -> x = min(n_props_a, n_props_b)
        "sum" -> x = n_props_a + n_props_b
    """
    if count_mode == "min":
        x_builder = lambda df: np.minimum(df["n_props_a"], df["n_props_b"])
        xlabel = "Min proposals across object pair"
        fname = "inter_similarity_vs_nproposals_min.png"
    elif count_mode == "sum":
        x_builder = lambda df: df["n_props_a"] + df["n_props_b"]
        xlabel = "Total proposals across object pair"
        fname = "inter_similarity_vs_nproposals_sum.png"
    else:
        raise ValueError(f"Unsupported count_mode: {count_mode}. Use 'min' or 'sum'.")
    return _plot_similarity_vs_count(
        inter_sim,
        out_dir,
        x_builder=x_builder,
        xlabel=xlabel,
        ylabel="Mean inter-object cosine similarity",
        title=(
            f"Inter-object similarity vs proposal count ({_emb_key_tag(emb_key)})\n"
            "flat = different objects stay equally separated as proposal count grows"
        ),
        fname=fname,
        legend_count_label="n_pairs",
        savefig=savefig,
        min_count_per_bin=2
    )

# =============================================================================
# Coherence by class (galaxy vs star)
# =============================================================================
def plot_coherence_by_class(coherence, out_dir, emb_key, savefig=False):
    labels_with_data = [l for l in coherence if len(coherence[l]) > 0]
    if not labels_with_data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.cm.tab10.colors
    class_names = {0: "Galaxy", 1: "Star"}
    for cls_id, ax in zip([0, 1], axes):
        for i, label in enumerate(labels_with_data):
            df = coherence[label]
            sub = df[df["gt_class"] == cls_id]["mean_cos_sim"]
            if len(sub) > 0:
                ax.hist(sub, bins=50, alpha=0.5, color=colors[i % len(colors)],
                        label=f"{label} (n={len(sub):,}, mean={sub.mean():.3f})")
        ax.set_xlabel("Intra-object mean cosine similarity")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.set_title(f"{class_names[cls_id]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Intra-object coherence by class (proposal-level, {_emb_key_tag(emb_key)})", fontsize=13)
    fig.tight_layout()
    if savefig:
        fname = os.path.join(out_dir, "coherence_by_class.png")
        fig.savefig(fname, dpi=SAVEFIG_DPI)
        print(f"  Saved {fname}")

# =============================================================================
# Inter-object similarity by class pair
# =============================================================================
def plot_inter_similarity_by_classpair(inter_sim, out_dir, emb_key, savefig=False):
    labels_with_data = [l for l in inter_sim if len(inter_sim[l]) > 0]
    if not labels_with_data:
        print("  [skip] No inter-object similarity data to plot.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = plt.cm.tab10.colors
    classpair_defs = [
        ("Galaxy-Galaxy", 0, 0),
        ("Galaxy-Star", 0, 1),
        ("Star-Star", 1, 1),
    ]

    for ax, (pair_name, cls_lo, cls_hi) in zip(axes, classpair_defs):
        for i, label in enumerate(labels_with_data):
            df = inter_sim[label].copy()
            cls_a = np.minimum(df["gt_class_a"].values, df["gt_class_b"].values)
            cls_b = np.maximum(df["gt_class_a"].values, df["gt_class_b"].values)
            mask = (cls_a == cls_lo) & (cls_b == cls_hi)
            sub = df.loc[mask, "mean_cos_sim"]
            if len(sub) > 0:
                ax.hist(sub, bins=50, alpha=0.5, color=colors[i % len(colors)],
                        label=f"{label} (n={len(sub):,}, mean={sub.mean():.3f})")
        ax.set_xlabel("Inter-object mean cosine similarity")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.set_title(pair_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Inter-object similarity by class pair (object-pair level, {_emb_key_tag(emb_key)})", fontsize=13)
    fig.tight_layout()
    if savefig:
        fname = os.path.join(out_dir, "inter_similarity_by_classpair.png")
        fig.savefig(fname, dpi=SAVEFIG_DPI)
        print(f"  Saved {fname}")

# =============================================================================
# Intra vs inter similarity (grouped bar chart)
# =============================================================================
def plot_intra_vs_inter(coherence, inter_sim, out_dir, emb_key, savefig=False):
    """
    Grouped bar chart: for each checkpoint, show mean intra-object similarity
    next to mean inter-object similarity. The gap between the two bars is what
    the contrastive loss is trying to maximize
    """
    labels_with_data = [l for l in coherence if len(coherence[l]) > 0 and len(inter_sim[l]) > 0]
    if not labels_with_data:
        print("  [skip] No data for intra vs inter plot.")
        return
    intra_vals = [coherence[l]["mean_cos_sim"].mean() for l in labels_with_data]
    inter_vals = [inter_sim[l]["mean_cos_sim"].mean() for l in labels_with_data]
    x = np.arange(len(labels_with_data))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_intra = ax.bar(x - width / 2, intra_vals, width, label="Intra-object (same GT)",
                        color="steelblue", alpha=0.8, edgecolor="black")
    bars_inter = ax.bar(x + width / 2, inter_vals, width, label="Inter-object (diff GT)",
                        color="lightcoral", alpha=0.8, edgecolor="black")
    # annotate values
    for bar in bars_intra:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_inter:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_with_data, fontsize=10)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title(
        f"Intra-object vs inter-object embedding similarity ({_emb_key_tag(emb_key)})\n"
        "(larger gap = better object-level discrimination)"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if savefig:
        fname = os.path.join(out_dir, "intra_vs_inter.png")
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname}")

# =============================================================================
# Dimensionality reduction for visualization (UMAP, PaCMAP, t-SNE)
# =============================================================================
# Helper functions
def _reduce(z, method, random_state=42, n_components=2):
    """Reduce z (N, d) to (N, n_components) using the requested method"""
    # - n_components is num of dim in output
    if method == 'umap':
        print(f"  Running UMAP on {len(z)} embeddings...")
        # - n_neighbors is number of neighbors for manifold approx (larger vals --> more global views of manifold while smaller vals --> more local views) default is 15
        # - metric is used to compute distances in high-d space
        # - min_dist is eff min dist between embedded points (default 0.1)
        #       smaller values will result in a more clustered/clumped embedding where nearby points on manifold are drawn closer together
        #       while larger values will result on a more even dispersal of points. Shld be set relative to spread value
        # - spread is eff scale of embedded points (default 1.0)
        #       with min_dist, this determines how clustered/clumped the embedded points are
        # - init is how to initialize the low-d embedding before optimization (default is 'spectral', but 'pca' is faster and often better for preserving global structure (Kobak and Linderman 2021))
        return UMAP(n_components=n_components, n_neighbors=30, metric='cosine', init='pca',
                    random_state=random_state).fit_transform(z)
    elif method == 'pacmap':
        print(f"  Running PaCMAP on {len(z)} embeddings...")
        # - n_neighbors is num of neighbors considered in k-Nearest neighbor graph and setting None enables auto-selection (10 for n < 10000)
        #   and n > 10000, it will be 10 + 15 * (log10(n) - 4)
        # - MN_ratio is ratio of num of mid-near pairs to num of neighbors (n_MN = MN_ratio * n_neighbors) default is 0.5
        # - FP_ratio is ratio of num of further pairs to num of neighbors (n_FP = FP_ratio * n_neighbors) default is 2.0
        # - distance is metric used for high-d space when constructing the neighbor graph (default is 'euclidean', but 'angular' probably better for CLIP embeddings)
        #   angular is cosine in PaCMAP
        # - apply_pca=False to avoid PaCMAP applying PCA by default since it does so for dims > 100
        #   we trained these 128 dims to be meaningful so we want to give PaCMAP the full representation to work with
        return PaCMAP(n_components=n_components, n_neighbors=None,
                      distance='angular', apply_pca=False,
                      random_state=random_state).fit_transform(z)
    elif method == 'tsne':
        print(f"  Running t-SNE on {len(z)} embeddings...")
        # - perplexity (default: 30) is related to number of nearest neighbors that's used in other manifold learning algos
        #       larger datasets usually require a larger perplexity (5 to 50)
        # - metric (default: 'euclidean') is used to compute distances in high-d space
        # - init is how to initialize the low-d embedding before optimization (default is 'pca')
        # - n_jobs is num of parallel jobs to run for neighbors search (default is None -> 1 unless in joblib.parallel backend, 
        #       but -1 uses all available cores)
        return TSNE(n_components=n_components, perplexity=30, metric='cosine',
                    random_state=random_state, n_jobs=-1).fit_transform(z)
    else:
        raise ValueError(f"Unknown method: {method}")

def _scan_manifest_for_objids(emb_dict, target_set):
    """
    Walk chunk manifest metadata (labels only, no embeddings) and return flat
    indices and objids for all proposals belonging to target_set.

    Parameters
    ----------
    emb_dict : dict
        Must contain '_chunk_manifest' and '_chunk_emb_dir'.
    target_set : set[int]

    Returns
    -------
    idxs : np.ndarray, shape (M,)
    objids : np.ndarray, shape (M,)
    """
    manifest = emb_dict["_chunk_manifest"]
    emb_dir  = emb_dict["_chunk_emb_dir"]
    gathered_idxs, gathered_objids = [], []
    offset = 0
    for rank_manifest in manifest.get("rank_manifests", []):
        for chunk_info in rank_manifest.get("chunk_meta", []):
            meta = _load_chunk_labels_metadata(chunk_info, emb_dir, manifest)
            chunk_objids = np.asarray(meta["gt_objid"].tolist(), dtype=np.int64)
            local_idxs = np.asarray(
                [i for i, oid in enumerate(chunk_objids) if oid in target_set],
                dtype=np.int64,
            )
            if local_idxs.size > 0:
                gathered_idxs.append(local_idxs + offset)
                gathered_objids.append(chunk_objids[local_idxs])
            offset += chunk_objids.shape[0]
            del meta

    if not gathered_idxs:
        return None, None
    return np.concatenate(gathered_idxs), np.concatenate(gathered_objids)

def _select_objids_to_indices(emb_dict, target_set):
    """
    Return flat indices and objids for all proposals belonging to target_set,
    handling both in-memory and chunk-manifest-backed dicts

    Parameters
    ----------
    emb_dict : dict
    target_set : set[int]

    Returns
    -------
    idxs : np.ndarray, shape (M,) or None
    objids : np.ndarray, shape (M,) or None
    """
    if "gt_objid" in emb_dict:
        objid_to_idx = _build_objid_to_idx(emb_dict["gt_objid"])
        valid = [oid for oid in target_set if oid in objid_to_idx]
        if not valid:
            return None, None
        idxs = np.concatenate([
            np.asarray(objid_to_idx[oid], dtype=np.int64) for oid in valid
        ])
        return idxs, emb_dict["gt_objid"].numpy()[idxs]

    if "_chunk_manifest" in emb_dict:
        return _scan_manifest_for_objids(emb_dict, target_set)

    return None, None

def _select_top_objects(emb_dict, coherence_df, n_objects=REDUCER_N_OBJECTS,
                        min_proposals=MIN_PROPOSALS_PLOT):
    """
    Select the top n_objects with the most proposals (at least min_proposals)
    Returns idxs into embedding tensor and corresponding objids
    """
    candidates = (
        coherence_df[coherence_df["n_proposals"] >= min_proposals]
        .sort_values(["n_proposals", "objid"], ascending=[False, True])
        .head(n_objects)
    )
    if len(candidates) == 0:
        return None, None
    return _select_objids_to_indices(emb_dict, set(candidates["objid"].values))

def _select_background_objects(emb_dict, inter_df, foreground_objids, n_bg=50, seed=seed):
    candidate_bg = (
        set(inter_df["objid_a"].unique()) | set(inter_df["objid_b"].unique())
    ) - foreground_objids
    if not candidate_bg:
        return None, None

    bg_objids = list(candidate_bg)
    if len(bg_objids) > n_bg:
        rng = np.random.default_rng(seed)
        bg_objids = rng.choice(bg_objids, size=n_bg, replace=False).tolist()

    return _select_objids_to_indices(emb_dict, set(bg_objids))

def _gather_embeddings_for_indices(emb_dict, emb_key, indices):
    """Fetch selected embeddings from dense tensors or chunk manifests."""
    if emb_key in emb_dict:
        return emb_dict[emb_key][indices].float()

    if "_chunk_manifest" not in emb_dict or emb_key not in _available_rep_keys(emb_dict):
        raise KeyError(
            f"Missing embedding key '{emb_key}'. "
            f"Available keys: {_available_rep_keys(emb_dict)}"
        )
        
    manifest = emb_dict["_chunk_manifest"]
    emb_dir = emb_dict["_chunk_emb_dir"]
    idx_arr = np.asarray(indices, dtype=np.int64)
    if idx_arr.size == 0:
        raise ValueError("indices is empty — nothing to gather.")
    
    order = np.argsort(idx_arr)
    inv_order = np.argsort(order)
    idx_sorted = idx_arr[order]

    z_parts = []
    offset = 0
    cursor = 0
    n = len(idx_sorted)
    for chunk in _iter_manifest_chunks(manifest, emb_dir):
        _assert_chunk_has_key(chunk, emb_key)
        chunk_len = int(chunk["gt_objid"].shape[0])
        chunk_end = offset + chunk_len
        start = cursor
        while cursor < n and idx_sorted[cursor] < chunk_end:
            cursor += 1
        if cursor > start:
            local_idx_np = idx_sorted[start:cursor] - offset
            local_idx = torch.from_numpy(local_idx_np).long()
            z_parts.append(chunk[emb_key][local_idx].float().cpu())

        offset = chunk_end
        del chunk
        if cursor >= n:
            break

    if cursor != n:
        raise ValueError(f"Failed to gather requested indices: got {cursor:,}, expected {n:,}.")

    gathered_sorted = torch.cat(z_parts, dim=0)
    inv_order_t = torch.from_numpy(inv_order).long()
    return gathered_sorted[inv_order_t]

def _setup_reducer_figure(labels_with_data):
    """
    Create the figure/axes grid and shared legend elements for reducer plots.
    Returns fig, axes (1D array).
    """
    n_panels = len(labels_with_data)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(PANEL_WIDTH * n_panels, PANEL_HEIGHT),
        squeeze=False,
    )
    return fig, axes[0]

def _reduce_fg_with_optional_bg(fg_embs, bg_embs, method, n_components):
    """
    If bg_embs is provided, reduce fg+bg jointly so they share a coordinate
    space. Returns (fg_reduced, bg_reduced) where bg_reduced may be None.
    """
    if bg_embs is not None and len(bg_embs) > 0:
        combined = np.concatenate([bg_embs, fg_embs], axis=0)
        reduced  = _reduce(combined, method, n_components=n_components)
        return reduced[len(bg_embs):], reduced[:len(bg_embs)]
    return _reduce(fg_embs, method, n_components=n_components), None

def _get_fg_inputs(data, coherence, label, emb_key, n_objects, min_proposals):
    """
    Shared per-panel setup: select top objects, gather embeddings and classes.
    Returns (fg_idxs, fg_objids, fg_embs, fg_classes) or None if no data.
    """
    fg_idxs, fg_objids = _select_top_objects(
        data[label], coherence[label],
        n_objects=n_objects, min_proposals=min_proposals,
    )
    if fg_idxs is None:
        return None
    fg_embs    = _gather_embeddings_for_indices(data[label], emb_key, fg_idxs).numpy()
    fg_classes = (
        data[label]["gt_classes"].numpy()[fg_idxs] if "gt_classes" in data[label]
        else np.zeros(len(fg_idxs), dtype=np.int64)
    )
    return fg_objids, fg_embs, fg_classes

def _get_bg_embs(data, inter_sim, label, emb_key, fg_objids):
    """
    Fetch background proposal embeddings from inter_sim pairs, excluding
    foreground objects. Returns bg_embs np.ndarray or None.
    """
    if inter_sim is None or label not in inter_sim or len(inter_sim[label]) == 0:
        return None
    bg_idxs, _ = _select_background_objects(
        data[label], inter_sim[label],
        foreground_objids=set(np.unique(fg_objids)),
    )
    if bg_idxs is None:
        return None
    return _gather_embeddings_for_indices(data[label], emb_key, bg_idxs).numpy()

def plot_reducer_comparison(data, coherence, emb_key, out_dir,
                            inter_sim=None, savefig=False,
                            n_objects=REDUCER_N_OBJECTS,
                            min_proposals=MIN_PROPOSALS_PLOT,
                            method=REDUCER_METHOD, n_components=2):
    """
    Side-by-side dimensionality reduction plots (PaCMAP/UMAP/t-SNE), one panel
    per checkpoint. Each point is a pre-NMS proposal where the color encodes GT
    object identity so same-color points should cluster tightly if the
    contrastive loss is working.

    If inter_sim is provided, a background scatter of proposals from objects
    appearing in sampled inter-object pairs is drawn in light gray first,
    giving a visual baseline for cross-object separation. Background and
    foreground are reduced jointly so the coordinate systems are shared.

    Parameters
    ----------
    data : dict[str, emb_dict]
    coherence : dict[str, pd.DataFrame]
        Output of compute_coherence_all_runs.
    emb_key : str
    out_dir : str
    inter_sim : dict[str, pd.DataFrame] or None
    savefig : bool
    n_objects : int
    min_proposals : int
    method : str  — 'pacmap', 'umap', or 'tsne'
    n_components : int — number of dimensions to reduce to (default 2 for easy plotting)
    """
    labels_with_data = [
        l for l in data
        if l in coherence and len(coherence[l]) > 0
        and emb_key in _available_rep_keys(data[l])
    ]
    if not labels_with_data:
        print("  [skip] No data available for reducer plot.")
        return
    palette         = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    marker_by_class = {0: "o", 1: "^"}
    fig, axes       = _setup_reducer_figure(labels_with_data)

    for ax, label in zip(axes, labels_with_data):
        print(f"\n--- {label} [{emb_key}] reducer (by objid) ---")
        result = _get_fg_inputs(data, coherence, label, emb_key, n_objects, min_proposals)
        if result is None:
            ax.set_title(f"{label}\n[no data]")
            continue
        fg_objids, fg_embs, fg_classes = result
        bg_embs = _get_bg_embs(data, inter_sim, label, emb_key, fg_objids)
        fg_reduced, bg_reduced = _reduce_fg_with_optional_bg(fg_embs, bg_embs, method, n_components)
        if bg_reduced is not None:
            ax.scatter(bg_reduced[:, 0], bg_reduced[:, 1],
                       c="lightgray", s=8, alpha=0.3,
                       marker="o", linewidths=0, zorder=1)

        selected_objids = list(dict.fromkeys(fg_objids.tolist()))
        objid_to_color  = {oid: palette[i % len(palette)] for i, oid in enumerate(selected_objids)}
        for oid in selected_objids:
            mask  = fg_objids == oid
            cls   = fg_classes[mask][0]
            ax.scatter(
                fg_reduced[mask, 0], fg_reduced[mask, 1],
                c=[objid_to_color[oid]], s=18, alpha=0.75,
                marker=marker_by_class.get(cls, "o"),
                linewidths=0, zorder=2,
            )

        ax.set_title(f"{label}\n{len(selected_objids)} objects, {len(fg_embs):,} proposals", fontsize=10)
        ax.set_xlabel(f"{method.upper()} 1", fontsize=9)
        ax.set_ylabel(f"{method.upper()} 2", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Galaxy"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=8, label="Star"),
    ]
    if inter_sim is not None:
        legend_elements.append(Patch(facecolor="lightgray", alpha=0.5, label="Inter-obj background"))
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{method.upper()} of top-{n_objects} multi-proposal objects "
        f"({_emb_key_tag(emb_key)}, pre-NMS)\n"
        "Each color = one GT object  |  ▲ = star  |  ● = galaxy  |  "
        "tighter clusters -> better intra-object coherence",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    if savefig:
        fname = os.path.join(out_dir, f"{method}_by_objid.png")
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
        print(f"  Saved {fname}")

def plot_reducer_centrality(data, coherence, emb_key, out_dir,
                            savefig=False, n_objects=REDUCER_N_OBJECTS,
                            min_proposals=MIN_PROPOSALS_PLOT,
                            method=REDUCER_METHOD, n_components=2):
    """
    Side-by-side dimensionality reduction plots colored by per-proposal
    centrality: cosine similarity of each proposal embedding to its GT object's
    mean embedding (centroid).

    High centrality (green) = proposal is representative of its object cluster.
    Low centrality (red)    = outlier proposal, pushed away from its own group.

    Under diagonal InfoNCE, same-object proposals are treated as negatives,
    so we expect more red outliers. SupCon should keep proposals tighter around
    their centroid (more green), since it explicitly pulls same-object proposals
    together.

    Parameters
    ----------
    data : dict[str, emb_dict]
    coherence : dict[str, pd.DataFrame]
        Output of compute_coherence_all_runs.
    emb_key : str
    out_dir : str
    savefig : bool
    n_objects : int
    min_proposals : int
    method : str
    n_components : int
    """
    labels_with_data = [
        l for l in data
        if l in coherence and len(coherence[l]) > 0
        and emb_key in _available_rep_keys(data[l])
    ]
    if not labels_with_data:
        print("  [skip] No data available for centrality reducer plot.")
        return

    marker_by_class = {0: "o", 1: "^"}
    cmap            = plt.cm.RdYlGn   # red=low centrality, green=high
    fig, axes       = _setup_reducer_figure(labels_with_data)
    # Precompute centrality for each panel so all subplots share one color scale.
    panel_data = {}
    global_min = np.inf
    global_max = -np.inf
    for label in labels_with_data:
        result = _get_fg_inputs(data, coherence, label, emb_key, n_objects, min_proposals)
        if result is None:
            panel_data[label] = None
            continue
        fg_objids, fg_embs, fg_classes = result

        # --- per-proposal centrality: cos sim to object centroid ---
        # fg_embs are already L2-normalized (from projection head), so
        # centroid = mean and renormalize, then dot product = cosine sim
        centrality = np.zeros(len(fg_embs), dtype=np.float32)
        for oid in np.unique(fg_objids):
            mask = fg_objids == oid
            z = fg_embs[mask]                        # (k, D)
            centroid = z.mean(axis=0, keepdims=True)        # (1, D)
            centroid = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8)
            centrality[mask] = (z * centroid).sum(axis=1)   # (k,) cosine sims
        panel_data[label] = (fg_objids, fg_embs, fg_classes, centrality)
        global_min = min(global_min, float(np.min(centrality)))
        global_max = max(global_max, float(np.max(centrality)))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        print("  [skip] No valid centrality values available for centrality reducer plot.")
        return
    if global_max <= global_min:
        global_max = global_min + 1e-8
    shared_norm = plt.Normalize(vmin=global_min, vmax=global_max)
    for ax, label in zip(axes, labels_with_data):
        print(f"\n--- {label} [{emb_key}] reducer (centrality) ---")
        result = panel_data[label]
        if result is None:
            ax.set_title(f"{label}\n[no data]")
            continue
        fg_objids, fg_embs, fg_classes, centrality = result
        fg_reduced, _ = _reduce_fg_with_optional_bg(fg_embs, None, method, n_components)
        # Scatter by class marker, while sharing one color normalization
        for cls_id, marker in marker_by_class.items():
            mask = fg_classes == cls_id
            if not mask.any():
                continue
            ax.scatter(
                fg_reduced[mask, 0], fg_reduced[mask, 1],
                c=centrality[mask], cmap=cmap,
                norm=shared_norm,
                s=18, alpha=0.85,
                marker=marker, linewidths=0, zorder=2,
            )
        ax.set_title(
            f"{label}\n{len(np.unique(fg_objids))} objects, {len(fg_embs):,} proposals",
            fontsize=10,
        )
        ax.set_xlabel(f"{method.upper()} 1", fontsize=9)
        ax.set_ylabel(f"{method.upper()} 2", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    colorbar_mappable = plt.cm.ScalarMappable(norm=shared_norm, cmap=cmap)
    colorbar_mappable.set_array([])
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(
        colorbar_mappable,
        cax=cbar_ax,
        label="Centrality (cos sim to centroid)",
    )

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Galaxy"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=8, label="Star"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{method.upper()} colored by centrality ({_emb_key_tag(emb_key)}, pre-NMS)\n"
        "green high = representative proposal  |  red low = outlier pushed from cluster  |  "
        "▲ = star  |  ● = galaxy",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 0.88, 1])
    if savefig:
        fname = os.path.join(out_dir, f"{method}_by_centrality.png")
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
        print(f"  Saved {fname}")

# =============================================================================
# Cross-modal matched-pair alignment
# =============================================================================
def _alignment_from_tensors(z_q, z_k, gt_objids):
    """
    Compute diagonal and off-diagonal cosine similarities from two
    L2-normalized tensors of shape (N, D)

    Off-diagonal pairs are formed by shifting z_k by N//2 to avoid pairing
    proposals from the same image (adjacent proposals in the tensor tend to
    come from the same image).

    Parameters
    ----------
    z_q : torch.Tensor, shape (N, D)
    z_k : torch.Tensor, shape (N, D)
    gt_objids : torch.Tensor or array-like, shape (N,)

    Returns
    -------
    dict with keys:
        'diagonal'     : np.ndarray (N,) — cos_sim(z_q_i, z_k_i)
        'off_diagonal' : np.ndarray (N,) — cos_sim(z_q_i, z_k_{i + N//2 mod N})
        'gt_objid'     : np.ndarray (N,) — GT object ID per proposal
    """
    diagonal     = (z_q * z_k).sum(dim=1).numpy()
    shifted      = torch.roll(z_k, shifts=z_k.shape[0] // 2, dims=0)
    off_diagonal = (z_q * shifted).sum(dim=1).numpy()
    return {
        "diagonal":     diagonal,
        "off_diagonal": off_diagonal,
        "gt_objid":     gt_objids.numpy() if isinstance(gt_objids, torch.Tensor) else np.asarray(gt_objids),
    }

def _stream_both_keys(emb_dict, q_key, k_key):
    """Stream chunks once, returning concatenated (z_q, z_k, gt_objids) tensors."""
    manifest = emb_dict["_chunk_manifest"]
    emb_dir  = emb_dict["_chunk_emb_dir"]
    q_parts, k_parts, oid_parts = [], [], []
    for chunk in _iter_manifest_chunks(manifest, emb_dir):
        _assert_chunk_has_key(chunk, q_key)
        _assert_chunk_has_key(chunk, k_key)
        q_parts.append(chunk[q_key].float().cpu())
        k_parts.append(chunk[k_key].float().cpu())
        oid_parts.append(chunk["gt_objid"].cpu())
        del chunk
    return (
        torch.cat(q_parts,   dim=0),
        torch.cat(k_parts,   dim=0),
        torch.cat(oid_parts, dim=0),
    )

def compute_cross_modal_alignment(emb_dict, q_key, k_key):
    """
    Compute diagonal (matched) and off-diagonal (unmatched) cross-modal cosine
    similarities, plus gt_objid per proposal for downstream joins.

    Parameters
    ----------
    emb_dict : dict
        Must contain q_key and k_key, each (N, D) L2-normalized.
        May be chunk-manifest-backed; both keys must live in the same chunk
        files (guaranteed when extracted with EMB_MODE='both').
    q_key : str  — e.g. 'z_q' or 'pooled_q'
    k_key : str  — e.g. 'z_k' or 'pooled_k'

    Returns
    -------
    dict with keys:
        'diagonal'     : np.ndarray (N,)
        'off_diagonal' : np.ndarray (N,)
        'gt_objid'     : np.ndarray (N,)
    """
    available = _available_rep_keys(emb_dict)
    if q_key not in available or k_key not in available:
        raise KeyError(
            f"Missing keys: need '{q_key}' and '{k_key}'. "
            f"Available: {available}"
        )
    if "_chunk_manifest" in emb_dict and q_key.startswith("pooled_") and k_key.startswith("pooled_"):
        z_q, z_k, gt_objids = _stream_both_keys(emb_dict, q_key, k_key)
        return _alignment_from_tensors(z_q, z_k, gt_objids)
    return _alignment_from_tensors(
        emb_dict[q_key].float(),
        emb_dict[k_key].float(),
        emb_dict["gt_objid"],
    )

def compute_cross_modal_alignment_all_runs(data, q_key, k_key):
    """Compute cross-modal alignment per run for a (q_key, k_key) pair."""
    alignment = {}
    for label, d in data.items():
        print(f"\n--- {label} [{q_key} vs {k_key}] ---")
        if q_key not in _available_rep_keys(d) or k_key not in _available_rep_keys(d):
            print(f"  [SKIP] missing '{q_key}' or '{k_key}'")
            alignment[label] = None
            continue
        result = compute_cross_modal_alignment(d, q_key, k_key)
        alignment[label] = result
        diag = result["diagonal"]
        off  = result["off_diagonal"]
        print(f"  diagonal:     mean={diag.mean():.4f}, std={diag.std():.4f}")
        print(f"  off-diagonal: mean={off.mean():.4f},  std={off.std():.4f}")
        print(f"  margin (diag - off): mean={(diag - off).mean():.4f}")
    return alignment

# =============================================================================
# Matched vs unmatched alignment distribution
# =============================================================================
def plot_cross_modal_alignment(alignment, out_dir, q_key, k_key, savefig=False):
    """
    For each checkpoint, violin+box of diagonal (matched) vs off-diagonal
    (unmatched) cross-modal cosine similarities.

    The gap between the two distributions is the margin the contrastive loss
    has learned. Larger gap = stronger cross-modal alignment.

    Parameters
    ----------
    alignment : dict[str, dict]  — output of compute_cross_modal_alignment_all_runs
    out_dir : str
    q_key : str
    k_key : str
    savefig : bool
    """
    labels_with_data = [l for l in alignment if alignment[l] is not None]
    if not labels_with_data:
        print("  [skip] No cross-modal alignment data to plot.")
        return

    colors = {"diagonal": "steelblue", "off_diagonal": "lightcoral"}
    fig, axes = _setup_reducer_figure(labels_with_data)
    for ax, label in zip(axes, labels_with_data):
        result = alignment[label]
        diag   = result["diagonal"]
        off    = result["off_diagonal"]
        margin = (diag - off).mean()

        parts = ax.violinplot([diag, off], positions=[1, 2], showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], [colors["diagonal"], colors["off_diagonal"]]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.boxplot(
            [diag, off], positions=[1, 2],
            widths=0.12, patch_artist=False,
            showfliers=False, zorder=3,
            medianprops=dict(color="black", linewidth=1.5),
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Matched\n(diagonal)", "Unmatched\n(off-diag)"], fontsize=9)
        ax.set_ylabel("Cosine similarity", fontsize=9)
        ax.set_title(f"{label}\nN={len(diag):,}  |  margin={margin:+.4f}", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.2, axis="y")
        ax.tick_params(labelsize=7)

    legend_elements = [
        Patch(facecolor=colors["diagonal"], alpha=0.7, label=f"Matched ({q_key}_i, {k_key}_i)"),
        Patch(facecolor=colors["off_diagonal"], alpha=0.7, label=f"Unmatched ({q_key}_i, {k_key}_j)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Cross-modal alignment: {_emb_key_tag(q_key)} vs {_emb_key_tag(k_key)}\n"
        "larger gap = stronger learned cross-modal margin",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    if savefig:
        fname = os.path.join(
            out_dir,
            f"cross_modal_alignment.png"
        )
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
        print(f"  Saved {fname}")

# =============================================================================
# Cross-modal alignment vs proposal count per object
# =============================================================================
def plot_cross_modal_alignment_vs_nproposals(alignment, coherence, out_dir,
                                             q_key, k_key, savefig=False,
                                             min_count_per_bin=2):
    """
    Mean diagonal cross-modal cosine similarity binned by number of proposals
    per GT object, one line per checkpoint.

    Objects with more proposals face stronger false-negative pressure under
    diagonal InfoNCE, so alignment should degrade with proposal count under
    InfoNCE but remain flat under SupCon.

    Parameters
    ----------
    alignment : dict[str, dict]  — output of compute_cross_modal_alignment_all_runs
    coherence : dict[str, pd.DataFrame]  — for proposal counts per object
    out_dir : str
    q_key : str
    k_key : str
    savefig : bool
    min_count_per_bin : int
        Minimum objects per bin to plot (suppresses noisy tails).
    """
    labels_with_data = [
        l for l in alignment
        if alignment[l] is not None and l in coherence and len(coherence[l]) > 0
    ]
    if not labels_with_data:
        print("  [skip] No data for alignment vs nproposals plot.")
        return

    # build a per-object mean-alignment df matching the shape _plot_similarity_vs_count expects:
    # one row per object with columns 'mean_cos_sim' and 'n_proposals'
    sim_dict_for_plot = {}
    for label in labels_with_data:
        result    = alignment[label]
        df_prop   = pd.DataFrame({
            "objid":    result["gt_objid"],
            "mean_cos_sim": result["diagonal"],
        })
        obj_mean = df_prop.groupby("objid")["mean_cos_sim"].mean().reset_index()
        sim_dict_for_plot[label] = obj_mean.merge(
            coherence[label][["objid", "n_proposals"]], on="objid", how="inner"
        )

    return _plot_similarity_vs_count(
        sim_dict_for_plot,
        out_dir,
        x_builder=lambda df: df["n_proposals"],
        xlabel="Number of proposals per GT object",
        ylabel="Mean diagonal cross-modal cosine similarity",
        title=(
            f"Cross-modal alignment vs proposal count\n"
            f"({_emb_key_tag(q_key)} vs {_emb_key_tag(k_key)})  |  "
            "degrading slope -> false-negative pressure on alignment"
        ),
        fname=f"cross_modal_alignment_vs_nproposals.png",
        legend_count_label="n_obj",
        savefig=savefig,
        min_count_per_bin=min_count_per_bin,
    )

# =============================================================================
# Cross-modal off-diagonal structure (weak positives vs. true negatives)
# =============================================================================
def _compute_offdiag_structure_from_tensors(z_q, z_k, gt_objids, n_sample=10000):
    """
    Classify off-diagonal pairs (i, j), i != j as weak positives
    (gt_objid[i] == gt_objid[j]) or true negatives (gt_objid[i] != gt_objid[j])
    and return their cosine similarities alongside the diagonal.

    Sampling is necessary because the full N x N matrix is O(N^2).
    We sample n_sample off-diagonal pairs, stratified 50/50 between
    weak positives and true negatives where possible.

    Parameters
    ----------
    z_q : torch.Tensor (N, D)
    z_k : torch.Tensor (N, D)
    gt_objids : np.ndarray (N,)
    n_sample : int
        Total off-diagonal pairs to sample across both categories.

    Returns
    -------
    dict with keys:
        'diagonal'       : np.ndarray (N,)
        'weak_positive'  : np.ndarray
        'true_negative'  : np.ndarray
    """
    N = z_q.shape[0]
    rng = np.random.default_rng(seed)
    # diagonal
    diagonal = (z_q * z_k).sum(dim=1).numpy()
    # build objid -> row indices map for fast weak-positive lookup
    objid_to_idx = _build_objid_to_idx(torch.from_numpy(gt_objids))
    # objects that have more than one proposal — only these can form weak positives
    multi_objids = [oid for oid, idxs in objid_to_idx.items() if len(idxs) > 1]

    # --- sample weak positive pairs (i, j): same objid, i != j ---
    wp_sims = []
    if multi_objids:
        n_wp = n_sample // 2
        wp_pairs_i, wp_pairs_j = [], []
        # sample objects proportional to their pair count
        obj_pair_counts = np.array([
            len(objid_to_idx[oid]) * (len(objid_to_idx[oid]) - 1)
            for oid in multi_objids
        ], dtype=np.float64)
        obj_pair_counts /= obj_pair_counts.sum()
        while len(wp_pairs_i) < n_wp:
            oid = multi_objids[rng.choice(len(multi_objids), p=obj_pair_counts)]
            idxs = objid_to_idx[oid]
            i, j = rng.choice(len(idxs), size=2, replace=False)
            wp_pairs_i.append(idxs[i])
            wp_pairs_j.append(idxs[j])
        wp_i = torch.tensor(wp_pairs_i[:n_wp], dtype=torch.long)
        wp_j = torch.tensor(wp_pairs_j[:n_wp], dtype=torch.long)
        wp_sims = (z_q[wp_i] * z_k[wp_j]).sum(dim=1).numpy()
    else:
        print("  [WARN] No multi-proposal objects found — weak positive distribution will be empty.")
        wp_sims = np.array([], dtype=np.float32)

    # --- sample true negative pairs (i, j): different objid, i != j ---
    n_tn = n_sample // 2
    tn_pairs_i = rng.integers(0, N, size=n_tn * 2)
    tn_pairs_j = rng.integers(0, N, size=n_tn * 2)
    # filter to i != j and different objid
    valid = (tn_pairs_i != tn_pairs_j) & (gt_objids[tn_pairs_i] != gt_objids[tn_pairs_j])
    tn_i = torch.tensor(tn_pairs_i[valid][:n_tn], dtype=torch.long)
    tn_j = torch.tensor(tn_pairs_j[valid][:n_tn], dtype=torch.long)
    tn_sims = (z_q[tn_i] * z_k[tn_j]).sum(dim=1).numpy()

    return {
        "diagonal":      diagonal,
        "weak_positive": wp_sims,
        "true_negative": tn_sims,
    }

def compute_offdiag_structure(emb_dict, q_key, k_key, n_sample=10000):
    """
    Compute diagonal, weak-positive, and true-negative cross-modal cosine
    similarity distributions.

    Parameters
    ----------
    emb_dict : dict
        May be chunk-manifest-backed; q_key and k_key must be in the same chunks.
    q_key : str  — e.g. 'z_q' or 'pooled_q'
    k_key : str  — e.g. 'z_k' or 'pooled_k'
    n_sample : int
        Total off-diagonal pairs to sample (split 50/50 wp vs tn).

    Returns
    -------
    dict with keys:
        'diagonal'       : np.ndarray (N,)
        'weak_positive'  : np.ndarray (~n_sample/2,)
        'true_negative'  : np.ndarray (~n_sample/2,)
    """
    available = _available_rep_keys(emb_dict)
    if q_key not in available or k_key not in available:
        raise KeyError(
            f"Missing keys: need '{q_key}' and '{k_key}'. "
            f"Available: {available}"
        )
    if "_chunk_manifest" in emb_dict and q_key.startswith("pooled_") and k_key.startswith("pooled_"):
        z_q, z_k, gt_objids = _stream_both_keys(emb_dict, q_key, k_key)
        return _compute_offdiag_structure_from_tensors(z_q, z_k, gt_objids.numpy(), n_sample=n_sample)
    return _compute_offdiag_structure_from_tensors(
        emb_dict[q_key].float(),
        emb_dict[k_key].float(),
        emb_dict["gt_objid"].numpy(),
        n_sample=n_sample,
    )
def compute_offdiag_structure_all_runs(data, q_key, k_key, n_sample=10000):
    """Compute off-diagonal structure per run for a (q_key, k_key) pair."""
    offdiag = {}
    for label, d in data.items():
        print(f"\n--- {label} [{q_key} vs {k_key}] ---")
        if q_key not in _available_rep_keys(d) or k_key not in _available_rep_keys(d):
            print(f"  [SKIP] missing '{q_key}' or '{k_key}'")
            offdiag[label] = None
            continue
        result = compute_offdiag_structure(d, q_key, k_key, n_sample=n_sample)
        offdiag[label] = result
        print(f"  diagonal:      mean={result['diagonal'].mean():.4f},      std={result['diagonal'].std():.4f}")
        print(f"  weak_positive: mean={result['weak_positive'].mean():.4f},  std={result['weak_positive'].std():.4f}  n={len(result['weak_positive']):,}")
        print(f"  true_negative: mean={result['true_negative'].mean():.4f},  std={result['true_negative'].std():.4f}  n={len(result['true_negative']):,}")
    return offdiag


# =============================================================================
# off-diagonal structure
# =============================================================================
def plot_offdiag_structure(offdiag, out_dir, q_key, k_key, savefig=False):
    """
    For each checkpoint, violin+box of three distributions:
      - diagonal    (matched pairs)        — what InfoNCE maximizes
      - weak positive (same obj, diff prop) — what InfoNCE wrongly minimizes
      - true negative (diff obj)            — what InfoNCE correctly minimizes

    Under diagonal InfoNCE: weak_positive ~ true_negative (both pushed down).
    Under SupCon:           weak_positive should shift toward diagonal.

    Parameters
    ----------
    offdiag : dict[str, dict]  — output of compute_offdiag_structure_all_runs
    out_dir : str
    q_key : str
    k_key : str
    savefig : bool
    """
    labels_with_data = [l for l in offdiag if offdiag[l] is not None]
    if not labels_with_data:
        print("  [skip] No off-diagonal structure data to plot.")
        return

    colors = {
        "diagonal":      "steelblue",
        "weak_positive": "mediumpurple",
        "true_negative": "lightcoral",
    }
    positions   = [1, 2, 3]
    xtick_labels = ["Matched\n(diagonal)", "Weak positive\n(same obj)", "True negative\n(diff obj)"]
    fig, axes   = _setup_reducer_figure(labels_with_data)

    for ax, label in zip(axes, labels_with_data):
        result = offdiag[label]
        groups = [result["diagonal"], result["weak_positive"], result["true_negative"]]

        parts = ax.violinplot(groups, positions=positions, showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], [colors["diagonal"], colors["weak_positive"], colors["true_negative"]]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.boxplot(
            groups, positions=positions,
            widths=0.12, patch_artist=False,
            showfliers=False, zorder=3,
            medianprops=dict(color="black", linewidth=1.5),
        )
        # annotate means
        for pos, grp in zip(positions, groups):
            ax.text(pos, grp.mean(), f"{grp.mean():.3f}",
                    ha="center", va="bottom", fontsize=7, color="black")

        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_ylabel("Cosine similarity", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.2, axis="y")
        ax.tick_params(labelsize=7)

    legend_elements = [
        Patch(facecolor=colors["diagonal"],      alpha=0.7, label=f"Matched ({q_key}_i, {k_key}_i)"),
        Patch(facecolor=colors["weak_positive"], alpha=0.7, label="Weak positive (same obj, diff prop)"),
        Patch(facecolor=colors["true_negative"], alpha=0.7, label="True negative (diff obj)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Cross-modal off-diagonal structure: {_emb_key_tag(q_key)} vs {_emb_key_tag(k_key)}\n"
        "SupCon should lift weak positives toward diagonal, true negatives stay low",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    if savefig:
        fname = os.path.join(
            out_dir,
            f"offdiag_structure.png"
        )
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
        print(f"  Saved {fname}")

# =============================================================================
# Joint cross-modal reducer (modality alignment)
# =============================================================================
def plot_joint_modal_reducer(data, out_dir, key_pairs, n_subsample=5000,
                             method=REDUCER_METHOD, n_components=2,
                             savefig=False):
    """
    For each (q_key, k_key) pair, run a joint dimensionality reduction on a
    subsample of concatenated Rubin (q) and Roman (k) embeddings, colored by
    modality. One panel per checkpoint, one figure per key pair.

    Interleaved points = modalities are aligned in embedding space.
    Separated clusters = contrastive loss did not bridge the modality gap.

    Parameters
    ----------
    data : dict[str, emb_dict]
    out_dir : str
    key_pairs : sequence of (q_key, k_key) tuples
        Representation levels to visualize, e.g. ('z_q','z_k') and ('pooled_q','pooled_k').
    n_subsample : int
        Number of proposals to subsample from each modality per checkpoint.
        Both modalities use the same sampled indices so pairs are preserved.
    method : str — 'pacmap', 'umap', or 'tsne'
    n_components : int
    savefig : bool
    """
    colors = {"q": "steelblue", "k": "tomato"}
    for q_key, k_key in key_pairs:
        labels_with_data = [
            l for l in data
            if q_key in _available_rep_keys(data[l])
            and k_key in _available_rep_keys(data[l])
        ]
        if not labels_with_data:
            print(f"  [skip] No data for joint reducer ({q_key}, {k_key}).")
            continue
        fig, axes = _setup_reducer_figure(labels_with_data)
        for ax, label in zip(axes, labels_with_data):
            print(f"\n--- {label} [{q_key} + {k_key}] joint reducer ---")
            d = data[label]
            # --- load both modalities ---
            if "_chunk_manifest" in d:
                z_q, z_k, _ = _stream_both_keys(d, q_key, k_key)
            else:
                z_q = d[q_key].float()
                z_k = d[k_key].float()
            N = z_q.shape[0]
            n = min(n_subsample, N)
            # same indices for both modalities so matched pairs stay together
            rng  = np.random.default_rng(seed)
            idxs = rng.choice(N, size=n, replace=False)
            idxs_t = torch.from_numpy(idxs).long()
            z_q_sub = z_q[idxs_t].numpy()   # (n, D)
            z_k_sub = z_k[idxs_t].numpy()   # (n, D)
            # joint reduction: stack both modalities so they share a coordinate space
            combined = np.concatenate([z_q_sub, z_k_sub], axis=0)   # (2n, D)
            reduced  = _reduce(combined, method, n_components=n_components)
            q_reduced = reduced[:n]   # (n, 2)
            k_reduced = reduced[n:]   # (n, 2)
            ax.scatter(q_reduced[:, 0], q_reduced[:, 1],
                       c=colors["q"], s=8, alpha=0.4,
                       linewidths=0, zorder=1, label=q_key)
            ax.scatter(k_reduced[:, 0], k_reduced[:, 1],
                       c=colors["k"], s=8, alpha=0.4,
                       linewidths=0, zorder=2, label=k_key)
            ax.set_title(f"{label}\nn={n:,} per modality", fontsize=10)
            ax.set_xlabel(f"{method.upper()} 1", fontsize=9)
            ax.set_ylabel(f"{method.upper()} 2", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
        legend_elements = [
            Patch(facecolor=colors["q"], alpha=0.7, label=f"Rubin ({q_key})"),
            Patch(facecolor=colors["k"], alpha=0.7, label=f"Roman ({k_key})"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f"Joint cross-modal {method.upper()}: {_emb_key_tag(q_key)} + {_emb_key_tag(k_key)}\n"
            "interleaved = modalities aligned  |  separated = modality gap remains",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        if savefig:
            fname = os.path.join(
                out_dir,
                f"joint_modal_reducer.png"
            )
            fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
            print(f"  Saved {fname}")

# =============================================================================
# Stats tests
# =============================================================================
def run_stat_tests(coherence, emb_key=None):
    labels_with_data = [l for l in coherence if len(coherence[l]) > 0]
    if len(labels_with_data) < 2:
        print("  [skip] Need at least 2 checkpoints.")
        return
    print("\n" + "=" * 70)
    if emb_key is None:
        print("Stats tests: intra-object coherence (proposal-level)")
    else:
        print(f"Stats tests: intra-object coherence (proposal-level) [{emb_key}]")
    print("=" * 70)

    for i, l1 in enumerate(labels_with_data):
        for l2 in labels_with_data[i + 1:]:
            x = coherence[l1]["mean_cos_sim"].values
            y = coherence[l2]["mean_cos_sim"].values
            # mannwhitneyu is a non-parametric test for difference in distributions (median shift)
            # basically do two independent samples come from distributions w/ diff medians?
            # so the mw_stat we get is the U statistic, which counts how many times a value from x is greater than a value from y
            # so in our case, if x (e.g. supCon) has higher coherence than y (e.g. flatten), 
            # we expect more pairs where x_i > y_j, leading to a higher U statistic            
            # two-sided b/c we don't want to assume direction of difference a priori.
            # So null hypothesis is that the two groups have the same distribution, and alternative is that they are not equal for at least one point in distribution
            mw_stat, mw_p = stats.mannwhitneyu(x, y, alternative="two-sided")
            n1, n2 = len(x), len(y)
            # rank-biserial correlation from Mann-Whitney U (essentially the size of effect) ranges from -1 to 1 
            # It's related to the prob that a value from one group will be greater than a value from the other group
            # 0 means no diff b/w the two groups, 1 indicates all values in one group are greater than the other, -1 means the opposite
            # In this two-sided test, r_rb is always <= 0 due to scipy returning the smaller U statistic between U(x, y) and U(y, x)
            # so that means the larger |r_rb| is, the greater the separation between groups, but the sign doesn't indicate which group is higher
            # You'll have to look at the medians/means directly to determine which group is higher
            r_rb = 1 - (2 * mw_stat) / (n1 * n2)
            
            # ks_2samp is the two-sample Kolmogorov-Smirnov test for equality of distributions
            # It tests the null hypothesis that the two samples are drawn from the same continuous distribution
            # ks_stat is the maximum difference between the empirical cumulative distribution functions of x and y
            # A larger ks_stat means greater separation between the distributions
            # The p-value (ks_p) indicates how likely it is to observe such a difference under the null hypothesis
            # so p=0 means we can reject the null hypothesis that the two groups come from the same distribution, and conclude they are different           
            ks_stat, ks_p = stats.ks_2samp(x, y)

            print(f"\n  {l1} (n={n1:,}) vs {l2} (n={n2:,}):")
            print(f"    median: {np.median(x):.4f} vs {np.median(y):.4f}")
            print(f"    mean: {np.mean(x):.4f} vs {np.mean(y):.4f}")
            print(f"    Mann-Whitney U = {mw_stat:.0f}, p = {mw_p:.2e}, r_rb = {r_rb:.3f}")
            print(f"    KS stat = {ks_stat:.4f}, p = {ks_p:.2e}")


# =============================================================================
# Summary
# =============================================================================
def print_summary(coherence, inter_sim, emb_key=None):
    print("\n" + "=" * 95)
    if emb_key is None:
        print("Summary: proposal-level coherence")
    else:
        print(f"Summary: proposal-level coherence [{emb_key}]")
    print(
        f"{'Checkpoint':<25} {'N obj':>8} {'N prop':>9} "
        f"{'Mean':>8} {'Median':>8} {'Std':>8} "
        f"{'InterMean':>10} {'InterMed':>10} {'InterStd':>10}"
    )
    print("-" * 95)
    for label in coherence:
        df = coherence[label]
        inter_df = inter_sim[label]
        if len(df) == 0:
            print(f"{label:<25} {'---':>8}")
            continue
        n_obj = len(df)
        n_prop = df["n_proposals"].sum()
        intra_mean = df["mean_cos_sim"].mean()
        intra_med = df["mean_cos_sim"].median()
        intra_std = df["mean_cos_sim"].std()
        inter_mean = inter_df["mean_cos_sim"].mean()
        inter_med = inter_df["mean_cos_sim"].median()
        inter_std = inter_df["mean_cos_sim"].std()
        print(
            f"{label:<25} {n_obj:>8,} {n_prop:>9,} "
            f"{intra_mean:>8.4f} {intra_med:>8.4f} {intra_std:>8.4f} "
            f"{inter_mean:>10.4f} {inter_med:>10.4f} {inter_std:>10.4f}"
        )
    print("=" * 95)


def get_analysis_keys(mode, loaded_data):
    """Resolve which embedding keys to run based on mode and loaded runs."""
    if mode in {"z_q", "pooled_q", "z_k", "pooled_k"}:
        return [mode]
    keys = []
    for key in ["z_q", "pooled_q", "z_k", "pooled_k"]:
        if any(key in _available_rep_keys(d) for d in loaded_data.values()):
            keys.append(key)
    return keys


def run_full_analysis_for_key(data, emb_key, emb_out_dir):
    """Run all coherence analyses for a single embedding representation key."""
    print("\n" + "#" * 90)
    print(f"Running coherence analysis for embedding key: {emb_key}")
    print(f"Output directory: {emb_out_dir}")
    print("#" * 90)

    os.makedirs(emb_out_dir, exist_ok=True)
    coherence = compute_coherence_all_runs(data, emb_key=emb_key)
    inter_sim = compute_inter_similarity_all_runs(data, emb_key=emb_key)
    # pooled_mode = emb_key in {"pooled_q", "pooled_k"}
    # reducer_method = "umap" if pooled_mode else REDUCER_METHOD
    # reducer_n_objects = 25 if pooled_mode else REDUCER_N_OBJECTS
    # joint_n_subsample = 2000 if emb_key == "pooled_q" else 5000
    k_key = None
    if emb_key in {"z_q", "pooled_q"}:
        k_key = emb_key.replace("_q", "_k")
        alignment = compute_cross_modal_alignment_all_runs(data, emb_key, k_key)
        plot_cross_modal_alignment(alignment, emb_out_dir, emb_key, k_key, savefig=True)
        plot_cross_modal_alignment_vs_nproposals(alignment, coherence, emb_out_dir, emb_key, k_key, savefig=True)
        offdiag = compute_offdiag_structure_all_runs(data, emb_key, k_key)
        plot_offdiag_structure(offdiag, emb_out_dir, emb_key, k_key, savefig=True)
    
    plot_coherence_comparison(coherence, emb_out_dir, emb_key=emb_key, savefig=True)
    plot_inter_similarity_distribution(inter_sim, emb_out_dir, emb_key=emb_key, savefig=True)
    
    plot_coherence_vs_nproposals(coherence, emb_out_dir, emb_key=emb_key, savefig=True)
    plot_inter_similarity_vs_nproposals(inter_sim, emb_out_dir, emb_key=emb_key, savefig=True)
    
    plot_coherence_by_class(coherence, emb_out_dir, emb_key=emb_key, savefig=True)
    plot_inter_similarity_by_classpair(inter_sim, emb_out_dir, emb_key=emb_key, savefig=True)
    
    plot_intra_vs_inter(coherence, inter_sim, emb_out_dir, emb_key=emb_key, savefig=True)
    # dimension reduction plots
    plot_reducer_comparison(data, coherence, emb_key, emb_out_dir, inter_sim=inter_sim, savefig=True)
    plot_reducer_centrality(data, coherence, emb_key, emb_out_dir, savefig=True)
    # plot_reducer_comparison(data, coherence, emb_key, emb_out_dir, inter_sim=inter_sim, savefig=True, n_objects=reducer_n_objects, method=reducer_method)
    # plot_reducer_centrality(data, coherence, emb_key, emb_out_dir, savefig=True, n_objects=reducer_n_objects, method=reducer_method)
    if k_key is not None:
        plot_joint_modal_reducer(data, emb_out_dir, key_pairs=((emb_key, k_key),), savefig=True)
        # plot_joint_modal_reducer(data, emb_out_dir, key_pairs=((emb_key, k_key),), savefig=True, n_subsample=joint_n_subsample, method=reducer_method)
    print_summary(coherence, inter_sim, emb_key=emb_key)
    # run_stat_tests(coherence, emb_key=emb_key)

analysis_keys = get_analysis_keys(EMB_MODE, data)
if not analysis_keys:
    print("[ERROR] No embedding keys available for analysis.")
else:
    for emb_key in analysis_keys:
        key_out_dir = out_dir if EMB_MODE != "both" else os.path.join(out_dir, emb_key)
        run_full_analysis_for_key(data, emb_key=emb_key, emb_out_dir=key_out_dir)

# include z_q/z_k weak/strong positives 

# def _select_top_objects(emb_dict, coherence_df, n_objects=50, min_proposals=3):
#     # sort by number of proposals, take the top n
#     candidates = coherence_df[coherence_df["n_proposals"] >= min_proposals]
#     candidates = candidates.nlargest(n_objects, "n_proposals")
#     if len(candidates) == 0:
#         return None, None, None
#     selected_objids = set(candidates["objid"].values)

#     if "gt_objid" in emb_dict: # unchunked case: just filter the dense tensor
#         objids = emb_dict["gt_objid"].numpy()
#         mask = np.isin(objids, list(selected_objids))
#         idxs = np.where(mask)[0]
#         return idxs, objids[idxs], candidates
#     if "_chunk_manifest" not in emb_dict:
#         return None, None, candidates

#     manifest = emb_dict["_chunk_manifest"]
#     emb_dir = emb_dict["_chunk_emb_dir"]
#     gathered_idxs = []
#     gathered_objids = []
#     offset = 0

#     for rank_manifest in manifest.get("rank_manifests", []):
#         for chunk_info in rank_manifest.get("chunk_meta", []):
#             meta = _load_chunk_labels_metadata(chunk_info, emb_dir, manifest)
#             chunk_objids = np.asarray(meta["gt_objid"].tolist(), dtype=np.int64)
#             local_idxs = [i for i, oid in enumerate(chunk_objids) if oid in selected_objids]
#             if local_idxs:
#                 local_idxs_np = np.asarray(local_idxs, dtype=np.int64)
#                 gathered_idxs.append(local_idxs_np + offset)
#                 gathered_objids.append(chunk_objids[local_idxs_np])
#             offset += int(chunk_objids.shape[0])
#             del meta
#     if not gathered_idxs:
#         return None, None, candidates
#     return np.concatenate(gathered_idxs), np.concatenate(gathered_objids), candidates


# def plot_reducer_by_objid(data, coherence, out_dir, emb_key="z_q", n_objects=50, min_proposals=3, method='pacmap', n_components=2, savefig=False):
#     """
#     Side-by-side dimensional reduction plots (UMAP/TSNE/PaCMAP) for each checkpoint, 
#     showing proposals from the top multi-proposal objects (each obj gets a distinct color)
#     If SupCon fixes false negatives, same-color points should cluster together
#     more tightly than under InfoNCE
#     """
#     labels_with_data = [l for l in data if l in coherence and len(coherence[l]) > 0]
#     if not labels_with_data:
#         print(f"  [skip] No data for {method.upper()} objid plot.")
#         return
#     n_plots = len(labels_with_data)
#     fig, axes = plt.subplots(1, n_plots, figsize=(PANEL_WIDTH * n_plots, PANEL_HEIGHT))
#     if n_plots == 1:
#         axes = [axes]
 
#     # build a consistent colormap across all panels: assign colors to objids
#     # from the first checkpoint that has them, reuse across all
#     all_unique_objids = set()
#     for label in labels_with_data:
#         indices, objids, _ = _select_top_objects(data[label], coherence[label], n_objects, min_proposals)
#         if indices is not None:
#             all_unique_objids.update(objids.tolist())
    
#     # deterministic color assignment
#     sorted_objids = sorted(all_unique_objids)
#     cmap = plt.cm.tab20
#     objid_to_color = {oid: cmap(i % 20) for i, oid in enumerate(sorted_objids)}
 
#     for ax, label in zip(axes, labels_with_data):
#         indices, objids, top_df = _select_top_objects(
#             data[label], coherence[label], n_objects, min_proposals
#         )
#         if indices is None or len(indices) == 0:
#             ax.set_title(f"{label}\n(no multi-proposal objects)")
#             ax.axis("off")
#             continue
 
#         if emb_key not in _available_rep_keys(data[label]):
#             ax.set_title(f"{label}\n(missing {emb_key})")
#             ax.axis("off")
#             continue

#         z = _gather_embeddings_for_indices(data[label], emb_key, indices).numpy()
#         print(f"  {label}: running {method.upper()} on {len(z)} proposals from {len(set(objids))} objects...")
#         z_2d = _reduce(z, method=method, n_components=n_components)
#         colors = [objid_to_color.get(oid, (0.5, 0.5, 0.5, 1.0)) for oid in objids]
#         ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, s=8, alpha=0.7)
#         ax.set_title(f"{label}\n({len(set(objids))} objects, {len(z)} proposals)")
#         ax.set_xlabel(f"{method.upper()} 1")
#         ax.set_ylabel(f"{method.upper()} 2")
 
#     fig.suptitle(
#         f"{method.upper()} of top-{n_objects} multi-proposal objects colored by gt_objid ({_emb_key_tag(emb_key)})\n"
#         "Same color = same GT object; tight clusters = good intra-object coherence",
#         fontsize=12
#     )
#     fig.tight_layout()
#     if savefig:
#         fname = os.path.join(out_dir, f"{method}_by_objid.png")
#         fig.savefig(fname, dpi=150)
#         print(f"  Saved {fname}")
 

# # =============================================================================
# # Dimensionality reduction of multi-proposal objects colored by centrality
# # =============================================================================
# def plot_reducer_by_centrality(data, coherence, out_dir, emb_key="z_q", n_objects=50, min_proposals=3, method='pacmap', n_components=2, savefig=False):
#     """
#     For each checkpoint, plot proposals from the top multi-proposal objects using a dimensionality reduction method (UMAP, PaCMAP, t-SNE, etc.),
#     colored by each proposal's cosine similarity to its object's centroid embedding ("centrality").
#     High centrality = proposal is representative of the object; low centrality = outlier proposal.
#     Under InfoNCE, we'd expect more red outliers within each object cluster (proposals being pushed away from their own group), 
#     while SupCon should keep them green. So, essentially, we're asking are any proposals being ejected from their object's cluster?
#     """

#     labels_with_data = [l for l in data if l in coherence and len(coherence[l]) > 0]
#     if not labels_with_data:
#         print(f"  [skip] No data for {method.upper()} centrality plot.")
#         return

#     # collect all centrality values for global colorbar scaling
#     all_centrality = []
#     per_panel = []  # store tuples of (z_2d, centrality, unique_oids, label)
#     for label in labels_with_data:
#         indices, objids, _ = _select_top_objects(
#             data[label], coherence[label], n_objects, min_proposals
#         )
#         if indices is None or len(indices) == 0:
#             per_panel.append(None)
#             continue
#         if emb_key not in _available_rep_keys(data[label]):
#             per_panel.append(None)
#             continue

#         z_emb = _gather_embeddings_for_indices(data[label], emb_key, indices)
#         z_np = z_emb.numpy()
#         z_2d = _reduce(z_np, method=method, n_components=n_components)
#         unique_oids = np.unique(objids)
#         centrality = np.zeros(len(indices))
#         for oid in unique_oids:
#             obj_mask = objids == oid
#             obj_embeddings = z_emb[obj_mask]
#             centroid = F.normalize(obj_embeddings.mean(dim=0, keepdim=True), dim=1)
#             sims = (obj_embeddings @ centroid.T).squeeze().numpy()
#             centrality[obj_mask] = sims
#         all_centrality.append(centrality)
#         per_panel.append((z_2d, centrality, unique_oids, label))

#     if not any(p is not None for p in per_panel):
#         print(f"  [skip] No valid data for {method.upper()} centrality plot.")
#         return

#     all_centrality_flat = np.concatenate([c for c in all_centrality if c is not None])
#     vmin = np.percentile(all_centrality_flat, 5)
#     vmax = np.percentile(all_centrality_flat, 95)

#     n_plots = len(labels_with_data)
#     fig, axes = plt.subplots(1, n_plots, figsize=(PANEL_WIDTH * n_plots, PANEL_HEIGHT))
#     if n_plots == 1:
#         axes = [axes]

#     for ax, panel in zip(axes, per_panel):
#         if panel is None:
#             ax.set_title("(no data)")
#             ax.axis("off")
#             continue
#         z_2d, centrality, unique_oids, label = panel
#         ax.scatter(z_2d[:, 0], z_2d[:, 1], c=centrality, cmap="RdYlGn",
#                         s=8, alpha=0.7, vmin=vmin, vmax=vmax)
#         ax.set_title(f"{label}\n({len(unique_oids)} objects, {len(z_2d)} proposals)")
#         ax.set_xlabel(f"{method.upper()} 1")
#         ax.set_ylabel(f"{method.upper()} 2")

#     # Add a single colorbar for all panels
#     fig.subplots_adjust(right=0.85)
#     cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
#     sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=vmin, vmax=vmax))
#     sm.set_array([])
#     fig.colorbar(sm, cax=cbar_ax, label="Cosine sim to object centroid")

#     fig.suptitle(
#         f"{method.upper()} of top-{n_objects} multi-proposal objects colored by centroid similarity ({_emb_key_tag(emb_key)})\n"
#         "Green = close to object centroid; Red = outlier proposal",
#         fontsize=12
#     )
#     fig.tight_layout(rect=[0, 0, 0.85, 1])

#     if savefig:
#         fname = os.path.join(out_dir, f"{method}_by_centrality.png")
#         fig.savefig(fname, dpi=150)
#         print(f"  Saved {fname}")
