"""
Star/galaxy classification accuracy from class parquets

For each matched (detection, truth) pair, compares the predicted morphological class
(star vs galaxy) to the truth class. Produces:

    1. Aggregate confusion matrices  — rows=truth class, cols=predicted class,
       one panel per (run, combo).  Shows raw counts + fraction.

    2. Accuracy-vs-magnitude curves  — galaxy accuracy and star accuracy binned
       by truth mag_i, one curve per (run, combo), on the same axes for comparison.

Class encoding in the parquets
-------------------------------
    DD det      : class  0=star,  1=galaxy
    LSST det    : class  0=star,  1=galaxy   (extendedness)
    LSST truth  : class  1=galaxy, 2=star    (truth_type)

Normalised internally to:  0=star, 1=galaxy  for both pred and truth
Only matched detections (prefix_is_matched == True) are used
The truth magnitude is used for binning (via matched_id lookup into truth rows)

Usage
-----
    python classify_plot.py \\
        --root-run-dir ~/lsst_runs \\
        --run-names "og_50ep flatten_15ep flatten_50ep supCon_15ep" \\
        --score-thresholds "0.5" --nms-thresholds "0.6" \\
        --linking-lengths "1.0 2.0" --mag-limit gold --buffers "1" \\
        --output-dir ~/lsst_runs/classifications

    # Cartesian product of thresholds:
    python classify_plot.py ... --combo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from metrics import split_truth_det, get_matched_truth_mag


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAG_LIMITS: Dict[str, float] = {
    "power_law": 26.22,
    "gold":      25.3,
    "nominal":   26.42,
}

CLASS_LABELS = ["Star", "Galaxy"]          # normalised index: 0=star, 1=galaxy

GALAXY_COLOR = "#2980b9"
STAR_COLOR   = "#e67e22"
OVERALL_COLOR = "#27ae60"

DEFAULT_LEGEND_NAME_MAP: Dict[str, str] = {
    "lsst5": "DeepDISC-LSST",
    "comb": "DeepDISC-Combined",
    "distill": "DeepDISC-KD",
}


def canonical_run_token(run_name: str) -> str:
    """Return a compact run token used for legend aliases and folder tags."""
    lower = run_name.lower()
    if lower.startswith("lsst5"):
        return "lsst5"
    if lower.startswith("comb"):
        return "comb"
    if lower.startswith("distill"):
        return "distill"
    return run_name.split("_")[0]


def parse_legend_name_map(raw_items: List[str]) -> Dict[str, str]:
    """Parse key=value entries into a legend display-name map."""
    out: Dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --legend-name-map entry '{item}'. Use key=value format."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid --legend-name-map entry '{item}'. Use key=value format."
            )
        out[key] = value
    return out


def get_display_name_for_run(run_name: str, legend_map: Dict[str, str]) -> str:
    """Return a user-facing display name for a run folder."""
    if run_name in legend_map:
        return legend_map[run_name]

    token = canonical_run_token(run_name)
    if token in legend_map:
        return legend_map[token]

    prefix = run_name.split("_")[0]
    if prefix in legend_map:
        return legend_map[prefix]

    return prefix


def build_comparison_tag(
    run_names: List[str],
    prefix: str,
    include_lsst_baseline: bool,
) -> str:
    """Build comparison-style subfolder tag, e.g. lsst5_comb_distill."""
    tokens: List[str] = []
    seen: set[str] = set()
    for run_name in run_names:
        token = canonical_run_token(run_name)
        if token not in seen:
            seen.add(token)
            tokens.append(token)

    tag = "_".join(tokens) if tokens else "runs"
    if prefix == "dd" and not include_lsst_baseline:
        tag = f"{tag}_no_lsst_pipe"
    return tag


def legend_name_from_panel_label(panel_label: str) -> str:
    """Extract compact legend label from multi-line panel label."""
    return panel_label.split("\n", 1)[0].strip()


def set_unique_legend(ax: plt.Axes, fontsize: int = 7, loc: str = "lower left") -> None:
    """Set legend with duplicate labels removed while preserving plot order."""
    handles, labels = ax.get_legend_handles_labels()
    unique_handles: List[object] = []
    unique_labels: List[str] = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, fontsize=fontsize, loc=loc)


# ---------------------------------------------------------------------------
# Class normalisation
# ---------------------------------------------------------------------------

def normalise_det_class(class_series: pd.Series, prefix: str) -> pd.Series:
    """
    Normalise detection class to 0=star, 1=galaxy.

    DD det   (prefix='dd')  : 0=galaxy, 1=star
    LSST det (prefix='lsst'): 0=star,   1=galaxy

    Both are mapped explicitly to DeepDISC convention:
        0=star, 1=galaxy

    Returns int Series; rows with unknown values set to -1.
    """
    out = class_series.copy()
    if prefix == "dd":
        # DD stores 0=galaxy, 1=star.
        mapped = out.map({0: 1, 1: 0})
    elif prefix == "lsst":
        # LSST det already uses 0=star, 1=galaxy.
        mapped = out.map({0: 0, 1: 1})
    else:
        mapped = pd.Series(-1, index=out.index)
    return mapped.fillna(-1).astype(int)


def normalise_truth_class(class_series: pd.Series) -> pd.Series:
    """
    Truth uses truth_type:  1=galaxy, 2=star.
    Map to:                  1->1 (galaxy), 2->0 (star), else -1.
    """
    out = class_series.map({1: 1, 2: 0}).fillna(-1).astype(int)
    return out


# ---------------------------------------------------------------------------
# Data loading (mirrors plot_fof_binned.py)
# ---------------------------------------------------------------------------

def build_combo_labels(
    score_thresholds: List[str],
    nms_thresholds: List[str],
    cartesian: bool,
) -> List[str]:
    if cartesian:
        return [f"s{s}_n{n}" for s in score_thresholds for n in nms_thresholds]
    if len(score_thresholds) != len(nms_thresholds):
        raise ValueError(
            "score and NMS threshold lists must have equal length unless --combo is set"
        )
    return [f"s{s}_n{n}" for s, n in zip(score_thresholds, nms_thresholds)]


def load_parquet(
    run_dir: Path,
    prefix: str,
    linking_length: str,
    combo: str,
    mag_limit_val: float,
) -> Optional[pd.DataFrame]:
    cats_dir = run_dir / "analysis_cats"
    if prefix == "dd":
        path = cats_dir / "dd" / linking_length / combo / f"class_{mag_limit_val:.2f}.parquet"
    else:
        path = cats_dir / "lsst" / linking_length / f"class_{mag_limit_val:.2f}.parquet"
    if not path.exists():
        print(f"  [WARN] not found: {path}")
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Classification extraction
# ---------------------------------------------------------------------------

def get_matched_class_pairs(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Return DataFrame of matched (det, truth) pairs with columns:
        mag_i      — truth magnitude (for binning)
        pred       — normalised predicted class (0=star, 1=galaxy)
        truth      — normalised truth class      (0=star, 1=galaxy)
        correct    — bool

    Only rows where both pred and truth are in {0,1} are returned.
    """
    truths, dets = split_truth_det(df, prefix)

    # Matched detections only
    matched_mask = dets[f"{prefix}_is_matched"].astype(bool)
    matched_dets = dets[matched_mask].copy()

    if matched_dets.empty:
        return pd.DataFrame(columns=["mag_i", "pred", "truth", "correct"])

    # Normalise predicted class to 0=star, 1=galaxy
    pred = normalise_det_class(matched_dets["class"], prefix)

    # Truth class via matched_id -> truths lookup
    matched_id_col = f"{prefix}_matched_id"
    truth_lookup = truths.set_index("row_index")

    truth_class_raw = matched_dets[matched_id_col].map(
        truth_lookup["class"]
    )
    truth_norm = normalise_truth_class(truth_class_raw)

    # Truth magnitude for binning via shared helper
    truth_mag = get_matched_truth_mag(matched_dets, truths, prefix)

    result = pd.DataFrame({
        "mag_i":   truth_mag.values,
        "pred":    pred.values,
        "truth":   truth_norm.values,
    }, index=matched_dets.index)

    # Drop rows with unknown class
    result = result[(result["pred"] >= 0) & (result["truth"] >= 0)].copy()
    result["correct"] = result["pred"] == result["truth"]
    return result


# ---------------------------------------------------------------------------
# Aggregate confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion(pairs: pd.DataFrame) -> np.ndarray:
    """
    Returns 2x2 confusion matrix C where:
        C[true_class, pred_class]  (0=star, 1=galaxy)
    """
    C = np.zeros((2, 2), dtype=int)
    for t in [0, 1]:
        for p in [0, 1]:
            C[t, p] = ((pairs["truth"] == t) & (pairs["pred"] == p)).sum()
    return C


def plot_confusion_matrices(
    run_combos: List[Tuple[str, str]],
    pairs_dict: Dict[Tuple[str, str], pd.DataFrame],
    col_labels: List[str],
    title: str,
    output_path: Optional[Path],
) -> plt.Figure:
    """One panel per (run, combo) showing star/galaxy confusion matrix in balanced grid."""
    n = len(run_combos)
    # Square-ish layout: use sqrt to balance rows and columns (2x2 for 4 items, not 3+1)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    figsize_width = 3.8 * ncols + 1.2
    figsize_height = 3.8 * nrows + 0.8
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_width, figsize_height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.flatten()
    im = None

    for j, key in enumerate(run_combos):
        ax = axes_flat[j]
        pairs = pairs_dict.get(key)
        label = legend_name_from_panel_label(col_labels[j])

        if pairs is None or pairs.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(label, fontsize=9)
            ax.axis("off")
            continue

        C = compute_confusion(pairs)
        total = C.sum()
        n_gal   = C[1, :].sum()
        n_star  = C[0, :].sum()

        # Normalise per true class (row-wise)
        C_frac = np.zeros_like(C, dtype=float)
        for i in [0, 1]:
            row_sum = C[i].sum()
            if row_sum > 0:
                C_frac[i] = C[i] / row_sum

        im = ax.imshow(C_frac, cmap="Blues", vmin=0, vmax=1, aspect="equal")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(CLASS_LABELS, fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(CLASS_LABELS, fontsize=9, rotation=90, va="center")
        row_idx = j // ncols
        col_idx = j % ncols
        if row_idx == nrows - 1:
            ax.set_xlabel("Predicted", fontsize=9)
        else:
            ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("True", fontsize=9)
        else:
            ax.set_ylabel("")

        # Cell annotations
        mid_val = 0.5
        for ti in [0, 1]:
            for pi in [0, 1]:
                frac = C_frac[ti, pi]
                cnt  = C[ti, pi]
                txt_color = "white" if frac > mid_val else "black"
                ax.text(pi, ti, f"{frac:.3f}\n({cnt:,})",
                        ha="center", va="center",
                        fontsize=8, color=txt_color, linespacing=1.4)

        # Overall accuracy in title
        acc = C.diagonal().sum() / total if total > 0 else 0
        ax.set_title(f"{label}\nacc={acc:.3f}  N={total:,}", fontsize=8)

    # blank out unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    
    fig.suptitle(title, fontsize=11, y=1.01)
    
    # Add colorbar as separate axes outside grid (cleaner layout)
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Fraction (row-norm)")
    
    fig.tight_layout(rect=[0, 0, 0.9, 0.98])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Accuracy-vs-magnitude curves
# ---------------------------------------------------------------------------

def binned_accuracy(
    pairs: pd.DataFrame,
    mag_bins: np.ndarray,
    class_val: Optional[int] = None,
    min_count: int = 5,
    bootstrap_ci: bool = True,
    bootstrap_samples: int = 500,
    bootstrap_alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute accuracy per mag bin.
    class_val: 0=star, 1=galaxy, None=overall.
    Returns (bin_centers, accuracy, counts, ci_lo, ci_hi).
    """
    if class_val is not None:
        subset = pairs[pairs["truth"] == class_val]
    else:
        subset = pairs

    centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    accs    = np.full(len(centers), np.nan)
    counts  = np.zeros(len(centers), dtype=int)
    ci_lo   = np.full(len(centers), np.nan)
    ci_hi   = np.full(len(centers), np.nan)

    if bootstrap_ci and rng is None:
        rng = np.random.default_rng()

    for i, (lo, hi) in enumerate(zip(mag_bins[:-1], mag_bins[1:])):
        mask = (subset["mag_i"] >= lo) & (subset["mag_i"] < hi)
        n = mask.sum()
        counts[i] = n
        if n < min_count:
            continue

        bin_correct = subset.loc[mask, "correct"].astype(float).to_numpy()
        accs[i] = float(bin_correct.mean())

        if bootstrap_ci and n > 0 and bootstrap_samples > 0:
            samples = rng.choice(
                bin_correct,
                size=(bootstrap_samples, n),
                replace=True,
            )
            means = samples.mean(axis=1)
            ci_lo[i] = float(np.quantile(means, bootstrap_alpha / 2.0))
            ci_hi[i] = float(np.quantile(means, 1.0 - bootstrap_alpha / 2.0))

    return centers, accs, counts, ci_lo, ci_hi


def plot_accuracy_curves(
    run_combos: List[Tuple[str, str]],
    pairs_dict: Dict[Tuple[str, str], pd.DataFrame],
    col_labels: List[str],
    mag_bins: np.ndarray,
    title: str,
    output_path: Optional[Path],
    min_count: int = 5,
    bootstrap_ci: bool = False,
    bootstrap_samples: int = 500,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: Optional[int] = None,
) -> plt.Figure:
    """
    Three-panel figure: overall / galaxy / star accuracy vs mag_i.
    Each (run, combo) gets its own line on all three panels.
    LSST baseline (key[0] == '__lsst_global__') plotted with dashed line behind all DD runs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    panel_specs = [
        ("Overall",  None,  OVERALL_COLOR),
        ("Galaxy",   1,     GALAXY_COLOR),
        ("Star",     0,     STAR_COLOR),
    ]

    # Use tab20 for distinct colors across many runs
    cmap = plt.cm.tab20
    n_series = len(run_combos)
    rng = np.random.default_rng(bootstrap_seed)

    for ax_idx, (panel_title, class_val, default_color) in enumerate(panel_specs):
        ax = axes[ax_idx]
        # Plot LSST baseline first (so it's behind DD lines)
        for j, key in enumerate(run_combos):
            pairs = pairs_dict.get(key)
            label = legend_name_from_panel_label(col_labels[j])
            
            # LSST baseline: dashed line, thinner, plot behind
            is_lsst = key[0] == "__lsst_global__"
            if not is_lsst:
                continue
            
            if pairs is None or pairs.empty:
                continue
            
            centers, accs, counts, ci_lo, ci_hi = binned_accuracy(
                pairs,
                mag_bins,
                class_val,
                min_count=min_count,
                bootstrap_ci=bootstrap_ci,
                bootstrap_samples=bootstrap_samples,
                bootstrap_alpha=bootstrap_alpha,
                rng=rng,
            )
            valid = np.isfinite(accs)
            if not valid.any():
                continue

            if bootstrap_ci:
                valid_ci = valid & np.isfinite(ci_lo) & np.isfinite(ci_hi)
                if valid_ci.any():
                    ax.fill_between(
                        centers[valid_ci],
                        ci_lo[valid_ci],
                        ci_hi[valid_ci],
                        color="black",
                        alpha=0.10,
                        linewidth=0,
                        zorder=0,
                    )
            
            ax.plot(centers[valid], accs[valid],
                    marker="o", markersize=3,
                    label=label, color="black", linewidth=1.8, linestyle="--",
                    zorder=1, alpha=0.7)
        
        # Now plot DD runs on top
        for j, key in enumerate(run_combos):
            pairs = pairs_dict.get(key)
            label = legend_name_from_panel_label(col_labels[j])
            
            is_lsst = key[0] == "__lsst_global__"
            if is_lsst:
                continue
            
            # Use tab20 for distinct colors
            color = cmap(j / max(n_series, 1))

            if pairs is None or pairs.empty:
                continue

            centers, accs, counts, ci_lo, ci_hi = binned_accuracy(
                pairs,
                mag_bins,
                class_val,
                min_count=min_count,
                bootstrap_ci=bootstrap_ci,
                bootstrap_samples=bootstrap_samples,
                bootstrap_alpha=bootstrap_alpha,
                rng=rng,
            )
            valid = np.isfinite(accs)
            if not valid.any():
                continue

            if bootstrap_ci:
                valid_ci = valid & np.isfinite(ci_lo) & np.isfinite(ci_hi)
                if valid_ci.any():
                    ax.fill_between(
                        centers[valid_ci],
                        ci_lo[valid_ci],
                        ci_hi[valid_ci],
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                        zorder=1,
                    )

            ax.plot(centers[valid], accs[valid],
                    marker="o", markersize=3.5,
                    label=label, color=color, linewidth=1.5,
                    zorder=2)

        ax.set_xlabel(r"mag_i (truth)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(panel_title, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.25)
        set_unique_legend(ax, fontsize=7, loc="lower left")

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Combined summary: confusion + curves on one figure
# ---------------------------------------------------------------------------

def plot_summary(
    run_combos: List[Tuple[str, str]],
    pairs_dict: Dict[Tuple[str, str], pd.DataFrame],
    col_labels: List[str],
    mag_bins: np.ndarray,
    title: str,
    output_path: Optional[Path],
    min_count: int = 5,
    bootstrap_ci: bool = False,
    bootstrap_samples: int = 500,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: Optional[int] = None,
) -> plt.Figure:
    """
    Top rows: confusion matrices (one per run/combo) in balanced grid.
    Bottom row: accuracy-vs-mag curves (overall / galaxy / star).
    """
    n_runs = len(run_combos)
    # Square-ish layout for confusion matrices (2x2 for 4 items, not 3+1)
    ncols_cm = int(np.ceil(np.sqrt(n_runs)))
    nrows_cm = int(np.ceil(n_runs / ncols_cm))
    # Bottom row has 3 accuracy panels, so total grid needs at least 3 columns.
    ncols_total = max(ncols_cm, 3)
    figsize_width = max(3.8 * ncols_total, 12)
    figsize_height = 3.8 * nrows_cm + 5.5
    fig = plt.figure(figsize=(figsize_width, figsize_height))
    gs  = gridspec.GridSpec(
        nrows_cm + 1, ncols_total, figure=fig,
        height_ratios=[1] * nrows_cm + [1.2],
        hspace=0.40, wspace=0.30,
    )

    # ---- Top rows: confusion matrices in 3xN grid ----
    cmap_cm = plt.cm.Blues
    last_im = None
    ref_cm_ax = None
    for j, key in enumerate(run_combos):
        row_idx = j // ncols_cm
        col_idx = j % ncols_cm
        if ref_cm_ax is None:
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ref_cm_ax = ax
        else:
            ax = fig.add_subplot(gs[row_idx, col_idx], sharex=ref_cm_ax, sharey=ref_cm_ax)
        pairs = pairs_dict.get(key)
        label = legend_name_from_panel_label(col_labels[j])

        if pairs is None or pairs.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.axis("off")
            continue

        C = compute_confusion(pairs)
        C_frac = np.zeros_like(C, dtype=float)
        for i in [0, 1]:
            rs = C[i].sum()
            if rs > 0:
                C_frac[i] = C[i] / rs

        im = ax.imshow(C_frac, cmap=cmap_cm, vmin=0, vmax=1, aspect="equal")
        last_im = im

        ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_LABELS, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_LABELS, fontsize=8, rotation=90, va="center")
        if row_idx == nrows_cm - 1:
            ax.set_xlabel("Predicted", fontsize=8)
        else:
            ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("True", fontsize=8)
        else:
            ax.set_ylabel("")

        for ti in [0, 1]:
            for pi in [0, 1]:
                frac = C_frac[ti, pi]
                txt_color = "white" if frac > 0.5 else "black"
                ax.text(pi, ti, f"{frac:.3f}\n({C[ti,pi]:,})",
                        ha="center", va="center",
                        fontsize=7, color=txt_color, linespacing=1.4)

        acc = C.diagonal().sum() / max(C.sum(), 1)
        ax.set_title(f"{label}\nacc={acc:.3f}", fontsize=8, pad=4)

    # blank out unused confusion matrix axes
    for j in range(n_runs, nrows_cm * ncols_cm):
        row_idx = j // ncols_cm
        col_idx = j % ncols_cm
        fig.add_subplot(gs[row_idx, col_idx]).axis("off")
    
    # Add colorbar for confusion matrices as separate axes (outside grid)
    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, nrows_cm / (nrows_cm + 1) + 0.01, 0.015, 0.25])
        fig.colorbar(last_im, cax=cbar_ax, label="Fraction (row-norm)")

    # ---- Bottom row: accuracy curves ----
    panel_defs = [
        ("Overall accuracy",  None, "tab:green"),
        ("Galaxy accuracy",   1,    GALAXY_COLOR),
        ("Star accuracy",     0,    STAR_COLOR),
    ]
    color_cycle = plt.cm.tab20
    rng = np.random.default_rng(bootstrap_seed)

    for col_idx, (panel_title, class_val, _) in enumerate(panel_defs):
        ax = fig.add_subplot(gs[nrows_cm, col_idx])
        
        # Plot LSST baseline first (so it's behind DD lines)
        for j, key in enumerate(run_combos):
            pairs = pairs_dict.get(key)
            label = legend_name_from_panel_label(col_labels[j])
            
            is_lsst = key[0] == "__lsst_global__"
            if not is_lsst:
                continue
            
            if pairs is None or pairs.empty:
                continue
            
            centers, accs, _, ci_lo, ci_hi = binned_accuracy(
                pairs,
                mag_bins,
                class_val,
                min_count=min_count,
                bootstrap_ci=bootstrap_ci,
                bootstrap_samples=bootstrap_samples,
                bootstrap_alpha=bootstrap_alpha,
                rng=rng,
            )
            valid = np.isfinite(accs)
            if valid.any():
                if bootstrap_ci:
                    valid_ci = valid & np.isfinite(ci_lo) & np.isfinite(ci_hi)
                    if valid_ci.any():
                        ax.fill_between(
                            centers[valid_ci],
                            ci_lo[valid_ci],
                            ci_hi[valid_ci],
                            color="black",
                            alpha=0.10,
                            linewidth=0,
                            zorder=0,
                        )
                ax.plot(centers[valid], accs[valid],
                        marker="o", markersize=3, label=label,
                        color="black", linewidth=1.8, linestyle="--",
                        zorder=1, alpha=0.7)
        
        # Now plot DD runs on top
        for j, key in enumerate(run_combos):
            pairs = pairs_dict.get(key)
            label = legend_name_from_panel_label(col_labels[j])
            color = color_cycle(j / max(n_runs, 1))
            
            is_lsst = key[0] == "__lsst_global__"
            if is_lsst:
                continue
            
            if pairs is None or pairs.empty:
                continue
            centers, accs, _, ci_lo, ci_hi = binned_accuracy(
                pairs,
                mag_bins,
                class_val,
                min_count=min_count,
                bootstrap_ci=bootstrap_ci,
                bootstrap_samples=bootstrap_samples,
                bootstrap_alpha=bootstrap_alpha,
                rng=rng,
            )
            valid = np.isfinite(accs)
            if valid.any():
                if bootstrap_ci:
                    valid_ci = valid & np.isfinite(ci_lo) & np.isfinite(ci_hi)
                    if valid_ci.any():
                        ax.fill_between(
                            centers[valid_ci],
                            ci_lo[valid_ci],
                            ci_hi[valid_ci],
                            color=color,
                            alpha=0.16,
                            linewidth=0,
                            zorder=1,
                        )
                ax.plot(centers[valid], accs[valid],
                        marker="o", markersize=3, label=label,
                        color=color, linewidth=1.5, zorder=2)

        ax.set_xlabel(r"$\mathrm{mag}_i$ (truth)", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(panel_title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.25)
        set_unique_legend(ax, fontsize=7, loc="lower left")

    # fill remaining bottom panels (3 accuracy panels + blanks)
    for col_idx in range(3, ncols_total):
        fig.add_subplot(gs[nrows_cm, col_idx]).axis("off")

    fig.suptitle(title, fontsize=12, y=0.995)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Star/galaxy classification confusion matrices and accuracy curves."
    )
    p.add_argument("--root-run-dir",      required=True)
    p.add_argument("--run-names",         required=True, nargs="+")
    p.add_argument("--score-thresholds",  required=True, nargs="+")
    p.add_argument("--nms-thresholds",    required=True, nargs="+")
    p.add_argument("--combo",             action="store_true",
                   help="Cartesian product of score x NMS thresholds")
    p.add_argument("--mag-limit",         default="gold",
                   choices=["gold", "power_law", "nominal"])
    p.add_argument("--buffers",           nargs="+", default=["1"])
    p.add_argument("--linking-lengths",   nargs="+", default=["1.0", "2.0"])
    p.add_argument("--prefix",            default="dd", choices=["dd", "lsst"])
    p.add_argument("--mag-min",           type=float, default=18.0)
    p.add_argument("--mag-max",           type=float, default=28.0)
    p.add_argument("--mag-bin-width",     type=float, default=0.5)
    p.add_argument("--min-count",         type=int,   default=5,
                   help="Min objects per mag bin for accuracy curves")
    p.add_argument("--output-dir",        default=None)
    p.add_argument("--no-save",           action="store_true")
    p.add_argument("--summary-only",      action="store_true",
                   help="Only produce combined summary figure, skip separate plots")
    p.add_argument("--include-lsst-baseline", action="store_true",
                   help="For DD plots, add threshold-agnostic LSST baseline per run")
    p.add_argument(
        "--legend-name-map",
        nargs="*",
        default=[],
        help=(
            "Optional legend aliases in key=value format. "
            "Keys can be full run names or tokens like lsst5, comb, distill."
        ),
    )
    p.add_argument("--bootstrap-ci", action="store_true", default=True,
                   help="Add bootstrap confidence intervals to accuracy-vs-mag curves")
    p.add_argument("--bootstrap-samples", type=int, default=500,
                   help="Number of bootstrap resamples per mag bin")
    p.add_argument("--bootstrap-alpha", type=float, default=0.05,
                   help="Two-sided alpha for bootstrap CI (e.g., 0.05 => 95%% CI)")
    p.add_argument("--bootstrap-seed", type=int, default=42,
                   help="Optional random seed for reproducible bootstrap CIs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root_dir   = Path(args.root_run_dir).expanduser()
    user_legend_map = parse_legend_name_map(args.legend_name_map)
    legend_name_map = DEFAULT_LEGEND_NAME_MAP.copy()
    legend_name_map.update(user_legend_map)

    comparison_tag = build_comparison_tag(
        args.run_names,
        args.prefix,
        args.include_lsst_baseline,
    )
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else root_dir / "classifications" / comparison_tag
    )

    combos   = build_combo_labels(args.score_thresholds, args.nms_thresholds, args.combo)
    mag_bins = np.arange(args.mag_min, args.mag_max + args.mag_bin_width, args.mag_bin_width)
    prefix   = args.prefix

    for buffer in args.buffers:
        mag_limit_val = MAG_LIMITS[args.mag_limit] + float(buffer)

        for ll in args.linking_lengths:
            tag = f"{args.mag_limit}_buf{buffer}_ll{ll}"

            # Load LSST baseline once globally if requested (threshold-agnostic)
            lsst_pairs = None
            if prefix == "dd" and args.include_lsst_baseline:
                # Use first run dir as source for LSST (same for all runs)
                if args.run_names:
                    first_run_dir = root_dir / args.run_names[0]
                    if first_run_dir.is_dir():
                        print(f"Loading global LSST baseline  LL={ll}  buf={buffer}")
                        lsst_df = load_parquet(
                            first_run_dir,
                            "lsst",
                            ll,
                            "lsst",
                            mag_limit_val,
                        )
                        if lsst_df is not None:
                            lsst_pairs = get_matched_class_pairs(lsst_df, "lsst")

            # Collect (run, combo) keys and load pairs
            run_combos: List[Tuple[str, str]] = []
            pairs_dict: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
            col_labels: List[str] = []

            for run_name in args.run_names:
                run_dir = root_dir / run_name
                if not run_dir.is_dir():
                    print(f"[WARN] run dir not found: {run_dir}")
                    continue

                display_name = get_display_name_for_run(run_name, legend_name_map)
                for combo in combos:
                    key = (run_name, combo)
                    run_combos.append(key)
                    col_labels.append(f"{display_name}\n{combo}")
                    print(f"Loading  {run_name}  combo={combo}  LL={ll}  buf={buffer}")
                    df = load_parquet(run_dir, prefix, ll, combo, mag_limit_val)
                    if df is not None:
                        pairs_dict[key] = get_matched_class_pairs(df, prefix)
                    else:
                        pairs_dict[key] = None

            # Add global LSST baseline once per LL/buffer (not per run)
            if lsst_pairs is not None:
                lsst_key = ("__lsst_global__", "lsst")
                run_combos.append(lsst_key)
                col_labels.append("LSST Pipeline")
                pairs_dict[lsst_key] = lsst_pairs

            if not run_combos:
                continue

            suptitle = (
                f"Star/Galaxy Classification — {prefix.upper()}  "
                f"LL={ll}\"  {args.mag_limit}+{buffer}  "
                f"{'combo' if len(combos) == 1 else 'combos'}="
                f"{' | '.join(combos)}"
            )

            # --- Combined summary figure (always produced) ---
            summary_path = (output_dir / tag / "summary.png") if not args.no_save else None
            plot_summary(
                run_combos, pairs_dict, col_labels, mag_bins,
                title=suptitle,
                output_path=summary_path,
                min_count=args.min_count,
                bootstrap_ci=args.bootstrap_ci,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_alpha=args.bootstrap_alpha,
                bootstrap_seed=args.bootstrap_seed,
            )

            if not args.summary_only:
                # --- Separate: confusion matrices ---
                cm_path = (output_dir / tag / "confusion_matrices.png") if not args.no_save else None
                plot_confusion_matrices(
                    run_combos, pairs_dict, col_labels,
                    title=suptitle, output_path=cm_path,
                )

                # --- Separate: accuracy curves ---
                acc_path = (output_dir / tag / "accuracy_vs_mag.png") if not args.no_save else None
                plot_accuracy_curves(
                    run_combos, pairs_dict, col_labels, mag_bins,
                    title=suptitle,
                    output_path=acc_path,
                    min_count=args.min_count,
                    bootstrap_ci=args.bootstrap_ci,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_alpha=args.bootstrap_alpha,
                    bootstrap_seed=args.bootstrap_seed,
                )

    if args.no_save:
        plt.show()


if __name__ == "__main__":
    main()