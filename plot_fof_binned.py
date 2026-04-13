"""
Plot binned FOF classification metrics from class parquet outputs.

This script is designed to mirror the threshold/buffer workflow used by
submit_run_fof_classify.sh while focusing on binned metric-vs-magnitude plots.

What it generates:
1. Per-run, per-combo plots for all binned metrics (DD and optional LSST)
   with one subplot per linking length.
2. Cross-run comparison plots (DD only) for configured run pairs, also
   with one subplot per linking length.

Output layout (default):
    {run_dir}/plots_auto/{mag_limit}_buf{buffer}/{combo}/{metric}.png

Comparison layout (default):
    {root_run_dir}/comparisons_auto/{runA}_vs_{runB}/{mag_limit}_buf{buffer}/
        {combo}/{metric}.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import (
    split_truth_det,
    binned_fraction,
    compute_completeness,
    compute_purity,
    compute_purity_by_det_mag,
    compute_blend_loss,
    compute_unrec_blend_frac,
    compute_shred_frac,
    compute_missed_frac,
    compute_spurious_frac,
    compute_resolved_frac,
    compute_shred_det_frac,
    compute_binned_f1,
)


MAG_LIMITS: Dict[str, float] = {
    "power_law": 26.22,
    "gold": 25.3,
    "nominal": 26.42,
}


@dataclass(frozen=True)
class MetricSpec:
    """Metadata for one binned metric plot."""

    key: str
    title: str
    y_label: str


METRICS: List[MetricSpec] = [
    MetricSpec("f1", "F1 Score vs Magnitude", "F1 Score"),
    MetricSpec("completeness", "Completeness vs Magnitude", "Completeness"),
    MetricSpec(
        "completeness_isolated",
        "Completeness (Isolated) vs Magnitude",
        "Completeness",
    ),
    MetricSpec(
        "completeness_blended",
        "Completeness (Blended) vs Magnitude",
        "Completeness",
    ),
    MetricSpec(
        "purity_by_truth_mag",
        "Purity (Truth Mag) vs Magnitude",
        "Purity",
    ),
    MetricSpec(
        "purity_by_det_mag",
        "Purity (Det Mag) vs Magnitude",
        "Purity",
    ),
    MetricSpec("blend_loss", "Blend Loss Fraction vs Magnitude", "Fraction"),
    MetricSpec(
        "unrec_blend_frac",
        "Unrecognized Blend Fraction vs Magnitude",
        "Fraction",
    ),
    MetricSpec(
        "unrec_blend_frac_blended",
        "Unrecognized Blend Fraction (Blended Truths) vs Magnitude",
        "Fraction",
    ),
    MetricSpec(
        "unrec_blend_frac_matched",
        "Unrecognized Blend Fraction (Matched Truths) vs Magnitude",
        "Fraction",
    ),
    MetricSpec("shred_frac", "Shred Fraction (Truth) vs Magnitude", "Fraction"),
    MetricSpec("missed_frac", "Missed Fraction vs Magnitude", "Fraction"),
    MetricSpec(
        "spurious_frac",
        "Spurious Fraction (Det) vs Magnitude",
        "Fraction",
    ),
    MetricSpec("resolved_frac", "Resolved Fraction vs Magnitude", "Fraction"),
    MetricSpec(
        "shred_det_frac",
        "Shred Fraction (Det) vs Magnitude",
        "Fraction",
    ),
]


FIXED_ONE_YMAX_METRICS = {
    "f1",
    "completeness",
    "completeness_isolated",
    "completeness_blended",
    "purity_by_truth_mag",
    "purity_by_det_mag",
    "resolved_frac",
}


def get_panel_ymax(metric_key: str, series_list: Sequence[np.ndarray]) -> float:
    """Return y-axis max for a panel using fixed or dynamic scaling."""
    if metric_key in FIXED_ONE_YMAX_METRICS:
        return 1.0

    finite_maxes: List[float] = []
    for series in series_list:
        arr = np.asarray(series, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            finite_maxes.append(float(np.max(finite)))

    if not finite_maxes:
        return 1.0

    return max(finite_maxes) + 0.3


def build_combo_labels(
    score_thresholds: Sequence[str],
    nms_thresholds: Sequence[str],
    cartesian: bool,
) -> List[str]:
    """Build threshold combo labels like s0.5_n0.6."""
    combos: List[str] = []
    if cartesian:
        for s in score_thresholds:
            for n in nms_thresholds:
                combos.append(f"s{s}_n{n}")
        return combos

    if len(score_thresholds) != len(nms_thresholds):
        raise ValueError(
            "Score and NMS threshold lists must have same length unless --combo is set"
        )
    for s, n in zip(score_thresholds, nms_thresholds):
        combos.append(f"s{s}_n{n}")
    return combos


def parse_run_pairs(raw_pairs: Sequence[str]) -> List[Tuple[str, str]]:
    """Parse run pair strings in the form runA,runB."""
    pairs: List[Tuple[str, str]] = []
    for raw in raw_pairs:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Invalid run pair '{raw}'. Use format: runA,runB"
            )
        pairs.append((parts[0], parts[1]))
    return pairs


def parse_combo_map(raw_maps: Sequence[str]) -> List[Tuple[str, str]]:
    """Parse combo map entries in the form left_combo:right_combo."""
    combo_pairs: List[Tuple[str, str]] = []
    for raw in raw_maps:
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Invalid combo map '{raw}'. Use format: left_combo:right_combo"
            )
        combo_pairs.append((parts[0], parts[1]))
    return combo_pairs


def get_mag_limit_value(mag_limit: str, buffer_val: int) -> float:
    """Return numeric mag limit after applying buffer."""
    return MAG_LIMITS[mag_limit] + float(buffer_val)


def load_analysis_df(
    run_dir: Path,
    prefix: str,
    linking_length: str,
    combo: str,
    mag_limit_val: float,
) -> Optional[pd.DataFrame]:
    """Load one classified analysis parquet for DD or LSST."""
    cats_dir = run_dir / "analysis_cats"
    if prefix == "dd":
        parquet = (
            cats_dir
            / "dd"
            / linking_length
            / combo
            / f"class_{mag_limit_val:.2f}.parquet"
        )
    else:
        parquet = (
            cats_dir
            / "lsst"
            / linking_length
            / f"class_{mag_limit_val:.2f}.parquet"
        )

    if not parquet.exists():
        return None
    return pd.read_parquet(parquet)


def compute_metric(
    metric_key: str,
    truths: pd.DataFrame,
    dets: pd.DataFrame,
    prefix: str,
    mag_bins: np.ndarray,
    min_count: int,
) -> Dict[str, np.ndarray]:
    """Compute one metric result dict with fraction and CI arrays."""
    if metric_key == "completeness":
        return compute_completeness(
            truths,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "f1":
        return compute_binned_f1(
            truths,
            dets,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "completeness_isolated":
        mask = ~truths[f"{prefix}_is_part_of_blend"].values.astype(bool)
        return compute_completeness(
            truths,
            prefix,
            mag_bins,
            subset_mask=mask,
            min_count=min_count,
        )

    if metric_key == "completeness_blended":
        mask = truths[f"{prefix}_is_part_of_blend"].values.astype(bool)
        return compute_completeness(
            truths,
            prefix,
            mag_bins,
            subset_mask=mask,
            min_count=min_count,
        )

    if metric_key == "purity_by_truth_mag":
        return compute_purity(
            dets,
            truths,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "purity_by_det_mag":
        return compute_purity_by_det_mag(
            dets,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "blend_loss":
        return compute_blend_loss(
            truths,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "unrec_blend_frac":
        return compute_unrec_blend_frac(
            truths,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "unrec_blend_frac_blended":
        blend_mask = truths[f"{prefix}_is_part_of_blend"].values.astype(bool)
        t = truths[blend_mask]
        if len(t) == 0:
            n_bins = len(mag_bins) - 1
            return {
                "bin_centers": 0.5 * (mag_bins[:-1] + mag_bins[1:]),
                "fractions": np.full(n_bins, np.nan),
                "ci_lo": np.full(n_bins, np.nan),
                "ci_hi": np.full(n_bins, np.nan),
                "counts": np.zeros(n_bins, dtype=int),
                "flagged_counts": np.zeros(n_bins, dtype=int),
            }
        return binned_fraction(
            t[f"{prefix}_is_unrec_blend"],
            t["mag_i"],
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "unrec_blend_frac_matched":
        matched_mask = truths[f"{prefix}_is_matched"].values.astype(bool)
        t = truths[matched_mask]
        if len(t) == 0:
            n_bins = len(mag_bins) - 1
            return {
                "bin_centers": 0.5 * (mag_bins[:-1] + mag_bins[1:]),
                "fractions": np.full(n_bins, np.nan),
                "ci_lo": np.full(n_bins, np.nan),
                "ci_hi": np.full(n_bins, np.nan),
                "counts": np.zeros(n_bins, dtype=int),
                "flagged_counts": np.zeros(n_bins, dtype=int),
            }
        return binned_fraction(
            t[f"{prefix}_is_unrec_blend"],
            t["mag_i"],
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "shred_frac":
        return compute_shred_frac(
            truths,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "missed_frac":
        return compute_missed_frac(
            truths,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "spurious_frac":
        return compute_spurious_frac(
            dets,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "resolved_frac":
        return compute_resolved_frac(
            truths,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    if metric_key == "shred_det_frac":
        return compute_shred_det_frac(
            dets,
            prefix,
            mag_bins,
            min_count=min_count,
        )

    raise ValueError(f"Unsupported metric key: {metric_key}")


def plot_metric_panels(
    metric: MetricSpec,
    linking_lengths: Sequence[str],
    dd_by_ll: Dict[str, Dict[str, np.ndarray]],
    lsst_by_ll: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    figure_title: str,
    include_lsst: bool,
) -> None:
    """Plot one metric with one panel per linking length."""
    n_ll = len(linking_lengths)
    fig, axes = plt.subplots(1, n_ll, figsize=(8 * n_ll, 6), squeeze=False)

    for idx, ll in enumerate(linking_lengths):
        ax = axes[0, idx]

        if ll in dd_by_ll:
            dd = dd_by_ll[ll]
            dd_mask = np.isfinite(dd["fractions"])
            ax.plot(
                dd["bin_centers"][dd_mask],
                dd["fractions"][dd_mask],
                color="tab:blue",
                linestyle="-",
                marker="o",
                markersize=4,
                label="DD",
            )
            ci_mask = (
                dd_mask
                & np.isfinite(dd["ci_lo"])
                & np.isfinite(dd["ci_hi"])
            )
            if np.any(ci_mask):
                ax.fill_between(
                    dd["bin_centers"][ci_mask],
                    dd["ci_lo"][ci_mask],
                    dd["ci_hi"][ci_mask],
                    color="tab:blue",
                    alpha=0.15,
                )

        if include_lsst and ll in lsst_by_ll:
            lsst = lsst_by_ll[ll]
            lsst_mask = np.isfinite(lsst["fractions"])
            ax.plot(
                lsst["bin_centers"][lsst_mask],
                lsst["fractions"][lsst_mask],
                color="tab:green",
                linestyle="--",
                marker="o",
                markersize=4,
                label="LSST",
            )
            ci_mask = (
                lsst_mask
                & np.isfinite(lsst["ci_lo"])
                & np.isfinite(lsst["ci_hi"])
            )
            if np.any(ci_mask):
                ax.fill_between(
                    lsst["bin_centers"][ci_mask],
                    lsst["ci_lo"][ci_mask],
                    lsst["ci_hi"][ci_mask],
                    color="tab:green",
                    alpha=0.15,
                )

        ax.set_xlabel("mag_i", fontsize=12)
        ax.set_ylabel(metric.y_label, fontsize=12)
        ax.set_title(f"LL={ll}\"", fontsize=12)
        ax.grid(True, alpha=0.3)
        panel_ymax = get_panel_ymax(
            metric.key,
            [
                dd_by_ll.get(ll, {}).get("fractions", np.asarray([])),
                lsst_by_ll.get(ll, {}).get("fractions", np.asarray([])),
            ],
        )
        ax.set_ylim(-0.02, panel_ymax)
        ax.legend(fontsize=9)

    fig.suptitle(figure_title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pair_metric_panels(
    metric: MetricSpec,
    linking_lengths: Sequence[str],
    run_a: str,
    run_b: str,
    run_a_by_ll: Dict[str, Dict[str, np.ndarray]],
    run_b_by_ll: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    figure_title: str,
) -> None:
    """Plot one cross-run metric figure with one panel per linking length."""
    n_ll = len(linking_lengths)
    fig, axes = plt.subplots(1, n_ll, figsize=(8 * n_ll, 6), squeeze=False)


    for idx, ll in enumerate(linking_lengths):
        ax = axes[0, idx]

        # Plot run A
        if ll in run_a_by_ll:
            a = run_a_by_ll[ll]
            mask = np.isfinite(a["fractions"])
            ax.plot(
                a["bin_centers"][mask],
                a["fractions"][mask],
                color="tab:blue",
                linestyle="-",
                marker="o",
                markersize=4,
                label=run_a,
            )
            # Error bands for run A
            if "ci_lo" in a and "ci_hi" in a:
                ci_mask = mask & np.isfinite(a["ci_lo"]) & np.isfinite(a["ci_hi"])
                if np.any(ci_mask):
                    ax.fill_between(
                        a["bin_centers"][ci_mask],
                        a["ci_lo"][ci_mask],
                        a["ci_hi"][ci_mask],
                        color="tab:blue",
                        alpha=0.18,
                        linewidth=0,
                        label=f"{run_a} CI"
                    )

        # Plot run B
        if ll in run_b_by_ll:
            b = run_b_by_ll[ll]
            mask = np.isfinite(b["fractions"])
            ax.plot(
                b["bin_centers"][mask],
                b["fractions"][mask],
                color="tab:red",
                linestyle="-",
                marker="o",
                markersize=4,
                label=run_b,
            )
            # Error bands for run B
            if "ci_lo" in b and "ci_hi" in b:
                ci_mask = mask & np.isfinite(b["ci_lo"]) & np.isfinite(b["ci_hi"])
                if np.any(ci_mask):
                    ax.fill_between(
                        b["bin_centers"][ci_mask],
                        b["ci_lo"][ci_mask],
                        b["ci_hi"][ci_mask],
                        color="tab:red",
                        alpha=0.18,
                        linewidth=0,
                        label=f"{run_b} CI"
                    )

        ax.set_xlabel("mag_i", fontsize=12)
        ax.set_ylabel(metric.y_label, fontsize=12)
        ax.set_title(f"LL={ll}\"", fontsize=12)
        ax.grid(True, alpha=0.3)
        panel_ymax = get_panel_ymax(
            metric.key,
            [
                run_a_by_ll.get(ll, {}).get("fractions", np.asarray([])),
                run_b_by_ll.get(ll, {}).get("fractions", np.asarray([])),
            ],
        )
        ax.set_ylim(-0.02, panel_ymax)
        ax.legend(fontsize=8)

    fig.suptitle(figure_title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_per_run_plots(
    root_run_dir: Path,
    run_names: Sequence[str],
    combos: Sequence[str],
    buffers: Sequence[int],
    linking_lengths: Sequence[str],
    mag_limit: str,
    mag_bins: np.ndarray,
    min_count: int,
    include_lsst: bool,
    output_subdir: str,
) -> None:
    """Generate per-run, per-combo all-metric plots."""
    for run_name in run_names:
        run_dir = root_run_dir / run_name
        if not run_dir.exists():
            print(f"WARNING: run directory not found, skipping: {run_dir}")
            continue

        for buffer_val in buffers:
            mag_limit_val = get_mag_limit_value(mag_limit, buffer_val)
            for combo in combos:
                print(
                    "Generating run plots:",
                    f"run={run_name}",
                    f"buffer={buffer_val}",
                    f"combo={combo}",
                )
                for metric in METRICS:
                    dd_results: Dict[str, Dict[str, np.ndarray]] = {}
                    lsst_results: Dict[str, Dict[str, np.ndarray]] = {}

                    for ll in linking_lengths:
                        dd_df = load_analysis_df(
                            run_dir,
                            "dd",
                            ll,
                            combo,
                            mag_limit_val,
                        )
                        if dd_df is not None:
                            truths_dd, dets_dd = split_truth_det(dd_df, "dd")
                            dd_results[ll] = compute_metric(
                                metric.key,
                                truths_dd,
                                dets_dd,
                                "dd",
                                mag_bins,
                                min_count,
                            )

                        if include_lsst:
                            lsst_df = load_analysis_df(
                                run_dir,
                                "lsst",
                                ll,
                                combo,
                                mag_limit_val,
                            )
                            if lsst_df is not None:
                                truths_lsst, dets_lsst = split_truth_det(
                                    lsst_df,
                                    "lsst",
                                )
                                lsst_results[ll] = compute_metric(
                                    metric.key,
                                    truths_lsst,
                                    dets_lsst,
                                    "lsst",
                                    mag_bins,
                                    min_count,
                                )

                    if not dd_results and not lsst_results:
                        continue

                    out_path = (
                        run_dir
                        / output_subdir
                        / f"{mag_limit}_buf{buffer_val}"
                        / combo
                        / f"{metric.key}.png"
                    )
                    figure_title = (
                        f"{run_name} | {combo} | {metric.title} | "
                        f"{mag_limit}+{buffer_val}"
                    )
                    plot_metric_panels(
                        metric,
                        linking_lengths,
                        dd_results,
                        lsst_results,
                        out_path,
                        figure_title,
                        include_lsst,
                    )


def run_pair_comparisons(
    root_run_dir: Path,
    run_pairs: Sequence[Tuple[str, str]],
    combos: Sequence[str],
    mapped_combo_pairs: Sequence[Tuple[str, str]],
    buffers: Sequence[int],
    linking_lengths: Sequence[str],
    mag_limit: str,
    mag_bins: np.ndarray,
    min_count: int,
    comparison_subdir: str,
) -> None:
    """Generate cross-run comparison plots (DD only)."""
    combo_items: List[Tuple[str, str, str]]
    if mapped_combo_pairs:
        combo_items = [
            (left_combo, right_combo, f"{left_combo}_vs_{right_combo}")
            for left_combo, right_combo in mapped_combo_pairs
        ]
    else:
        combo_items = [(combo, combo, combo) for combo in combos]

    for run_a, run_b in run_pairs:
        run_a_dir = root_run_dir / run_a
        run_b_dir = root_run_dir / run_b
        if not run_a_dir.exists() or not run_b_dir.exists():
            print(
                f"WARNING: comparison pair skipped (missing run dir): {run_a}, {run_b}"
            )
            continue

        for buffer_val in buffers:
            mag_limit_val = get_mag_limit_value(mag_limit, buffer_val)
            for combo_a, combo_b, combo_tag in combo_items:
                combo_has_data = False
                for metric in METRICS:
                    a_results: Dict[str, Dict[str, np.ndarray]] = {}
                    b_results: Dict[str, Dict[str, np.ndarray]] = {}

                    for ll in linking_lengths:
                        dd_a = load_analysis_df(
                            run_a_dir,
                            "dd",
                            ll,
                            combo_a,
                            mag_limit_val,
                        )
                        dd_b = load_analysis_df(
                            run_b_dir,
                            "dd",
                            ll,
                            combo_b,
                            mag_limit_val,
                        )
                        if dd_a is None or dd_b is None:
                            # In mapped mode, skip when either mapped combo is missing.
                            continue

                        truths_a, dets_a = split_truth_det(dd_a, "dd")
                        truths_b, dets_b = split_truth_det(dd_b, "dd")
                        a_results[ll] = compute_metric(
                            metric.key,
                            truths_a,
                            dets_a,
                            "dd",
                            mag_bins,
                            min_count,
                        )
                        b_results[ll] = compute_metric(
                            metric.key,
                            truths_b,
                            dets_b,
                            "dd",
                            mag_bins,
                            min_count,
                        )

                    if not a_results or not b_results:
                        continue

                    combo_has_data = True
                    out_path = (
                        root_run_dir
                        / comparison_subdir
                        / f"{run_a}_vs_{run_b}"
                        / f"{mag_limit}_buf{buffer_val}"
                        / combo_tag
                        / f"{metric.key}.png"
                    )
                    title = (
                        f"{run_a} ({combo_a}) vs {run_b} ({combo_b}) | "
                        f"{metric.title} | "
                        f"{mag_limit}+{buffer_val}"
                    )
                    plot_pair_metric_panels(
                        metric,
                        linking_lengths,
                        run_a,
                        run_b,
                        a_results,
                        b_results,
                        out_path,
                        title,
                    )

                if not combo_has_data:
                    print(
                        "WARNING: no matched DD data for comparison:",
                        f"pair={run_a},{run_b}",
                        f"buffer={buffer_val}",
                        f"combo_a={combo_a}",
                        f"combo_b={combo_b}",
                    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Plot all binned FOF metrics for run/combo configurations."
    )
    parser.add_argument("--root-run-dir", default="~/lsst_runs")
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--score-thresholds", nargs="+", required=True)
    parser.add_argument("--nms-thresholds", nargs="+", required=True)
    parser.add_argument("--combo", action="store_true")
    parser.add_argument("--buffers", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--linking-lengths", nargs="+", default=["1.0", "2.0"])
    parser.add_argument(
        "--mag-limit",
        choices=list(MAG_LIMITS.keys()),
        default="gold",
    )
    parser.add_argument("--mag-min", type=float, default=18.0)
    parser.add_argument("--mag-max", type=float, default=28.0)
    parser.add_argument("--mag-bin-width", type=float, default=0.5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--skip-lsst", action="store_true")
    parser.add_argument(
        "--output-subdir",
        default="plots_auto",
        help="Output subdir inside each run directory for per-run plots.",
    )
    parser.add_argument(
        "--comparison-subdir",
        default="comparisons_auto",
        help="Output subdir inside root-run-dir for cross-run comparisons.",
    )
    parser.add_argument(
        "--comparison-pairs",
        nargs="*",
        default=[],
        help="Run pairs in format runA,runB. Example: a,b c,d",
    )
    parser.add_argument(
        "--comparison-combo-map",
        nargs="*",
        default=[],
        help=(
            "Optional non-matching combo map in format "
            "left_combo:right_combo. "
            "Example: s0.5_n0.6:s0.55_n0.65"
        ),
    )

    parser.add_argument(
        "--only-comparison",
        action="store_true",
        help="Only generate comparison plots (skip per-run plots)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    root_run_dir = Path(args.root_run_dir).expanduser().resolve()
    if not root_run_dir.exists():
        raise FileNotFoundError(f"root run directory not found: {root_run_dir}")

    for b in args.buffers:
        if b not in (0, 1, 2):
            raise ValueError(f"buffer must be one of 0,1,2 (got {b})")

    combos = build_combo_labels(
        args.score_thresholds,
        args.nms_thresholds,
        args.combo,
    )
    mag_bins = np.arange(
        args.mag_min,
        args.mag_max + 0.01,
        args.mag_bin_width,
    )

    run_pairs = parse_run_pairs(args.comparison_pairs)
    mapped_combo_pairs = parse_combo_map(args.comparison_combo_map)

    print("=" * 80)
    print("PLOT FOF BINNED METRICS")
    print("=" * 80)
    print(f"Root run dir:     {root_run_dir}")
    print(f"Run names:        {args.run_names}")
    print(f"Combos:           {combos}")
    print(f"Buffers:          {args.buffers}")
    print(f"Linking lengths:  {args.linking_lengths}")
    print(
        f"Mag bins:         {args.mag_min} to {args.mag_max} "
        f"(width {args.mag_bin_width})"
    )
    print(f"Include LSST:     {not args.skip_lsst}")
    print(f"Comparison pairs: {run_pairs if run_pairs else '<none>'}")
    print(
        "Comparison combo map: "
        f"{mapped_combo_pairs if mapped_combo_pairs else '<exact-match>'}"
    )


    if not args.only_comparison:
        run_per_run_plots(
            root_run_dir=root_run_dir,
            run_names=args.run_names,
            combos=combos,
            buffers=args.buffers,
            linking_lengths=args.linking_lengths,
            mag_limit=args.mag_limit,
            mag_bins=mag_bins,
            min_count=args.min_count,
            include_lsst=not args.skip_lsst,
            output_subdir=args.output_subdir,
        )

    if run_pairs:
        run_pair_comparisons(
            root_run_dir=root_run_dir,
            run_pairs=run_pairs,
            combos=combos,
            mapped_combo_pairs=mapped_combo_pairs,
            buffers=args.buffers,
            linking_lengths=args.linking_lengths,
            mag_limit=args.mag_limit,
            mag_bins=mag_bins,
            min_count=args.min_count,
            comparison_subdir=args.comparison_subdir,
        )

    print("Done.")


if __name__ == "__main__":
    main()
