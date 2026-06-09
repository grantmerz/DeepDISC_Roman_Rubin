"""Utilities for plotting 2D/3D embedding spaces.

This module provides notebook-independent helpers for reducing embedding vectors
with UMAP, PaCMAP, or t-SNE and plotting either 2D or 3D scatter figures.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pacmap import PaCMAP
from sklearn.manifold import TSNE
from umap import UMAP


def get_matched_truth_col(
    dets: pd.DataFrame,
    truths: pd.DataFrame,
    prefix: str,
    col: str = "mag_i",
) -> pd.Series:
    """Return a truth-column series aligned to detection rows.

    Parameters
    ----------
    dets:
        Detection rows.
    truths:
        Truth rows.
    prefix:
        Prefix for match columns, typically ``dd`` or ``lsst``.
    col:
        Truth column to pull values from.

    Returns
    -------
    pd.Series
        Series aligned with ``dets.index``; unmatched rows are NaN.
    """
    truth_col_lookup = truths.set_index("row_index")[col]
    matched_id_col = f"{prefix}_matched_id"
    matched_mask = dets[f"{prefix}_is_matched"]

    matched_truth_vals = dets.loc[matched_mask, matched_id_col].map(
        truth_col_lookup
    )
    result = pd.Series(np.nan, index=dets.index)
    result.loc[matched_mask] = matched_truth_vals.values
    return result


def _make_embedding(
    z: np.ndarray,
    method: str,
    random_state: int = 42,
    n_components: int = 2,
) -> np.ndarray:
    """Reduce ``z`` from shape ``(N, d)`` to ``(N, n_components)``."""
    if method == "umap":
        print(f"  Running UMAP on {len(z)} embeddings...")
        return UMAP(
            n_components=n_components,
            n_neighbors=30,
            metric="cosine",
            init="pca",
            random_state=random_state,
        ).fit_transform(z)

    if method == "pacmap":
        print(f"  Running PaCMAP on {len(z)} embeddings...")
        return PaCMAP(
            n_components=n_components,
            n_neighbors=None,
            distance="angular",
            apply_pca=False,
            random_state=random_state,
        ).fit_transform(z)

    if method == "tsne":
        print(f"  Running t-SNE on {len(z)} embeddings...")
        return TSNE(
            n_components=n_components,
            perplexity=30,
            metric="cosine",
            random_state=random_state,
            n_jobs=-1,
        ).fit_transform(z)

    raise ValueError(f"Unknown method: {method}")


def _plot_2d(
    z_2d: np.ndarray,
    df: pd.DataFrame,
    method: str,
    out_dir: Optional[str],
    mag_col: str,
    savefig: bool,
    title_suffix: str = "",
    fname_suffix: str = "",
    size_col: str = "size",
    z_col: str = "z",
    matched: bool = False,
) -> None:
    """Plot 2D embedding diagnostics."""
    if matched:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = np.asarray(axes).ravel()

    tag = method.upper()

    classes = df["class"].values
    colors_cls = ["royalblue" if c == 0 else "tomato" for c in classes]
    axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=colors_cls, s=4, alpha=0.5)
    axes[0].set_title(f"{tag} colored by class")
    axes[0].set_xlabel(f"{tag} dim 1")
    axes[0].set_ylabel(f"{tag} dim 2")
    axes[0].legend(
        handles=[
            Patch(color="royalblue", label="galaxy"),
            Patch(color="tomato", label="star"),
        ]
    )

    def _plot_continuous(
        ax: plt.Axes,
        value_col: str,
        label: str,
        cmap: str = "viridis_r",
    ) -> None:
        if value_col in df.columns:
            vals = df[value_col].values.astype(float)
            valid = np.isfinite(vals)
            if valid.any():
                vmin, vmax = np.nanpercentile(vals[valid], [5, 95])
                sc = ax.scatter(
                    z_2d[:, 0],
                    z_2d[:, 1],
                    c=vals,
                    cmap=cmap,
                    s=4,
                    alpha=0.5,
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(sc, ax=ax, label=label)
                ax.set_title(f"{tag} colored by {label}")
            else:
                ax.scatter(
                    z_2d[:, 0],
                    z_2d[:, 1],
                    color="lightgray",
                    s=4,
                    alpha=0.5,
                )
                ax.set_title(f"{tag} ({label} all-NaN)")
        else:
            ax.scatter(
                z_2d[:, 0],
                z_2d[:, 1],
                color="lightgray",
                s=4,
                alpha=0.5,
            )
            ax.set_title(f"{tag} (no {label} column available)")

        ax.set_xlabel(f"{tag} dim 1")
        ax.set_ylabel(f"{tag} dim 2")

    _plot_continuous(axes[1], mag_col, mag_col, cmap="viridis_r")

    if matched:
        _plot_continuous(axes[2], size_col, size_col, cmap="magma")
        _plot_continuous(axes[3], z_col, z_col, cmap="coolwarm")

    suptitle = f"CLIP model: z_q embedding space via {tag} (n={len(z_2d):,})"
    if title_suffix:
        suptitle += f" ({title_suffix})"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()

    if savefig:
        if not out_dir:
            raise ValueError("out_dir must be provided when savefig=True")
        fname = os.path.join(out_dir, f"{method}_clip{fname_suffix}.png")
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname}")


def _plot_3d(
    z_3d: np.ndarray,
    df: pd.DataFrame,
    method: str,
    out_dir: Optional[str],
    mag_col: str,
    savefig: bool,
    title_suffix: str = "",
    fname_suffix: str = "",
    size_col: str = "size",
    z_col: str = "z",
    matched: bool = False,
) -> None:
    """Plot 3D embedding diagnostics."""
    if matched:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 12),
            subplot_kw={"projection": "3d"},
        )
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 6),
            subplot_kw={"projection": "3d"},
        )
        axes = np.asarray(axes).ravel()

    tag = method.upper()

    classes = df["class"].values
    colors_cls = ["royalblue" if c == 0 else "tomato" for c in classes]
    axes[0].scatter(
        z_3d[:, 0],
        z_3d[:, 1],
        z_3d[:, 2],
        c=colors_cls,
        s=4,
        alpha=0.5,
    )
    axes[0].set_title(f"{tag} 3D colored by class")
    axes[0].set_xlabel(f"{tag} dim 1")
    axes[0].set_ylabel(f"{tag} dim 2")
    axes[0].set_zlabel(f"{tag} dim 3")
    axes[0].legend(
        handles=[
            Patch(color="royalblue", label="galaxy"),
            Patch(color="tomato", label="star"),
        ]
    )

    def _plot_continuous(
        ax: plt.Axes,
        value_col: str,
        label: str,
        cmap: str = "viridis_r",
    ) -> None:
        if value_col in df.columns:
            vals = df[value_col].values.astype(float)
            valid = np.isfinite(vals)
            if valid.any():
                vmin, vmax = np.nanpercentile(vals[valid], [5, 95])
                sc = ax.scatter(
                    z_3d[:, 0],
                    z_3d[:, 1],
                    z_3d[:, 2],
                    c=vals,
                    cmap=cmap,
                    s=4,
                    alpha=0.5,
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(sc, ax=ax, label=label, fraction=0.04, pad=0.08)
                ax.set_title(f"{tag} 3D colored by {label}")
            else:
                ax.scatter(
                    z_3d[:, 0],
                    z_3d[:, 1],
                    z_3d[:, 2],
                    color="lightgray",
                    s=4,
                    alpha=0.5,
                )
                ax.set_title(f"{tag} 3D ({label} all-NaN)")
        else:
            ax.scatter(
                z_3d[:, 0],
                z_3d[:, 1],
                z_3d[:, 2],
                color="lightgray",
                s=4,
                alpha=0.5,
            )
            ax.set_title(f"{tag} 3D (no {label} column available)")

        ax.set_xlabel(f"{tag} dim 1")
        ax.set_ylabel(f"{tag} dim 2")
        ax.set_zlabel(f"{tag} dim 3")

    _plot_continuous(axes[1], mag_col, mag_col, cmap="viridis_r")

    if matched:
        _plot_continuous(axes[2], size_col, size_col, cmap="magma")
        _plot_continuous(axes[3], z_col, z_col, cmap="coolwarm")

    suptitle = (
        f"CLIP model: z_q embedding space via {tag} "
        f"({z_3d.shape[1]}D, n={len(z_3d):,})"
    )
    if title_suffix:
        suptitle += f" ({title_suffix})"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()

    if savefig:
        if not out_dir:
            raise ValueError("out_dir must be provided when savefig=True")
        fname = os.path.join(
            out_dir,
            f"{method}_clip{fname_suffix}_{z_3d.shape[1]}d.png",
        )
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname}")


def plot_embedding_space(
    emb_data: dict,
    det_df: pd.DataFrame,
    out_dir: Optional[str] = None,
    reducer: Sequence[str] = ("pacmap", "umap"),
    n_subsample: int = 3000,
    mag_col: str = "mag_i",
    savefig: bool = False,
    fof_df: Optional[pd.DataFrame] = None,
    n_components: int = 2,
    seed: int = 42,
) -> None:
    """Plot embedding space in 2D or 3D for one or more reducers.

    Parameters
    ----------
    emb_data:
        Dictionary containing at least ``emb_q`` (torch tensor).
    det_df:
        Detection dataframe aligned to embedding rows.
    out_dir:
        Directory for output images when ``savefig=True``.
    reducer:
        Reducer methods to run: ``umap``, ``pacmap``, ``tsne``.
    n_subsample:
        Number of rows to subsample for plotting.
    mag_col:
        Magnitude column name used for continuous coloring.
    savefig:
        Whether to save plots to ``out_dir``.
    fof_df:
        Optional FoF dataframe for matched-only plots.
    n_components:
        Output dimensions to visualize, either 2 or 3.
    seed:
        RNG seed used for subsampling.
    """
    if not isinstance(reducer, (list, tuple)) or len(reducer) == 0:
        raise ValueError("reducer must be a non-empty list/tuple of methods.")
    if n_components not in {2, 3}:
        raise ValueError("n_components must be either 2 or 3.")

    allowed_methods = {"umap", "pacmap", "tsne"}
    methods = []
    for method in reducer:
        if method not in allowed_methods:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Choose from {sorted(allowed_methods)}"
            )
        methods.append(method)

    z_q = emb_data["emb_q"].float()
    n = min(len(z_q), n_subsample)

    np.random.seed(seed)
    idx = np.random.choice(len(z_q), n, replace=False)
    z = z_q[idx].numpy()
    df = det_df.iloc[idx].reset_index(drop=True)

    for method in methods:
        z_low = _make_embedding(z, method, random_state=seed, n_components=n_components)
        if n_components == 2:
            _plot_2d(z_low, df, method, out_dir, mag_col, savefig)
        else:
            _plot_3d(z_low, df, method, out_dir, mag_col, savefig)

    if fof_df is None:
        print("  [info] No fof_df provided, skipping matched-only visualisation.")
        return

    truths = fof_df[fof_df["catalog_key"] == "lsst_truth"]
    dets = fof_df[fof_df["catalog_key"] == "dd_det"].copy()
    dets["truth_mag_i"] = get_matched_truth_col(
        dets,
        truths,
        prefix="dd",
        col=mag_col,
    )
    dets["truth_size"] = get_matched_truth_col(
        dets,
        truths,
        prefix="dd",
        col="size",
    )
    dets["truth_z"] = get_matched_truth_col(
        dets,
        truths,
        prefix="dd",
        col="z",
    )

    matched = dets[dets["dd_is_matched"] == True]
    if len(matched) == 0:
        print("  [skip] No matched detections in fof_df")
        return

    matched_idx = matched["row_index"].values
    n_matched = min(len(matched_idx), n_subsample)

    np.random.seed(seed)
    chosen = np.random.choice(len(matched_idx), n_matched, replace=False)
    matched_idx_sub = matched_idx[chosen]
    matched_df_sub = matched.iloc[chosen].reset_index(drop=True)

    plot_df = pd.DataFrame(
        {
            "class": matched_df_sub["class"].values,
            mag_col: matched_df_sub["truth_mag_i"].values,
            "size": matched_df_sub["truth_size"].values,
            "z": matched_df_sub["truth_z"].values,
        },
        index=range(n_matched),
    )

    z_matched = z_q[matched_idx_sub].numpy()
    print(f"  Running matched-only visualisation ({n_matched:,} detections)...")

    for method in methods:
        z_low = _make_embedding(
            z_matched,
            method,
            random_state=seed,
            n_components=n_components,
        )
        if n_components == 2:
            _plot_2d(
                z_low,
                plot_df,
                method,
                out_dir,
                mag_col,
                savefig,
                title_suffix="Matched",
                fname_suffix="_matched",
                size_col="size",
                z_col="z",
                matched=True,
            )
        else:
            _plot_3d(
                z_low,
                plot_df,
                method,
                out_dir,
                mag_col,
                savefig,
                title_suffix="Matched",
                fname_suffix="_matched",
                size_col="size",
                z_col="z",
                matched=True,
            )


__all__ = [
    "get_matched_truth_col",
    "plot_embedding_space",
]
