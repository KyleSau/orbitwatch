"""Before/after change statistics and a PNG change map."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Chart colours: sequential single-hue ramp for index values, a blue/red diverging
# ramp with a neutral gray midpoint for change, and a distinct muted gray for masked
# pixels (clouds, shadows, nodata) so "no data" never reads as "no change".
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MASKED = "#898781"
SEQUENTIAL = ["#f3f8ee", "#cfe6bd", "#9fcf86", "#6bb257", "#3d8f35", "#1f6b20", "#0d4712"]
DIVERGING = ["#8f1d1d", "#e34948", "#f3b2a9", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"]


def describe(values: np.ndarray) -> dict[str, float | None]:
    """Mean/median/percentiles over finite values, rounded for display."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "std": None}
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return {
        "mean": round(float(v.mean()), 4),
        "median": round(float(p50), 4),
        "p10": round(float(p10), 4),
        "p90": round(float(p90), 4),
        "std": round(float(v.std()), 4),
    }


def summarize_change(before: np.ndarray, after: np.ndarray, threshold: float) -> dict:
    """Statistics of ``after - before`` over pixels valid in both images."""
    if before.shape != after.shape:
        raise ValueError(f"Shape mismatch: {before.shape} vs {after.shape}")
    both = np.isfinite(before) & np.isfinite(after)
    n_total = int(before.size)
    n_valid = int(both.sum())
    delta = np.where(both, after - before, np.nan)
    d = delta[both]

    def pct(mask: np.ndarray) -> float | None:
        return round(100.0 * float(mask.sum()) / n_valid, 2) if n_valid else None

    return {
        "pixels_total": n_total,
        "pixels_valid_both": n_valid,
        "valid_fraction": round(n_valid / n_total, 4) if n_total else 0.0,
        "before": describe(np.where(both, before, np.nan)),
        "after": describe(np.where(both, after, np.nan)),
        "delta": describe(delta),
        "threshold": threshold,
        "pct_pixels_decreased": pct(d < -threshold),
        "pct_pixels_increased": pct(d > threshold),
        "pct_pixels_stable": pct(np.abs(d) <= threshold),
    }


def render_change_png(
    before: np.ndarray,
    after: np.ndarray,
    path: Path,
    index: str,
    before_label: str,
    after_label: str,
    title: str,
    footnote: str,
    pixel_size_m: float,
) -> Path:
    """Three panels (before, after, after - before) with shared colour scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    seq = LinearSegmentedColormap.from_list("seq", SEQUENTIAL).with_extremes(bad=MASKED)
    div = LinearSegmentedColormap.from_list("div", DIVERGING).with_extremes(bad=MASKED)

    rows, cols = before.shape
    extent_km = (0, cols * pixel_size_m / 1000, 0, rows * pixel_size_m / 1000)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0), facecolor=SURFACE)
    # Leave a strip at the bottom for the attribution footnote.
    fig.set_layout_engine("constrained", rect=(0, 0.05, 1, 0.95))
    label = index.upper()
    panels = [
        (before, seq, (-0.1, 0.9), f"Before: {before_label}", label),
        (after, seq, (-0.1, 0.9), f"After: {after_label}", label),
        (after - before, div, (-0.5, 0.5), "Change", f"{label} change (after − before)"),
    ]
    for ax, (data, cmap, (vmin, vmax), subtitle, cbar_label) in zip(axes, panels, strict=True):
        im = ax.imshow(
            data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_km, interpolation="nearest"
        )
        ax.set_title(subtitle, fontsize=9, color=INK_SECONDARY, loc="left")
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.set_xlabel("km", color=INK_SECONDARY, fontsize=8)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
        cb.set_label(cbar_label, fontsize=8, color=INK_SECONDARY)
        cb.outline.set_edgecolor("#c3c2b7")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.012, footnote, fontsize=7.5, color=INK_SECONDARY, ha="left", va="bottom")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    _shrink_png(path)
    return path


def _shrink_png(path: Path) -> None:
    """Re-save as a 256-colour palette PNG: colormapped maps survive this nearly unchanged
    and the file is a fraction of the size (keeps the example in docs/ small)."""
    from PIL import Image

    with Image.open(path) as img:
        small = img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    small.save(path, optimize=True)
