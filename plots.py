"""Figure generation for the motorcycle price project. Writes PNGs for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868"]


def save_figures(summary: dict, figdir: Path) -> list[Path]:
    figdir.mkdir(parents=True, exist_ok=True)
    return [
        _price_distribution(summary, figdir / "price_distribution.png"),
        _price_vs_age(summary, figdir / "price_vs_age.png"),
        _model_comparison(summary, figdir / "model_comparison.png"),
        _actual_vs_predicted(summary, figdir / "actual_vs_predicted.png"),
        _feature_importance(summary, figdir / "feature_importance.png"),
    ]


def _finish(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _price_distribution(s: dict, path: Path) -> Path:
    prices = s["df"][s["price_col"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(prices / 1000, bins=50, color=PALETTE[0], edgecolor="white")
    ax.set_title("Resale price distribution")
    ax.set_xlabel("Selling price (Rs thousands)")
    ax.set_ylabel("Listings")
    return _finish(fig, path)


def _price_vs_age(s: dict, path: Path) -> Path:
    df = s["df"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for owner, color in zip(sorted(df["owner"].unique()), PALETTE + ["#8172B3"]):
        sub = df[df["owner"] == owner]
        ax.scatter(sub["bike_age"], sub[s["price_col"]] / 1000, s=10, alpha=0.6, label=owner, color=color)
    ax.set_title("Resale price vs bike age, by owner type")
    ax.set_xlabel("Bike age (years)")
    ax.set_ylabel("Selling price (Rs thousands)")
    ax.legend(fontsize=8)
    return _finish(fig, path)


def _model_comparison(s: dict, path: Path) -> Path:
    names = list(s["results"])
    r2 = [s["results"][n]["r2"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, r2, color=[PALETTE[0], PALETTE[1], PALETTE[2]])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in r2])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R-squared (test set)")
    ax.set_title("Model comparison")
    return _finish(fig, path)


def _actual_vs_predicted(s: dict, path: Path) -> Path:
    a, p = s["actual"] / 1000, s["best_pred"] / 1000
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(a, p, s=12, alpha=0.6, color=PALETTE[0])
    lim = [0, max(a.max(), p.max()) * 1.05]
    ax.plot(lim, lim, "--", color="grey", label="perfect prediction")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual price (Rs thousands)")
    ax.set_ylabel("Predicted price (Rs thousands)")
    ax.set_title(f"Actual vs predicted: {s['best']}")
    ax.legend()
    return _finish(fig, path)


def _feature_importance(s: dict, path: Path) -> Path:
    top = s["drivers"].head(8)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(top.index, top.values, color=PALETTE[0])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in top.values], fontsize=8)
    ax.set_title("Price drivers (Random Forest importance)")
    ax.set_xlabel("Importance")
    return _finish(fig, path)


def save_console_image(text: str, path: Path) -> Path:
    """Render the console output as an image so it can be shown in the README."""
    lines = text.splitlines()
    width = max(len(line) for line in lines) if lines else 40
    fig_w = max(6, width * 0.085)
    fig_h = max(2, len(lines) * 0.22 + 0.4)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1e1e1e")
    fig.text(0.02, 0.97, "\n".join(lines), family="monospace", fontsize=9,
             color="#d4d4d4", va="top", ha="left")
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
