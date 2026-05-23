from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MPL_DIR = ROOT / ".mpl-cache"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = ROOT / "experimental_results.csv"
IMG_DIR = ROOT / "img"


FAMILY_COLORS = {
    "MLP": "#2F5D8A",
    "LSTM": "#2A7F62",
    "StockMixer": "#B36A1E",
}

OBJECTIVE_LABELS = {
    "MSE baseline": "MSE",
    "MAE baseline": "MAE",
    "Sharpe-loss": "Sharpe",
    "Sortino-loss": "Sortino",
}

OBJECTIVE_MARKERS = {
    "MSE baseline": "o",
    "MAE baseline": "s",
    "Sharpe-loss": "^",
    "Sortino-loss": "D",
}


def load_results() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["plot_label"] = df["family"] + " / " + df["objective"].map(OBJECTIVE_LABELS)
    df["color"] = df["family"].map(FAMILY_COLORS)
    return df


def save_metric_panels(df: pd.DataFrame) -> None:
    metrics = [
        ("net_sharpe", "Net Sharpe"),
        ("net_sortino", "Net Sortino"),
        ("alpha_ir", "Alpha IR"),
        ("net_cum", "Net Cum"),
    ]

    ordered = df.sort_values("net_sharpe", ascending=True).reset_index(drop=True)
    y_labels = ordered["plot_label"]
    y_positions = range(len(ordered))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (metric_key, metric_title) in zip(axes.flat, metrics):
        values = ordered[metric_key]
        ax.barh(y_positions, values, color=ordered["color"], alpha=0.9)
        ax.set_title(metric_title, fontsize=13, weight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9)

        if metric_key == "alpha_ir":
            ax.axvline(0, color="#444444", linewidth=0.9, alpha=0.8)

        for y_pos, value in zip(y_positions, values):
            offset = 0.02 if value >= 0 else -0.02
            ha = "left" if value >= 0 else "right"
            ax.text(value + offset, y_pos, f"{value:.3f}", va="center", ha=ha, fontsize=8)

    for ax in axes[:, 0]:
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(y_labels, fontsize=9)

    for ax in axes[:, 1]:
        ax.set_yticks(list(y_positions))
        ax.tick_params(axis="y", labelleft=False)

    fig.suptitle("Porovnanie hlavných metrík na testovacej množine", fontsize=16, weight="bold")
    fig.subplots_adjust(left=0.24, right=0.98, top=0.92, bottom=0.06, wspace=0.08, hspace=0.15)

    fig.savefig(IMG_DIR / "results_metric_panels.pdf", bbox_inches="tight")
    fig.savefig(IMG_DIR / "results_metric_panels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_turnover_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    fig.patch.set_facecolor("white")

    size_base = 260
    sizes = size_base + df["net_cum"].clip(lower=0).to_numpy() * 420

    x_min = df["turnover"].min() - 0.004
    x_max = df["turnover"].max() + 0.004
    y_min = df["net_sharpe"].min() - 0.06
    y_max = df["net_sharpe"].max() + 0.08

    for _, row in df.iterrows():
        ax.scatter(
            row["turnover"],
            row["net_sharpe"],
            s=sizes[df.index.get_loc(row.name)],
            color=row["color"],
            marker=OBJECTIVE_MARKERS[row["objective"]],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.9,
        )

        dx = 0.0015
        dy = 0.012
        ha = "left"
        va = "bottom"

        if row["turnover"] > 0.093:
            dx = -0.0017
            ha = "right"
        if row["net_sharpe"] > 1.40:
            dy = -0.018
            va = "top"

        ax.text(
            row["turnover"] + dx,
            row["net_sharpe"] + dy,
            row["plot_label"],
            fontsize=8.5,
            color="#222222",
            ha=ha,
            va=va,
        )

    ax.set_title("Vzťah medzi turnoverom a čistým Sharpeho pomerom", fontsize=15, weight="bold")
    ax.set_xlabel("Priemerný turnover")
    ax.set_ylabel("Net Sharpe")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    family_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=family, markerfacecolor=color, markersize=9)
        for family, color in FAMILY_COLORS.items()
    ]
    objective_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#666666",
            linestyle="None",
            label=OBJECTIVE_LABELS[objective],
            markersize=8,
        )
        for objective, marker in OBJECTIVE_MARKERS.items()
    ]

    legend1 = ax.legend(handles=family_handles, title="Rodina modelu", loc="lower left")
    ax.add_artist(legend1)
    ax.legend(handles=objective_handles, title="Cieľ učenia", loc="upper right")

    fig.tight_layout()
    fig.savefig(IMG_DIR / "results_turnover_scatter.pdf", bbox_inches="tight")
    fig.savefig(IMG_DIR / "results_turnover_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    save_metric_panels(df)
    save_turnover_scatter(df)
    print("Saved:")
    print(IMG_DIR / "results_metric_panels.pdf")
    print(IMG_DIR / "results_metric_panels.png")
    print(IMG_DIR / "results_turnover_scatter.pdf")
    print(IMG_DIR / "results_turnover_scatter.png")


if __name__ == "__main__":
    main()
