from contextlib import nullcontext
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from . import theme


def plot_regression_outliers(
    df: pd.DataFrame,
    x_col: str = "dE",
    y_col: str = "dG",
    xlabel: str = "dE, kcal/mol",
    ylabel: str = "dG, kcal/mol",
    font_size: int = 14,
    label_col: str = "substrate_name",
    rpos_col: str = "rpos",
    method: str = "spearman",
    num_outliers: int = 2,
    size: tuple = (8, 6),
    plot_1x: bool = False,
    equal_axis: bool = False,
    regression_text: str = "legend",
    regression_text_loc: Union[str, Tuple[float, float]] = "best",
    rmsd_unit: str = "kcal/mol",
) -> pd.DataFrame:
    """Plot x vs y with linear fit, score outliers, and annotate top points.

    Args:
        df (pd.DataFrame): Input data.
        x_col (str): Name of the column to use for x values. Defaults to "dG".
        y_col (str): Name of the column to use for y values. Defaults to "dE".
        label_col (str): Column used for point labels. Defaults to
            "substrate_name".
        rpos_col (str): Column used for position annotations. Defaults to
            "rpos".
        method (str, optional): Scoring method, "pearson" or "spearman".
            Defaults to "spearman".
        num_outliers (int, optional): Number of top outliers to annotate.
            Defaults to 2.
        regression_text (str, optional): Where to place regression statistics.
            Use "legend" for the historical behavior, "plot" to place them
            inside the axes, "both" for both locations, or "none" to hide
            them. Defaults to "legend".
        regression_text_loc (str or tuple, optional): Location for in-plot
            regression text. Named locations are "best", "upper left",
            "upper right", "lower left", and "lower right". A tuple is
            interpreted as axes-fraction coordinates. Defaults to "best".
        rmsd_unit (str, optional): Unit displayed after RMSD values. Use an
            empty string to omit the unit. Defaults to "kcal/mol".

    Returns:
        pd.DataFrame: DataFrame of the top outliers sorted by score.
    """
    for col in (x_col, y_col, label_col, rpos_col):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Invalid method: {method}")
    if regression_text not in ("legend", "plot", "both", "none"):
        raise ValueError(f"Invalid regression_text: {regression_text}")

    data = df.copy()
    x = data[x_col]
    y = data[y_col]

    lr = linregress(x, y)
    y_fit = lr.slope * x + lr.intercept
    rho, _ = spearmanr(x, y)
    
    c = float(np.mean(y - x))
    y_hat = x + c

    # Metrics
    y_arr = np.asarray(y, dtype=float)
    yfit_arr = np.asarray(y_fit, dtype=float)
    yhat_arr = np.asarray(y_hat, dtype=float)

    rmsd_fit = float(np.sqrt(np.mean((y_arr - yfit_arr) ** 2)))
    rmsd_hat = float(np.sqrt(np.mean((y_arr - yhat_arr) ** 2)))

    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    sse_hat = float(np.sum((y_arr - yhat_arr) ** 2))
    r2_hat = 1.0 - (sse_hat / sst) if sst > 0 else np.nan

    rho_hat, _ = spearmanr(y_hat, y)

    # Print equations to stdout (not on the plot)
    eq_label = (f"y = {lr.slope:.2f}x "
                f"{'+' if lr.intercept >= 0 else '-'} "
                f"{abs(lr.intercept):.2f}")
    print("[INFO]: Linear relation:", eq_label)
    eq2_label = (f"y = 1x "
                 f"{'+' if c >= 0 else '-'} "
                 f"{abs(c):.2f}")
    print("[INFO]: Error relationship: ", eq2_label)

    rmsd_unit_suffix = f" {rmsd_unit}" if rmsd_unit else ""
    fit_stats_label = (f"$R^2$={lr.rvalue**2:.3f}, "
                       f"spearman={rho:.3f}, "
                       f"RMSD={rmsd_fit:.3f}{rmsd_unit_suffix}")
    offset_stats_label = (f"$R^2$={r2_hat:.3f}, "
                          f"spearman={rho_hat:.3f}, "
                          f"RMSD={rmsd_hat:.3f}{rmsd_unit_suffix}")

    if method == "pearson":
        data["score"] = (y - y_fit).abs()
    else:
        data["rank_x"] = x.rank()
        data["rank_y"] = y.rank()
        data["score"] = (data["rank_y"] - data["rank_x"]).abs()

    outliers = data.nlargest(num_outliers, "score")

    style_ctx = (plt.style.context('dark_background')
                 if theme.darkmode else nullcontext())
    with style_ctx:
        fig, ax = plt.subplots(figsize=size)
        ax.scatter(x, y, alpha=0.7)
        ax.plot(
            x, y_fit, color="red", marker="",
            label=fit_stats_label
            if regression_text in ("legend", "both") else "linear fit"
        )
        if plot_1x:
            ax.plot(
                x, y_hat, marker="",
                label=offset_stats_label
                if regression_text in ("legend", "both") else "1:1 offset"
            )
            
        for _, row in outliers.iterrows():
            label = f"{row[label_col]}-r{int(row[rpos_col])}"
            ax.annotate(
                label,
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.5)
            )
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.legend(fontsize=font_size)
        ax.grid(True)
        if equal_axis:
            xmin = min(x.min(), y.min())
            xmax = max(x.max(), y.max())

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(xmin, xmax)
            ax.set_aspect("equal", adjustable="box")

        if regression_text in ("plot", "both"):
            text_lines = [
                f"Fit: {eq_label}",
                f"$R^2$={lr.rvalue**2:.3f}, spearman={rho:.3f}",
                f"RMSD={rmsd_fit:.3f}{rmsd_unit_suffix}",
            ]
            if plot_1x:
                text_lines.extend([
                    f"Offset fit: {eq2_label}",
                    f"$R^2$={r2_hat:.3f}, spearman={rho_hat:.3f}",
                    f"RMSD={rmsd_hat:.3f}{rmsd_unit_suffix}",
                ])

            if isinstance(regression_text_loc, tuple):
                x_text, y_text = regression_text_loc
                ha = "left" if x_text <= 0.5 else "right"
                va = "bottom" if y_text <= 0.5 else "top"
            else:
                loc_lookup = {
                    "upper left": (0.03, 0.97, "left", "top"),
                    "upper right": (0.97, 0.97, "right", "top"),
                    "lower left": (0.03, 0.03, "left", "bottom"),
                    "lower right": (0.97, 0.03, "right", "bottom"),
                }
                loc = regression_text_loc
                if loc == "best":
                    loc = _least_crowded_text_loc(x, y)
                if loc not in loc_lookup:
                    raise ValueError(
                        f"Invalid regression_text_loc: {regression_text_loc}"
                    )
                x_text, y_text, ha, va = loc_lookup[loc]

            ax.text(
                x_text, y_text, "\n".join(text_lines),
                transform=ax.transAxes,
                ha=ha,
                va=va,
                fontsize=max(font_size - 2, 8),
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "black" if theme.darkmode else "white",
                    "edgecolor": "0.5",
                    "alpha": 0.85,
                },
            )

        fig.tight_layout()
        plt.show()
        
    return None

def _least_crowded_text_loc(x: pd.Series, y: pd.Series) -> str:
    """Find the least crowded plot corner for an annotation box.

    Parameters
    ----------
    x
        X values shown in the plot.
    y
        Y values shown in the plot.

    Returns
    -------
    str
        Matplotlib-style corner name.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_span = np.ptp(x_arr)
    y_span = np.ptp(y_arr)
    if x_span == 0 or y_span == 0:
        return "upper left"

    x_norm = (x_arr - np.min(x_arr)) / x_span
    y_norm = (y_arr - np.min(y_arr)) / y_span
    corners = {
        "upper left": (0.0, 1.0),
        "upper right": (1.0, 1.0),
        "lower left": (0.0, 0.0),
        "lower right": (1.0, 0.0),
    }

    def score(corner: Tuple[float, float]) -> float:
        x_corner, y_corner = corner
        dist = np.hypot(x_norm - x_corner, y_norm - y_corner)
        return float(np.sum(1.0 / np.maximum(dist, 0.08)))

    return min(corners, key=lambda loc: score(corners[loc]))
