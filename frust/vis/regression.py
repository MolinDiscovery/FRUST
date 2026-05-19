from contextlib import nullcontext
from numbers import Integral
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
    xlabel: str = r"$\Delta$E, kcal/mol",
    ylabel: str = r"$\Delta$G, kcal/mol",
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
    scaled: Union[bool, int] = False,
) -> None:
    """Plot x vs y with linear fit, score outliers, and annotate top points.

    Parameters
    ----------
    df
        Input data.
    x_col
        Name of the column to use for x values.
    y_col
        Name of the column to use for y values.
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.
    font_size
        Font size used for labels, ticks, and legends.
    label_col
        Column used for point labels.
    rpos_col
        Column used for position annotations.
    method
        Outlier scoring method. Use ``"pearson"`` for absolute residuals to
        the fitted line or ``"spearman"`` for absolute rank differences.
    num_outliers
        Number of top outliers to annotate.
    size
        Matplotlib figure size.
    plot_1x
        Whether to add a unit-slope offset line, ``y = x + mean(y - x)``.
    equal_axis
        Whether to force equal x/y limits and aspect ratio.
    regression_text
        Where to place regression statistics. Use ``"legend"``, ``"plot"``,
        ``"both"``, or ``"none"``.
    regression_text_loc
        Location for in-plot regression text. Named locations are ``"best"``,
        ``"upper left"``, ``"upper right"``, ``"lower left"``, and
        ``"lower right"``. A tuple is interpreted as axes-fraction
        coordinates.
    rmsd_unit
        Unit displayed after RMSD values. Use an empty string to omit units.
    scaled
        If ``False``, use the raw x-values. If ``True``, fit the raw x/y
        values first and use that full-precision linear relation to transform
        the x-values before plotting. If an integer is passed, round the
        fitted slope and intercept to that many significant figures before
        applying the transform.

        The reported main RMSD is the direct prediction error between the
        plotted x-values and y, not the residual to the plotted regression
        line. For example, with ``scaled=True`` it reports
        ``sqrt(mean((y - x_scaled)**2))``.

    Returns
    -------
    None
    """
    for col in (x_col, y_col, label_col, rpos_col):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Invalid method: {method}")
    if regression_text not in ("legend", "plot", "both", "none"):
        raise ValueError(f"Invalid regression_text: {regression_text}")

    data = df.dropna(subset=[x_col, y_col]).copy()
    if len(data) < 2:
        raise ValueError("At least two finite x/y pairs are required")

    x_raw = data[x_col]
    y = data[y_col]

    if isinstance(scaled, bool):
        scaled_sig_figs = None
        use_scaled = scaled
    elif isinstance(scaled, Integral):
        if scaled < 1:
            raise ValueError("scaled must be False, True, or a positive int")
        scaled_sig_figs = scaled
        use_scaled = True
    else:
        raise ValueError("scaled must be False, True, or a positive int")

    if use_scaled:
        scale_lr = linregress(x_raw, y)
        scale_slope = scale_lr.slope
        scale_intercept = scale_lr.intercept
        if scaled_sig_figs is not None:
            scale_slope = _round_to_sig_figs(scale_slope, scaled_sig_figs)
            scale_intercept = _round_to_sig_figs(scale_intercept, scaled_sig_figs)
        x = scale_slope * x_raw + scale_intercept
        scale_eq_label = _format_line_equation(
            scale_slope,
            scale_intercept,
            sig_figs=scaled_sig_figs,
        )
        print("[INFO]: Scaling relation:", scale_eq_label)
    else:
        x = x_raw

    lr = linregress(x, y)
    y_fit = lr.slope * x + lr.intercept
    rho, _ = spearmanr(x, y)
    line_order = np.argsort(np.asarray(x, dtype=float))

    c = float(np.mean(y - x))
    y_hat = x + c

    # Metrics
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    yhat_arr = np.asarray(y_hat, dtype=float)

    rmsd_direct = float(np.sqrt(np.mean((y_arr - x_arr) ** 2)))
    rmsd_hat = float(np.sqrt(np.mean((y_arr - yhat_arr) ** 2)))

    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    sse_hat = float(np.sum((y_arr - yhat_arr) ** 2))
    r2_hat = 1.0 - (sse_hat / sst) if sst > 0 else np.nan

    rho_hat, _ = spearmanr(y_hat, y)

    # Print equations to stdout (not on the plot)
    eq_label = _format_line_equation(lr.slope, lr.intercept)
    print("[INFO]: Linear relation:", eq_label)
    eq2_label = _format_line_equation(1.0, c)
    print("[INFO]: Error relationship: ", eq2_label)

    rmsd_unit_suffix = f" {rmsd_unit}" if rmsd_unit else ""
    fit_stats_label = (f"$R^2$={lr.rvalue**2:.3f}, "
                       f"spearman={rho:.3f}, "
                       f"RMSD$_{{direct}}$={rmsd_direct:.3f}"
                       f"{rmsd_unit_suffix}")
    offset_stats_label = (f"$R^2$={r2_hat:.3f}, "
                          f"spearman={rho_hat:.3f}, "
                          f"RMSD$_{{offset}}$={rmsd_hat:.3f}"
                          f"{rmsd_unit_suffix}")

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
            np.asarray(x)[line_order],
            np.asarray(y_fit)[line_order],
            color="red",
            marker="",
            label=fit_stats_label
            if regression_text in ("legend", "both") else "linear fit"
        )
        if plot_1x:
            ax.plot(
                np.asarray(x)[line_order],
                np.asarray(y_hat)[line_order],
                marker="",
                label=offset_stats_label
                if regression_text in ("legend", "both") else "1:1 offset"
            )

        for _, row in outliers.iterrows():
            label = f"{row[label_col]}-r{int(row[rpos_col])}"
            ax.annotate(
                label,
                (x.loc[row.name], row[y_col]),
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
                f"RMSD$_{{direct}}$={rmsd_direct:.3f}{rmsd_unit_suffix}",
            ]
            if plot_1x:
                text_lines.extend([
                    f"Offset fit: {eq2_label}",
                    f"$R^2$={r2_hat:.3f}, spearman={rho_hat:.3f}",
                    f"RMSD$_{{offset}}$={rmsd_hat:.3f}{rmsd_unit_suffix}",
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


def _format_line_equation(
    slope: float,
    intercept: float,
    sig_figs: int | None = None,
) -> str:
    """Format a linear equation for logs and legend labels."""
    if sig_figs is None:
        return (
            f"y = {slope:.2f}x "
            f"{'+' if intercept >= 0 else '-'} "
            f"{abs(intercept):.2f}"
        )

    slope_fmt = _format_sig_figs(slope, sig_figs)
    intercept_fmt = _format_sig_figs(abs(intercept), sig_figs)
    return (
        f"y = {slope_fmt}x "
        f"{'+' if intercept >= 0 else '-'} "
        f"{intercept_fmt}"
    )


def _format_sig_figs(value: float, sig_figs: int) -> str:
    """Format a float with a fixed number of significant figures."""
    rounded = _round_to_sig_figs(value, sig_figs)
    return format(rounded, f".{sig_figs}g")


def _round_to_sig_figs(value: float, sig_figs: int) -> float:
    """Round a float to a fixed number of significant figures."""
    if not np.isfinite(value) or value == 0.0:
        return float(value)

    exponent = int(np.floor(np.log10(abs(value))))
    decimals = sig_figs - exponent - 1
    return float(np.round(value, decimals))


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
