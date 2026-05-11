def _annotate_energy_only(
    ax_,
    xi,
    Ei,
    alpha,
    color,
    placement_counts,
    is_dummy,
    decimals: int,
    label_offset_up: int,
    label_offset_down: int,
    dummy_alpha: float,
    energy_fontsize: float,
):
    top_n = placement_counts["top"]
    bottom_n = placement_counts["bottom"]
    left_n = placement_counts["left"]
    right_n = placement_counts["right"]

    dx = 0
    dy = 0
    ha = "center"
    va = "center"

    if left_n:
        dx = -12 * left_n
        ha = "right"
    elif right_n:
        dx = 12 * right_n
        ha = "left"

    if top_n:
        dy = abs(label_offset_up) * top_n
        va = "bottom"
    elif bottom_n:
        dy = -abs(label_offset_down) * bottom_n
        va = "top"

    add_arrow = max(top_n, bottom_n, left_n, right_n) > 1

    text = f"{Ei:.{decimals}f}"

    a = (dummy_alpha if is_dummy else 1.0) * alpha

    arrowprops = None
    if add_arrow:
        arrowprops = {
            "arrowstyle": "->",
            "lw": 0.8,
            "alpha": a * 0.8,
            "shrinkA": 0,
            "shrinkB": 6,
            "mutation_scale": 8,
        }

    ax_.annotate(
        text,
        (xi, Ei),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        va=va,
        alpha=a,
        arrowprops=arrowprops,
        color=color,
        fontsize=energy_fontsize,
    )
