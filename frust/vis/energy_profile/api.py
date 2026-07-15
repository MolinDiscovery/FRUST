import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from .layout import (
    _build_energy_map,
    _compute_x_from_reference,
    _compute_x_single,
    _dedup_for_interp,
)
from .parsing import (
    _is_product,
    _norm_label,
    _parse_placement,
    parse_profile,
)
from .render import _annotate_energy_only
from .styles import _resolve_colors, _style_meta_for_side_label


def _explicit_overlay_colors(overlay_colors, profile_name, overlay_idx):
    if overlay_colors is None:
        return None

    spec = None
    if isinstance(overlay_colors, dict):
        spec = overlay_colors.get(profile_name)
    elif (
        isinstance(overlay_colors, (list, tuple))
        and overlay_idx < len(overlay_colors)
    ):
        spec = overlay_colors[overlay_idx]

    if spec is None:
        return None

    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return spec[0], spec[1]

    return spec, spec


def _plan_profile_colors(profiles, side_token, overlay_colors):
    color_pairs = []
    next_color_idx = 1

    for profile_idx, (profile_name, profile_states) in enumerate(profiles):
        parsed = parse_profile(profile_states, side_token)
        needs_side = any(sid == 1 for sid in parsed.segment_ids)

        if profile_idx == 0:
            color_pairs.append(("C0", "C1" if needs_side else "C0"))
            next_color_idx = 2 if needs_side else 1
            continue

        overlay_idx = profile_idx - 1
        explicit = _explicit_overlay_colors(
            overlay_colors,
            profile_name,
            overlay_idx,
        )
        if explicit is not None:
            color_pairs.append(explicit)
        else:
            main_color = f"C{next_color_idx % 10}"
            side_color = (
                f"C{(next_color_idx + 1) % 10}"
                if needs_side
                else main_color
            )
            color_pairs.append((main_color, side_color))

        next_color_idx += 2 if needs_side else 1

    return color_pairs


def _parse_product_reference_spec(product_reference):
    if product_reference is None:
        return None, "compact"

    if isinstance(product_reference, str):
        reference_label = product_reference
        display = "compact"
    elif isinstance(product_reference, tuple) and len(product_reference) == 2:
        reference_label, display = product_reference
        if not isinstance(reference_label, str) or not isinstance(display, str):
            raise TypeError(
                "product_reference tuple values must both be strings, such as "
                "('Cat', 'connector')."
            )
    else:
        raise TypeError(
            "product_reference must be a state-label string, a "
            "(state_label, display) tuple, or None."
        )

    reference_label = reference_label.strip()
    if not reference_label:
        raise ValueError("product_reference state label cannot be empty.")

    display = display.strip()
    valid_displays = {"compact", "connector"}
    if display not in valid_displays:
        raise ValueError(
            "product_reference display must be 'compact' or 'connector'."
        )

    return reference_label, display


def plot_energy_profile(
    states,
    ylabel: str = "ΔG (kcal/mol)",
    n_points: int = 500,
    figsize=(8, 3.5),
    annotate_energies: bool = True,
    decimals: int = 1,
    int_prefix: str = "int",
    label_offset_up: int = 8,
    label_offset_down: int = 12,
    hide_y_ticks: bool = True,
    hide_x_ticks: bool = True,
    hide_spines: bool = True,
    grid: bool = False,
    ax=None,
    dummy_substr: str = "dummy",
    dummy_alpha: float = 0.5,
    side_token: str = "side-rxn",
    show_main_to_product: bool = True,
    main_to_product_alpha: float = 1,
    main_to_product_linestyle: str = ":",
    main_to_product_lw: float = 3.0,
    main_to_product_bow: float = 0.5,
    main_to_product_drop_frac: float = 0.65,
    main_to_product_drop_points: int | None = None,
    main_to_product_flat_points: int | None = None,
    product_x_offset: float = 0.18,
    # --- multi-molecule overlays ---
    overlay: str = "auto",  # "auto" | "off" | "on"
    overlay_annotate: str = "energy",  # "none" | "energy" | "full"
    overlay_alpha: float = 0.35,
    overlay_lw_scale: float = 1.0,
    marker: str = "o",
    overlay_markers=None,
    show_legend: bool = True,
    profile_label: str | None = None,
    overlay_colors=None,
    same_energy_tol: float = 1e-3,
    same_energy_mode: str = "hide",  # "hide" | "tag" | "show"
    same_energy_tag: str = "≡",
    # --- NEW: bottom state labels (recommended for overlays) ---
    show_state_labels: bool | None = None,
    state_label_rotation: float = 0.0,
    font_size: float | None = None,
    state_label_fontsize: float | None = None,
    energy_fontsize: float | None = None,
    axis_label_fontsize: float | None = None,
    tick_label_fontsize: float | None = None,
    legend_fontsize: float | None = None,
    same_energy_tag_fontsize: float | None = None,
    state_label_pad: float = 6.0,
    product_reference: str | tuple[str, str] | None = None,
):
    """Plot one or more reaction energy profiles.

    Parameters
    ----------
    states
        Single profile as a sequence of ``(label, energy[, placement])`` entries,
        or multiple profiles as a mapping/list of ``(profile_name, states)``.
        A string marker such as ``"side-rxn@int2@0.6#Side product"`` starts a
        side-reaction segment. Place ``"main-to-product@0.8"`` between the last
        continuous-path state and the final Product to draw the existing dotted
        connector without adding a side reaction. Its optional fraction
        overrides `main_to_product_drop_frac` for that profile. End a profile
        with ``"no-product"`` when its Product energy is intentionally unknown;
        that profile stops at its preceding state and skips Product references.
    ylabel
        Y-axis label.
    n_points
        Number of interpolation points used for smooth profile curves.
    figsize
        Figure size used when `ax` is not provided.
    annotate_energies
        Whether to annotate energies for the reference profile.
    decimals
        Number of decimal places shown in energy labels.
    int_prefix
        Label prefix treated as an intermediate for default label placement.
    label_offset_up, label_offset_down
        Point offsets used for top and bottom annotations.
    hide_y_ticks, hide_x_ticks, hide_spines
        Axis cleanup options.
    grid
        Whether to show the Matplotlib grid.
    ax
        Existing Matplotlib axes. If omitted, a new figure and axes are created.
    dummy_substr
        Substring used to detect dummy states that should render with reduced
        annotation alpha.
    dummy_alpha
        Alpha multiplier for dummy-state annotations.
    side_token
        String token that starts side-reaction parsing.
    show_main_to_product
        Whether to draw the dotted main-path connection to the product after a
        side-reaction branch.
    main_to_product_alpha, main_to_product_linestyle, main_to_product_lw
        Style controls for the main-to-product connector.
    main_to_product_drop_frac
        Fraction of the connector x-distance kept flat before dropping to the
        product energy.
    main_to_product_drop_points, main_to_product_flat_points
        Optional explicit point counts for the connector segments.
    product_x_offset
        Horizontal spacing between multiple product-like states.
    overlay
        Overlay mode: ``"auto"``, ``"off"``, or ``"on"``.
    overlay_annotate
        Annotation mode for overlay profiles: ``"none"``, ``"energy"``, or
        ``"full"``.
    overlay_alpha, overlay_lw_scale
        Alpha and line-width scaling for overlay profiles.
    marker, overlay_markers
        Marker style for reference and overlay points.
    show_legend
        Whether to draw a legend.
    profile_label
        Legend label for a single profile.
    overlay_colors
        Optional overlay color mapping or sequence. A two-item tuple sets
        ``(main_color, side_color)``.
    same_energy_tol, same_energy_mode, same_energy_tag
        Controls matching overlay energies. ``"hide"`` suppresses overlay
        energy labels that match the reference profile at the same state,
        ``"tag"`` suppresses those labels and marks the reference label with
        `same_energy_tag`, and ``"show"`` annotates matching overlay energies.
    show_state_labels, state_label_rotation, state_label_pad
        X-axis state-label controls.
    font_size, state_label_fontsize, energy_fontsize, axis_label_fontsize,
    tick_label_fontsize, legend_fontsize, same_energy_tag_fontsize
        Font-size controls. Specific values override `font_size`.
    product_reference
        Optional additional reference for the first product-like state. Pass a
        state label such as ``"Cat"`` for a compact stacked annotation, or a
        ``(state_label, display)`` tuple. ``display="compact"`` stacks
        ``Product - Cat`` beneath the original Product value;
        ``display="connector"`` draws a vertical dotted connector to that value at
        its referenced y-position. Later product-like states such as
        ``"Product + int2"`` are unchanged.

    Returns
    -------
    tuple
        ``(fig, ax)`` for the Matplotlib figure and axes.

    Examples
    --------
    Show a Product energy relative to both Dimer and Cat::

        states = [
            ("Dimer", 0.0),
            ("Cat", 5.6),
            ("Product", 3.3),
        ]
        fig, ax = plot_energy_profile(
            states,
            product_reference=("Cat", "connector"),
        )

    The Product point remains at 3.3 while a dotted connector ends at -2.3,
    the Product energy relative to Cat.
    """
    base_fontsize = 12.0 if font_size is None else float(font_size)
    state_label_fontsize = (
        base_fontsize
        if state_label_fontsize is None
        else float(state_label_fontsize)
    )
    energy_fontsize = (
        base_fontsize
        if energy_fontsize is None
        else float(energy_fontsize)
    )
    axis_label_fontsize = (
        base_fontsize
        if axis_label_fontsize is None
        else float(axis_label_fontsize)
    )
    tick_label_fontsize = (
        base_fontsize
        if tick_label_fontsize is None
        else float(tick_label_fontsize)
    )
    legend_fontsize = (
        base_fontsize
        if legend_fontsize is None
        else float(legend_fontsize)
    )
    same_energy_tag_fontsize = (
        energy_fontsize
        if same_energy_tag_fontsize is None
        else float(same_energy_tag_fontsize)
    )

    valid_same_energy_modes = {"hide", "tag", "show"}
    if same_energy_mode not in valid_same_energy_modes:
        allowed = "', '".join(["hide", "tag", "show"])
        raise ValueError(f"same_energy_mode must be one of '{allowed}'.")

    product_reference_label, product_reference_display = (
        _parse_product_reference_spec(product_reference)
    )
    product_reference_key = (
        None
        if product_reference_label is None
        else _norm_label(product_reference_label)
    )

    def _plot_one(
        profile_name,
        profile_states,
        ax_,
        is_reference,
        ref_x_map,
        ref_prod_xs,
        ref_energy_map,
        overlay_idx,
        color_pair=None,
    ):
        parsed = parse_profile(profile_states, side_token)
        entries = parsed.entries
        seg_ids = parsed.segment_ids
        side_anchor_label = parsed.side_anchor_label
        side_connector_rise_frac = parsed.side_connector_rise_frac
        side_legend_label = parsed.side_legend_label
        main_product_anchor_idx = parsed.main_to_product_anchor_idx
        local_main_product_drop_frac = parsed.main_to_product_drop_frac
        no_product = parsed.no_product

        names = [e[0] for e in entries]
        E = np.array([e[1] for e in entries], dtype=float)

        if is_reference or not ref_x_map:
            x = _compute_x_single(entries, product_x_offset)
        else:
            x = _compute_x_from_reference(
                entries,
                ref_x_map,
                ref_prod_xs,
                product_x_offset,
            )

        profile_energy_map = _build_energy_map(entries)

        product_indices = [i for i, lab in enumerate(names) if _is_product(lab)]
        main_product_idx = product_indices[0] if product_indices else (len(entries) - 1)
        side_product_idx = (
            product_indices[1] if len(product_indices) >= 2 else main_product_idx
        )

        product_relative_energy = None
        if product_reference_key is not None:
            profile_description = (
                "the profile"
                if profile_name is None
                else f"profile {profile_name!r}"
            )
            if not product_indices:
                if not no_product:
                    raise ValueError(
                        f"product_reference={product_reference_label!r} requires a "
                        f"product-like state in {profile_description}. End an "
                        "intentionally incomplete profile with 'no-product'."
                    )
            elif product_reference_key not in profile_energy_map:
                available = ", ".join(str(name) for name in names)
                raise ValueError(
                    f"product_reference={product_reference_label!r} was not found in "
                    f"{profile_description}. Available labels: {available}."
                )
            else:
                product_relative_energy = (
                    float(E[main_product_idx])
                    - float(profile_energy_map[product_reference_key])
                )

        def _annotation_matches_reference(entry_idx, energy, key):
            if is_reference or ref_energy_map is None:
                return False

            ref_e = ref_energy_map.get(key)
            matches_reference = (
                ref_e is not None
                and abs(float(energy) - float(ref_e)) <= float(same_energy_tol)
            )
            if (
                matches_reference
                and product_relative_energy is not None
                and entry_idx == main_product_idx
            ):
                ref_product_relative_energy = (
                    float(ref_e) - float(ref_energy_map[product_reference_key])
                )
                matches_reference = (
                    abs(product_relative_energy - ref_product_relative_energy)
                    <= float(same_energy_tol)
                )

            return matches_reference

        side_start_idx = None
        for i, sid in enumerate(seg_ids):
            if sid == 1:
                side_start_idx = i
                break

        if color_pair is None:
            main_color, side_color = _resolve_colors(
                overlay_colors,
                profile_name,
                is_reference,
                overlay_idx,
                side_start_idx is not None,
            )
        else:
            main_color, side_color = color_pair
        point_colors = [main_color] * len(entries)
        a = 1.0 if is_reference else float(overlay_alpha)
        lw = (1.5 * float(overlay_lw_scale)) if not is_reference else 1.5
        z_line = 5 if is_reference else 3
        z_scatter = 6 if is_reference else 4
        z_conn = 2.5 if is_reference else 2.0
        legend_marker = marker if is_reference else (
            overlay_markers.get(profile_name, marker)
            if isinstance(overlay_markers, dict)
            else marker
        )
        side_legend_meta = (
            {
                "profile_name": None if profile_name is None else str(profile_name),
                "label": str(side_legend_label) if side_legend_label is not None else None,
                "color": side_color,
                "alpha": a,
                "marker": legend_marker,
            }
            if side_start_idx is not None
            else None
        )
        profile_legend_meta = {
            "profile_name": None if profile_name is None else str(profile_name),
            "color": main_color,
            "alpha": a,
            "marker": legend_marker,
        }

        def _draw_main_product_connector(
            anchor_idx,
            product_idx,
            connector_color,
            drop_frac,
        ):
            x0 = float(x[anchor_idx])
            y0 = float(E[anchor_idx])
            x1 = float(x[product_idx])
            y1 = float(E[product_idx])

            frac = min(max(float(drop_frac), 0.0), 1.0)
            x_drop = x0 + frac * (x1 - x0)

            n_flat = (
                int(main_to_product_flat_points)
                if main_to_product_flat_points is not None
                else max(20, int(n_points * 0.15))
            )
            n_drop = (
                int(main_to_product_drop_points)
                if main_to_product_drop_points is not None
                else max(80, int(n_points * 0.35))
            )

            xs_flat = np.linspace(x0, x_drop, max(2, n_flat), endpoint=False)
            ys_flat = np.full_like(xs_flat, y0, dtype=float)

            xs_drop = np.linspace(x_drop, x1, max(2, n_drop))
            denom = x1 - x_drop
            if denom == 0:
                ys_drop = np.full_like(xs_drop, y1, dtype=float)
            else:
                t = (xs_drop - x_drop) / denom
                t = np.clip(t, 0.0, 1.0)
                s = t * t * (3.0 - 2.0 * t)
                ys_drop = y0 + (y1 - y0) * s

            ax_.plot(
                np.concatenate([xs_flat, xs_drop]),
                np.concatenate([ys_flat, ys_drop]),
                linestyle=main_to_product_linestyle,
                linewidth=main_to_product_lw,
                alpha=main_to_product_alpha * a,
                marker="",
                color=connector_color,
                zorder=z_conn,
            )

        if side_start_idx is None:
            continuous_end_idx = (
                len(entries) - 1
                if main_product_anchor_idx is None
                else main_product_anchor_idx
            )
            x_continuous = x[: continuous_end_idx + 1]
            E_continuous = E[: continuous_end_idx + 1]
            if len(x_continuous) >= 2:
                x_i, E_i = _dedup_for_interp(x_continuous, E_continuous)
                xs = np.linspace(x_i.min(), x_i.max(), int(n_points))
                interp = PchipInterpolator(x_i, E_i)
                Es = interp(xs)

                ax_.plot(
                    xs,
                    Es,
                    marker="",
                    alpha=a,
                    linewidth=lw,
                    color=main_color,
                    zorder=z_line,
                )
            m = marker if is_reference else (
                overlay_markers.get(profile_name, marker)
                if isinstance(overlay_markers, dict)
                else marker
            )
            ax_.scatter(
                x,
                E,
                zorder=z_scatter,
                color=main_color,
                alpha=a,
                marker=m,
                s=30,
            )
            if show_main_to_product and main_product_anchor_idx is not None:
                drop_frac = (
                    main_to_product_drop_frac
                    if local_main_product_drop_frac is None
                    else local_main_product_drop_frac
                )
                _draw_main_product_connector(
                    main_product_anchor_idx,
                    main_product_idx,
                    main_color,
                    drop_frac,
                )
        else:
            if side_start_idx == 0:
                raise ValueError(f"{side_token!r} cannot be the first entry.")

            main_end = side_start_idx - 1

            x_main = x[: main_end + 1]
            E_main = E[: main_end + 1]
            x_main_i, E_main_i = _dedup_for_interp(x_main, E_main)

            xs_main = np.linspace(
                x_main_i.min(), x_main_i.max(), max(2, int(n_points * 0.6))
            )
            interp_main = PchipInterpolator(x_main_i, E_main_i)
            Es_main = interp_main(xs_main)

            ax_.plot(
                xs_main,
                Es_main,
                marker="",
                alpha=a,
                linewidth=lw,
                color=main_color,
                zorder=z_line
            )
            m = marker if is_reference else (
                overlay_markers.get(profile_name, marker)
                if isinstance(overlay_markers, dict)
                else marker
            )            
            ax_.scatter(
                x_main,
                E_main,
                zorder=z_scatter,
                color=main_color,
                alpha=a,
                marker=m,
                s=30,
            )

            side_anchor_idx = main_end
            if side_anchor_label is not None:
                target = side_anchor_label.lower().strip()
                for j, (lab, _, _) in enumerate(entries):
                    if _norm_label(lab) == target:
                        side_anchor_idx = j
                        break
                else:
                    raise ValueError(
                        f"side-rxn anchor {side_anchor_label!r} not found among labels."
                    )

            if side_anchor_idx >= side_start_idx:
                raise ValueError(
                    f"side-rxn anchor {side_anchor_label!r} must be before side segment."
                )

            side_idxs = [i for i in range(side_start_idx, len(entries))]
            if main_product_idx in side_idxs and main_product_idx != side_product_idx:
                side_idxs = [i for i in side_idxs if i != main_product_idx]

            if side_product_idx not in side_idxs and side_product_idx >= side_start_idx:
                side_idxs.append(side_product_idx)
                side_idxs = sorted(set(side_idxs))

            for idx in side_idxs:
                point_colors[idx] = side_color
            point_colors[main_product_idx] = main_color

            x_side_main = x[side_idxs]
            E_side_main = E[side_idxs]
            x_side_i, E_side_i = _dedup_for_interp(x_side_main, E_side_main)

            xs_side = np.linspace(
                float(x_side_i.min()),
                float(x_side_i.max()),
                max(2, int(n_points * 0.6)),
            )
            interp_side = PchipInterpolator(x_side_i, E_side_i)
            Es_side = interp_side(xs_side)

            ax_.plot(
                xs_side,
                Es_side,
                marker="",
                alpha=a,
                linewidth=lw,
                color=side_color,
                zorder=z_line
            )
            m = marker if is_reference else (
                overlay_markers.get(profile_name, marker)
                if isinstance(overlay_markers, dict)
                else marker
            )            
            ax_.scatter(
                x_side_main,
                E_side_main,
                zorder=z_scatter,
                color=side_color,
                alpha=a,
                marker=m,
                s=30,
            )

            x0 = float(x[side_anchor_idx])
            y0 = float(E[side_anchor_idx])
            x1c = float(x[side_start_idx])
            y1c = float(E[side_start_idx])

            frac = (
                0.0
                if side_connector_rise_frac is None
                else float(side_connector_rise_frac)
            )
            frac = min(max(frac, 0.0), 1.0)

            x_rise = x0 + frac * (x1c - x0)

            xs_flat = np.linspace(x0, x_rise, 60, endpoint=False)
            ys_flat = np.full_like(xs_flat, y0, dtype=float)

            xs_rise = np.linspace(x_rise, x1c, 120)
            denom = (x1c - x_rise)
            if denom == 0:
                ys_rise = np.full_like(xs_rise, y1c, dtype=float)
            else:
                t = (xs_rise - x_rise) / denom
                t = np.clip(t, 0.0, 1.0)
                s = t * t * (3.0 - 2.0 * t)
                ys_rise = y0 + (y1c - y0) * s

            xs_conn = np.concatenate([xs_flat, xs_rise])
            ys_conn = np.concatenate([ys_flat, ys_rise])

            ax_.plot(
                xs_conn,
                ys_conn,
                linestyle=":",
                linewidth=3.0,
                alpha=a,
                marker="",
                color=side_color,
                zorder=z_conn
            )

            if show_main_to_product and len(x) >= 2:
                mp_color = "C0" if is_reference else main_color
                _draw_main_product_connector(
                    main_end,
                    main_product_idx,
                    mp_color,
                    main_to_product_drop_frac,
                )
                m = marker if is_reference else (
                    overlay_markers.get(profile_name, marker)
                    if isinstance(overlay_markers, dict)
                    else marker
                )                
                ax_.scatter(
                    [float(x[main_product_idx])],
                    [float(E[main_product_idx])],
                    zorder=z_scatter,
                    color=mp_color,
                    alpha=a,
                    marker=m,
                    s=30,
                )

        # --- Energy annotations (labels are handled on x-axis if enabled) ---
        # --- Annotations ---
        if is_reference:
            do_annotate = bool(annotate_energies)
        else:
            do_annotate = overlay_annotate in {"energy", "full"}

        product_annotation_is_hidden = (
            product_relative_energy is not None
            and _annotation_matches_reference(
                main_product_idx,
                E[main_product_idx],
                _norm_label(names[main_product_idx]),
            )
            and same_energy_mode in {"hide", "tag"}
        )
        if (
            do_annotate
            and product_reference_display == "connector"
            and product_relative_energy is not None
            and not product_annotation_is_hidden
        ):
            product_x = float(x[main_product_idx])
            product_energy = float(E[main_product_idx])
            product_color = point_colors[main_product_idx]
            ax_.plot(
                [product_x, product_x],
                [product_energy, product_relative_energy],
                linestyle=":",
                linewidth=1.5,
                alpha=a,
                marker="",
                color=product_color,
                zorder=z_conn,
            )

            relative_is_below = product_relative_energy <= product_energy
            ax_.annotate(
                f"({product_relative_energy:.{decimals}f})",
                (product_x, product_relative_energy),
                textcoords="offset points",
                xytext=(0, -5 if relative_is_below else 5),
                ha="center",
                va="top" if relative_is_below else "bottom",
                alpha=a,
                color=product_color,
                fontsize=energy_fontsize,
            )

        if do_annotate:
            for i, (xi, Ei, label) in enumerate(zip(x, E, names), start=1):
                key = _norm_label(label)
                is_dummy = dummy_substr.lower() in key

                matches_reference = _annotation_matches_reference(i - 1, Ei, key)
                if matches_reference and same_energy_mode in {"hide", "tag"}:
                    continue

                placement_counts = _parse_placement(entries[i - 1][2])
                if placement_counts is None:
                    is_int = key.startswith(int_prefix.lower())
                    if i == 1:
                        placement_counts = {"left": 1, "right": 0, "top": 0, "bottom": 0}
                    elif i == len(entries):
                        placement_counts = {"right": 1, "left": 0, "top": 0, "bottom": 0}
                    elif is_int:
                        placement_counts = {"bottom": 1, "top": 0, "left": 0, "right": 0}
                    else:
                        placement_counts = {"top": 1, "bottom": 0, "left": 0, "right": 0}

                txt_color = point_colors[i - 1]
                alpha = 1.0 if is_reference else float(overlay_alpha)

                if not multi:
                    # SINGLE-MOLECULE: restore original label+energy annotations
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

                    if annotate_energies:
                        energy_lines = [f"{Ei:.{decimals}f}"]
                        if (
                            product_relative_energy is not None
                            and i - 1 == main_product_idx
                            and product_reference_display == "compact"
                        ):
                            energy_lines.append(
                                f"({product_relative_energy:.{decimals}f})"
                            )
                        text = f"{label}\n" + "\n".join(energy_lines)
                    else:
                        text = f"{label}"

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
                        color=txt_color,
                        fontsize=energy_fontsize,
                    )
                else:
                    # MULTI-MOLECULE: energy-only (state names are on x-axis)
                    _annotate_energy_only(
                        ax_=ax_,
                        xi=float(xi),
                        Ei=float(Ei),
                        alpha=alpha,
                        color=txt_color,
                        placement_counts=placement_counts,
                        is_dummy=is_dummy,
                        decimals=decimals,
                        label_offset_up=label_offset_up,
                        label_offset_down=label_offset_down,
                        dummy_alpha=dummy_alpha,
                        energy_fontsize=energy_fontsize,
                        additional_energy=(
                            product_relative_energy
                            if (
                                i - 1 == main_product_idx
                                and product_reference_display == "compact"
                            )
                            else None
                        ),
                    )

        if is_reference:
            x_map = {}
            prod_xs = []
            ordered = []
            for xi, lab in zip(x, names):
                k = _norm_label(lab)
                x_map[k] = float(xi)
                ordered.append((float(xi), str(lab)))
                if _is_product(lab):
                    prod_xs.append(float(xi))
            return (
                x_map,
                prod_xs,
                profile_energy_map,
                ordered,
                side_legend_meta,
                profile_legend_meta,
            )

        return (
            None,
            None,
            profile_energy_map,
            None,
            side_legend_meta,
            profile_legend_meta,
        )

    # ---- Detect multi-profile input (no breaking of current list input) ----
    multi = False
    profiles = None

    if isinstance(states, dict):
        profiles = list(states.items())
        multi = True
    elif isinstance(states, (list, tuple)) and states:
        first = states[0]
        if (
            isinstance(first, (list, tuple))
            and len(first) == 2
            and isinstance(first[0], str)
            and isinstance(first[1], (list, tuple))
        ):
            profiles = list(states)
            multi = True

    if overlay == "off":
        multi = False
        profiles = None
    elif overlay == "on":
        if not multi:
            raise ValueError("overlay='on' requires dict or list-of-(name, states).")

    if show_state_labels is None:
        show_state_labels = bool(multi)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ref_x_map = {}
    ref_prod_xs = []
    ref_energy_map = None
    ref_ordered = None
    side_legend_metas = []
    profile_legend_metas = []

    if not multi:
        _, _, _, _, side_meta, _ = _plot_one(
            profile_name=None,
            profile_states=states,
            ax_=ax,
            is_reference=True,
            ref_x_map=ref_x_map,
            ref_prod_xs=ref_prod_xs,
            ref_energy_map=None,
            overlay_idx=0,
        )
        if side_meta is not None:
            side_legend_metas.append(side_meta)
    else:
        profile_color_pairs = _plan_profile_colors(
            profiles,
            side_token,
            overlay_colors,
        )
        ref_name, ref_states = profiles[0]
        (
            ref_x_map,
            ref_prod_xs,
            ref_energy_map,
            ref_ordered,
            side_meta,
            profile_meta,
        ) = _plot_one(
            profile_name=ref_name,
            profile_states=ref_states,
            ax_=ax,
            is_reference=True,
            ref_x_map=ref_x_map,
            ref_prod_xs=ref_prod_xs,
            ref_energy_map=None,
            overlay_idx=0,
            color_pair=profile_color_pairs[0],
        )
        profile_legend_metas.append(profile_meta)
        if side_meta is not None:
            side_legend_metas.append(side_meta)

        overlay_energy_maps = []
        for k, (name, st) in enumerate(profiles[1:], start=0):
            _, _, e_map, _, side_meta, profile_meta = _plot_one(
                profile_name=name,
                profile_states=st,
                ax_=ax,
                is_reference=False,
                ref_x_map=ref_x_map,
                ref_prod_xs=ref_prod_xs,
                ref_energy_map=ref_energy_map,
                overlay_idx=k,
                color_pair=profile_color_pairs[k + 1],
            )
            overlay_energy_maps.append(e_map)
            profile_legend_metas.append(profile_meta)
            if side_meta is not None:
                side_legend_metas.append(side_meta)

        ref_main_product_key = next(
            (
                _norm_label(label)
                for _, label in ref_ordered
                if _is_product(label)
            ),
            None,
        )

        if same_energy_mode == "tag" and annotate_energies and ref_energy_map is not None:
            for key, ref_e in ref_energy_map.items():
                matched = False
                for om in overlay_energy_maps:
                    oe = om.get(key) if om is not None else None
                    if oe is None:
                        continue
                    energies_match = (
                        abs(float(oe) - float(ref_e)) <= float(same_energy_tol)
                    )
                    if (
                        energies_match
                        and product_reference_key is not None
                        and key == ref_main_product_key
                    ):
                        ref_relative_energy = (
                            float(ref_e)
                            - float(ref_energy_map[product_reference_key])
                        )
                        overlay_relative_energy = (
                            float(oe) - float(om[product_reference_key])
                        )
                        energies_match = (
                            abs(ref_relative_energy - overlay_relative_energy)
                            <= float(same_energy_tol)
                        )
                    if energies_match:
                        matched = True
                        break
                if not matched:
                    continue

                xi = float(ref_x_map[key])
                yi = float(ref_e)
                ax.annotate(
                    same_energy_tag,
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(8, 0),
                    ha="left",
                    va="center",
                    alpha=1.0,
                    color="C0",
                    fontsize=same_energy_tag_fontsize,
                )

        if show_legend:
            handles = []
            labels = []
            for meta in profile_legend_metas:
                name = meta["profile_name"]
                if name is None:
                    continue

                h = plt.Line2D(
                    [0],
                    [0],
                    color=meta["color"],
                    alpha=meta["alpha"],
                    marker=meta["marker"],
                    linestyle="-",
                )
                handles.append(h)
                labels.append(str(name))
            for meta in side_legend_metas:
                label = meta["label"]
                if label is None:
                    continue

                style_meta = _style_meta_for_side_label(
                    label,
                    meta,
                    side_legend_metas,
                )
                h = plt.Line2D(
                    [0],
                    [0],
                    color=style_meta["color"],
                    alpha=style_meta["alpha"],
                    marker=style_meta["marker"],
                    linestyle="-",
                )
                handles.append(h)
                labels.append(label)
            if handles:
                ax.legend(handles, labels, frameon=False, fontsize=legend_fontsize)

    if not multi and show_legend:
        handles = []
        labels = []

        if profile_label is not None:
            h = plt.Line2D(
                [0],
                [0],
                color="C0",
                alpha=1.0,
                marker=marker,
                linestyle="-",
            )
            handles.append(h)
            labels.append(str(profile_label))

        for meta in side_legend_metas:
            label = meta["label"]
            if label is None:
                continue

            h = plt.Line2D(
                [0],
                [0],
                color=meta["color"],
                alpha=meta["alpha"],
                marker=meta["marker"],
                linestyle="-",
            )
            handles.append(h)
            labels.append(label)

        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=legend_fontsize)

    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

    # --- Bottom labels (states) ---
    if show_state_labels:
        # Use reference ordering if available (multi); otherwise derive from single.
        if ref_ordered is None:
            entries = parse_profile(states, side_token).entries
            x_single = _compute_x_single(entries, product_x_offset)
            ref_ordered = [(float(xi), str(lab)) for xi, (lab, _, _) in zip(x_single, entries)]

        # If there are duplicated x (multiple products), matplotlib will still
        # accept them; labels may overlap, but the offsets typically separate them.
        xs = [p[0] for p in ref_ordered]
        labs = [p[1] for p in ref_ordered]

        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=state_label_rotation,
                           fontsize=state_label_fontsize)
        ax.tick_params(axis="x", pad=state_label_pad)

        hide_x_ticks = False

    # --- Limits ---
    if ref_x_map:
        xmin = min(ref_x_map.values())
        xmax = max(ref_x_map.values())
    else:
        xmin, xmax = ax.get_xlim()

    left_pad = 1.05
    right_pad = 0.8
    ax.set_xlim(xmin - left_pad, xmax + right_pad)

    ax.grid(bool(grid))
    ax.set_facecolor("white")

    if hide_x_ticks:
        ax.set_xticks([])
    elif not show_state_labels:
        ax.tick_params(axis="x", labelsize=tick_label_fontsize)

    if hide_y_ticks:
        ax.set_yticks([])
    else:
        ax.tick_params(axis="y", labelsize=tick_label_fontsize)

    if hide_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if created_fig:
        fig.tight_layout()

    return fig, ax
