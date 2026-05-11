from .parsing import _norm_label


def _resolve_colors(
    overlay_colors,
    profile_name,
    is_reference,
    overlay_idx,
    needs_side: bool,
):
    if overlay_colors is not None and not is_reference:
        if isinstance(overlay_colors, dict):
            spec = overlay_colors.get(profile_name)
            if spec is not None:
                if isinstance(spec, (tuple, list)) and len(spec) == 2:
                    return spec[0], spec[1]
                return spec, spec

        elif isinstance(overlay_colors, (list, tuple)):
            if overlay_idx < len(overlay_colors):
                spec = overlay_colors[overlay_idx]
                if isinstance(spec, (tuple, list)) and len(spec) == 2:
                    return spec[0], spec[1]
                return spec, spec

    if is_reference:
        return "C0", "C1"

    base_idx = 2 * overlay_idx + 1
    main_color = f"C{base_idx % 10}"
    side_color = f"C{(base_idx + 1) % 10}" if needs_side else main_color
    return main_color, side_color


def _style_meta_for_side_label(label, default_meta, side_metas):
    label_norm = _norm_label(label)
    matches = []
    for meta in side_metas:
        profile_name = meta.get("profile_name")
        if profile_name is None:
            continue
        profile_norm = _norm_label(profile_name)
        if profile_norm and profile_norm in label_norm:
            matches.append((len(profile_norm), meta))

    if not matches:
        return default_meta

    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]
