from .models import ParsedProfile


def _parse_placement(value):
    if value is None:
        return None

    s = str(value).lower().strip()
    if not s:
        return None

    aliases = {
        "t": "top",
        "top": "top",
        "b": "bottom",
        "bottom": "bottom",
        "l": "left",
        "left": "left",
        "r": "right",
        "right": "right",
    }

    parts = [p for p in s.replace("_", "-").replace(" ", "-").split("-") if p]

    expanded = []
    for p in parts:
        p = p.strip().lower()
        if p and all(ch in {"t", "b", "l", "r"} for ch in p) and p not in aliases:
            expanded.extend(list(p))
        else:
            expanded.append(p)

    parts = [aliases.get(p, p) for p in expanded]

    allowed = {"top", "bottom", "left", "right"}
    if any(p not in allowed for p in parts):
        return None

    counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    for p in parts:
        counts[p] += 1

    if counts["top"] and counts["bottom"]:
        return None
    if counts["left"] and counts["right"]:
        return None
    if sum(counts.values()) == 0:
        return None

    return counts


def _norm_label(label):
    return str(label).strip().lower()


def _is_product(label):
    return _norm_label(label).startswith("product")


def parse_profile(
    profile_states,
    side_token: str = "side-rxn",
    main_to_product_token: str = "main-to-product",
    no_product_token: str = "no-product",
) -> ParsedProfile:
    """Parse raw energy-profile input into normalized entries.

    Parameters
    ----------
    profile_states
        Sequence of state tuples and optional side-reaction marker strings.
    side_token
        Marker token that starts a side-reaction segment.
    main_to_product_token
        Marker token that replaces the final continuous segment with the
        main-to-product dotted connector. An optional ``@<fraction>`` suffix
        sets the flat fraction locally.
    no_product_token
        Final marker indicating that the profile intentionally has no Product
        state or Product connector.

    Returns
    -------
    ParsedProfile
        Parsed entries with segment identifiers and side-reaction metadata.
    """
    profile_states = list(profile_states)
    entries = []
    seg_ids = []
    seg = 0
    token = str(side_token).lower().strip()
    main_product_token = str(main_to_product_token).lower().strip()
    no_product_marker = str(no_product_token).lower().strip()

    side_anchor_label = None
    side_connector_rise_frac = None
    embedded_side_label = None
    main_to_product_anchor_idx = None
    main_to_product_drop_frac = None
    no_product = False

    for item_idx, item in enumerate(profile_states):
        if isinstance(item, str):
            side_spec, legend_spec = (
                item.split("#", 1)
                if "#" in item
                else (item, None)
            )
            parsed_legend = (
                legend_spec.strip()
                if legend_spec is not None and legend_spec.strip()
                else None
            )
            s = side_spec.lower().strip()

            if s == no_product_marker:
                if item_idx != len(profile_states) - 1:
                    raise ValueError(
                        f"{no_product_token!r} must be the final entry in its "
                        "profile."
                    )
                if not entries:
                    raise ValueError(f"{no_product_token!r} cannot be the first entry.")
                if parsed_legend is not None:
                    raise ValueError(
                        f"{no_product_token!r} does not accept a legend suffix."
                    )
                if seg == 1 or main_to_product_anchor_idx is not None:
                    raise ValueError(
                        f"{no_product_token!r} cannot be combined with "
                        f"{side_token!r} or {main_to_product_token!r} in the "
                        "same profile."
                    )
                no_product = True
                continue

            if s == main_product_token or s.startswith(main_product_token + "@"):
                if main_to_product_anchor_idx is not None:
                    raise ValueError(
                        f"Only one {main_to_product_token!r} marker is allowed "
                        "per profile."
                    )
                if not entries:
                    raise ValueError(
                        f"{main_to_product_token!r} cannot be the first entry."
                    )
                if parsed_legend is not None:
                    raise ValueError(
                        f"{main_to_product_token!r} does not accept a legend suffix."
                    )

                main_to_product_anchor_idx = len(entries) - 1
                main_to_product_drop_frac = None
                if s.startswith(main_product_token + "@"):
                    fraction_text = side_spec.split("@", 1)[1].strip()
                    try:
                        main_to_product_drop_frac = float(fraction_text)
                    except ValueError as exc:
                        raise ValueError(
                            f"{main_to_product_token!r} requires a numeric flat "
                            "fraction, such as 'main-to-product@0.8'."
                        ) from exc
                    if not 0.0 <= main_to_product_drop_frac <= 1.0:
                        raise ValueError(
                            f"{main_to_product_token!r} flat fraction must be "
                            "between 0 and 1."
                        )
                continue

            if s == token:
                embedded_side_label = parsed_legend
                seg = 1
                continue

            if s.startswith(token + "@") or s.startswith(token + ":"):
                rest = (
                    side_spec.split("@", 1)[1]
                    if "@" in side_spec
                    else side_spec.split(":", 1)[1]
                )
                parts = [p.strip() for p in str(rest).split("@") if p.strip()]

                side_anchor_label = parts[0] if len(parts) >= 1 else None
                side_connector_rise_frac = None
                if len(parts) >= 2:
                    side_connector_rise_frac = float(parts[1])

                embedded_side_label = parsed_legend
                seg = 1
                continue

            raise ValueError(
                f"Unknown string entry in states: {item!r}. "
                f"Only {side_token!r} (optionally with @{'<label>'} and "
                f"a #legend suffix) or {main_to_product_token!r} "
                f"(optionally with @<fraction>) or {no_product_token!r} is "
                "supported."
            )

        label = item[0]
        energy = item[1]
        placement = item[2] if len(item) >= 3 else None

        entries.append((label, energy, placement))
        seg_ids.append(seg)

    if no_product and any(_is_product(entry[0]) for entry in entries):
        raise ValueError(
            f"{no_product_token!r} cannot be used when the profile already "
            "contains a product-like state."
        )

    if main_to_product_anchor_idx is not None:
        target_idx = main_to_product_anchor_idx + 1
        if target_idx >= len(entries) or not _is_product(entries[target_idx][0]):
            raise ValueError(
                f"{main_to_product_token!r} must be followed directly by a "
                "product-like state."
            )
        if target_idx != len(entries) - 1:
            raise ValueError(
                f"The product following {main_to_product_token!r} must be the "
                "final state in that profile."
            )
        if any(segment_id == 1 for segment_id in seg_ids):
            raise ValueError(
                f"{main_to_product_token!r} cannot be combined with "
                f"{side_token!r} in the same profile."
            )

    return ParsedProfile(
        entries=entries,
        segment_ids=seg_ids,
        side_anchor_label=side_anchor_label,
        side_connector_rise_frac=side_connector_rise_frac,
        side_legend_label=embedded_side_label,
        main_to_product_anchor_idx=main_to_product_anchor_idx,
        main_to_product_drop_frac=main_to_product_drop_frac,
        no_product=no_product,
    )


def _parse_entries(profile_states, side_token: str = "side-rxn"):
    parsed = parse_profile(profile_states, side_token)
    return (
        parsed.entries,
        parsed.segment_ids,
        parsed.side_anchor_label,
        parsed.side_connector_rise_frac,
        parsed.side_legend_label,
    )
