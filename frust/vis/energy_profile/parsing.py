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


def parse_profile(profile_states, side_token: str = "side-rxn") -> ParsedProfile:
    """Parse raw energy-profile input into normalized entries.

    Parameters
    ----------
    profile_states
        Sequence of state tuples and optional side-reaction marker strings.
    side_token
        Marker token that starts a side-reaction segment.

    Returns
    -------
    ParsedProfile
        Parsed entries with segment identifiers and side-reaction metadata.
    """
    entries = []
    seg_ids = []
    seg = 0
    token = str(side_token).lower().strip()

    side_anchor_label = None
    side_connector_rise_frac = None
    embedded_side_label = None

    for item in profile_states:
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
                "a #legend suffix) is supported."
            )

        label = item[0]
        energy = item[1]
        placement = item[2] if len(item) >= 3 else None

        entries.append((label, energy, placement))
        seg_ids.append(seg)

    return ParsedProfile(
        entries=entries,
        segment_ids=seg_ids,
        side_anchor_label=side_anchor_label,
        side_connector_rise_frac=side_connector_rise_frac,
        side_legend_label=embedded_side_label,
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
