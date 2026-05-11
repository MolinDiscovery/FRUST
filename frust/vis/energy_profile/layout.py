import numpy as np

from .parsing import _is_product, _norm_label


def _dedup_for_interp(xv, yv):
    seen = set()
    x_out = []
    y_out = []
    for xx, yy in zip(xv, yv):
        key = float(xx)
        if key in seen:
            continue
        seen.add(key)
        x_out.append(float(xx))
        y_out.append(float(yy))
    return np.array(x_out, dtype=float), np.array(y_out, dtype=float)


def _compute_x_single(entries, product_x_offset: float):
    names = [e[0] for e in entries]

    x_list = []
    product_indices = []
    for i, label in enumerate(names):
        if _is_product(label):
            product_indices.append(i)

    n_prod = len(product_indices)
    base_x = float(product_indices[0]) if n_prod else None

    product_rank = {idx: k for k, idx in enumerate(product_indices)}

    for i, label in enumerate(names):
        if _is_product(label) and n_prod:
            k = product_rank[i]
            shift = (k - (n_prod - 1) / 2.0) * float(product_x_offset)
            x_list.append(base_x + shift)
        else:
            x_list.append(float(i))

    return np.array(x_list, dtype=float)


def _compute_x_from_reference(entries, ref_x_map, ref_prod_xs, product_x_offset: float):
    names = [e[0] for e in entries]
    x_list = []

    prod_labels = [lab for lab in names if _is_product(lab)]
    prod_rank = {lab: k for k, lab in enumerate(prod_labels)}
    n_prod = len(prod_labels)

    if ref_prod_xs:
        base_prod_x = float(np.mean(ref_prod_xs))
    else:
        base_prod_x = float(max(ref_x_map.values())) if ref_x_map else 0.0

    for lab in names:
        key = _norm_label(lab)
        if key in ref_x_map:
            x_list.append(float(ref_x_map[key]))
            continue

        if _is_product(lab) and n_prod:
            k = prod_rank[lab]
            shift = (k - (n_prod - 1) / 2.0) * float(product_x_offset)
            x_list.append(base_prod_x + shift)
            continue

        raise ValueError(
            "Overlay mode: label not found in reference profile: "
            f"{lab!r}. Either add it to the reference profile, or "
            "turn overlay='off'."
        )

    return np.array(x_list, dtype=float)


def _build_energy_map(entries):
    out = {}
    for lab, en, _ in entries:
        out[_norm_label(lab)] = float(en)
    return out
