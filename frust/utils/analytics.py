import re
import pandas as pd
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Geometry import rdGeometry
import html
import math
import base64
import io
import zipfile
import cairosvg

def summarize_ts_vibrations(
    df: pd.DataFrame,
    col: str = "DFT-wB97X-D3-6-31G**-OptTS-vibs",
    max_rows: int = 5
):
    true_ts_count = 0
    non_ts_count = 0
    rows = []

    for idx, row in df.iterrows():
        ligand = row.get("ligand_name", "")
        rpos   = row.get("rpos", "")
        vibs   = row[col]

        freqs = [entry.get('frequency') for entry in vibs]
        neg_freqs = [f for f in freqs if f < 0]
        pos_freqs = [f for f in freqs if f >= 0]

        is_true_ts = len(neg_freqs) == 1
        status = "✅ True TS" if is_true_ts else f"❌ Not TS ({len(neg_freqs)} neg)"

        if is_true_ts:
            true_ts_count += 1
        else:
            non_ts_count += 1

        if neg_freqs:
            neg_str = ", ".join(f"{f:.2f}" for f in neg_freqs[:3])
            if len(neg_freqs) > 3:
                neg_str += " ..."
        else:
            neg_str = "No negatives"
        neg_str += " |"

        if pos_freqs:
            pos_str = ", ".join(f"{f:.1f}" for f in pos_freqs[:3])
            if len(pos_freqs) > 3:
                pos_str += " ..."
        else:
            pos_str = "No positives"
        pos_str += " |"

        rows.append({
            "Structure": idx,
            "Ligand": ligand,
            "RPOS": rpos,
            "Status": status,
            "Neg. freqs": neg_str,
            "Pos. freqs": pos_str
        })

    result_df = pd.DataFrame(rows)

    print(result_df.head(max_rows).to_string(index=False))
    if len(result_df) > max_rows:
        print(f"\n... and {len(result_df) - max_rows} more rows.")

    print("\nSummary:")
    print(f"  ✅ True TSs : {true_ts_count}")
    print(f"  ❌ Non-TSs  : {non_ts_count}")


# # an old more simple function for annotation
# def _svg_annotated_smi(
#         smi, pos_list, dE_list,
#         size=(250, 250), highlight_color=(1, 0, 0),
#         show_rpos: bool = False, step_list: list[str] | None = None,
#         fixed_bond_px: float | None = 25.0,
#         note_font_px: float | None = None,
#         annotation_scale: float = 1):
#     """Return an SVG string of the molecule with per-atom ΔE labels.

#     Args:
#         smi: SMILES string.
#         pos_list: Atom indices to annotate.
#         dE_list: Values to display next to each atom.
#         size: (width, height) of the SVG in px.
#         highlight_color: Unused unless highlight code is enabled.
#         show_rpos: If True, append ' (rX)' after the value.
#         step_list: Optional per-position step tags (e.g., 'ts1').
#             When provided, labels become like '20.10(ts1)'.
#         fixed_bond_px: If set, draw with a fixed px/bond to keep drawings
#             on a common scale (RDKit may still shrink to fit if canvas is
#             too small).
#         note_font_px: If set, enforce an absolute pixel size for annotation
#             text (via fixedFontSize * annotationFontScale).
#         annotation_scale: Multiplier for annotation size relative to the
#             base label font.
#     """
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None:
#         return ""
#     rdDepictor.Compute2DCoords(mol)

#     for i, (p, e) in enumerate(zip(pos_list, dE_list)):
#         try:
#             p_int = int(p)
#         except (TypeError, ValueError):
#             continue

#         try:
#             val = float(e)
#             note = f"{val:.2f}"
#         except (TypeError, ValueError):
#             note = f"{e}"
    
#         if step_list is not None and i < len(step_list):
#             note += f" [{step_list[i]}]"
#         if show_rpos:
#             note += f" (r{p_int})"

#         mol.GetAtomWithIdx(p_int).SetProp("atomNote", note)

#     drawer = rdMolDraw2D.MolDraw2DSVG(*size)
#     opts = drawer.drawOptions()
#     opts.drawAtomNotes = True

#     # Keep all molecules at the same scale (px per bond).
#     if fixed_bond_px is not None:
#         opts.fixedBondLength = float(fixed_bond_px)

#     # Control annotation size deterministically.
#     # If note_font_px is given, make the final annotation size = note_font_px.
#     scale = float(annotation_scale) if annotation_scale else 0.4
#     opts.annotationFontScale = scale
#     use_svg_patch = False
#     if note_font_px is not None:
#         # final px = annotationFontScale * fixedFontSize
#         base_font_px = max(1, int(round(float(note_font_px) / max(scale, 1e-6))))
#         if hasattr(opts, "fixedFontSize"):
#             opts.fixedFontSize = base_font_px  # <- must be INT
#             # (optional clamps; harmless when fixedFontSize is honored)
#             if hasattr(opts, "minFontSize"):
#                 opts.minFontSize = base_font_px
#             if hasattr(opts, "maxFontSize"):
#                 opts.maxFontSize = base_font_px
#         else:
#             # very old RDKit fallback: patch SVG after drawing
#             use_svg_patch = True

#     drawer.DrawMolecule(mol)
#     drawer.FinishDrawing()
#     svg = drawer.GetDrawingText()

#     if note_font_px is not None and use_svg_patch:
#         px = float(note_font_px)
#         # Patch both style: and attribute-based font declarations.
#         svg = re.sub(r'(font-size\s*:\s*)[\d.]+px', rf'\1{px}px', svg)
#         svg = re.sub(r'(font-size\s*=\s*["\'])[\d.]+(px)?(["\'])',
#                      rf'\1{px}\3', svg)

#     return svg


def _mol_annotated_smi(
    smi,
    pos_list,
    dE_list,
    show_rpos: bool = False,
    step_list: list[str] | None = None,
):
    """Internal: build an RDKit Mol with atomNote props set (no drawing)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    rdDepictor.Compute2DCoords(mol)

    for i, (p, e) in enumerate(zip(pos_list, dE_list)):
        try:
            p_int = int(p)
        except (TypeError, ValueError):
            continue

        try:
            val = float(e)
            note = f"{val:.2f}"
        except (TypeError, ValueError):
            note = f"{e}"

        if step_list is not None and i < len(step_list):
            note += f" [{step_list[i]}]"
        if show_rpos:
            note += f" (r{p_int})"

        try:
            mol.GetAtomWithIdx(p_int).SetProp("atomNote", note)
        except Exception:
            continue

    return mol


# def _svg_annotated_smi(
#     smi,
#     pos_list,
#     dE_list,
#     size=(250, 250),
#     highlight_color=(1, 0, 0),
#     show_rpos: bool = False,
#     step_list: list[str] | None = None,
#     fixed_bond_px: float | None = 25.0,
#     note_font_px: float | None = None,
#     annotation_scale: float = 1,
# ):
#     """Return an SVG string of the molecule with per-atom ΔE labels."""
#     mol = _mol_annotated_smi(
#         smi,
#         pos_list,
#         dE_list,
#         show_rpos=show_rpos,
#         step_list=step_list,
#     )
#     if mol is None:
#         return ""

#     drawer = rdMolDraw2D.MolDraw2DSVG(*size)
#     opts = drawer.drawOptions()
#     opts.drawAtomNotes = True

#     if fixed_bond_px is not None:
#         opts.fixedBondLength = float(fixed_bond_px)

#     scale = float(annotation_scale) if annotation_scale else 0.4
#     opts.annotationFontScale = scale
#     use_svg_patch = False

#     if note_font_px is not None:
#         base_font_px = max(
#             1,
#             int(round(float(note_font_px) / max(scale, 1e-6))),
#         )
#         if hasattr(opts, "fixedFontSize"):
#             opts.fixedFontSize = base_font_px
#             if hasattr(opts, "minFontSize"):
#                 opts.minFontSize = base_font_px
#             if hasattr(opts, "maxFontSize"):
#                 opts.maxFontSize = base_font_px
#         else:
#             use_svg_patch = True

#     drawer.DrawMolecule(mol)
#     drawer.FinishDrawing()
#     svg = drawer.GetDrawingText()

#     if note_font_px is not None and use_svg_patch:
#         px = float(note_font_px)
#         svg = re.sub(r"(font-size\s*:\s*)[\d.]+px", rf"\1{px}px", svg)
#         svg = re.sub(
#             r'(font-size\s*=\s*["\'])[\d.]+(px)?(["\'])',
#             rf"\1{px}\3",
#             svg,
#         )

#     return svg

from rdkit.Geometry import rdGeometry

def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&apos;"))

import math


def _layout_robust_labels(
    drawer,
    mol,
    size: tuple[int, int],
    atom_ids: list[int],
    labels: list[str],
    note_font_px: float | None,
    label_offset_px: float,
    relax_iters: int = 40,
    min_gap_px: float = 2.0,
):
    atom_xy = []
    for idx in atom_ids:
        p = drawer.GetDrawCoords(idx)
        atom_xy.append((float(p.x), float(p.y)))

    if not atom_xy:
        return []

    font_px = float(note_font_px) if note_font_px is not None else 14.0

    # --- NEW: use molecule centroid (all atoms) as "inside" direction ---
    all_xy = []
    for a in mol.GetAtoms():
        p = drawer.GetDrawCoords(a.GetIdx())
        all_xy.append((float(p.x), float(p.y)))

    mol_cx = sum(x for x, _ in all_xy) / len(all_xy)
    mol_cy = sum(y for _, y in all_xy) / len(all_xy)
    # -------------------------------------------------------------------

    def _angle(ax: float, ay: float, bx: float, by: float) -> float:
        a = math.atan2(by - ay, bx - ax)
        if a < 0:
            a += 2 * math.pi
        return a

    def _ang_dist(a: float, b: float) -> float:
        d = abs(a - b) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

    def _best_dir(atom_idx: int, ax: float, ay: float) -> tuple[float, float]:
        # Neighbor bond angles (occupied)
        nb_angs: list[float] = []
        a = mol.GetAtomWithIdx(int(atom_idx))
        for nb in a.GetNeighbors():
            nb_idx = nb.GetIdx()
            xnb, ynb = all_xy[nb_idx]
            nb_angs.append(_angle(ax, ay, xnb, ynb))

        # "Inside" angle: toward molecule centroid
        in_ang = _angle(ax, ay, mol_cx, mol_cy)

        # If something weird happens, fall back to outward-from-centroid
        if not nb_angs:
            dx = ax - mol_cx
            dy = ay - mol_cy
            n = (dx * dx + dy * dy) ** 0.5
            if n < 1e-6:
                return 0.0, -1.0
            return dx / n, dy / n

        # --- Compute largest gap between occupied angles (neighbors + inside) ---
        occ = sorted(nb_angs + [in_ang])

        gaps: list[tuple[float, float]] = []
        for i in range(len(occ)):
            a1 = occ[i]
            a2 = occ[(i + 1) % len(occ)]
            gap = a2 - a1
            if gap <= 0:
                gap += 2 * math.pi
            gaps.append((gap, a1))

        gaps.sort(reverse=True)  # biggest first
        theta0 = gaps[0][1] + 0.5 * gaps[0][0]

        # --- NEW: candidate angles + bond keep-out scoring ---
        # Keep-out around neighbor bond directions (degrees -> radians)
        keepout = math.radians(28.0)

        # Candidate angles: best gap bisector + slight rotations,
        # plus (optionally) second-best gap bisector.
        cand = [theta0,
                theta0 + math.radians(20.0),
                theta0 - math.radians(20.0),
                theta0 + math.radians(40.0),
                theta0 - math.radians(40.0)]

        if len(gaps) > 1:
            theta1 = gaps[1][1] + 0.5 * gaps[1][0]
            cand.extend([theta1, theta1 + math.radians(20.0),
                         theta1 - math.radians(20.0)])

        # Normalize candidates to [0, 2pi)
        cand = [(t % (2 * math.pi)) for t in cand]

        def _score(theta: float) -> float:
            # Prefer far from neighbor bonds and far from inside direction
            min_nb = min(_ang_dist(theta, a) for a in nb_angs)
            dist_in = _ang_dist(theta, in_ang)

            # Hard penalty if we're near a bond direction
            if min_nb < keepout:
                return -1e6 + min_nb

            # Soft preference: away from inside + away from bonds
            return (1.3 * min_nb) + (0.7 * dist_in)

        best_theta = max(cand, key=_score)
        ux = math.cos(best_theta)
        uy = math.sin(best_theta)

        if abs(ux) < 1e-6 and abs(uy) < 1e-6:
            return 0.0, -1.0
        return ux, uy

    label_xy = []
    label_wh = []

    w_px, h_px = size
    margin = 6.0

    def _clamp_point(x: float, y: float, w: float, h: float) -> tuple[float, float]:
        x = min(max(x, margin + w / 2), w_px - margin - w / 2)
        y = min(max(y, margin + h / 2), h_px - margin - h / 2)
        return x, y

    # --- Optional: outward walk for clearance (kept minimal & stable) ---
    # Note: This only adjusts distance, not direction.
    bond_segs = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        x1, y1 = all_xy[i]
        x2, y2 = all_xy[j]
        bond_segs.append((x1, y1, x2, y2))

    def _pt_seg_dist(px: float, py: float,
                     x1: float, y1: float, x2: float, y2: float) -> float:
        vx = x2 - x1
        vy = y2 - y1
        wx = px - x1
        wy = py - y1
        vv = vx * vx + vy * vy
        if vv < 1e-12:
            dx = px - x1
            dy = py - y1
            return (dx * dx + dy * dy) ** 0.5

        t = (wx * vx + wy * vy) / vv
        if t <= 0.0:
            cxp, cyp = x1, y1
        elif t >= 1.0:
            cxp, cyp = x2, y2
        else:
            cxp = x1 + t * vx
            cyp = y1 + t * vy

        dx = px - cxp
        dy = py - cyp
        return (dx * dx + dy * dy) ** 0.5

    def _min_bond_dist(px: float, py: float) -> float:
        if not bond_segs:
            return 1e9
        return min(_pt_seg_dist(px, py, *seg) for seg in bond_segs)

    def _min_atom_dist(px: float, py: float, self_idx: int) -> float:
        best = 1e9
        for j, (ax2, ay2) in enumerate(all_xy):
            if j == self_idx:
                continue
            dx = px - ax2
            dy = py - ay2
            d = (dx * dx + dy * dy) ** 0.5
            if d < best:
                best = d
        return best

    bond_clear = 0.85 * font_px
    atom_clear = 0.95 * font_px
    # -------------------------------------------------------------------

    for atom_idx, (ax, ay), s in zip(atom_ids, atom_xy, labels):
        ux, uy = _best_dir(int(atom_idx), ax, ay)

        w = max(8.0, 0.60 * font_px * max(1, len(str(s))))
        h = 1.10 * font_px

        base_off = float(label_offset_px) + 0.12 * w

        # walk outward until clear (distance only)
        step = 2.0
        max_extra = 60.0

        best_lx, best_ly = None, None
        best_score = None

        d = base_off
        while d <= base_off + max_extra + 1e-6:
            lx = ax + ux * d
            ly = ay + uy * d
            lx, ly = _clamp_point(lx, ly, w, h)

            db = _min_bond_dist(lx, ly)
            da = _min_atom_dist(lx, ly, int(atom_idx))

            score = 0.0
            if db < bond_clear:
                score += (bond_clear - db) ** 2 * 3.0
            if da < atom_clear:
                score += (atom_clear - da) ** 2 * 2.0

            if best_score is None or score < best_score:
                best_score = score
                best_lx, best_ly = lx, ly

            if score <= 1e-6:
                break

            d += step

        lx, ly = best_lx, best_ly

        label_xy.append([lx, ly])
        label_wh.append([w, h])

    def _clamp(i: int) -> None:
        w, h = label_wh[i]
        x, y = label_xy[i]
        x, y = _clamp_point(x, y, w, h)
        label_xy[i][0] = x
        label_xy[i][1] = y

    for i in range(len(label_xy)):
        _clamp(i)

    # label-label relaxation (unchanged)
    for _ in range(max(0, int(relax_iters))):
        moved = False
        for i in range(len(label_xy)):
            xi, yi = label_xy[i]
            wi, hi = label_wh[i]
            for j in range(i + 1, len(label_xy)):
                xj, yj = label_xy[j]
                wj, hj = label_wh[j]

                ox = (wi + wj) / 2 + min_gap_px - abs(xi - xj)
                oy = (hi + hj) / 2 + min_gap_px - abs(yi - yj)
                if ox > 0 and oy > 0:
                    if ox < oy:
                        sgn = 1.0 if xi >= xj else -1.0
                        push = 0.5 * ox
                        label_xy[i][0] += sgn * push
                        label_xy[j][0] -= sgn * push
                    else:
                        sgn = 1.0 if yi >= yj else -1.0
                        push = 0.5 * oy
                        label_xy[i][1] += sgn * push
                        label_xy[j][1] -= sgn * push

                    _clamp(i)
                    _clamp(j)
                    moved = True

        if not moved:
            break

    out = []
    for (ax, ay), (lx, ly), s in zip(atom_xy, label_xy, labels):
        out.append({"ax": ax, "ay": ay, "lx": lx, "ly": ly, "text": s})
    return out


def _inject_robust_labels_svg(
    svg: str,
    layout: list[dict],
    note_font_px: float | None,
    leader_line: bool,
):
    if not layout:
        return svg

    def _estimate_label_wh(text: str) -> tuple[float, float]:
        # Must match your placement heuristic: include brackets etc.
        w = max(8.0, 0.60 * font_px * max(1, len(text)))
        h = 1.10 * font_px
        return w, h

    def _ray_rect_edge_point(
        cx: float,
        cy: float,
        ux: float,
        uy: float,
        hw: float,
        hh: float,
        pad: float = 2.0,
    ) -> tuple[float, float]:
        ex = hw + pad
        ey = hh + pad
        tx = float("inf") if abs(ux) < 1e-12 else ex / abs(ux)
        ty = float("inf") if abs(uy) < 1e-12 else ey / abs(uy)
        t = min(tx, ty)
        return cx + ux * t, cy + uy * t

    font_px = float(note_font_px) if note_font_px is not None else 14.0

    # Arrow styling knobs
    arrow_opacity = 0.55
    arrow_stroke = "#000000"
    arrow_stroke_width = 1.2
    arrow_head_len = 4.0     # px
    arrow_head_width = 6.0   # px
    arrow_gap_px = 4.0       # gap between atom point and arrow head tip
    arrow_text_gap_px = 2.0   # gap between the text and the arrow start
    min_shaft_len_px = 0.0

    overlay = [
        ("""
    <style type="text/css"><![CDATA[
    .robust-note-halo {
        font-family: sans-serif;
        font-size: %0.2fpx;
        fill: none;
        stroke: #ffffff;
        stroke-width: 3px;
        stroke-linejoin: round;
        stroke-linecap: round;
    }
    .robust-note-text {
        font-family: sans-serif;
        font-size: %0.2fpx;
        fill: #000000;
    }
    .robust-note-arrow {
        stroke: %s;
        stroke-width: %0.2fpx;
        opacity: %0.2f;
        fill: none;
        stroke-linecap: round;
    }
    .robust-note-arrowhead {
        fill: %s;
        opacity: %0.2f;
    }
    ]]></style>
    """ % (
            font_px,
            font_px,
            arrow_stroke,
            arrow_stroke_width,
            arrow_opacity,
            arrow_stroke,
            arrow_opacity,
        )).strip()
    ]

    overlay.append('<g class="robust-notes">')

    for it in layout:
        ax = float(it["ax"])
        ay = float(it["ay"])
        lx = float(it["lx"])
        ly = float(it["ly"])
        txt = _xml_escape(str(it["text"]))

        if leader_line:
            # Vector from label -> atom
            dx = ax - lx
            dy = ay - ly
            n = (dx * dx + dy * dy) ** 0.5

            if n > 1e-6:
                ux = dx / n
                uy = dy / n

                tip_x = ax - ux * arrow_gap_px
                tip_y = ay - uy * arrow_gap_px

                base_x = tip_x - ux * arrow_head_len
                base_y = tip_y - uy * arrow_head_len

                # Use RAW text for bbox estimation (escaped text can be longer, e.g. &gt;)
                raw_text = str(it["text"])
                w, h = _estimate_label_wh(raw_text)
                hw = w / 2.0
                hh = h / 2.0

                # Start at bbox edge along ray label->atom, then move outward by the gap
                edge_x, edge_y = _ray_rect_edge_point(
                    cx=lx, cy=ly, ux=ux, uy=uy, hw=hw, hh=hh, pad=2.0
                )
                start_x = edge_x + ux * arrow_text_gap_px
                start_y = edge_y + uy * arrow_text_gap_px

                # NEW: skip if arrowhead would be too close to (or inside) the label area
                edge_to_base_dx = base_x - edge_x
                edge_to_base_dy = base_y - edge_y
                edge_to_base = (edge_to_base_dx * edge_to_base_dx +
                                edge_to_base_dy * edge_to_base_dy) ** 0.5
                min_head_clear_px = arrow_head_len + arrow_text_gap_px + 0.0

                shaft_dx = base_x - start_x
                shaft_dy = base_y - start_y
                shaft_len2 = shaft_dx * shaft_dx + shaft_dy * shaft_dy

                if edge_to_base < min_head_clear_px:
                    pass
                elif shaft_len2 < (min_shaft_len_px ** 2):
                    pass
                else:
                    px = -uy
                    py = ux
                    left_x = base_x + px * (arrow_head_width / 2.0)
                    left_y = base_y + py * (arrow_head_width / 2.0)
                    right_x = base_x - px * (arrow_head_width / 2.0)
                    right_y = base_y - py * (arrow_head_width / 2.0)

                    pad = 2.0
                    x0 = lx - hw - pad
                    x1 = lx + hw + pad
                    y0 = ly - hh - pad
                    y1 = ly + hh + pad

                    def _in_box(x: float, y: float) -> bool:
                        return (x0 <= x <= x1) and (y0 <= y <= y1)

                    # Any arrowhead vertex inside label box?
                    if _in_box(tip_x, tip_y) or _in_box(left_x, left_y) or _in_box(right_x, right_y):
                        pass
                    else:
                        # Also check bbox overlap (cheap)
                        tri_x0 = min(tip_x, left_x, right_x)
                        tri_x1 = max(tip_x, left_x, right_x)
                        tri_y0 = min(tip_y, left_y, right_y)
                        tri_y1 = max(tip_y, left_y, right_y)

                        boxes_overlap = not (tri_x1 < x0 or tri_x0 > x1 or tri_y1 < y0 or tri_y0 > y1)
                        if boxes_overlap:
                            pass
                        else:
                            overlay.append(
                                f'<line class="robust-note-arrow" '
                                f'x1="{start_x:.2f}" y1="{start_y:.2f}" '
                                f'x2="{base_x:.2f}" y2="{base_y:.2f}" />'
                            )
                            overlay.append(
                                f'<polygon class="robust-note-arrowhead" points="'
                                f'{tip_x:.2f},{tip_y:.2f} '
                                f'{left_x:.2f},{left_y:.2f} '
                                f'{right_x:.2f},{right_y:.2f}" />'
                            )

        overlay.append(
            f'<text class="robust-note-halo" '
            f'x="{lx:.2f}" y="{ly:.2f}" '
            f'text-anchor="middle" dominant-baseline="middle">{txt}</text>'
        )

        overlay.append(
            f'<text class="robust-note-text" '
            f'x="{lx:.2f}" y="{ly:.2f}" '
            f'text-anchor="middle" dominant-baseline="middle">{txt}</text>'
        )        

    overlay.append("</g>")
    overlay_str = "\n".join(overlay)

    # Insert at the end (draw on top of RDKit background + molecule)
    m = re.search(r"</svg\s*>", svg, flags=re.IGNORECASE)
    if not m:
        return svg

    insert_at = m.start()
    return svg[:insert_at] + overlay_str + svg[insert_at:]


def _draw_robust_atomnotes(
    drawer,
    mol,
    size: tuple[int, int],
    atom_ids: list[int],
    labels: list[str],
    note_font_px: float | None,
    label_offset_px: float = 18.0,
    leader_line: bool = True,
    relax_iters: int = 40,
    min_gap_px: float = 2.0,
):
    opts = drawer.drawOptions()

    if note_font_px is not None:
        # Some RDKit builds respect SetFontSize more reliably for DrawString.
        if hasattr(drawer, "SetFontSize"):
            try:
                drawer.SetFontSize(float(note_font_px))
            except Exception:
                pass    

    if note_font_px is not None and hasattr(opts, "fixedFontSize"):
        opts.fixedFontSize = int(round(float(note_font_px)))

    atom_xy = []
    for idx in atom_ids:
        p = drawer.GetDrawCoords(idx)
        atom_xy.append((float(p.x), float(p.y)))

    if not atom_xy:
        return

    cx = sum(x for x, _ in atom_xy) / len(atom_xy)
    cy = sum(y for _, y in atom_xy) / len(atom_xy)

    label_xy = []
    label_wh = []
    font_px = float(note_font_px) if note_font_px is not None else 14.0

    for (ax, ay), s in zip(atom_xy, labels):
        dx = ax - cx
        dy = ay - cy
        n = (dx * dx + dy * dy) ** 0.5
        if n < 1e-6:
            ux, uy = 0.0, -1.0
        else:
            ux, uy = dx / n, dy / n

        w = max(8.0, 0.60 * font_px * max(1, len(s)))
        h = 1.05 * font_px
        off = float(label_offset_px) + 0.12 * w

        lx = ax + ux * off
        ly = ay + uy * off

        label_xy.append([lx, ly])
        label_wh.append([w, h])

    w_px, h_px = size
    margin = 6.0

    def _clamp(i: int) -> None:
        w, h = label_wh[i]
        label_xy[i][0] = min(max(label_xy[i][0], margin + w / 2),
                             w_px - margin - w / 2)
        label_xy[i][1] = min(max(label_xy[i][1], margin + h / 2),
                             h_px - margin - h / 2)

    for i in range(len(label_xy)):
        _clamp(i)

    for _ in range(max(0, int(relax_iters))):
        moved = False
        for i in range(len(label_xy)):
            xi, yi = label_xy[i]
            wi, hi = label_wh[i]
            for j in range(i + 1, len(label_xy)):
                xj, yj = label_xy[j]
                wj, hj = label_wh[j]

                ox = (wi + wj) / 2 + min_gap_px - abs(xi - xj)
                oy = (hi + hj) / 2 + min_gap_px - abs(yi - yj)
                if ox > 0 and oy > 0:
                    if ox < oy:
                        sgn = 1.0 if xi >= xj else -1.0
                        push = 0.5 * ox
                        label_xy[i][0] += sgn * push
                        label_xy[j][0] -= sgn * push
                    else:
                        sgn = 1.0 if yi >= yj else -1.0
                        push = 0.5 * oy
                        label_xy[i][1] += sgn * push
                        label_xy[j][1] -= sgn * push

                    _clamp(i)
                    _clamp(j)
                    xi, yi = label_xy[i]
                    moved = True

        if not moved:
            break

    for (ax, ay), (lx, ly), s, (w, h) in zip(atom_xy, label_xy,
                                            labels, label_wh):
        pa = rdGeometry.Point2D(ax, ay)

        if leader_line:
            pl = rdGeometry.Point2D(lx, ly)
            drawer.DrawLine(pa, pl)

        # MolDraw2D anchors strings in a slightly annoying way; this is a
        # reasonable baseline offset for “centered-ish” labels.
        pt = rdGeometry.Point2D(lx - w / 2, ly + h / 3)
        drawer.DrawString(s, pt)


def _tighten_svg_viewbox(
    svg: str,
    minx: float,
    miny: float,
    w: float,
    h: float,
    set_size: bool = False,
) -> str:
    def _repl(m: re.Match) -> str:
        tag = m.group(0)

        if re.search(r'\bviewBox\s*=\s*["\']', tag):
            tag = re.sub(
                r'(\bviewBox\s*=\s*["\'])[^"\']*(["\'])',
                rf'\g<1>{minx:.2f} {miny:.2f} {w:.2f} {h:.2f}\g<2>',
                tag,
            )
        else:
            tag = tag[:-1] + (
                f' viewBox="{minx:.2f} {miny:.2f} {w:.2f} {h:.2f}">'
            )

        if set_size:
            if re.search(r'\bwidth\s*=\s*["\']', tag):
                tag = re.sub(
                    r'(\bwidth\s*=\s*["\'])[^"\']*(["\'])',
                    rf'\g<1>{w:.0f}px\g<2>',
                    tag,
                )
            else:
                tag = tag[:-1] + f' width="{w:.0f}px">'

            if re.search(r'\bheight\s*=\s*["\']', tag):
                tag = re.sub(
                    r'(\bheight\s*=\s*["\'])[^"\']*(["\'])',
                    rf'\g<1>{h:.0f}px\g<2>',
                    tag,
                )
            else:
                tag = tag[:-1] + f' height="{h:.0f}px">'

        return tag

    return re.sub(r"<svg[^>]*>", _repl, svg, count=1)


def _svg_annotated_smi(
    smi,
    pos_list,
    dE_list,
    size=(250, 250),
    show_rpos: bool = False,
    step_list: list[str] | None = None,
    fixed_bond_px: float | None = 25.0,
    note_font_px: float | None = None,
    annotation_scale: float = 1,
    robust_notes: bool = False,
    label_offset_px: float = 18.0,
    leader_line: bool = True,
    tight_viewbox: bool = True,
    tight_padding_px: float = 12.0,
    tight_set_size: bool = False,
    highlight_atoms: list[int] | None = None,
    highlight_color=(1, 0, 0),
):
    """Return an SVG string of the molecule with per-atom ΔE labels."""
    mol = _mol_annotated_smi(
        smi,
        pos_list,
        dE_list,
        show_rpos=show_rpos,
        step_list=step_list,
    )
    if mol is None:
        return ""

    robust_atom_ids: list[int] = []
    robust_labels: list[str] = []
    if robust_notes:
        for a in mol.GetAtoms():
            if a.HasProp("atomNote"):
                robust_atom_ids.append(a.GetIdx())
                robust_labels.append(a.GetProp("atomNote"))
                a.ClearProp("atomNote")

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    opts = drawer.drawOptions()

    if fixed_bond_px is not None:
        opts.fixedBondLength = float(fixed_bond_px)

    scale = float(annotation_scale) if annotation_scale else 0.4
    opts.annotationFontScale = scale

    # RDKit notes only in non-robust mode
    opts.drawAtomNotes = not robust_notes

    ha = list({int(x) for x in (highlight_atoms or [])})
    hac = {int(a): highlight_color for a in ha} if ha else None

    if ha:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=ha,
            highlightAtomColors=hac,
        )
    else:
        drawer.DrawMolecule(mol)

    layout = []
    if robust_notes and robust_atom_ids:
        layout = _layout_robust_labels(
            drawer,
            mol,
            size=size,
            atom_ids=robust_atom_ids,
            labels=robust_labels,
            note_font_px=note_font_px,
            label_offset_px=float(label_offset_px),
        )

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    if robust_notes and layout:
        svg = _inject_robust_labels_svg(
            svg,
            layout=layout,
            note_font_px=note_font_px,
            leader_line=leader_line,
        )

    # --- NEW: tighten viewBox around molecule + labels ---
    if tight_viewbox:
        # Include all atoms (not just annotated) so the ring always fits.
        pts = []
        for a in mol.GetAtoms():
            p = drawer.GetDrawCoords(a.GetIdx())
            pts.append((float(p.x), float(p.y)))

        # Include label bboxes (estimate, same heuristic as layout)
        font_px_local = float(note_font_px) if note_font_px is not None else 14.0

        def _label_wh(s: str) -> tuple[float, float]:
            w = max(8.0, 0.60 * font_px_local * max(1, len(s)))
            h = 1.10 * font_px_local
            return w, h

        for it in layout:
            lx = float(it["lx"])
            ly = float(it["ly"])
            raw = str(it["text"])
            w, h = _label_wh(raw)
            hw = w / 2.0
            hh = h / 2.0
            pts.extend(
                [
                    (lx - hw, ly - hh),
                    (lx - hw, ly + hh),
                    (lx + hw, ly - hh),
                    (lx + hw, ly + hh),
                ]
            )

        if pts:
            minx = min(x for x, _ in pts) - float(tight_padding_px)
            maxx = max(x for x, _ in pts) + float(tight_padding_px)
            miny = min(y for _, y in pts) - float(tight_padding_px)
            maxy = max(y for _, y in pts) + float(tight_padding_px)

            w = max(1.0, maxx - minx)
            h = max(1.0, maxy - miny)

            svg = _tighten_svg_viewbox(
                svg,
                minx=minx,
                miny=miny,
                w=w,
                h=h,
                set_size=bool(tight_set_size),
            )        

    return svg


def build_annotated_montage(
    df: pd.DataFrame,
    ligand_col: str = "ligand_name",
    smi_col: str = "smiles",
    pos_col: str = "rpos",
    energy_col: str = "dE",
    output_path: str | None = None,
    step_col: str | None = None,
    show_rpos: bool = False,
    energy_cols: list[str] | None = None,
    subimg_size: tuple[int, int] = (250, 250),
    fixed_bond_px: float | None = 25.0,
    note_font_px: float | None = 14.0,
    annotation_scale: float = 1,
    columns: int = 5,
    show_name: bool = True,
    show_smiles: bool = True,
    smiles_wrap: int = 32,
    padding: float | None = None,
) -> tuple[pd.DataFrame, str]:
    """One combined SVG image containing a grid of all annotated molecules."""
    for col in (ligand_col, smi_col, pos_col):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    work = df.copy()

    energy_col_local = energy_col
    step_col_local = step_col

    if energy_col_local not in work.columns:
        if not energy_cols:
            raise ValueError(
                "Energy column not found and no energy_cols provided. "
                f"Missing: {energy_col!r}"
            )
        missing = [c for c in energy_cols if c not in work.columns]
        if missing:
            raise ValueError(f"Missing energy columns: {missing}")

        vals = work[energy_cols].apply(pd.to_numeric, errors="coerce")
        work["_energy_max_tmp"] = vals.max(axis=1, skipna=True)
        try:
            step_idx = vals.idxmax(axis=1, skipna=True)
        except Exception:
            step_idx = vals.apply(
                lambda r: (r.idxmax(skipna=True) if r.notna().any() else None),
                axis=1,
            )
        work["_step_tmp"] = step_idx
        energy_col_local = "_energy_max_tmp"
        if step_col_local is None:
            step_col_local = "_step_tmp"

    if step_col_local is not None and step_col_local not in work.columns:
        raise ValueError(f"Column not found: {step_col_local}")

    def _norm_step(s: object) -> str:
        if not isinstance(s, str):
            return str(s)
        t = s.strip()
        t = t.replace("dG_", "").replace("dE_", "").lower()
        if t.startswith("ts"):
            t = t[2:]
        return t

    def _wrap(s: str, n: int) -> str:
        if n <= 0:
            return s
        return "\n".join(s[i:i + n] for i in range(0, len(s), n))

    mols = []
    legends = []
    rows = []

    for idx, (lig, grp) in enumerate(work.groupby(ligand_col), start=1):
        smi = grp[smi_col].iloc[0]
        pos = grp[pos_col].astype(int).tolist()
        dE_vals = grp[energy_col_local].tolist()

        steps = None
        if step_col_local is not None:
            steps = [_norm_step(x) for x in grp[step_col_local].tolist()]

        mol = _mol_annotated_smi(
            smi,
            pos,
            dE_vals,
            show_rpos=show_rpos,
            step_list=steps,
        )
        if mol is None:
            continue

        mols.append(mol)

        legend_parts = []
        if show_name:
            legend_parts.append(str(lig))
        if show_smiles:
            legend_parts.append(_wrap(str(smi), int(smiles_wrap)))
        legends.append("\n".join(legend_parts))

        rows.append({"i": idx, ligand_col: lig, smi_col: smi})

    result_df = pd.DataFrame(rows)

    if not mols:
        return result_df, ""

    n = len(mols)
    cols = max(1, int(columns))
    n_rows = int(math.ceil(n / cols))

    width = cols * int(subimg_size[0])
    height = n_rows * int(subimg_size[1])

    # --- draw one combined SVG using RDKit's grid helper (version-stable) ---
    dopts = rdMolDraw2D.MolDrawOptions()
    dopts.drawAtomNotes = True

    if fixed_bond_px is not None and hasattr(dopts, "fixedBondLength"):
        dopts.fixedBondLength = float(fixed_bond_px)

    scale = float(annotation_scale) if annotation_scale else 0.4
    if hasattr(dopts, "annotationFontScale"):
        dopts.annotationFontScale = scale

    if padding is not None and hasattr(dopts, "padding"):
        dopts.padding = float(padding)

    use_svg_patch = False
    if note_font_px is not None:
        base_font_px = max(
            1,
            int(round(float(note_font_px) / max(scale, 1e-6))),
        )
        if hasattr(dopts, "fixedFontSize"):
            dopts.fixedFontSize = base_font_px
            if hasattr(dopts, "minFontSize"):
                dopts.minFontSize = base_font_px
            if hasattr(dopts, "maxFontSize"):
                dopts.maxFontSize = base_font_px
        else:
            use_svg_patch = True

    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=subimg_size,
        legends=legends,
        useSVG=True,
        drawOptions=dopts,
    )

    # RDKit sometimes returns an object with .data
    if hasattr(svg, "data"):
        svg = svg.data

    if note_font_px is not None and use_svg_patch:
        px = float(note_font_px)
        svg = re.sub(r"(font-size\s*:\s*)[\d.]+px", rf"\1{px}px", svg)
        svg = re.sub(
            r'(font-size\s*=\s*["\'])[\d.]+(px)?(["\'])',
            rf"\1{px}\3",
            svg,
        )

    if output_path:
        if output_path.lower().endswith(".html"):
            title = output_path.split(".html")[0]
            full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
</head>
<body>
{svg}
</body>
</html>"""
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_html)
        else:
            # default: write raw SVG (best for papers)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(svg)

    return result_df, svg


def build_annotated_frame(
    df: pd.DataFrame,
    ligand_col: str = "ligand_name",
    smi_col: str = "smiles",
    pos_col: str = "rpos",
    energy_col: str = "dE",
    output_path: str | None = None,
    step_col: str | None = None,
    show_rpos: bool = False,
    energy_cols: list[str] | None = None,
    size: tuple[int, int] = (250, 250),
    fixed_bond_px: float | None = 25.0,
    note_font_px: float | None = 14.0,
    annotation_scale: float = 1,
    label_offset_px: int = 22,
) -> tuple[pd.DataFrame, str]:
    """One row per ligand + an SVG column with per-atom energy annotations.

    Groups the input DataFrame by `ligand_col` and renders one annotated SVG
    per ligand. Annotations are attached to the atom indices in `pos_col` and
    display the corresponding energies. If `energy_col` is missing, the
    function can compute a per-row maximum across `energy_cols` and optionally
    annotate which column produced that maximum (via `step_col`).

    If `output_path` is set, writes a standalone HTML file containing the
    rendered table.

    Args:
        df (pd.DataFrame): Input data.
        ligand_col (str): Column name for ligand grouping.
        smi_col (str): Column name for SMILES.
        pos_col (str): Column name for annotation atom indices.
        energy_col (str): Column name for energy values. If this column is not
            present, `energy_cols` must be provided and the per-row maximum is
            used.
        output_path (str | None): File path to write HTML. No file written if
            None.
        step_col (str | None): Optional column with per-row step labels. When
            `energy_col` is missing and `energy_cols` are used, a temporary
            step label column is created from `idxmax` unless `step_col` is
            provided.
        show_rpos (bool): If True, append ' (rX)' to each label.
        energy_cols (list[str] | None): Columns to compute per-row maxima when
            `energy_col` is not available.
        size (tuple[int, int]): SVG canvas size in pixels (width, height).
        fixed_bond_px (float | None): Fixed bond length in pixels passed to
            `_svg_annotated_smi`. Set to None to use RDKit defaults.
        note_font_px (float | None): Target annotation font size (px) passed to
            `_svg_annotated_smi`. If None, RDKit defaults are used.
        annotation_scale (float): RDKit annotation font scale passed to
            `_svg_annotated_smi`.
        label_offset_px (int): Base offset (px) used for robust label placement
            passed to `_svg_annotated_smi`.

    Returns:
        tuple[pd.DataFrame, str]:
            - DataFrame with one row per ligand and an "annotated_svg" column.
            - HTML table fragment as a string.
    """
    for col in (ligand_col, smi_col, pos_col):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    work = df.copy()

    energy_col_local = energy_col
    step_col_local = step_col

    if energy_col_local not in work.columns:
        if not energy_cols:
            raise ValueError(
                "Energy column not found and no energy_cols provided. "
                f"Missing: {energy_col!r}"
            )
        missing = [c for c in energy_cols if c not in work.columns]
        if missing:
            raise ValueError(f"Missing energy columns: {missing}")

        vals = work[energy_cols].apply(pd.to_numeric, errors="coerce")
        work["_energy_max_tmp"] = vals.max(axis=1, skipna=True)
        try:
            step_idx = vals.idxmax(axis=1, skipna=True)
        except Exception:
            step_idx = vals.apply(
                lambda r: (r.idxmax(skipna=True)
                           if r.notna().any() else None),
                axis=1
            )
        work["_step_tmp"] = step_idx
        energy_col_local = "_energy_max_tmp"
        if step_col_local is None:
            step_col_local = "_step_tmp"

    if step_col_local is not None and step_col_local not in work.columns:
        raise ValueError(f"Column not found: {step_col_local}")

    def _norm_step(s: object) -> str:
        if not isinstance(s, str):
            return str(s)
        t = s.strip()
        t = t.replace("dG_", "").replace("dE_", "").lower()
        if t.startswith("ts"):
            t = t[2:]  # remove the 'ts' prefix
        return t

    rows = []
    for idx, (lig, grp) in enumerate(work.groupby(ligand_col), start=1):
        smi = grp[smi_col].iloc[0]
        pos = grp[pos_col].astype(int).tolist()
        dE_vals = grp[energy_col_local].tolist()

        steps = None
        if step_col_local is not None:
            steps = [_norm_step(x) for x in grp[step_col_local].tolist()]

        svg = _svg_annotated_smi(
            smi, pos, dE_vals,
            show_rpos=show_rpos,
            step_list=steps,
            fixed_bond_px=fixed_bond_px,
            note_font_px=note_font_px,
            annotation_scale=annotation_scale,
            size=size,
            robust_notes=True,
            label_offset_px=label_offset_px,
            leader_line=True,
        )
        rows.append({"i": idx, ligand_col: lig, smi_col: smi, "annotated_svg": svg})

    result_df = pd.DataFrame(rows)
    html_table = result_df.to_html(
        escape=False,
        formatters={"annotated_svg": lambda x: x},
        index=False
    )

    if output_path:
        title = output_path.split(".html")[0]
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
</head>
<body>
<p>{title}</p>
{html_table}
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

    return result_df, html_table


import argparse
from pathlib import Path
from typing import Sequence, Union
import pandas as pd


def merge_parquet_dir(
    input_dir: Union[str, Path],
    output: Union[str, Path] = "merged.parquet",
    require_normal_termination: bool = False,
    normal_term_cols: Sequence[str] | None = None,
    recursive: bool = False,
) -> Path:
    """Merge multiple Parquet files with identical schemas into one file.

    Args:
        input_dir: Directory containing .parquet files to merge.
        output: Output Parquet file path.
        require_normal_termination: If True, skip any file whose DataFrame
            has a column ending with 'normal_termination' containing any
            False (or NaN). If no such column exists, the file is kept.
        normal_term_cols: Optional explicit list of columns to treat as
            'normal_termination' columns. If None, columns ending with
            'normal_termination' are auto-detected.
        recursive: If True, also include .parquet files found in
            subdirectories (uses rglob).

    Returns:
        Path to the merged Parquet file.

    Raises:
        FileNotFoundError: If the input directory does not exist or contains
            no .parquet files.
        ValueError: If no files pass the filter or the merged DataFrame is
            empty.
    """
    in_path = Path(input_dir)
    if not in_path.is_dir():
        raise FileNotFoundError(f"Input directory '{in_path}' not found.")

    pattern = "*.parquet"
    files = sorted(
        in_path.rglob(pattern) if recursive else in_path.glob(pattern)
    )
    if not files:
        raise FileNotFoundError(f"No .parquet files in '{in_path}'.")

    dfs: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_parquet(str(fp))
        if require_normal_termination:
            cols = (list(normal_term_cols)
                    if normal_term_cols is not None
                    else [c for c in df.columns
                          if str(c).endswith("normal_termination")])
            if cols:
                subset = (df[cols]
                          .replace({None: False})
                          .fillna(False))
                try:
                    subset = subset.astype(bool)
                except Exception:
                    subset = subset.applymap(
                        lambda x: bool(x) if pd.notna(x) else False
                    )
                if not subset.all().all():
                    continue
        dfs.append(df)

    if not dfs:
        raise ValueError("No files to merge after filtering.")

    merged = pd.concat(dfs, ignore_index=True)
    if merged.empty:
        raise ValueError("Merged DataFrame is empty.")

    out_path = Path(output)
    merged.to_parquet(out_path)
    return out_path


def build_annotated_grid(
    df: pd.DataFrame,
    ligand_col: str = "ligand_name",
    smi_col: str = "smiles",
    pos_col: str = "rpos",
    energy_col: str = "dE",
    output_path: str | None = None,
    step_col: str | None = None,
    show_rpos: bool = False,
    energy_cols: list[str] | None = None,
    size: tuple[int, int] = (250, 250),
    fixed_bond_px: float | None = 25.0,
    note_font_px: float | None = 14.0,
    annotation_scale: float = 1,
    columns: int = 4,
    gap_px: int = 12,
    card_padding_px: int = 8,
    show_index: bool = False,
    show_name: bool = True,
    show_smiles: bool = True,
    include_style: bool = True,
    label_offset_px: int = 22,
    tight_viewbox: bool = True,
    tight_padding_px: float = 12.0,
    tight_set_size: bool = False,
    include_download_button: bool = False,
    download_format: str = "png",
    png_scale: float = 2.0,
    highlight_n_lowest: int | None = None,
    highlight_edge_ranks: list[tuple[str, int]] | None = None,
    highlight_color: tuple[float, float, float] = (1, 0, 0),
) -> tuple[pd.DataFrame, str]:
    """Render annotated molecules as a responsive HTML/CSS grid.

    Groups the input DataFrame by `ligand_col` and produces one "card" per
    ligand. Each card contains an annotated SVG depiction (per-atom energy
    labels) and optional metadata (index, ligand name, and SMILES).

    If `energy_col` is missing, `energy_cols` can be provided and the function
    uses the per-row maximum across those columns. In that case, a step label
    can be taken from `step_col`, or inferred from the column name that
    produced the maximum.

    SVGs can optionally be "tightened" by updating their viewBox (and
    optionally their pixel width/height) so the molecule + labels fill the
    available space with minimal whitespace.

    If `include_download_button` is True, the returned HTML includes a button
    that downloads a ZIP file containing one file per ligand in the requested
    `download_format` (either SVG or PNG). It also includes per-card download
    buttons (same format).

    Args:
        df (pd.DataFrame): Input data.
        ligand_col (str): Column name for ligand grouping.
        smi_col (str): Column name for SMILES.
        pos_col (str): Column name for annotation atom indices.
        energy_col (str): Column name for energy values. If this column is not
            present, `energy_cols` must be provided and the per-row maximum is
            used.
        output_path (str | None): File path to write a standalone HTML page. No
            file written if None.
        step_col (str | None): Optional column with per-row step labels. When
            `energy_col` is missing and `energy_cols` are used, a temporary
            step label column is created from `idxmax` unless `step_col` is
            provided.
        show_rpos (bool): If True, append ' (rX)' to each label.
        energy_cols (list[str] | None): Columns to compute per-row maxima when
            `energy_col` is not available.
        size (tuple[int, int]): SVG canvas size in pixels (width, height).
        fixed_bond_px (float | None): Fixed bond length in pixels passed to
            `_svg_annotated_smi`. Set to None to use RDKit defaults.
        note_font_px (float | None): Target annotation font size (px) passed to
            `_svg_annotated_smi`. If None, RDKit defaults are used.
        annotation_scale (float): RDKit annotation font scale passed to
            `_svg_annotated_smi`.
        columns (int): Number of grid columns.
        gap_px (int): Gap between cards in pixels.
        card_padding_px (int): Card padding in pixels.
        show_index (bool): If True, show the per-card index.
        show_name (bool): If True, show the ligand name under the SVG.
        show_smiles (bool): If True, show the SMILES string under the SVG.
        include_style (bool): If True, embed CSS needed for the grid/cards in
            the returned HTML.
        label_offset_px (int): Base offset (px) used for robust label placement
            passed to `_svg_annotated_smi`.
        tight_viewbox (bool): If True, update the SVG viewBox to tightly bound
            the molecule and labels (reduces whitespace and visually "zooms"
            the drawing within the fixed canvas size).
        tight_padding_px (float): Padding (px) added around the computed
            content bounding box when `tight_viewbox` is enabled.
        tight_set_size (bool): If True, also set the SVG width/height to the
            tightened bounding box dimensions (in pixels). If False, keep the
            original `size` and only adjust the viewBox.
        include_download_button (bool): If True, embed a button in the returned
            HTML that downloads a ZIP of all depictions, and per-card download
            buttons.
        download_format (str): Download file format. Supported: "png" or "svg"
            (case-insensitive). Controls both per-card downloads and the ZIP.
        png_scale (float): PNG render scale factor used when
            `download_format="png"`.
        highlight_n_lowest (int | None): If set, highlight N lowest-energy
            annotated positions per molecule.
        highlight_edge_ranks (list[tuple[str, int]] | None): Overrides to
            highlight a specific rank (0-based among lowest energies) for a
            specific molecule, keyed by canonical SMILES or ligand name.
        highlight_color (tuple[float, float, float]): RGB float triple (0..1)
            used for highlighted atoms.

    Returns:
        tuple[pd.DataFrame, str]:
            - DataFrame with one row per ligand and an "annotated_svg" column.
            - HTML grid fragment as a string (with optional embedded CSS).
    """
    for col in (ligand_col, smi_col, pos_col):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    work = df.copy()

    energy_col_local = energy_col
    step_col_local = step_col

    if energy_col_local not in work.columns:
        if not energy_cols:
            raise ValueError(
                "Energy column not found and no energy_cols provided. "
                f"Missing: {energy_col!r}"
            )
        missing = [c for c in energy_cols if c not in work.columns]
        if missing:
            raise ValueError(f"Missing energy columns: {missing}")

        vals = work[energy_cols].apply(pd.to_numeric, errors="coerce")
        work["_energy_max_tmp"] = vals.max(axis=1, skipna=True)
        try:
            step_idx = vals.idxmax(axis=1, skipna=True)
        except Exception:
            step_idx = vals.apply(
                lambda r: (r.idxmax(skipna=True) if r.notna().any() else None),
                axis=1,
            )
        work["_step_tmp"] = step_idx
        energy_col_local = "_energy_max_tmp"
        if step_col_local is None:
            step_col_local = "_step_tmp"

    if step_col_local is not None and step_col_local not in work.columns:
        raise ValueError(f"Column not found: {step_col_local}")

    def _maybe_smiles(s: str) -> bool:
        from rdkit import rdBase

        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(s) is not None

    def _canon_smi(s: str) -> str:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s
        return Chem.MolToSmiles(m, canonical=True)

    def _norm_name(s: object) -> str:
        return str(s).strip().lower()

    edge_smi_map: dict[str, int] = {}
    edge_name_map: dict[str, int] = {}

    if highlight_edge_ranks:
        items = highlight_edge_ranks
        if isinstance(items, tuple) and len(items) == 2:
            items = [items]

        for key, r in items:
            try:
                r_int = int(r)
            except Exception:
                continue

            key_str = str(key)

            if _maybe_smiles(key_str):
                edge_smi_map[_canon_smi(key_str)] = r_int
            else:
                edge_name_map[_norm_name(key_str)] = r_int

    def _norm_step(s: object) -> str:
        if not isinstance(s, str):
            return str(s)
        t = s.strip()
        t = t.replace("dG_", "").replace("dE_", "").lower()
        if t.startswith("ts"):
            t = t[2:]
        return t

    rows = []
    for idx, (lig, grp) in enumerate(work.groupby(ligand_col), start=1):
        smi = grp[smi_col].iloc[0]
        pos = grp[pos_col].astype(int).tolist()
        dE_vals = grp[energy_col_local].tolist()

        steps = None
        if step_col_local is not None:
            steps = [_norm_step(x) for x in grp[step_col_local].tolist()]

        highlight_atoms: set[int] = set()

        # Highlight N lowest energies (per ligand)
        if highlight_n_lowest is not None:
            try:
                n_low = int(highlight_n_lowest)
            except Exception:
                n_low = 0

            if n_low > 0:
                pairs = []
                for p, e in zip(pos, dE_vals):
                    try:
                        pairs.append((float(e), int(p)))
                    except Exception:
                        continue
                pairs.sort(key=lambda t: t[0])  # lowest first
                for _, p_int in pairs[:n_low]:
                    highlight_atoms.add(p_int)

        # Edge overrides: rank among LOWEST energies (0-based)
        canon = _canon_smi(str(smi))
        lig_norm = _norm_name(lig)

        r = None
        if canon in edge_smi_map:
            r = edge_smi_map[canon]
        elif lig_norm in edge_name_map:
            r = edge_name_map[lig_norm]

        if r is not None:
            try:
                r = int(r)
            except Exception:
                r = -1

            if r >= 0:
                pairs = []
                for p, e in zip(pos, dE_vals):
                    try:
                        pairs.append((float(e), int(p)))
                    except Exception:
                        continue

                pairs.sort(key=lambda t: t[0])  # lowest first
                if 0 <= r < len(pairs):
                    highlight_atoms.add(pairs[r][1])

        svg = _svg_annotated_smi(
            smi,
            pos,
            dE_vals,
            show_rpos=show_rpos,
            step_list=steps,
            fixed_bond_px=fixed_bond_px,
            note_font_px=note_font_px,
            annotation_scale=annotation_scale,
            size=size,
            robust_notes=True,
            label_offset_px=label_offset_px,
            leader_line=True,
            tight_viewbox=tight_viewbox,
            tight_padding_px=tight_padding_px,
            tight_set_size=tight_set_size,
            highlight_atoms=sorted(highlight_atoms),
            highlight_color=highlight_color,
        )
        rows.append(
            {
                "i": idx,
                ligand_col: lig,
                smi_col: smi,
                "annotated_svg": svg,
            }
        )

    result_df = pd.DataFrame(rows)

    zip_b64 = None
    zip_name = None

    if include_download_button:
        fmt = str(download_format).strip().lower()
        if fmt not in {"png", "svg"}:
            raise ValueError("download_format must be 'png' or 'svg'")

        def _safe_name(s: str) -> str:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
            return safe or "ligand"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for _, r in result_df.iterrows():
                lig_safe = _safe_name(str(r[ligand_col]))
                stem = f"{int(r['i']):03d}_{lig_safe}"

                svg_txt = r["annotated_svg"] or ""
                if fmt == "svg":
                    zf.writestr(f"{stem}.svg", svg_txt)
                else:
                    png_bytes = cairosvg.svg2png(
                        bytestring=svg_txt.encode("utf-8"),
                        scale=float(png_scale),
                    )
                    zf.writestr(f"{stem}.png", png_bytes)

        zip_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        zip_name = f"annotated_{fmt}s.zip"

    css = f"""
.mol-grid {{
  display: grid;
  grid-template-columns: repeat({int(columns)}, minmax(0, 1fr));
  gap: {int(gap_px)}px;
  align-items: start;
}}
.mol-card {{
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: {int(card_padding_px)}px;
}}
.mol-svg {{
  line-height: 0;
}}
.mol-svg svg {{
  max-width: 100%;
  height: auto;
  display: block;
}}
.mol-meta {{
  margin-top: 6px;
}}
.mol-name {{
  font-weight: 600;
  font-size: 12px;
  margin-top: 4px;
}}
.mol-smiles {{
  font-size: 10px;
  margin-top: 2px;
  overflow-wrap: anywhere;
  word-break: break-word;
}}
.mol-idx {{
  font-size: 10px;
  opacity: 0.75;
  margin-top: 4px;
}}
.mol-actions {{
  margin-top: 6px;
  display: flex;
  gap: 6px;
}}
.mol-dl-btn {{
  font-size: 11px;
  padding: 4px 8px;
  border: 1px solid #bbb;
  border-radius: 6px;
  background: #f7f7f7;
  cursor: pointer;
}}
.mol-dl-btn:hover {{
  background: #eee;
}}
"""

    parts: list[str] = []
    if include_style:
        parts.append(f"<style>\n{css}\n</style>")

    if include_download_button and zip_b64 and zip_name:
        parts.append(
            f"""
    <div style="margin: 10px 0; display: flex; gap: 8px; align-items: center;">
    <button id="download-annotated-all"
            class="mol-dl-btn"
            data-mol-download="all"
            data-zip-b64="{html.escape(zip_b64)}"
            data-zip-name="{html.escape(zip_name)}"
            data-format="{html.escape(str(download_format).strip().lower())}"
            data-png-scale="{float(png_scale)}"
            style="font-weight: 600;">
        Download all {html.escape(str(download_format).strip().upper())}s (.zip)
    </button>
    </div>
    """.strip()
        )

        fmt = str(download_format).strip().lower()
        if fmt not in {"png", "svg"}:
            raise ValueError("download_format must be 'png' or 'svg'")

        zip_b64_js = zip_b64 or ""
        zip_name_js = zip_name or ""

    if include_download_button:
        parts.append(
            f"""
    <script type="text/javascript">
    (function() {{
    // Bind once per page. Use capture so we run before older/bubbled handlers.
    if (window.__molGridDlBound) return;
    window.__molGridDlBound = true;

    function safeFileName(name) {{
        return (name || "molecule")
        .replace(/[^A-Za-z0-9._-]+/g, "_")
        .replace(/^_+|_+$/g, "");
    }}

    function downloadBlob(blob, fileName) {{
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }}

    function b64ToUint8Array(b64) {{
        const bin = atob(b64);
        const len = bin.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
        return bytes;
    }}

    function serializeSvg(svgEl) {{
        const serializer = new XMLSerializer();
        let svgText = serializer.serializeToString(svgEl);

        if (!svgText.match(/^<svg[^>]+xmlns=/)) {{
        svgText = svgText.replace(
            "<svg",
            '<svg xmlns="http://www.w3.org/2000/svg"'
        );
        }}
        if (svgText.indexOf("xlink:") !== -1 &&
            !svgText.match(/^<svg[^>]+xmlns:xlink=/)) {{
        svgText = svgText.replace(
            "<svg",
            '<svg xmlns:xlink="http://www.w3.org/1999/xlink"'
        );
        }}
        return svgText;
    }}

    function svgToPngBlob(svgEl, scale, cb) {{
        const svgText = serializeSvg(svgEl);
        const svgBlob = new Blob([svgText], {{type: "image/svg+xml;charset=utf-8"}});
        const url = URL.createObjectURL(svgBlob);

        const img = new Image();
        img.onload = function() {{
        const rect = svgEl.getBoundingClientRect();
        let w = rect.width;
        let h = rect.height;

        if ((!w || !h) && svgEl.viewBox && svgEl.viewBox.baseVal) {{
            w = svgEl.viewBox.baseVal.width;
            h = svgEl.viewBox.baseVal.height;
        }}
        w = Math.max(1, w);
        h = Math.max(1, h);

        const canvas = document.createElement("canvas");
        canvas.width = Math.round(w * scale);
        canvas.height = Math.round(h * scale);

        const ctx = canvas.getContext("2d");
        ctx.setTransform(scale, 0, 0, scale, 0, 0);
        ctx.drawImage(img, 0, 0);

        canvas.toBlob(function(pngBlob) {{
            URL.revokeObjectURL(url);
            cb(pngBlob);
        }}, "image/png");
        }};
        img.onerror = function() {{
        URL.revokeObjectURL(url);
        cb(null);
        }};
        img.src = url;
    }}

    function stopAll(ev) {{
        ev.preventDefault();
        ev.stopPropagation();
        if (ev.stopImmediatePropagation) ev.stopImmediatePropagation();
    }}

    // Capture-phase click handler so we beat older notebook outputs.
    document.addEventListener("click", function(ev) {{
        const allBtn = ev.target.closest('[data-mol-download="all"]');
        if (allBtn) {{
        stopAll(ev);
        const b64 = allBtn.getAttribute("data-zip-b64") || "";
        const name = allBtn.getAttribute("data-zip-name") || "annotated.zip";
        if (!b64) return;

        const bytes = b64ToUint8Array(b64);
        const blob = new Blob([bytes], {{ type: "application/zip" }});
        downloadBlob(blob, name);
        return;
        }}

        const oneBtn = ev.target.closest('[data-mol-download="one"]');
        if (!oneBtn) return;

        stopAll(ev);

        const format = String(oneBtn.getAttribute("data-format") || "png").toLowerCase();
        const pngScale = Number(oneBtn.getAttribute("data-png-scale") || "2.0");

        const card = oneBtn.closest(".mol-card");
        if (!card) return;

        const svgEl = card.querySelector(".mol-svg svg");
        if (!svgEl) return;

        const stem = safeFileName(oneBtn.getAttribute("data-stem") || "molecule");

        if (format === "svg") {{
        const svgText = serializeSvg(svgEl);
        const blob = new Blob([svgText], {{type: "image/svg+xml;charset=utf-8"}});
        downloadBlob(blob, stem + ".svg");
        return;
        }}

        svgToPngBlob(svgEl, pngScale, function(pngBlob) {{
        if (!pngBlob) return;
        downloadBlob(pngBlob, stem + ".png");
        }});
    }}, true);
    }})();
    </script>
    """.strip()
        )

    parts.append('<div class="mol-grid">')

    for _, r in result_df.iterrows():
        lig = html.escape(str(r[ligand_col]))
        smi = html.escape(str(r[smi_col]))
        svg = r["annotated_svg"] or ""

        meta_bits: list[str] = ['<div class="mol-meta">']
        if show_index:
            meta_bits.append(f'<div class="mol-idx">{int(r["i"])}</div>')
        if show_name:
            meta_bits.append(f'<div class="mol-name">{lig}</div>')
        if show_smiles:
            meta_bits.append(f'<div class="mol-smiles"><code>{smi}</code></div>')
        meta_bits.append("</div>")
        meta_html = "\n".join(meta_bits)

        actions_html = ""
        if include_download_button:
            lig_safe_for_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(r[ligand_col]))
            lig_safe_for_stem = lig_safe_for_stem.strip("_") or "ligand"
            stem = f"{int(r['i']):03d}_{lig_safe_for_stem}"
            actions_html = ""
            if include_download_button:
                fmt = str(download_format).strip().lower()
                stem = f"{int(r['i']):03d}_{lig}"
                actions_html = (
                    f'<div class="mol-actions">'
                    f'<button type="button" class="mol-dl-btn" '
                    f'data-mol-download="one" '
                    f'data-format="{html.escape(fmt)}" '
                    f'data-png-scale="{float(png_scale)}" '
                    f'data-stem="{html.escape(stem)}">'
                    f'Download</button>'
                    f"</div>"
                )

        parts.append(
            "\n".join(
                [
                    '<div class="mol-card">',
                    f'<div class="mol-svg">{svg}</div>',
                    actions_html,
                    meta_html,
                    "</div>",
                ]
            )
        )

    parts.append("</div>")

    html_grid = "\n".join(parts)

    if output_path:
        title = output_path.split(".html")[0]
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
</head>
<body>
<p>{html.escape(title)}</p>
{html_grid}
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

    return result_df, html_grid


def main_merge_parquet(argv: Sequence[str] | None = None) -> int:
    """CLI entry for merging Parquet files.

    Args:
        argv: Optional list of CLI args for testing or entry points.

    Returns:
        Process exit code (0 on success, nonzero on error).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple Parquet files with the same schema into one "
            "file."
        )
    )
    default_in = str(Path(__file__).resolve().parent.parent / "results")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=default_in,
        help="Directory containing .parquet files to merge.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="merged.parquet",
        help="Output Parquet file path.",
    )
    parser.add_argument(
        "--require-normal-termination",
        action="store_true",
        help=("Skip any file where a '*normal_termination' column contains "
              "False/NaN."),
    )
    parser.add_argument(
        "--normal-term-cols",
        nargs="*",
        default=None,
        help=("Explicit columns to treat as normal_termination (override "
              "auto-detect)."),
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Include .parquet files in subdirectories.",
    )

    args = parser.parse_args(argv)
    try:
        out = merge_parquet_dir(
            args.input_dir,
            args.output,
            require_normal_termination=args.require_normal_termination,
            normal_term_cols=args.normal_term_cols,
            recursive=args.recursive,
        )
    except Exception as e:
        print(str(e))
        return 1
    print(f"Merged files into '{out}'.")
    return 0