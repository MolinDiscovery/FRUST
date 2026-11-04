"""Build hero and scientific pipeline assets for README.md and docs/index.md.

Run from the repository root:

    conda run -n UMA python scripts/build_readme_assets.py

To also refresh the widescreen hero banner and GitHub social preview from a
rendered Blender frame (or directly from the journal-cover ``.blend`` scene):

    conda run -n UMA python scripts/build_readme_assets.py --hero-source /path/to/raw.png
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from xyzrender import load as xyz_load, render as xyz_render

import frust as ft


ASSET_DIR = Path("docs/assets/readme")

BLENDER_BIN = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def render_blender_hero(blend_path: Path, raw_out: Path, samples: int = 256) -> Path:
    """Render the trademark-free Factory artwork layer in widescreen Cycles.

    Parameters
    ----------
    blend_path : Path
        Path to ``chemical_science_artwork_v3.blend`` (opened read-only; never
        re-saved).
    raw_out : Path
        Destination PNG path for the ``2400 x 1260`` raw render.
    samples : int, default 256
        Cycles sample count (with OpenImageDenoise enabled in the scene).

    Returns
    -------
    Path
        Written raw PNG path.
    """
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    expr = f"""
import bpy
s = bpy.context.scene
c = s.camera
s.view_layers["Cover graphics"].use = False
s.view_layers["Factory artwork"].layer_collection.children["Chemical Science Decor"].exclude = True
tree = s.compositing_node_group
linear = tree.nodes["Factory · display to linear"]
out = tree.nodes["Group Output"]
for link in list(out.inputs["Image"].links):
    tree.links.remove(link)
tree.links.new(linear.outputs["Image"], out.inputs["Image"])
vh = 14.84
sw = vh * (2200.0 / 800.0)
c.data.sensor_fit = "HORIZONTAL"
c.data.sensor_width = sw
c.data.shift_x = 7.071 / sw
c.data.shift_y = -0.3235 / sw
s.render.engine = "CYCLES"
s.cycles.samples = {int(samples)}
s.render.resolution_x = 2200
s.render.resolution_y = 800
s.render.resolution_percentage = 100
s.render.filepath = {str(raw_out)!r}
bpy.ops.render.render(write_still=True)
"""
    subprocess.run(
        [str(BLENDER_BIN), "--background", str(blend_path), "--python-expr", expr],
        check=True,
    )
    return raw_out


def export_hero_variants(raw_path: Path, out_dir: Path = ASSET_DIR) -> list[Path]:
    """Export compact hero banner and GitHub social-preview JPEGs.

    Parameters
    ----------
    raw_path : Path
        Raw RGB/RGBA render of the Factory artwork scene (e.g. ``2200 x 800``).
    out_dir : Path, default ``Path("docs/assets/readme")``
        Output asset directory.

    Returns
    -------
    list of Path
        Written ``frust-hero.jpg`` and ``frust-social-preview.jpg`` paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = Image.open(raw_path).convert("RGB")
    w, h = raw.size

    hero_path = out_dir / "frust-hero.jpg"
    hero = raw.resize((2000, int(round(2000 * h / w))), Image.Resampling.LANCZOS)
    hero.save(hero_path, quality=93, subsampling=0, optimize=True)

    if w / h >= 2.0:
        target_w = int(round(h * 2.0))
        left = max(0, int(round((w - target_w) * 0.58)))
        crop_box = (left, 0, left + target_w, h)
    else:
        target_h = int(round(w / 2.0))
        top = max(0, (h - target_h) // 2)
        crop_box = (0, top, w, top + target_h)

    social = raw.crop(crop_box).resize((1280, 640), Image.Resampling.LANCZOS)
    social_path = out_dir / "frust-social-preview.jpg"
    social.save(social_path, quality=94, subsampling=0, optimize=True)
    return [hero_path, social_path]


def _adapt_rdkit_grid_for_theme(
    grid_img: Image.Image, *, dark: bool, card_rgb: tuple[int, int, int]
) -> np.ndarray:
    """Map an RDKit grid image onto the target light or dark card background."""
    arr = np.asarray(grid_img.convert("RGB")).astype(np.float32)
    if not dark:
        # Blend pure white background smoothly into the light card fill
        white_mask = (arr.min(axis=-1) > 248)[:, :, None]
        bg = np.array(card_rgb, dtype=np.float32)[None, None, :]
        arr = np.where(white_mask, bg, arr)
        return np.clip(arr, 0, 255).astype(np.uint8)

    # Dark mode: remap low-saturation greyscale [black..white] -> [light ink..card bg]
    # while boosting saturated heteroatom & C-H highlight colors so they stay vivid.
    cmax = arr.max(axis=-1)
    cmin = arr.min(axis=-1)
    chroma = (cmax - cmin) / 255.0  # 0 for neutral greyscale, >0.15 for colored features

    lum = arr.mean(axis=-1) / 255.0  # 1.0 = white bg, 0.0 = black bond/text
    bg = np.array(card_rgb, dtype=np.float32)
    ink = np.array([232.0, 238.0, 246.0], dtype=np.float32)
    neutral_mapped = ink[None, None, :] * (1.0 - lum[:, :, None]) + bg[
        None, None, :
    ] * (lum[:, :, None])

    # For colored pixels (pink/red C-H highlights, blue N, red O), darken light pastel
    # fills slightly against the dark card while keeping bright saturated strokes.
    vivid = arr.copy()
    # RDKit default highlight is pastel red ~(255, 204, 204) -> rich coral glow on dark
    pastel_red = (arr[:, :, 0] > 220) & (arr[:, :, 1] < 215) & (arr[:, :, 0] - arr[:, :, 1] > 25)
    vivid[pastel_red] = np.array([175.0, 58.0, 72.0], dtype=np.float32)
    # Boost dark blue N / red O text slightly for dark-mode legibility
    blue_ink = (arr[:, :, 2] > arr[:, :, 0] + 50) & (arr[:, :, 0] < 120)
    vivid[blue_ink] = np.array([96.0, 165.0, 250.0], dtype=np.float32)
    red_ink = (arr[:, :, 0] > arr[:, :, 2] + 60) & (arr[:, :, 1] < 120)
    vivid[red_ink] = np.array([248.0, 113.0, 113.0], dtype=np.float32)

    weight = np.clip((chroma - 0.06) / 0.16, 0.0, 1.0)[:, :, None]
    out = neutral_mapped * (1.0 - weight) + vivid * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def _orient_ts1_plane(coords: np.ndarray, roles: dict[str, int]) -> np.ndarray:
    """Rotate 3D coordinates so the B-C-H-N reactive ring and aromatic rings read cleanly."""
    i_b = roles["cat_B"]
    i_n = roles["cat_N"]
    i_c = roles["substrate_C"]
    i_h = roles["transfer_H"]

    center = coords[[i_b, i_n, i_c, i_h]].mean(axis=0)
    shifted = coords - center

    # Horizontal axis: substrate_C (left) -> cat_N (right), matching the hero banner
    ex = shifted[i_n] - shifted[i_c]
    ex = ex / np.linalg.norm(ex)

    # Vertical axis: transfer_H/substrate_C (lower) -> cat_B (upper)
    v_up = shifted[i_b] - 0.5 * (shifted[i_c] + shifted[i_h])
    ey = v_up - np.dot(v_up, ex) * ex
    ey = ey / np.linalg.norm(ey)

    ez = np.cross(ex, ey)
    aligned = shifted @ np.stack([ex, ey, ez], axis=1)

    # Gentle yaw/pitch so both the 1-methylpyrrole and phenylene rings open toward camera
    yaw = np.deg2rad(22.0)
    ry = np.array(
        [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
    )
    pitch = np.deg2rad(10.0)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    return aligned @ ry @ rx


def _render_ts1_xyzrender(
    ts1_row: pd.Series, *, dark: bool, card_bg: str, card_rgb: tuple[int, int, int]
) -> np.ndarray:
    """Render the real tsguess2 TS1 3D geometry with xyzrender and crop tightly."""
    atoms: list[str] = list(ts1_row["atoms"])
    bonds: list[tuple[int, int]] = [
        (int(b[0]), int(b[1])) for b in ts1_row["connectivity_bonds"]
    ]
    roles: dict[str, int] = dict(ts1_row["constraint_roles"])
    coords = _orient_ts1_plane(np.asarray(ts1_row["coords_embedded"], dtype=float), roles)

    i_b = roles["cat_B"]
    i_n = roles["cat_N"]
    i_c = roles["substrate_C"]
    i_h = roles["transfer_H"]

    # Show BH2 hydrogens + transferring proton (1-based indices for xyzrender)
    bh_1based = [
        (v + 1 if u == i_b else u + 1)
        for u, v in bonds
        if (u == i_b or v == i_b) and atoms[v if u == i_b else u] == "H"
    ]
    hy_1based = sorted(set(bh_1based + [i_h + 1]))

    b1, n1, c1, h1 = i_b + 1, i_n + 1, i_c + 1, i_h + 1
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        xyz_path = tmp / "ts1.xyz"
        xyz_lines = [f"{len(atoms)}", "TS1"] + [
            f"{sym} {x:.5f} {y:.5f} {z:.5f}" for sym, (x, y, z) in zip(atoms, coords)
        ]
        xyz_path.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")

        png_path = tmp / "ts1.png"
        mol = xyz_load(xyz_path)
        xyz_render(
            mol,
            orient=False,
            hy=hy_1based,
            bo=True,
            ts_bonds=[(b1, c1), (c1, h1), (n1, h1)],
            unbond=[f"{b1}-{c1}", f"{c1}-{h1}", f"{n1}-{h1}"],
            ts_color="#4ade80" if dark else "#16a34a",
            bond_color="#cbd5e1" if dark else "#1e293b",
            atom_stroke_width=1.6,
            labels=[f"{b1} B", f"{c1} C2", f"{h1} H", f"{n1} N"],
            label_font_size=46,
            fog=True,
            fog_strength=0.42,
            fog_color=card_bg,
            background=card_bg,
            canvas_size=820,
            output=png_path,
        )
        img = Image.open(png_path).convert("RGB")

    arr = np.asarray(img)
    bg_vec = np.array(card_rgb, dtype=np.int16)[None, None, :]
    fg_mask = np.abs(arr.astype(np.int16) - bg_vec).max(axis=-1) > 4
    ys = np.where(fg_mask.any(axis=1))[0]
    xs = np.where(fg_mask.any(axis=0))[0]
    pad = 18
    cropped = img.crop(
        (
            max(0, int(xs[0]) - pad),
            max(0, int(ys[0]) - pad),
            min(img.width, int(xs[-1]) + pad + 1),
            min(img.height, int(ys[-1]) + pad + 1),
        )
    )
    return np.asarray(cropped)


def build_pipeline_showcase(out_dir: Path = ASSET_DIR) -> list[Path]:
    """Generate light and dark 3-panel FRUST pipeline figures sized for GitHub.

    Parameters
    ----------
    out_dir : Path, default ``Path("docs/assets/readme")``
        Destination directory for the showcase PNG files.

    Returns
    -------
    list of Path
        Written light and dark showcase image paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Single iconic 1-methylpyrrole site-discovery drawing (cropping RDKit's tiny legend)
    raw_ch = ft.DrawUniqueChGrid(
        "CN1C=CC=C1",
        mols_per_row=1,
        sub_img_size=(380, 480),
    )
    ch_arr = np.asarray(raw_ch.convert("RGB"))
    ink_rows = np.where((ch_arr.min(axis=-1) < 245).any(axis=1))[0]
    row_gaps = np.where(np.diff(ink_rows) > 30)[0]
    mol_y1 = int(ink_rows[row_gaps[0]]) if len(row_gaps) > 0 else int(ink_rows[-1])
    mol_y0 = int(ink_rows[0])
    ink_cols = np.where((ch_arr[mol_y0 : mol_y1 + 1].min(axis=-1) < 245).any(axis=0))[0]
    pad = 16
    ch_cropped = raw_ch.crop(
        (
            max(0, int(ink_cols[0]) - pad),
            max(0, mol_y0 - pad),
            min(raw_ch.width, int(ink_cols[-1]) + pad + 1),
            min(raw_ch.height, mol_y1 + pad + 1),
        )
    )

    # 2. Real 3D TS1 guess for 1-methylpyrrole + TMP-Bcat via FRUST public API
    components = pd.DataFrame(
        [
            {
                "role": "substrate",
                "smiles": "CN1C=CC=C1",
                "compound_name": "n_methyl_pyrrole",
                "rpos": "2",
            },
            {
                "role": "catalyst",
                "smiles": "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B",
                "compound_name": "tmp_bcat",
                "rpos": "",
            },
        ]
    )
    systems = ft.screen.expand(components)
    ts1_df = ft.screen.create_ts_guesses(
        systems, ts_types=["TS1"], n_confs=1, random_seed=42
    )["TS1"]
    ts1_row = ts1_df.iloc[0]

    # 3. Essential catalytic milestones + competing bisarylation branch (uncluttered ticks)
    rxn_profiles = {
        "C–H cycle": [
            ("Dimer", 0.0),
            ("TS1", 29.6),
            ("int2", 5.2),
            ("TS3", 19.6),
            "side-rxn@int2@0.75#Bisarylation",
            ("TS5", 45.4),
            ("Prod", 2.8, "r"),
        ],
    }

    written: list[Path] = []
    for dark, suffix in [(False, "light"), (True, "dark")]:
        canvas_bg = "#0d1117" if dark else "#ffffff"
        card_bg = "#161b22" if dark else "#f8fafc"
        card_rgb = (22, 27, 34) if dark else (248, 250, 252)
        border_col = "#30363d" if dark else "#cbd5e1"
        title_col = "#f0f6fc" if dark else "#0f172a"
        sub_col = "#8b949e" if dark else "#475569"
        accent_col = "#3fb950" if dark else "#15803d"

        # 9.85 in * 96 CSS px/in = 946 CSS px (1:1 match to GitHub README width!)
        fig = plt.figure(figsize=(9.85, 3.85), dpi=200, facecolor=canvas_bg)
        bg_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=-10)
        bg_ax.set_xlim(0.0, 1.0)
        bg_ax.set_ylim(0.0, 1.0)
        bg_ax.axis("off")

        card_specs = [
            (
                0.010,
                0.312,
                "1 · Unique Reactive Sites",
                "ft.DrawUniqueChGrid",
            ),
            (
                0.340,
                0.316,
                "2 · 3D TS1 Assembly",
                "ft.screen.create_ts_guesses",
            ),
            (
                0.674,
                0.316,
                "3 · Cycle & Side Rxns",
                "ft.plot_energy_profile",
            ),
        ]

        for x0, width, heading, api_call in card_specs:
            card = FancyBboxPatch(
                (x0, 0.03),
                width,
                0.94,
                boxstyle="round,pad=0.0,rounding_size=0.024",
                facecolor=card_bg,
                edgecolor=border_col,
                linewidth=1.25,
                zorder=1,
            )
            bg_ax.add_patch(card)
            fig.text(
                x0 + 0.018,
                0.908,
                heading,
                fontsize=10.8,
                fontweight="bold",
                color=title_col,
                va="center",
            )
            fig.text(
                x0 + 0.018,
                0.842,
                api_call,
                fontsize=8.8,
                fontweight="bold",
                family="monospace",
                color=accent_col,
                va="center",
            )

        # Panel 1: Crisp single 1-methylpyrrole structure + legible caption
        ax1 = fig.add_axes([0.028, 0.135, 0.276, 0.665], facecolor=card_bg, zorder=5)
        themed_grid = _adapt_rdkit_grid_for_theme(
            ch_cropped, dark=dark, card_rgb=card_rgb
        )
        ax1.imshow(themed_grid, interpolation="bilinear")
        ax1.axis("off")
        fig.text(
            0.166,
            0.075,
            "1-methylpyrrole · unique C–H: 2, 3",
            fontsize=8.8,
            color=sub_col,
            ha="center",
            va="center",
        )

        # Panel 2: 3D TS1 guess rendered with xyzrender
        ax2 = fig.add_axes([0.348, 0.125, 0.300, 0.680], facecolor=card_bg, zorder=5)
        ts1_img = _render_ts1_xyzrender(
            ts1_row, dark=dark, card_bg=card_bg, card_rgb=card_rgb
        )
        ax2.imshow(ts1_img, interpolation="bilinear")
        ax2.axis("off")
        fig.text(
            0.498,
            0.075,
            "TMP-Bcat + C2 · dynamic role core",
            fontsize=8.8,
            color=sub_col,
            ha="center",
            va="center",
        )

        # Panel 3: Clean milestone energy profile
        ax3 = fig.add_axes([0.720, 0.175, 0.256, 0.610], facecolor=card_bg, zorder=5)
        with ft.vis.use_darkmode(dark):
            ft.plot_energy_profile(
                rxn_profiles,
                ax=ax3,
                ylabel="ΔG (kcal/mol)",
                state_label_rotation=32,
                font_size=8.8,
                energy_fontsize=8.6,
                state_label_fontsize=8.8,
                overlay_alpha=1.0,
                show_legend=True,
            )
        ax3.set_facecolor(card_bg)
        ax3.yaxis.label.set_color(title_col)
        ax3.yaxis.label.set_fontsize(9.0)
        ax3.tick_params(axis="x", colors=title_col)
        for tick_lbl in ax3.get_xticklabels():
            tick_lbl.set_color(title_col)
        leg = ax3.get_legend()
        if leg is not None:
            leg.set_frame_on(False)
            for txt in leg.get_texts():
                txt.set_fontsize(8.4)
                txt.set_color(title_col)

        out_path = out_dir / f"pipeline-showcase-{suffix}.png"
        fig.savefig(
            out_path,
            dpi=200,
            facecolor=canvas_bg,
            pil_kwargs={"optimize": True, "compress_level": 9},
        )
        plt.close(fig)
        written.append(out_path)

    return written


def build_workflow_funnel(out_dir: Path = ASSET_DIR) -> list[Path]:
    """Generate light and dark composed catalyst_screen architecture figures.

    Parameters
    ----------
    out_dir : Path, default ``Path("docs/assets/readme")``
        Destination directory for the funnel PNG files.

    Returns
    -------
    list of Path
        Written light and dark workflow funnel image paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    stages = [
        {
            "pill": "screen.csv · tsguess2",
            "title": "1 · Build Targets",
            "lines": [
                "Catalysts × Substrates",
                "3D TS1–TS8 + PRISM",
            ],
            "footer": "Conformer ensemble",
            "color_light": ("#0f766e", "#ccfbf1"),
            "color_dark": ("#2dd4bf", "#11322f"),
        },
        {
            "pill": "screening=",
            "title": "2 · Screen",
            "lines": [
                "GFN-FF / g-xTB  or  UMA",
                "Constrained core opt [C]",
            ],
            "footer": "→ Keep top ~20",
            "color_light": ("#b45309", "#fef3c7"),
            "color_dark": ("#fbbf24", "#3b2506"),
        },
        {
            "pill": "ranking=",
            "title": "3 · Rerank",
            "lines": [
                "UMA (ALPB)  or  DFT SP",
                "Solvent-aware scoring",
            ],
            "footer": "→ Shortlist 1–3",
            "color_light": ("#7e22ce", "#f3e8ff"),
            "color_dark": ("#c084fc", "#2e1065"),
        },
        {
            "pill": "method=",
            "title": "4 · Validate ΔG‡",
            "lines": [
                "UMA  or  ωB97X-D3",
                "OptTS + Freq + Review",
            ],
            "footer": "→ Result bundle",
            "color_light": ("#15803d", "#dcfce7"),
            "color_dark": ("#4ade80", "#0f2e1c"),
        },
    ]

    for dark, suffix in [(False, "light"), (True, "dark")]:
        canvas_bg = "#0d1117" if dark else "#ffffff"
        card_bg = "#161b22" if dark else "#f8fafc"
        subbar_bg = "#11161d" if dark else "#f1f5f9"
        border_col = "#30363d" if dark else "#cbd5e1"
        title_col = "#f0f6fc" if dark else "#0f172a"
        body_col = "#c9d1d9" if dark else "#334155"
        muted_col = "#8b949e" if dark else "#475569"
        arrow_col = "#58a6ff" if dark else "#2563eb"
        ref_accent = "#38bdf8" if dark else "#0284c7"
        ref_fill = "#0c2d48" if dark else "#e0f2fe"

        # 9.85 in * 96 CSS px/in = 946 CSS px (1:1 GitHub column scale!)
        fig = plt.figure(figsize=(9.85, 3.25), dpi=200, facecolor=canvas_bg)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=canvas_bg)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

        n = len(stages)
        card_w = 0.226
        gap = 0.024
        start_x = 0.012
        card_y = 0.32
        card_h = 0.64

        xs: list[float] = []
        for idx, st in enumerate(stages):
            x0 = start_x + idx * (card_w + gap)
            xs.append(x0)
            fg_acc, bg_acc = st["color_dark"] if dark else st["color_light"]

            card = FancyBboxPatch(
                (x0, card_y),
                card_w,
                card_h,
                boxstyle="round,pad=0.0,rounding_size=0.024",
                facecolor=card_bg,
                edgecolor=fg_acc if idx == n - 1 else border_col,
                linewidth=1.6 if idx == n - 1 else 1.25,
                zorder=2,
            )
            ax.add_patch(card)

            pill = FancyBboxPatch(
                (x0 + 0.014, card_y + card_h - 0.165),
                card_w - 0.028,
                0.120,
                boxstyle="round,pad=0.0,rounding_size=0.016",
                facecolor=bg_acc,
                edgecolor=fg_acc,
                linewidth=1.0,
                zorder=3,
            )
            ax.add_patch(pill)
            ax.text(
                x0 + card_w / 2.0,
                card_y + card_h - 0.105,
                st["pill"],
                fontsize=9.0,
                fontweight="bold",
                family="monospace",
                color=fg_acc,
                ha="center",
                va="center",
                zorder=4,
            )

            ax.text(
                x0 + card_w / 2.0,
                card_y + card_h - 0.265,
                st["title"],
                fontsize=10.5,
                fontweight="bold",
                color=title_col,
                ha="center",
                va="center",
                zorder=4,
            )

            for l_i, line in enumerate(st["lines"]):
                ax.text(
                    x0 + card_w / 2.0,
                    card_y + card_h - 0.390 - l_i * 0.105,
                    line,
                    fontsize=8.9,
                    color=body_col,
                    ha="center",
                    va="center",
                    zorder=4,
                )

            ax.text(
                x0 + card_w / 2.0,
                card_y + 0.072,
                st["footer"],
                fontsize=8.9,
                fontweight="bold",
                color=fg_acc,
                ha="center",
                va="center",
                zorder=4,
            )

            if idx < n - 1:
                ax_x0 = x0 + card_w + 0.003
                ax_x1 = x0 + card_w + gap - 0.003
                arr = FancyArrowPatch(
                    (ax_x0, card_y + card_h * 0.50),
                    (ax_x1, card_y + card_h * 0.50),
                    arrowstyle="-|>,head_length=5.5,head_width=4.2",
                    linewidth=1.9,
                    color=arrow_col,
                    zorder=5,
                )
                ax.add_patch(arr)

        # Bottom Left (aligned under Cards 1-2): Execution parity
        exec_w = xs[1] + card_w - xs[0]
        exec_box = FancyBboxPatch(
            (xs[0], 0.045),
            exec_w,
            0.205,
            boxstyle="round,pad=0.0,rounding_size=0.022",
            facecolor=subbar_bg,
            edgecolor=border_col,
            linewidth=1.15,
            zorder=2,
        )
        ax.add_patch(exec_box)
        ax.text(
            xs[0] + exec_w / 2.0,
            0.182,
            "Local ⇄ Slurm Execution Parity",
            fontsize=9.4,
            fontweight="bold",
            color=title_col,
            ha="center",
            va="center",
            zorder=4,
        )
        ax.text(
            xs[0] + exec_w / 2.0,
            0.100,
            "wf.plan()  →  wf.run(debug=True)  ⇄  wf.submit(array=True)",
            fontsize=8.4,
            family="monospace",
            color=muted_col,
            ha="center",
            va="center",
            zorder=4,
        )

        # Bottom Right (aligned under Cards 3-4): FRUST_REFERENCE_STORE
        ref_x0 = xs[2]
        ref_w = (xs[3] + card_w) - ref_x0
        ref_box = FancyBboxPatch(
            (ref_x0, 0.045),
            ref_w,
            0.205,
            boxstyle="round,pad=0.0,rounding_size=0.022",
            facecolor=ref_fill,
            edgecolor=ref_accent,
            linewidth=1.25,
            zorder=2,
        )
        ax.add_patch(ref_box)
        ax.text(
            ref_x0 + ref_w / 2.0,
            0.182,
            "FRUST_REFERENCE_STORE  ·  Reference Library",
            fontsize=9.2,
            fontweight="bold",
            family="monospace",
            color=ref_accent,
            ha="center",
            va="center",
            zorder=4,
        )
        ax.text(
            ref_x0 + ref_w / 2.0,
            0.100,
            "Reuses reviewed catalyst dimers, substrates, HBpin & H₂",
            fontsize=8.8,
            color=body_col,
            ha="center",
            va="center",
            zorder=4,
        )

        up_x = xs[3] + card_w * 0.50
        up_arr = FancyArrowPatch(
            (up_x, 0.252),
            (up_x, card_y - 0.004),
            arrowstyle="-|>,head_length=5.5,head_width=4.2",
            linewidth=1.9,
            color=ref_accent,
            zorder=5,
        )
        ax.add_patch(up_arr)

        out_path = out_dir / f"workflow-funnel-{suffix}.png"
        fig.savefig(
            out_path,
            dpi=200,
            facecolor=canvas_bg,
            pil_kwargs={"optimize": True, "compress_level": 9},
        )
        plt.close(fig)
        written.append(out_path)

    return written


def main() -> None:
    """Build README and documentation landing-page visual assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hero-source",
        type=Path,
        default=None,
        help="Optional 2400x1260 raw render PNG to export hero/social JPEGs from.",
    )
    parser.add_argument(
        "--blend-path",
        type=Path,
        default=None,
        help="Optional path to chemical_science_artwork_v3.blend for headless Cycles rendering.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Cycles sample count when --blend-path is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSET_DIR,
        help="Output asset directory (default: docs/assets/readme).",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.blend_path is not None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = render_blender_hero(
                args.blend_path.resolve(), Path(tmp) / "hero_raw.png", samples=args.samples
            )
            paths.extend(export_hero_variants(raw, out_dir=args.output))
    elif args.hero_source is not None:
        paths.extend(export_hero_variants(args.hero_source.resolve(), out_dir=args.output))

    paths.extend(build_pipeline_showcase(out_dir=args.output))
    paths.extend(build_workflow_funnel(out_dir=args.output))
    for path in paths:
        size_kb = path.stat().st_size / 1024.0
        print(f"{path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
