"""Build documentation assets for the energy-profile tutorial.

Run from the repository root:

    python scripts/build_energy_profile_assets.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from frust.vis import plot_energy_profile


ASSET_DIR = Path("docs/assets")


def _save(fig, name: str) -> Path:
    path = ASSET_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def build_assets() -> list[Path]:
    paths: list[Path] = []

    simple_states = [
        ("Dimer", 0.0),
        ("Cat", 5.6),
        ("TS1", 25.5, "l"),
        ("int1", 1.3, "ll"),
        ("TS2", 19.1, "ll"),
        ("int2", 4.7, "lll"),
        ("TS3", 19.5, "lll"),
        ("Product", -0.2, "tr"),
    ]
    fig, _ = plot_energy_profile(
        simple_states,
        figsize=(8, 3.3),
        state_label_rotation=35,
        font_size=11,
    )
    paths.append(_save(fig, "energy-profile-minimal.png"))

    overlay_profiles = {
        "Boc-N-pyrrole": [
            ("Dimer", 0.0),
            ("Cat", 5.6),
            ("TS1", 25.5, "l"),
            ("int1", 1.3, "ll"),
            ("TS2", 19.1, "ll"),
            ("int2", 4.7, "lll"),
            ("TS3", 19.5, "lll"),
            ("Product", -0.2, "tr"),
        ],
        "Trimethoxybenzene": [
            ("Dimer", 0.0),
            ("Cat", 5.6),
            ("TS1", 25.4, "r"),
            ("int1", 5.9, "t"),
            ("TS2", 23.6),
            ("int2", 8.1, "t"),
            ("TS3", 21.6),
            ("Product", -3.6, "r"),
        ],
        "Phenyl-pyrrole": [
            ("Dimer", 0.0),
            ("Cat", 5.6),
            ("TS1", 25.0),
            ("int1", -0.6),
            ("TS2", 17.5, "b"),
            ("int2", 1.7),
            ("TS3", 17.0, "b"),
            ("Product", -0.8, "r"),
        ],
    }
    fig, _ = plot_energy_profile(
        overlay_profiles,
        figsize=(10, 3.5),
        overlay_alpha=1.0,
        state_label_rotation=35,
        font_size=11,
    )
    paths.append(_save(fig, "energy-profile-overlay.png"))

    side_reaction_profiles = {
        "DFT": [
            ("Dimer", 0.0),
            ("Cat", 5.6, "tl"),
            ("TS1", 29.6),
            ("int1", 7.4),
            ("TS2", 21.6),
            ("int2", 5.2),
            ("TS3", 19.6),
            ("int3", 16.9),
            ("TS4", 19.8),
            "side-rxn@int2@0.8#Bisarylation",
            ("TS5", 45.4),
            ("int4", 20.3),
            ("TS6", 39.4),
            ("Product", 2.8, "b"),
            ("Product + int2", -9.4),
        ],
        "Constrained-xTB/SP": [
            ("Dimer", 0.0),
            ("Cat", 5.6, "tl"),
            ("TS1", 26.9, "b"),
            ("int1", 7.4),
            ("TS2", 20.9, "b"),
            ("int2", 5.2),
            ("TS3", 17.4, "b"),
            ("int3", 16.9),
            ("TS4", 18.8, "b"),
            "side-rxn@int2@0.8#Bisarylation",
            ("TS5", 45.4),
            ("int4", 20.3),
            ("TS6", 39.4),
            ("Product", 2.8, "b"),
            ("Product + int2", -9.4),
        ],
    }
    fig, _ = plot_energy_profile(
        side_reaction_profiles,
        figsize=(12, 5),
        main_to_product_drop_frac=0.7,
        product_x_offset=0.5,
        same_energy_mode="hide",
        state_label_rotation=45,
        overlay_alpha=1.0,
        overlay_colors={"Constrained-xTB/SP": "tab:green"},
        font_size=11,
    )
    paths.append(_save(fig, "energy-profile-side-reaction.png"))

    return paths


if __name__ == "__main__":
    for path in build_assets():
        print(path)
