from frust.schema import latest_opt_coords_column

from . import theme


def plot_vibs(
    df,
    row_index=0,
    vId: int = 0,
    width: float = 600,
    height: float = 400,
    numFrames: int = 20,
    amplitude: float = 1,
    transparent: bool = True,
    fps: float | None = None,
    reps: int = 50,
    custom_coords_col_name: str | None = None,
    row_indices: list[int] | None = None,
    viewergrid: tuple[int, int] | None = None,
    linked: bool = True,
    freq_label: bool = True,
    legends: list[str] | None = None,
    legend_screen_offset: dict | None = None,  # e.g. {'x': 10, 'y': 6}
    export_HTML: str = ""
):
    import math

    import py3Dmol
    from tooltoad.vis import ac2xyz, show_vibs

    vibs_col = [c for c in df.columns if "vibs" in c][-1]
    print(f"vibs col {vibs_col}")
    vibs_col_pre = vibs_col.split("vibs")[0]
    coords_col = latest_opt_coords_column(vibs_col_pre, df) or vibs_col_pre + "oc"
    if custom_coords_col_name:
        coords_col = custom_coords_col_name

    # ---- Single-row (unchanged rendering, improved label) ----
    if row_indices is None or (isinstance(row_indices, list)
                               and len(row_indices) == 1):
        if isinstance(row_indices, list) and len(row_indices) == 1:
            row_index = row_indices[0]

        atoms = df["atoms"].iloc[row_index]
        vibs = df[vibs_col].iloc[row_index]
        coords = df[coords_col].iloc[row_index]

        view_dict = {"atoms": atoms, "opt_coords": coords, "vibs": vibs}
        bg = "0x000000" if theme.darkmode else "0xeeeeee"

        view = show_vibs(
            view_dict,
            vId,
            width,
            height,
            numFrames,
            amplitude,
            transparent,
            fps,
            reps,
            background_color=bg,
            export_HTML=export_HTML,
        )

        if freq_label:
            vib = vibs[vId]
            freq = vib["frequency"]
            label_text = (
                legends[0] if legends and len(legends) >= 1
                else f"{freq:.1f} cm^-1"
            )
            # Screen-anchored like MolTo3DGrid
            font_color = "white" if theme.darkmode else "black"
            back_color = "black" if theme.darkmode else "white"
            offs = legend_screen_offset or {"x": 10, "y": 6}
            view.addLabel(
                label_text,
                {
                    "useScreen": True,
                    "inFront": True,
                    "fontSize": 14,
                    "fontColor": font_color,
                    "backgroundColor": back_color,
                    "borderColor": font_color,
                    "borderWidth": 1,
                    "screenOffset": offs,  # top-left corner offset
                },
            )
        return view

    # ---- Grid across multiple rows ----
    idxs = list(row_indices)
    if legends is not None and len(legends) != len(idxs):
        raise ValueError("Length of legends must match row_indices.")

    interval_ms = 50 if fps is None else max(1, int(1000.0 / fps))

    n = len(idxs)
    if viewergrid is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    else:
        rows, cols = viewergrid

    bg = "0x000000" if theme.darkmode else "0xeeeeee"
    alpha = 0 if transparent else 1

    p = py3Dmol.view(width=width, height=height,
                     viewergrid=(rows, cols), linked=linked)

    for i, ri in enumerate(idxs):
        atoms = df["atoms"].iloc[ri]
        vibs = df[vibs_col].iloc[ri]
        coords = df[coords_col].iloc[ri]

        vib = vibs[vId]
        mode = vib["mode"]
        freq = vib["frequency"]

        xyz = ac2xyz(atoms, coords)

        r, c = divmod(i, cols)
        p.addModel(xyz, "xyz", viewer=(r, c))

        propmap = [
            {"index": j, "props": {"dx": m[0], "dy": m[1], "dz": m[2]}}
            for j, m in enumerate(mode)
        ]
        p.mapAtomProperties(propmap, viewer=(r, c))

        p.vibrate(numFrames, amplitude, True, viewer=(r, c))
        p.animate(
            {"loop": "backAndForth", "interval": interval_ms, "reps": reps},
            viewer=(r, c),
        )

        p.setStyle({"sphere": {"radius": 0.4}, "stick": {}}, viewer=(r, c))
        p.setBackgroundColor(bg, alpha, viewer=(r, c))
        p.zoomTo(viewer=(r, c))

        if freq_label:
            label_text = (
                legends[i] if legends else f"{freq:.1f} cm^-1"
            )
            font_color = "white" if theme.darkmode else "black"
            back_color = "black" if theme.darkmode else "white"
            offs = legend_screen_offset or {"x": 10, "y": 6}
            p.addLabel(
                label_text,
                {
                    "useScreen": True,
                    "inFront": True,
                    "fontSize": 14,
                    "fontColor": font_color,
                    "backgroundColor": back_color,
                    "borderColor": font_color,
                    "borderWidth": 1,
                    "screenOffset": offs,  # per-cell top-left
                },
                viewer=(r, c),
            )

    return p
