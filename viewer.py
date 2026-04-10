import os
import sys
import glob
import numpy as np
import panel as pn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import Turbo256

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

pn.extension()

# ----------------------------
# Input path
# ----------------------------
if len(sys.argv) > 1:
    log_folder = sys.argv[1]
else:
    log_folder = "fsc_logs"

# ----------------------------
# Load static
# ----------------------------
static = np.load(os.path.join(log_folder, "run_static.npz"), allow_pickle=True)
static_data = static["static_data"][0]

# Quantum-layer info
Qsites = np.array(static_data["Qsites"], dtype=int)
coords_q = np.array(static_data["coords_q"], dtype=float)   # (x, y)

# Full-system info for surface cuts
site_ids_all = np.array(static_data["site_ids_all"], dtype=int)
coords_all = np.array(static_data["coords_all"], dtype=float)  # (x, y, z)

# ----------------------------
# Load snapshots
# ----------------------------
files = sorted(
    f for f in glob.glob(os.path.join(log_folder, "*.npz"))
    if "run_static" not in f
)

file_map = {os.path.basename(f): f for f in files}
snapshot_names = list(file_map.keys())

if len(snapshot_names) == 0:
    raise RuntimeError(f"No snapshot files found in {log_folder}")

# ----------------------------
# Helpers
# ----------------------------
def load_snapshot(name):
    return np.load(file_map[name], allow_pickle=True)

def get_prev_snapshot(name):
    idx = snapshot_names.index(name)
    if idx == 0:
        return None
    return snapshot_names[idx - 1]

def qprime_mask_from_data(data):
    return np.isin(Qsites, np.asarray(data["Qprime"], dtype=int))

def set_mapper_range_from_qprime(vals, qmask):
    vals = np.asarray(vals, dtype=float)

    vals_q = vals[qmask]
    vals_q = vals_q[np.isfinite(vals_q)]

    if vals_q.size == 0:
        vals_q = vals[np.isfinite(vals)]

    if vals_q.size == 0:
        low, high = 0.0, 1.0
    else:
        low = float(np.nanmin(vals_q))
        high = float(np.nanmax(vals_q))
        if np.isclose(low, high):
            pad = 1e-12 if low == 0 else 1e-6 * abs(low)
            low -= pad
            high += pad

    mapper.low = low
    mapper.high = high

def ldos_at_energy(data, energy_mode="Ui"):
    """
    Evaluate LDOS on each quantum site at either:
    - energy_mode = "Ui"  -> rho(E = Ui_site)
    - energy_mode = "0"   -> rho(E = 0)
    """
    if "ildos" not in data:
        return np.full(len(Qsites), np.nan, dtype=float)

    ildos = data["ildos"]
    Ui_all = np.asarray(data["Ui"], dtype=float)

    vals = np.full(len(Qsites), np.nan, dtype=float)

    for sidx, site in enumerate(Qsites):
        try:
            E = np.asarray(ildos[sidx][0], dtype=float)
            rho = np.asarray(ildos[sidx][1], dtype=float)

            if E.size < 2:
                continue

            if energy_mode == "Ui":
                target_E = float(Ui_all[site])
            elif energy_mode == "0":
                target_E = 0.0
            else:
                target_E = 0.0

            vals[sidx] = np.interp(target_E, E, rho, left=np.nan, right=np.nan)
        except Exception:
            vals[sidx] = np.nan

    return vals

def scalar_values_quantum(data, prop_name, snapshot_name):
    qmask = qprime_mask_from_data(data)

    if prop_name == "Ui":
        vals = np.asarray(data["Ui"], dtype=float)[Qsites]

    elif prop_name == "ni":
        vals = np.asarray(data["ni"], dtype=float)[Qsites]

    elif prop_name == "Ci":
        vals = np.full(len(Qsites), np.nan, dtype=float)
        Ci = np.asarray(data["Ci"], dtype=float)
        Qp = np.asarray(data["Qprime"], dtype=int)
        for i, s in enumerate(Qp[:len(Ci)]):
            idx = np.where(Qsites == s)[0]
            if len(idx):
                vals[idx[0]] = Ci[i]

    elif prop_name == "Qprime_mask":
        vals = qmask.astype(float)

    elif prop_name == "ΔUi":
        prev = get_prev_snapshot(snapshot_name)
        if prev is None:
            vals = np.zeros(len(Qsites), dtype=float)
        else:
            prev_data = load_snapshot(prev)
            vals = (
                np.asarray(data["Ui"], dtype=float)[Qsites]
                - np.asarray(prev_data["Ui"], dtype=float)[Qsites]
            )

    elif prop_name == "LDOS@0":
        vals = ldos_at_energy(data, energy_mode="0")

    elif prop_name == "LDOS@Ui":
        vals = ldos_at_energy(data, energy_mode="Ui")

    else:
        vals = np.zeros(len(Qsites), dtype=float)

    return vals

def scalar_values_surface(data, prop_name, snapshot_name):
    ids = site_ids_all

    if prop_name == "Ui":
        return np.asarray(data["Ui"], dtype=float)[ids]

    elif prop_name == "ni":
        return np.asarray(data["ni"], dtype=float)[ids]

    elif prop_name == "ΔUi":
        prev = get_prev_snapshot(snapshot_name)
        if prev is None:
            return np.zeros(len(ids), dtype=float)
        prev_data = load_snapshot(prev)
        return (
            np.asarray(data["Ui"], dtype=float)[ids]
            - np.asarray(prev_data["Ui"], dtype=float)[ids]
        )

    elif prop_name == "Ci":
        vals = np.full(len(ids), np.nan, dtype=float)
        Ci = np.asarray(data["Ci"], dtype=float)
        Qp = np.asarray(data["Qprime"], dtype=int)
        id_to_all = {sid: i for i, sid in enumerate(ids)}
        for i, sid in enumerate(Qp[:len(Ci)]):
            if sid in id_to_all:
                vals[id_to_all[sid]] = Ci[i]
        return vals

    else:
        return np.full(len(ids), np.nan, dtype=float)

def get_surface_cut_indices(cut_direction, cut_center, cut_width):
    """
    Select all full-system sites in a slab.

    cut_direction == "x": keep |y - cut_center| <= cut_width/2, plot (x, z)
    cut_direction == "y": keep |x - cut_center| <= cut_width/2, plot (y, z)
    """
    x = coords_all[:, 0]
    y = coords_all[:, 1]
    z = coords_all[:, 2]

    if cut_direction == "x":
        mask = np.abs(y - cut_center) <= cut_width / 2
        coord_along = x[mask]
        coord_vert = z[mask]
        idx = np.where(mask)[0]
    else:
        mask = np.abs(x - cut_center) <= cut_width / 2
        coord_along = y[mask]
        coord_vert = z[mask]
        idx = np.where(mask)[0]

    return idx, coord_along, coord_vert

def get_quantum_line_cut_indices(cut_direction, cut_center, cut_width):
    """
    Select quantum-layer sites near a line cut in the 2D layer.
    """
    x = coords_q[:, 0]
    y = coords_q[:, 1]

    if cut_direction == "x":
        mask = np.abs(y - cut_center) <= cut_width / 2
        coord_along = x[mask]
        idx = np.where(mask)[0]
    else:
        mask = np.abs(x - cut_center) <= cut_width / 2
        coord_along = y[mask]
        idx = np.where(mask)[0]

    order = np.argsort(coord_along)
    return idx[order], coord_along[order]

# ----------------------------
# Widgets
# ----------------------------
snapshot_slider = pn.widgets.DiscreteSlider(
    name="Snapshot",
    options=snapshot_names,
    value=snapshot_names[0],
    width=320
)

snapshot_player = pn.widgets.Player(
    name="Play snapshots",
    start=0,
    end=max(len(snapshot_names) - 1, 0),
    value=0,
    interval=500,
    loop_policy="loop",
    width=320
)

# Main left heatmap property
prop_select = pn.widgets.Select(
    name="Main heatmap",
    options=["Ui", "ni", "Ci", "Qprime_mask", "ΔUi", "LDOS@0", "LDOS@Ui"],
    value="Ui",
    width=320
)

# Surface plot property
surface_prop_select = pn.widgets.Select(
    name="Surface property",
    options=["Ui", "ni", "Ci", "ΔUi"],
    value="Ui",
    width=320
)

site_select = pn.widgets.IntInput(
    name="Site",
    value=int(Qsites[0]),
    width=320
)

cut_direction_select = pn.widgets.Select(
    name="Cut direction",
    options=["x", "y"],
    value="x",
    width=320
)

cut_center_input = pn.widgets.FloatInput(
    name="Cut center",
    value=float(np.median(coords_q[:, 1])),
    step=0.01,
    width=320
)

cut_width_input = pn.widgets.FloatInput(
    name="Cut width",
    value=0.05,
    step=0.01,
    width=320
)

info_panel = pn.pane.Markdown("", width=320)
bounds_panel = pn.pane.Markdown("", width=320)

# ----------------------------
# Bokeh sources
# ----------------------------
main_source = ColumnDataSource(data=dict(x=[], y=[], site=[], value=[]))
selected_source = ColumnDataSource(data=dict(x=[], y=[]))
max_ui_source = ColumnDataSource(data=dict(x=[], y=[]))

# ----------------------------
# Main quantum heatmap
# ----------------------------
fig = figure(
    width=900,
    height=900,
    tools="pan,wheel_zoom,box_zoom,reset,tap",
    active_scroll="wheel_zoom",
    match_aspect=True
)

mapper = LinearColorMapper(palette=Turbo256)

renderer = fig.scatter(
    "x", "y",
    source=main_source,
    size=15,
    marker="circle",
    fill_color={"field": "value", "transform": mapper},
    line_color=None
)

fig.scatter(
    "x", "y", source=selected_source,
    size=16, marker="cross",
    line_color="red"
)

fig.scatter(
    "x", "y", source=max_ui_source,
    size=18, marker="diamond",
    line_color="orange", fill_color=None, line_width=3
)

fig.add_layout(ColorBar(color_mapper=mapper), "right")

# ----------------------------
# Snapshot player linkage
# ----------------------------
def _player_to_slider(event):
    snapshot_slider.value = snapshot_names[event.new]

snapshot_player.param.watch(_player_to_slider, "value")

def _slider_to_player(event):
    try:
        snapshot_player.value = snapshot_names.index(event.new)
    except ValueError:
        pass

snapshot_slider.param.watch(_slider_to_player, "value")

# ----------------------------
# Update main heatmap
# ----------------------------
def update_system():
    data = load_snapshot(snapshot_slider.value)
    qmask = qprime_mask_from_data(data)

    vals = scalar_values_quantum(data, prop_select.value, snapshot_slider.value)
    set_mapper_range_from_qprime(vals, qmask)

    main_source.data = dict(
        x=coords_q[:, 0],
        y=coords_q[:, 1],
        site=Qsites,
        value=vals
    )

    Ui_vals = np.asarray(data["Ui"], dtype=float)[Qsites]
    max_idx = int(np.argmax(Ui_vals))
    max_ui_source.data = dict(
        x=[coords_q[max_idx, 0]],
        y=[coords_q[max_idx, 1]]
    )

    update_selected_marker()
    update_bounds_panel()

# ----------------------------
# Energy bounds panel
# ----------------------------
def update_bounds_panel():
    data = load_snapshot(snapshot_slider.value)

    if "energy_bounds" in data and len(data["energy_bounds"]) == 2:
        emin, emax = data["energy_bounds"]
        bounds_panel.object = (
            f"### Energy bounds\n"
            f"- Emin = {emin:.4g}\n"
            f"- Emax = {emax:.4g}"
        )
    else:
        bounds_panel.object = "### Energy bounds\n- N/A"

# ----------------------------
# Selection handling
# ----------------------------
def update_selected_marker():
    site = int(site_select.value)
    idx = np.where(Qsites == site)[0]
    if len(idx) == 0:
        selected_source.data = dict(x=[], y=[])
        return
    idx = idx[0]
    selected_source.data = dict(
        x=[coords_q[idx, 0]],
        y=[coords_q[idx, 1]]
    )

def on_click(attr, old, new):
    if len(new) > 0:
        idx = new[0]
        site_select.value = int(main_source.data["site"][idx])

renderer.data_source.selected.on_change("indices", on_click)

# ----------------------------
# Local selected-site plots
# ----------------------------
@pn.depends(snapshot_slider, site_select)
def local_plot(snapshot_name, site):
    data = load_snapshot(snapshot_name)

    if site not in Qsites:
        return "Invalid site"

    if "ildos" not in data:
        return "No ildos saved"

    sidx = np.where(Qsites == site)[0][0]
    ildos = data["ildos"]

    E = np.asarray(ildos[sidx][0], dtype=float)
    rho = np.asarray(ildos[sidx][1], dtype=float)

    Ui = float(data["Ui"][site])
    ni = float(data["ni"][site])

    rho_ui = np.interp(Ui, E, rho, left=np.nan, right=np.nan)
    rho_0 = np.interp(0.0, E, rho, left=np.nan, right=np.nan)

    fig_ldos, ax_ldos = plt.subplots(figsize=(6, 3.5))
    ax_ldos.plot(E, rho, label="LDOS")
    ax_ldos.axvline(Ui, ls="--", color="r", label="Ui")
    ax_ldos.axvline(0.0, ls=":", color="k", label="E=0")
    ax_ldos.set_title(f"LDOS (site {site})")
    ax_ldos.set_xlabel("Energy")
    ax_ldos.set_ylabel("LDOS")
    ax_ldos.legend()
    fig_ldos.tight_layout()

    Qp = np.asarray(data["Qprime"], dtype=int)

    if site not in Qp:
        info_panel.object = (
            f"### Site {site}\n"
            f"- Ui = {Ui:.4g}\n"
            f"- ni = {ni:.4g}\n"
            f"- LDOS(0) = {rho_0:.4g}\n"
            f"- LDOS(Ui) = {rho_ui:.4g}\n\n"
            f"Not in Qprime"
        )
        return pn.Column(pn.pane.Matplotlib(fig_ldos, tight=True))

    qidx = np.where(Qp == site)[0][0]
    Ci = float(data["Ci"][qidx])

    x = E
    ildos_int = np.zeros_like(rho)
    ildos_int[1:] = np.cumsum(
        0.5 * (rho[1:] + rho[:-1]) * np.diff(x)
    )

    poisson = x * Ci + ni
    diff = np.abs(poisson - ildos_int)
    imin = int(np.argmin(diff))

    info_panel.object = (
        f"### Site {site}\n"
        f"- Ui = {Ui:.4g}\n"
        f"- ni = {ni:.4g}\n"
        f"- LDOS(0) = {rho_0:.4g}\n"
        f"- LDOS(Ui) = {rho_ui:.4g}\n"
        f"- Ci = {Ci:.4g}\n"
        f"- best ΔU ≈ {x[imin]:.4g}"
    )

    fig_local, ax_local = plt.subplots(figsize=(6, 4))
    ax_local.plot(x, poisson, label="Poisson")
    ax_local.plot(x, ildos_int, label="Integrated LDOS")
    ax_local.axvline(x[imin], ls="--", color="k")
    ax_local.set_xlabel("Energy / ΔU axis")
    ax_local.set_ylabel("Density")
    ax_local.legend()
    fig_local.tight_layout()

    return pn.Column(
        pn.pane.Matplotlib(fig_local, tight=True),
        pn.pane.Matplotlib(fig_ldos, tight=True)
    )

# ----------------------------
# Surface plot on the left
# ----------------------------
@pn.depends(snapshot_slider, cut_direction_select, cut_center_input, cut_width_input, surface_prop_select)
def surface_plot(snapshot_name, cut_direction, cut_center, cut_width, surface_prop):
    data = load_snapshot(snapshot_name)

    idx_cut, coord_along, coord_vert = get_surface_cut_indices(
        cut_direction, cut_center, cut_width
    )

    if len(idx_cut) == 0:
        fig_empty, ax_empty = plt.subplots(figsize=(7, 4))
        ax_empty.text(0.5, 0.5, "No sites in cut", ha="center", va="center")
        ax_empty.axis("off")
        fig_empty.tight_layout()
        return pn.pane.Matplotlib(fig_empty, tight=True)

    vals_all = scalar_values_surface(data, surface_prop, snapshot_name)
    vals_cut = vals_all[idx_cut]

    fig_cut, ax_cut = plt.subplots(figsize=(7, 5))
    sc = ax_cut.scatter(
        coord_along,
        coord_vert,
        c=vals_cut,
        cmap="turbo",
        s=35
    )
    plt.colorbar(sc, ax=ax_cut, pad=0.02, label=surface_prop)
    ax_cut.set_xlabel("x" if cut_direction == "x" else "y")
    ax_cut.set_ylabel("z")
    ax_cut.set_title(f"Surface cut heatmap: {surface_prop}")
    ax_cut.set_aspect("equal")
    fig_cut.tight_layout()
    return pn.pane.Matplotlib(fig_cut, tight=True)

# ----------------------------
# LDOS line-cut heatmap on the right
# ----------------------------
@pn.depends(snapshot_slider, cut_direction_select, cut_center_input, cut_width_input)
def ldos_cut_plot(snapshot_name, cut_direction, cut_center, cut_width):
    data = load_snapshot(snapshot_name)

    if "ildos" not in data:
        fig_empty, ax_empty = plt.subplots(figsize=(7, 4))
        ax_empty.text(0.5, 0.5, "No ildos saved", ha="center", va="center")
        ax_empty.axis("off")
        fig_empty.tight_layout()
        return pn.pane.Matplotlib(fig_empty, tight=True)

    idx_cut, coord_along = get_quantum_line_cut_indices(
        cut_direction, cut_center, cut_width
    )

    if len(idx_cut) == 0:
        fig_empty, ax_empty = plt.subplots(figsize=(7, 4))
        ax_empty.text(0.5, 0.5, "No quantum sites in cut", ha="center", va="center")
        ax_empty.axis("off")
        fig_empty.tight_layout()
        return pn.pane.Matplotlib(fig_empty, tight=True)

    ildos = data["ildos"]
    E0 = np.asarray(ildos[idx_cut[0]][0], dtype=float)

    rho_mat = []
    ui_cut = []

    for i in idx_cut:
        E = np.asarray(ildos[i][0], dtype=float)
        rho = np.asarray(ildos[i][1], dtype=float)

        if len(E) != len(E0) or not np.allclose(E, E0):
            rho_interp = np.interp(E0, E, rho, left=np.nan, right=np.nan)
            rho_mat.append(rho_interp)
        else:
            rho_mat.append(rho)

        site_id = Qsites[i]
        ui_cut.append(float(data["Ui"][site_id]))

    rho_mat = np.asarray(rho_mat, dtype=float).T  # (nE, nPos)
    ui_cut = np.asarray(ui_cut, dtype=float)

    fig_hm, ax_hm = plt.subplots(figsize=(7, 5))
    extent = [coord_along.min(), coord_along.max(), E0.min(), E0.max()]
    im = ax_hm.imshow(
        rho_mat,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="turbo"
    )
    plt.colorbar(im, ax=ax_hm, pad=0.02, label="LDOS")

    # Overlay Ui line
    ui_plot = ui_cut.copy()
    ui_plot[(ui_plot < E0.min()) | (ui_plot > E0.max())] = np.nan
    ax_hm.plot(coord_along, ui_plot, color="white", lw=1, ls="--", label="Ui")

    ax_hm.set_xlim(coord_along.min(), coord_along.max())
    ax_hm.legend(loc="upper right")

    ax_hm.set_xlabel("x" if cut_direction == "x" else "y")
    ax_hm.set_ylabel("Energy")
    ax_hm.set_ylim(0.5*(E0.min()),0.5*E0.max())
    ax_hm.set_title("LDOS heatmap along line cut")
    fig_hm.tight_layout()
    return pn.pane.Matplotlib(fig_hm, tight=True)

# ----------------------------
# Watchers
# ----------------------------
snapshot_slider.param.watch(lambda e: update_system(), "value")
prop_select.param.watch(lambda e: update_system(), "value")
site_select.param.watch(lambda e: update_selected_marker(), "value")

update_system()

# ----------------------------
# Layout
# ----------------------------
left_controls = pn.Column(
    snapshot_slider,
    snapshot_player,
    prop_select,
    surface_prop_select,
    site_select,
    cut_direction_select,
    cut_center_input,
    cut_width_input,
    bounds_panel,
    info_panel,
    width=340
)

left_plots = pn.Column(
    fig,
    surface_plot
)

right_plots = pn.Column(
    local_plot,
    ldos_cut_plot
)

layout = pn.Row(
    left_controls,
    left_plots,
    right_plots
)

layout.servable()