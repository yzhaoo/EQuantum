import os
import sys
import glob
import numpy as np
import panel as pn
from scipy.interpolate import interp1d

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import Viridis256

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

Qsites = np.array(static_data["Qsites"], dtype=int)
coords = np.array(static_data["coords_q"], dtype=float)

# ----------------------------
# Load snapshots
# ----------------------------
files = sorted(
    f for f in glob.glob(os.path.join(log_folder, "*.npz"))
    if "run_static" not in f
)

file_map = {os.path.basename(f): f for f in files}
snapshot_names = list(file_map.keys())

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
    """
    Set colorbar limits using only values inside Qprime.
    Falls back to all finite values if Qprime is empty or invalid.
    """
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

# ----------------------------
# Widgets
# ----------------------------
snapshot_select = pn.widgets.Select(
    name="Snapshot",
    options=snapshot_names,
    width=300
)

prop_select = pn.widgets.Select(
    name="Property",
    options=["Ui", "ni", "Ci", "Qprime_mask", "ΔUi"],
    value="Ui",
    width=300
)

site_select = pn.widgets.IntInput(
    name="Site",
    value=int(Qsites[0]),
    width=300
)

info_panel = pn.pane.Markdown("", width=300)
bounds_panel = pn.pane.Markdown("", width=300)

# ----------------------------
# Bokeh sources
# ----------------------------
main_source = ColumnDataSource(data=dict(x=[], y=[], site=[], value=[]))
qprime_source = ColumnDataSource(data=dict(x=[], y=[]))
selected_source = ColumnDataSource(data=dict(x=[], y=[]))
max_ui_source = ColumnDataSource(data=dict(x=[], y=[]))

# ----------------------------
# Plot setup
# ----------------------------
fig = figure(
    width=900,
    height=900,
    tools="pan,wheel_zoom,box_zoom,reset,tap",
    active_scroll="wheel_zoom",
    match_aspect=True
)

mapper = LinearColorMapper(palette=Viridis256)

renderer = fig.scatter(
    "x", "y",
    source=main_source,
    size=15,
    marker="circle",
    fill_color={"field": "value", "transform": mapper},
    line_color=None
)

# fig.scatter(
#     "x", "y", source=qprime_source,
#     size=12, marker="circle",
#     line_color="black", fill_color=None
# )

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
# Update system plot
# ----------------------------
def update_system():
    data = load_snapshot(snapshot_select.value)
    qmask = qprime_mask_from_data(data)

    if prop_select.value == "Ui":
        vals = np.asarray(data["Ui"], dtype=float)[Qsites]

    elif prop_select.value == "ni":
        vals = np.asarray(data["ni"], dtype=float)[Qsites]

    elif prop_select.value == "Ci":
        vals = np.full(len(Qsites), np.nan, dtype=float)
        Ci = np.asarray(data["Ci"], dtype=float)
        Qp = np.asarray(data["Qprime"], dtype=int)
        for i, s in enumerate(Qp[:len(Ci)]):
            idx = np.where(Qsites == s)[0]
            if len(idx):
                vals[idx[0]] = Ci[i]

    elif prop_select.value == "Qprime_mask":
        vals = qmask.astype(float)

    elif prop_select.value == "ΔUi":
        prev = get_prev_snapshot(snapshot_select.value)
        if prev is None:
            vals = np.zeros(len(Qsites), dtype=float)
        else:
            prev_data = load_snapshot(prev)
            vals = (
                np.asarray(data["Ui"], dtype=float)[Qsites]
                - np.asarray(prev_data["Ui"], dtype=float)[Qsites]
            )

    # --- scale colorbar ONLY from Qprime region ---
    set_mapper_range_from_qprime(vals, qmask)

    main_source.data = dict(
        x=coords[:, 0],
        y=coords[:, 1],
        site=Qsites,
        value=vals
    )

    qprime_source.data = dict(
        x=coords[qmask, 0],
        y=coords[qmask, 1]
    )

    # --- max Ui site over full Qsites (marker behavior unchanged) ---
    Ui_vals = np.asarray(data["Ui"], dtype=float)[Qsites]
    max_idx = int(np.argmax(Ui_vals))
    max_ui_source.data = dict(
        x=[coords[max_idx, 0]],
        y=[coords[max_idx, 1]]
    )

    update_selected_marker()
    update_bounds_panel()

# ----------------------------
# Energy bounds panel
# ----------------------------
def update_bounds_panel():
    data = load_snapshot(snapshot_select.value)

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
    site = site_select.value
    idx = np.where(Qsites == site)[0][0]
    selected_source.data = dict(
        x=[coords[idx, 0]],
        y=[coords[idx, 1]]
    )

def on_click(attr, old, new):
    if len(new) > 0:
        idx = new[0]
        site_select.value = int(main_source.data["site"][idx])

renderer.data_source.selected.on_change("indices", on_click)

# ----------------------------
# Local + LDOS plot
# ----------------------------
@pn.depends(snapshot_select, site_select)
def local_plot(snapshot_name, site):
    data = load_snapshot(snapshot_name)

    if site not in Qsites:
        return "Invalid site"

    if "ildos" not in data:
        return "No ildos saved"

    sidx = np.where(Qsites == site)[0][0]
    ildos = data["ildos"]

    E = ildos[sidx][0]
    rho = ildos[sidx][1]

    Ui = data["Ui"][site]
    ni = data["ni"][site]

    # -------- LDOS plot --------
    fig_ldos, ax_ldos = plt.subplots(figsize=(6, 3.5))
    ax_ldos.plot(E, rho, label="LDOS")
    ax_ldos.axvline(Ui, ls="--", color="r", label="Ui")
    ax_ldos.set_title(f"LDOS (site {site})")
    ax_ldos.legend()

    Qp = np.asarray(data["Qprime"], dtype=int)

    if site not in Qp:
        info_panel.object = f"### Site {site}\n\nNot in Qprime"
        return pn.Column(pn.pane.Matplotlib(fig_ldos, tight=True))

    qidx = np.where(Qp == site)[0][0]
    Ci = data["Ci"][qidx]

    x = E - 0 * Ui

    ildos_int = np.zeros_like(rho)
    ildos_int[1:] = np.cumsum(
        0.5 * (rho[1:] + rho[:-1]) * np.diff(x)
    )

    poisson = x * Ci + ni
    diff = np.abs(poisson - ildos_int)
    imin = np.argmin(diff)

    info_panel.object = (
        f"### Site {site}\n"
        f"- Ui = {Ui:.4g}\n"
        f"- ni = {ni:.4g}\n"
        f"- Ci = {Ci:.4g}\n"
        f"- best ΔU ≈ {x[imin]:.4g}"
    )

    fig_local, ax_local = plt.subplots(figsize=(6, 4))
    ax_local.plot(x, poisson, label="Poisson")
    ax_local.plot(x, ildos_int, label="Integrated LDOS")
    ax_local.axvline(x[imin], ls="--", color="k")
    ax_local.legend()

    return pn.Column(
        pn.pane.Matplotlib(fig_local, tight=True),
        pn.pane.Matplotlib(fig_ldos, tight=True)
    )

# ----------------------------
# Watchers
# ----------------------------
snapshot_select.param.watch(lambda e: update_system(), "value")
prop_select.param.watch(lambda e: update_system(), "value")
site_select.param.watch(lambda e: update_selected_marker(), "value")

update_system()

# ----------------------------
# Layout
# ----------------------------
layout = pn.Row(
    pn.Column(
        snapshot_select,
        prop_select,
        site_select,
        bounds_panel,
        info_panel,
        width=320
    ),
    fig,
    local_plot
)

layout.servable()