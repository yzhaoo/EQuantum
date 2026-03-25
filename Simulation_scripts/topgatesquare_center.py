import os
import sys
import json
import csv
import time
import inspect
from datetime import datetime

import numpy as np

# --------------------------------------------------
# helpers
# --------------------------------------------------
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if callable(obj):
        return getattr(obj, "__name__", "<callable>")
    return obj

def get_function_source(func):
    try:
        return inspect.getsource(func)
    except Exception:
        return f"<source unavailable: {getattr(func, '__name__', 'anonymous')}>"

def write_scan_manifest(
    out_folder,
    density_function,
    geoparams,
    phis,
    Vbgs,
    config_file,
    fixed_params,
):
    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": os.path.abspath(__file__) if "__file__" in globals() else "<interactive>",
        "output_folder": os.path.abspath(out_folder),
        "config_file": os.path.abspath(config_file),
        "density_function_name": getattr(density_function, "__name__", "anonymous"),
        "density_function_source": get_function_source(density_function),
        "geoparams": sanitize_for_json(geoparams),
        "scan_ranges": {
            "phis": sanitize_for_json(phis),
            "Vbgs": sanitize_for_json(Vbgs),
        },
        "fixed_params": sanitize_for_json(fixed_params),
    }

    with open(os.path.join(out_folder, "scan_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

def append_progress_row(out_folder, row):
    csv_path = os.path.join(out_folder, "scan_progress.csv")
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "timestamp",
        "phi",
        "Vbg",
        "outfile",
        "converged",
        "runtime_sec",
        "iter_qprime",
        "iter_poisson",
        "iter_quantum",
        "qprime_len",
        "ui_maxdiff",
        "ni_maxdiff",
        "ildos_maxdiff",
        "status",
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)




import os
import sys
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "Equantum"))

from sites import Site
from fsc import FSC
from EQsystem import System
import poissonsolver as psolver
import qbuilder

# --------------------------------------------------
# setup
# --------------------------------------------------
def density_function(z):
    spacing0 = 0.011
    k = 0.2
    if abs(z) < 3 * spacing0:
        return spacing0
    else:
        return spacing0 + k * z

geoparams = {
    "lattice_type": "square",
    "box_size": ((-0.6, 0.6), (-0.45, 0.45), (-0.08, 0.08)),
    "sampling_density_function": density_function,
    "quantum_center": (0, 0, 0),
}

setuppathupdate = "/scratch/zhaoyuha/Datas/EQuantum_data/squaregate_center/setup/setup_ae7d9fc59f6721d485f0e595c14ca989"
config_file = setuppathupdate + "/updated_sites_square.json"
out_folder = setuppathupdate + "/fsc_logs_scan"
os.makedirs(out_folder, exist_ok=True)

phis = np.linspace(0.05, 0.12, 10)
Vbgs = np.linspace(0.5, 2.5, 10)

fixed_params = {
    "gate_potential": -0.5,
    "dielectric_constant": 3.2,
    "Ncore": 20,
    "convergence_tol": 3e-2,
    "ldos_method": "ED",
    "eta": 0.00015,
    "M": 256,
    "eps": 0.05,
    "kernel": "jackson",
    "snapshot_mode": "final_only",
    "save_ildos": True,
}

write_scan_manifest(
    out_folder=out_folder,
    density_function=density_function,
    geoparams=geoparams,
    phis=phis,
    Vbgs=Vbgs,
    config_file=config_file,
    fixed_params=fixed_params,
)

# --------------------------------------------------
# initialize system once
# --------------------------------------------------
syst = System(
    geoparams,
    config_file=config_file,
    ifqsystem=True,
    quantum_builder="default"
)

qparams = {"Ufunc": lambda x: 0, "phi": float(phis[0])}
fsc = FSC(syst, ifinitial=False, qparams=qparams)

fsc.update_BC(syst, "gate", "potential", fixed_params["gate_potential"])
fsc.update_BC(syst, "dielectric", "dielectric_constant", fixed_params["dielectric_constant"])
fsc.update_BC(syst, "backgate", "potential", float(Vbgs[0]), ifinitial=True)

fsc.Ncore = fixed_params["Ncore"]
fsc.convergence_tol = fixed_params["convergence_tol"]

# --------------------------------------------------
# scan loop
# --------------------------------------------------
for phi in phis:
    for Vbg in Vbgs:
        phi = float(phi)
        Vbg = float(Vbg)

        print(f"\n=== Running phi={phi:.4f}, Vbg={Vbg:.4f} ===")
        t0 = time.perf_counter()

        status = "ok"
        converged = False
        outfile = f"phi_{phi:.4f}_Vbg_{Vbg:.4f}.npz"

        try:
            # update parameters
            fsc.update_qparams(syst, {"phi": phi}, ifinitial=False)
            fsc.update_BC(syst, "backgate", "potential", Vbg, ifinitial=False)

            # run FSC
            fsc.solve(
                syst,
                save=True,
                snapshot_mode="final_only",
                snapshot_folder=out_folder,
                save_ildos=True,
                ldos_method="ED",
                eta=0.00015,
                M=256,
                eps=0.05,
                kernel="jackson",
            )

            # rename the latest final snapshot to parameter-tagged filename
            npz_files = [
                f for f in os.listdir(out_folder)
                if f.endswith(".npz") and f != "run_static.npz"
            ]
            latest_file = max(
                npz_files,
                key=lambda x: os.path.getmtime(os.path.join(out_folder, x))
            )
            src = os.path.join(out_folder, latest_file)
            dst = os.path.join(out_folder, outfile)
            if os.path.abspath(src) != os.path.abspath(dst):
                os.replace(src, dst)

            # infer convergence from filename if your solve uses _conv/_max/_stop tags
            converged = ("_conv" in latest_file)

        except Exception as e:
            status = f"error: {type(e).__name__}: {e}"
            print(status)

        runtime_sec = time.perf_counter() - t0

        row = {
            "timestamp": datetime.now().isoformat(),
            "phi": phi,
            "Vbg": Vbg,
            "outfile": outfile if status == "ok" else "",
            "converged": converged,
            "runtime_sec": runtime_sec,
            "iter_qprime": fsc.log["Qprime_len"][-1] if len(fsc.log["Qprime_len"]) else "",
            "iter_poisson": len(fsc.log["timing_poisson"]),
            "iter_quantum": len(fsc.log["timing_quantum"]),
            "qprime_len": len(fsc.Qprime),
            "ui_maxdiff": fsc.log["Ui_maxdiff"][-1] if len(fsc.log["Ui_maxdiff"]) else "",
            "ni_maxdiff": fsc.log["ni_maxdiff"][-1] if len(fsc.log["ni_maxdiff"]) else "",
            "ildos_maxdiff": fsc.log["ildos_maxdiff"][-1] if len(fsc.log["ildos_maxdiff"]) else "",
            "status": status,
        }

        append_progress_row(out_folder, row)
        print(f"✔ Logged phi={phi:.4f}, Vbg={Vbg:.4f}")