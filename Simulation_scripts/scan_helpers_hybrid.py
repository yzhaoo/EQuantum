import os
import json
import csv
import time
import math
import inspect
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


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
    script_path=None,
):
    nworkers = int(fixed_params["Ncore"])
    total_points = int(len(phis) * len(Vbgs))
    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": script_path or "<interactive>",
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
        "parallelization_mode": "hybrid_chunked_warm_start",
        "max_parallel_jobs": nworkers,
        "solver_cores_per_job": int(fixed_params.get("solver_Ncore", 1)),
        "expected_total_points": total_points,
        "expected_chunk_count": nworkers,
        "expected_points_per_chunk": math.ceil(total_points / nworkers) if nworkers > 0 else total_points,
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
        "restart_mode",
        "worker_pid",
        "chunk_id",
        "chunk_index",
        "chunk_size",
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def reset_fsc_log(fsc):
    fsc.log = {
        "Qprime_len": [len(fsc.Qprime) if hasattr(fsc, "Qprime") else 0],
        "Ui_maxdiff": [],
        "Ui_l2diff": [],
        "ni_maxdiff": [],
        "ni_l2diff": [],
        "ildos_maxdiff": [],
        "timing_poisson": [],
        "timing_quantum": [],
        "timing_total": [],
    }


def build_fresh_fsc(FSC_cls, syst, phi, Vbg, fixed_params):
    qparams = {"Ufunc": lambda x: 0, "phi": phi}

    fsc = FSC_cls(syst, ifinitial=False, qparams=qparams)
    fsc.update_BC(syst, "gate", "potential", fixed_params["gate_potential"])
    fsc.update_BC(syst, "dielectric", "dielectric_constant", fixed_params["dielectric_constant"])
    fsc.update_BC(syst, "backgate", "potential", Vbg, ifinitial=True)
    fsc.Ncore = int(fixed_params.get("solver_Ncore", 1))
    fsc.convergence_tol = fixed_params["convergence_tol"]
    reset_fsc_log(fsc)
    return fsc


def update_existing_fsc(fsc, syst, phi, Vbg):
    fsc.Qprime = fsc.Qsites.copy()
    fsc.qsystem.update_active_sites(fsc.Qprime)
    fsc.Qp_in_Q = {ii: fsc.qsite_id_to_idx[sid] for ii, sid in enumerate(fsc.Qprime)}
    fsc.N_indices = syst.N_indices.copy()
    fsc.D_indices = syst.D_indices.copy()
    fsc.update_qparams(syst, {"phi": phi}, ifinitial=False)
    fsc.update_BC(syst, "backgate", "potential", Vbg, ifinitial=True)
    reset_fsc_log(fsc)


def _solve_current_point(fsc, syst, out_folder, fixed_params, outfile_prefix):
    snapshot_folder = os.path.join(out_folder, f"_worker_tmp_{outfile_prefix}_pid_{os.getpid()}")
    os.makedirs(snapshot_folder, exist_ok=True)

    fsc.solve(
        syst,
        save=True,
        snapshot_mode="final_only",
        snapshot_folder=snapshot_folder,
        save_ildos=fixed_params["save_ildos"],
        ldos_method=fixed_params["ldos_method"],
        eta=fixed_params["eta"],
        M=fixed_params["M"],
        eps=fixed_params["eps"],
        kernel=fixed_params["kernel"],
    )

    new_files = sorted(
        f for f in os.listdir(snapshot_folder)
        if f.endswith(".npz") and f != "run_static.npz"
    )
    if len(new_files) != 1:
        raise RuntimeError(f"Expected exactly 1 new snapshot, got {new_files}")

    latest_file = new_files[0]
    src = os.path.join(snapshot_folder, latest_file)
    return src, latest_file, snapshot_folder


def _run_chunk(
    FSC_cls,
    System_cls,
    geoparams,
    config_file,
    out_folder,
    fixed_params,
    chunk_id,
    jobs,
):
    worker_pid = os.getpid()
    syst = System_cls(
        geoparams,
        config_file=config_file,
        ifqsystem=True,
        quantum_builder="default",
    )

    rows = []
    fsc = None
    need_fresh_fsc = True
    chunk_size = len(jobs)

    for chunk_index, (phi, Vbg) in enumerate(jobs, start=1):
        phi = float(phi)
        Vbg = float(Vbg)
        t0 = time.perf_counter()
        status = "ok"
        converged = False
        outfile = f"phi_{phi:.4f}_Vbg_{Vbg:.4f}.npz"
        restart_mode = "fresh" if need_fresh_fsc or fsc is None else "warm"

        try:
            if need_fresh_fsc or fsc is None:
                fsc = build_fresh_fsc(FSC_cls, syst, phi, Vbg, fixed_params)
                need_fresh_fsc = False
                restart_mode = "fresh"
            else:
                update_existing_fsc(fsc, syst, phi, Vbg)
                restart_mode = "warm"

            prefix = f"chunk_{chunk_id}_idx_{chunk_index}_phi_{phi:.4f}_Vbg_{Vbg:.4f}"
            src, latest_file, snapshot_folder = _solve_current_point(
                fsc,
                syst,
                out_folder,
                fixed_params,
                prefix,
            )

            dst = os.path.join(out_folder, outfile)
            if os.path.abspath(src) != os.path.abspath(dst):
                os.replace(src, dst)

            try:
                os.rmdir(snapshot_folder)
            except OSError:
                pass

            converged = "_conv" in latest_file

        except Exception as e:
            status = f"error: {type(e).__name__}: {e}"
            need_fresh_fsc = True

        runtime_sec = time.perf_counter() - t0

        rows.append({
            "timestamp": datetime.now().isoformat(),
            "phi": phi,
            "Vbg": Vbg,
            "outfile": outfile if status == "ok" else "",
            "converged": converged,
            "runtime_sec": runtime_sec,
            "iter_qprime": len(fsc.log["Qprime_len"]) if fsc is not None else "",
            "iter_poisson": len(fsc.log["timing_poisson"]) if fsc is not None else "",
            "iter_quantum": len(fsc.log["timing_quantum"]) if fsc is not None else "",
            "qprime_len": len(fsc.Qprime) if (fsc is not None and hasattr(fsc, "Qprime")) else "",
            "ui_maxdiff": fsc.log["Ui_maxdiff"][-1] if (fsc is not None and len(fsc.log["Ui_maxdiff"])) else "",
            "ni_maxdiff": fsc.log["ni_maxdiff"][-1] if (fsc is not None and len(fsc.log["ni_maxdiff"])) else "",
            "ildos_maxdiff": fsc.log["ildos_maxdiff"][-1] if (fsc is not None and len(fsc.log["ildos_maxdiff"])) else "",
            "status": status,
            "restart_mode": restart_mode,
            "worker_pid": worker_pid,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "chunk_size": chunk_size,
        })

    return rows


def _partition_jobs(phis, Vbgs, nchunks):
    jobs = [(float(phi), float(Vbg)) for phi in phis for Vbg in Vbgs]
    raw_chunks = np.array_split(np.array(jobs, dtype=float), nchunks)
    chunks = []
    for i, arr in enumerate(raw_chunks):
        if len(arr) == 0:
            continue
        chunks.append((i, [(float(phi), float(Vbg)) for phi, Vbg in arr.tolist()]))
    return chunks


def run_scan(
    FSC_cls,
    System_cls,
    geoparams,
    config_file,
    phis,
    Vbgs,
    out_folder,
    fixed_params,
):
    os.makedirs(out_folder, exist_ok=True)

    max_workers = int(fixed_params["Ncore"])
    chunks = _partition_jobs(phis, Vbgs, max_workers)
    total_points = sum(len(jobs) for _, jobs in chunks)

    print(f"Launching hybrid parameter scan with {len(chunks)} workers over {total_points} points.")
    print("Each worker gets one chunk: first point is cold-started, later points in the same chunk use warm start.")
    print(f"Internal solver cores per point: {int(fixed_params.get('solver_Ncore', 1))}")

    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(
                _run_chunk,
                FSC_cls,
                System_cls,
                geoparams,
                config_file,
                out_folder,
                fixed_params,
                chunk_id,
                jobs,
            ): (chunk_id, len(jobs))
            for chunk_id, jobs in chunks
        }

        completed_points = 0
        for future in as_completed(futures):
            chunk_id, chunk_size = futures[future]
            print(f"\n=== Chunk {chunk_id} finished ({chunk_size} points) ===")
            try:
                rows = future.result()
            except Exception as e:
                rows = [{
                    "timestamp": datetime.now().isoformat(),
                    "phi": "",
                    "Vbg": "",
                    "outfile": "",
                    "converged": False,
                    "runtime_sec": "",
                    "iter_qprime": "",
                    "iter_poisson": "",
                    "iter_quantum": "",
                    "qprime_len": "",
                    "ui_maxdiff": "",
                    "ni_maxdiff": "",
                    "ildos_maxdiff": "",
                    "status": f"error: ChunkFailure: {type(e).__name__}: {e}",
                    "restart_mode": "",
                    "worker_pid": "",
                    "chunk_id": chunk_id,
                    "chunk_index": "",
                    "chunk_size": chunk_size,
                }]

            for row in rows:
                append_progress_row(out_folder, row)
                if row["phi"] != "":
                    completed_points += 1
                    print(
                        f"[{completed_points}/{total_points}] phi={float(row['phi']):.4f}, "
                        f"Vbg={float(row['Vbg']):.4f}, restart={row['restart_mode']}, "
                        f"status={row['status']}, pid={row['worker_pid']}"
                    )
                else:
                    print(f"chunk {chunk_id} status={row['status']}")
