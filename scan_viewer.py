import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


FNAME_RE = re.compile(r"phi_([0-9eE+\-\.]+)_Vbg_([0-9eE+\-\.]+)\.npz$")


def parse_filename(fname):
    m = FNAME_RE.match(os.path.basename(fname))
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2))


def load_static_qsites(scan_folder):
    static_path = os.path.join(scan_folder, "run_static.npz")
    if not os.path.exists(static_path):
        return None
    data = np.load(static_path, allow_pickle=True)
    static_data = data["static_data"][0]
    if "Qsites" not in static_data:
        return None
    return np.asarray(static_data["Qsites"], dtype=int)


def ensure_q_local(arr, qsites):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")

    if len(arr) == len(qsites):
        return arr

    if len(arr) >= np.max(qsites) + 1:
        return arr[qsites]

    raise ValueError(
        f"Cannot map array of length {len(arr)} to Qsites of length {len(qsites)}"
    )


def ldos_at_energy_for_snapshot(npz_path, qsites_static=None):
    data = np.load(npz_path, allow_pickle=True)

    if "ildos" not in data:
        raise ValueError(f"No 'ildos' in {npz_path}")
    if "Ui" not in data:
        raise ValueError(f"No 'Ui' in {npz_path}")

    ildos = data["ildos"]

    if "Qsites" in data:
        qsites = np.asarray(data["Qsites"], dtype=int)
    elif qsites_static is not None:
        qsites = np.asarray(qsites_static, dtype=int)
    else:
        raise ValueError(f"No Qsites found in {npz_path} or run_static.npz")

    Ui_local = ensure_q_local(data["Ui"], qsites)

    n_q = len(qsites)
    if len(ildos) != n_q:
        raise ValueError(
            f"ildos length {len(ildos)} does not match len(Qsites)={n_q} in {npz_path}"
        )

    rho0 = np.full(n_q, np.nan, dtype=float)
    rhoUi = np.full(n_q, np.nan, dtype=float)

    for i in range(n_q):
        E = np.asarray(ildos[i][0], dtype=float)
        rho = np.asarray(ildos[i][1], dtype=float)

        if E.ndim != 1 or rho.ndim != 1 or len(E) < 2 or len(E) != len(rho):
            continue

        rho0[i] = np.interp(0.0, E, rho, left=np.nan, right=np.nan)
        rhoUi[i] = np.interp(Ui_local[i], E, rho, left=np.nan, right=np.nan)

    qprime_mask = None
    if "Qprime" in data:
        Qprime = np.asarray(data["Qprime"], dtype=int)
        qprime_mask = np.isin(qsites, Qprime)

    return {
        "qsites": qsites,
        "Ui_local": Ui_local,
        "rho0": rho0,
        "rhoUi": rhoUi,
        "qprime_mask": qprime_mask,
    }


def reduce_ldos(snapshot_dict, mode="mean_qprime", site_id=None):
    rho0 = snapshot_dict["rho0"]
    rhoUi = snapshot_dict["rhoUi"]
    qsites = snapshot_dict["qsites"]
    qprime_mask = snapshot_dict["qprime_mask"]

    if mode == "site":
        if site_id is None:
            raise ValueError("mode='site' requires --site-id")
        idx = np.where(qsites == site_id)[0]
        if len(idx) == 0:
            return np.nan, np.nan
        i = idx[0]
        return rho0[i], rhoUi[i]

    if mode == "mean_all":
        return np.nanmean(rho0), np.nanmean(rhoUi)

    if mode == "max_all":
        return np.nanmax(rho0), np.nanmax(rhoUi)

    if qprime_mask is None:
        # fallback if Qprime not saved
        if mode == "mean_qprime":
            return np.nanmean(rho0), np.nanmean(rhoUi)
        if mode == "max_qprime":
            return np.nanmax(rho0), np.nanmax(rhoUi)

    rho0_q = rho0[qprime_mask]
    rhoUi_q = rhoUi[qprime_mask]

    if rho0_q.size == 0:
        return np.nan, np.nan

    if mode == "mean_qprime":
        return np.nanmean(rho0_q), np.nanmean(rhoUi_q)

    if mode == "max_qprime":
        return np.nanmax(rho0_q), np.nanmax(rhoUi_q)

    raise ValueError(f"Unknown mode: {mode}")


def build_maps(scan_folder, mode="mean_qprime", site_id=None):
    qsites_static = load_static_qsites(scan_folder)

    files = []
    for fname in os.listdir(scan_folder):
        if not fname.endswith(".npz"):
            continue
        if fname == "run_static.npz":
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        files.append((fname, *parsed))

    if len(files) == 0:
        raise RuntimeError(f"No scan npz files found in {scan_folder}")

    phis = sorted({phi for _, phi, _ in files})
    vbgs = sorted({vbg for _, _, vbg in files})

    phi_to_i = {v: i for i, v in enumerate(phis)}
    vbg_to_j = {v: j for j, v in enumerate(vbgs)}

    map_rho0 = np.full((len(vbgs), len(phis)), np.nan, dtype=float)
    map_rhoUi = np.full((len(vbgs), len(phis)), np.nan, dtype=float)

    for fname, phi, vbg in files:
        full_path = os.path.join(scan_folder, fname)
        snap = ldos_at_energy_for_snapshot(full_path, qsites_static=qsites_static)
        val0, valUi = reduce_ldos(snap, mode=mode, site_id=site_id)

        j = vbg_to_j[vbg]
        i = phi_to_i[phi]
        map_rho0[j, i] = val0
        map_rhoUi[j, i] = valUi

    return np.array(phis), np.array(vbgs), map_rho0, map_rhoUi


def plot_maps(phis, vbgs, map_rho0, map_rhoUi, mode_label):
    extent = [phis.min(), phis.max(), vbgs.min(), vbgs.max()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        map_rho0,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    axes[0].set_title(f"LDOS@0 ({mode_label})")
    axes[0].set_xlabel("phi")
    axes[0].set_ylabel("backgate voltage")
    plt.colorbar(im0, ax=axes[0], pad=0.02)

    im1 = axes[1].imshow(
        map_rhoUi,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    axes[1].set_title(f"LDOS@Ui ({mode_label})")
    axes[1].set_xlabel("phi")
    axes[1].set_ylabel("backgate voltage")
    plt.colorbar(im1, ax=axes[1], pad=0.02)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_folder", help="Folder containing scan .npz files")
    parser.add_argument(
        "--mode",
        default="mean_qprime",
        choices=["mean_qprime", "max_qprime", "mean_all", "max_all", "site"],
        help="How to reduce site-resolved LDOS to one value per scan point",
    )
    parser.add_argument(
        "--site-id",
        type=int,
        default=None,
        help="Global site id for mode='site'",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output image path, e.g. ldos_scan.png",
    )
    args = parser.parse_args()

    phis, vbgs, map_rho0, map_rhoUi = build_maps(
        args.scan_folder,
        mode=args.mode,
        site_id=args.site_id,
    )

    extent = [phis.min(), phis.max(), vbgs.min(), vbgs.max()]
    mode_label = f"site {args.site_id}" if args.mode == "site" else args.mode

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        map_rho0,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    axes[0].set_title(f"LDOS@0 ({mode_label})")
    axes[0].set_xlabel("phi")
    axes[0].set_ylabel("backgate voltage")
    plt.colorbar(im0, ax=axes[0], pad=0.02)

    im1 = axes[1].imshow(
        map_rhoUi,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    axes[1].set_title(f"LDOS@Ui ({mode_label})")
    axes[1].set_xlabel("phi")
    axes[1].set_ylabel("backgate voltage")
    plt.colorbar(im1, ax=axes[1], pad=0.02)

    if args.save:
        fig.savefig(args.save, dpi=200)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()