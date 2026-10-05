"""
Download LIDC-IDRI CT scans and turn them into small nodule cubes.

The full collection is ~128 GB of chest CT. We never keep that: each scan is
downloaded, cropped around its nodules, and deleted before the next one starts.
Peak disk use is one scan (~100-400 MB); the output for ALL 2651 nodules is
about 1.3 GB.

    scan (100-400 MB)  ->  64x64x64 cube per nodule (512 KB)  ->  scan deleted

Each cube is resampled to 1 mm isotropic voxels first, so "64" means 64 mm in
every direction regardless of the scanner's slice thickness (which varies from
0.6 to 5 mm across the collection).

Usage
-----
    python scripts/prepare_lidc.py --n-patients 50
    python scripts/prepare_lidc.py --n-patients 50 --min-annotations 3
    python scripts/prepare_lidc.py --all

Outputs (under --out, default data/processed/lidc):
    cubes/<patient>_<nodule>.npy    int16 HU, shape (64, 64, 64)
    nodules.csv                     one row per cube: concepts + label
    .done                           patients already processed (resumable)

Re-running skips patients listed in .done, so it is safe to interrupt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# pylidc 0.2.3 predates the removal of these NumPy aliases. Must run before
# `import pylidc`, and before anything else imports it.
for _name, _builtin in [("int", int), ("float", float), ("bool", bool)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

import pylidc as pl                                            # noqa: E402
from scipy.ndimage import zoom                                 # noqa: E402

TCIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
AIR_HU = -1024                      # what to pad with when a cube runs off the scan

RATINGS = ["subtlety", "internalStructure", "calcification", "sphericity",
           "margin", "lobulation", "spiculation", "texture"]
CONTINUOUS = ["diameter", "volume", "surface_area"]


# ---------------------------------------------------------------------------
# TCIA
# ---------------------------------------------------------------------------

def fetch_series_index(cache: Path) -> dict[str, dict]:
    """Map patient_id -> series metadata for every LIDC-IDRI CT scan."""
    if cache.exists():
        series = json.loads(cache.read_text())
    else:
        print("fetching series index from TCIA ...")
        url = f"{TCIA}/getSeries?Collection=LIDC-IDRI"
        with urllib.request.urlopen(url, timeout=120) as r:
            series = json.loads(r.read().decode())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(series))
    ct = [s for s in series if s.get("Modality") == "CT"]
    return {s["PatientID"]: s for s in ct}


def download_series(uid: str, dest_zip: Path, retries: int = 3,
                    verbose: bool = False) -> None:
    url = f"{TCIA}/getImage?SeriesInstanceUID={uid}"

    # TCIA does not send Content-Length, so there is no percentage to show.
    # Report every 10 MB, and only redraw in place when attached to a terminal
    # (otherwise a piped log fills up with thousands of partial lines).
    tty = sys.stdout.isatty()
    state = {"next": 10e6}

    def hook(block, block_size, total):
        if not verbose:
            return
        got = block * block_size
        if got < state["next"]:
            return
        state["next"] += 10e6
        end = "\r" if tty else "\n"
        print(f"    downloading {got/1e6:6.0f} MB ...", end=end, flush=True)

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest_zip, reporthook=hook)
            if verbose:
                mb = dest_zip.stat().st_size / 1e6
                print(f"    downloaded {mb:6.1f} MB" + " " * 12)
            return
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"    download failed ({exc}), retrying ...")
            time.sleep(5 * (attempt + 1))


# ---------------------------------------------------------------------------
# Volume handling
# ---------------------------------------------------------------------------

def to_isotropic(volume: np.ndarray, scan) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample to 1 mm isotropic voxels.

    CT voxels are not cubes: in-plane spacing is ~0.5-0.9 mm while slices are
    0.6-5 mm apart. Without this, a fixed-size voxel crop covers a different
    physical volume in every scan, and shape concepts like sphericity become
    meaningless.

    Returns the resampled volume and the zoom factors, which are also needed to
    move the nodule centroids into the new index space.
    """
    factors = np.array([scan.pixel_spacing, scan.pixel_spacing,
                        scan.slice_thickness], dtype=float)
    out = zoom(volume.astype(np.float32), factors, order=1)
    return out.astype(np.int16), factors


def crop_cube(volume: np.ndarray, centre: np.ndarray, size: int) -> np.ndarray:
    """Cut a `size`^3 cube centred on `centre`, padding with air if it runs off."""
    half = size // 2
    cube = np.full((size, size, size), AIR_HU, dtype=np.int16)
    starts = [int(round(c)) - half for c in centre]

    src, dst = [], []
    for axis, start in enumerate(starts):
        s0, s1 = max(0, start), min(volume.shape[axis], start + size)
        if s0 >= s1:
            return cube                                  # nodule outside volume
        src.append(slice(s0, s1))
        dst.append(slice(s0 - start, s0 - start + (s1 - s0)))

    cube[tuple(dst)] = volume[tuple(src)]
    return cube


# ---------------------------------------------------------------------------
# Per-patient work
# ---------------------------------------------------------------------------

def process_patient(scan, series, out_dir: Path, size: int,
                    min_annotations: int, verbose: bool = False,
                    scan_tag: str = "0") -> list[dict]:
    """Download one scan, write its nodule cubes, delete the scan. Returns rows."""
    tmp = Path(tempfile.mkdtemp(prefix="lidc_"))
    try:
        zip_path = tmp / "series.zip"
        download_series(series["SeriesInstanceUID"], zip_path, verbose=verbose)

        dicom_dir = (tmp / "dicom" / scan.patient_id /
                     scan.study_instance_uid / scan.series_instance_uid)
        dicom_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.lower().endswith(".dcm"):
                    (dicom_dir / os.path.basename(name)).write_bytes(z.read(name))
        zip_path.unlink()
        if verbose:
            print(f"    {len(list(dicom_dir.iterdir()))} DICOM slices extracted")

        # Point pylidc at our temp folder instead of ~/.pylidcrc, so this script
        # never touches the user's home directory config.
        sys.modules["pylidc.Scan"]._get_dicom_file_path_from_config_file = (
            lambda: str(tmp / "dicom"))

        volume = scan.to_volume(verbose=False)
        if verbose:
            print(f"    volume {volume.shape}  spacing {scan.pixel_spacing:.3f} mm "
                  f"in-plane, {scan.slice_thickness} mm slices")
        volume, factors = to_isotropic(volume, scan)
        if verbose:
            print(f"    resampled to 1 mm isotropic -> {volume.shape}")

        rows = []
        for idx, anns in enumerate(scan.cluster_annotations()):
            if len(anns) < min_annotations:
                if verbose:
                    print(f"      nodule {idx}: skipped "
                          f"({len(anns)} < --min-annotations {min_annotations})")
                continue

            centre = np.mean([a.centroid for a in anns], axis=0) * factors
            cube = crop_cube(volume, centre, size)

            # Include a short scan tag: 8 patients have two CTs, and without
            # this the second scan silently overwrites the first one's cubes.
            name = f"{scan.patient_id}_s{scan_tag}_{idx:02d}.npy"
            np.save(out_dir / "cubes" / name, cube)
            if verbose:
                mal = np.mean([a.malignancy for a in anns])
                dia = np.mean([a.diameter for a in anns])
                print(f"      nodule {idx}: {len(anns)} radiologist(s), "
                      f"{dia:5.1f} mm, malignancy {mal:.2f} -> {name}")

            row = {"file": name, "patient_id": scan.patient_id,
                   "scan_idx": scan_tag, "nodule_idx": idx,
                   "n_annotations": len(anns)}
            for f in RATINGS:
                v = np.array([getattr(a, f) for a in anns], dtype=float)
                row[f] = round(float(v.mean()), 4)
                row[f + "_std"] = round(float(v.std()), 4)
            mal = np.array([a.malignancy for a in anns], dtype=float)
            row["malignancy"] = round(float(mal.mean()), 4)
            row["malignancy_std"] = round(float(mal.std()), 4)
            # LIDC convention: >3 malignant, <3 benign, ==3 ambiguous (drop it)
            row["label"] = 1 if mal.mean() > 3 else (0 if mal.mean() < 3 else -1)
            for f in CONTINUOUS:
                try:    row[f] = round(float(np.mean([getattr(a, f) for a in anns])), 4)
                except Exception: row[f] = ""
            rows.append(row)
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)          # the scan never survives


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/processed/lidc", type=Path)
    ap.add_argument("--n-patients", type=int, default=50,
                    help="How many scans to process (default 50, ~5 GB downloaded).")
    ap.add_argument("--all", action="store_true",
                    help="Process all 1018 scans (1010 patients; 8 have two).")
    ap.add_argument("--cube", type=int, default=64, help="Cube side in voxels = mm.")
    ap.add_argument("--min-annotations", type=int, default=1,
                    help="Skip nodules seen by fewer than this many radiologists.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show download progress, volume shapes and per-nodule detail.")
    args = ap.parse_args()

    out = args.out
    (out / "cubes").mkdir(parents=True, exist_ok=True)
    done_file = out / ".done"
    done = set(done_file.read_text().split()) if done_file.exists() else set()

    index = fetch_series_index(out / "series_index.json")
    all_scans = [s for s in pl.query(pl.Scan).all() if s.patient_id in index]

    # 8 patients have two CTs. Number each patient's scans so their cubes get
    # distinct filenames, and key resume state on the scan, not the patient.
    seen: dict[str, int] = {}
    tagged = []
    for sc in all_scans:
        n = seen.get(sc.patient_id, 0)
        seen[sc.patient_id] = n + 1
        tagged.append((sc, str(n)))

    scans = [(sc, tag) for sc, tag in tagged
             if f"{sc.patient_id}_s{tag}" not in done]
    if not args.all:
        scans = scans[:args.n_patients]

    if not scans:
        print("Nothing to do — every requested scan is already in .done")
        return

    est = sum(int(index[sc.patient_id]["FileSize"]) for sc, _ in scans) / 1e9
    print(f"{len(scans)} scans to process (~{est:.1f} GB will be downloaded "
          f"and discarded)\nwriting cubes to {out/'cubes'}\n")

    csv_path, all_rows, t0 = out / "nodules.csv", [], time.time()
    for i, (scan, tag) in enumerate(scans, 1):
        pid = scan.patient_id
        mb = int(index[pid]["FileSize"]) / 1e6
        extra = f"  [scan {int(tag)+1}]" if tag != "0" else ""
        print(f"[{i}/{len(scans)}] {pid}{extra}  ({mb:.0f} MB)", flush=True)
        try:
            rows = process_patient(scan, index[pid], out, args.cube,
                                   args.min_annotations, verbose=args.verbose,
                                   scan_tag=tag)
        except Exception as exc:
            print(f"    SKIPPED: {type(exc).__name__}: {exc}")
            continue

        all_rows += rows
        print(f"    {len(rows)} nodule cubes  ({time.time()-t0:.0f}s elapsed)")

        # Append as we go so an interrupted run keeps its work.
        write_header = not csv_path.exists()
        if rows:
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if write_header:
                    w.writeheader()
                w.writerows(rows)
        done.add(f"{pid}_s{tag}")
        done_file.write_text("\n".join(sorted(done)))

    size_mb = sum(p.stat().st_size for p in (out/"cubes").glob("*.npy")) / 1e6
    print(f"\nDone. {len(all_rows)} new cubes in {time.time()-t0:.0f}s")
    print(f"{len(list((out/'cubes').glob('*.npy')))} cubes total, {size_mb:.0f} MB on disk")
    print(f"metadata: {csv_path}")


if __name__ == "__main__":
    main()
