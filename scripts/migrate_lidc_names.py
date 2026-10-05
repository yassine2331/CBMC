"""
One-off migration for output produced before the multi-scan fix.

Earlier runs named cubes "<patient>_<nodule>.npy" and recorded resume state as
bare patient ids. Eight LIDC patients have two CT scans, so the second scan
silently overwrote the first one's cubes. The fixed script includes a scan tag:
"<patient>_s<scan>_<nodule>.npy".

This renames existing output to the new scheme so a rerun resumes instead of
starting over. Everything already on disk is assumed to be scan 0, which is
true for every patient except any of the 8 whose SECOND scan was also processed
— those are reported so you can redo just them.

    python scripts/migrate_lidc_names.py --out data/processed/lidc --dry-run
    python scripts/migrate_lidc_names.py --out data/processed/lidc
"""

import argparse
import csv
import re
from pathlib import Path

DUAL_SCAN = {"LIDC-IDRI-0132", "LIDC-IDRI-0151", "LIDC-IDRI-0315", "LIDC-IDRI-0332",
             "LIDC-IDRI-0355", "LIDC-IDRI-0365", "LIDC-IDRI-0442", "LIDC-IDRI-0484"}
OLD = re.compile(r"^(LIDC-IDRI-\d+)_(\d+)\.npy$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/processed/lidc"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out, dry = args.out, args.dry_run
    cubes, done_file, csv_path = out / "cubes", out / ".done", out / "nodules.csv"

    if not cubes.exists():
        print(f"No cubes directory at {cubes} — nothing to migrate.")
        return

    todo = [p for p in sorted(cubes.glob("*.npy")) if OLD.match(p.name)]
    already = [p for p in cubes.glob("*_s*_*.npy")]
    if not todo:
        print(f"Nothing to migrate ({len(already)} files already use the new names).")
        return

    print(f"{len(todo)} cubes to rename, {len(already)} already migrated")
    if dry:
        for p in todo[:5]:
            m = OLD.match(p.name)
            print(f"  {p.name}  ->  {m.group(1)}_s0_{m.group(2)}.npy")
        print("  ...")

    mapping = {}
    for p in todo:
        m = OLD.match(p.name)
        new = f"{m.group(1)}_s0_{m.group(2)}.npy"
        mapping[p.name] = new
        if not dry:
            p.rename(cubes / new)
    print(f"{'would rename' if dry else 'renamed'} {len(mapping)} cubes")

    # .done : "LIDC-IDRI-0001" -> "LIDC-IDRI-0001_s0"
    if done_file.exists():
        ids = [x for x in done_file.read_text().split() if x]
        new_ids = sorted({x if "_s" in x else f"{x}_s0" for x in ids})
        if not dry:
            done_file.write_text("\n".join(new_ids))
        print(f"{'would update' if dry else 'updated'} .done: {len(new_ids)} entries")

    # nodules.csv : rewrite the file column, add scan_idx
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and "scan_idx" not in rows[0]:
            for r in rows:
                r["file"] = mapping.get(r["file"], r["file"])
                r["scan_idx"] = "0"
            cols = list(rows[0].keys())
            cols.insert(cols.index("nodule_idx"), cols.pop(cols.index("scan_idx")))
            if not dry:
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    w.writerows(rows)
            print(f"{'would update' if dry else 'updated'} nodules.csv: "
                  f"{len(rows)} rows, added scan_idx")
        else:
            print("nodules.csv already has scan_idx — left alone")

    # Warn about any of the 8 that may already be corrupted
    if csv_path.exists():
        seen = {}
        for r in rows:
            seen.setdefault(r["patient_id"], []).append(r["file"])
        hit = [p for p in seen if p in DUAL_SCAN and len(seen[p]) != len(set(seen[p]))]
        if hit:
            print("\nWARNING: these dual-scan patients have duplicate filenames and "
                  "should be deleted and redone:")
            for p in hit:
                print(f"  {p}")
        else:
            print("\nNo dual-scan patient shows duplicate filenames — nothing corrupted.")


if __name__ == "__main__":
    main()
