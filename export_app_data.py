r"""Export Experiment C H_peak arrays to text for the app data layer.

Reads cached Experiment A hpeak .npy files and writes text versions plus a
manifest. Read-only: no simulation, no repo changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from urbanflood.app_data import SCENARIO_NAMES, hpeak_filename, write_hpeak_text, write_manifest
from urbanflood.config import build_default_config


def _check_expa_cache(expa_dir: Path) -> None:
    """Reuse the run_experiment_c.py cache guard: run_meta.txt steps=720, dt=30.0."""
    meta_path = expa_dir / "run_meta.txt"
    if not meta_path.exists():
        print(
            "ERROR: expA run_meta.txt not found at %s. Run Experiment A first."
            % meta_path,
            file=sys.stderr,
        )
        sys.exit(1)
    meta_dict = {}
    for line in meta_path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta_dict[k.strip()] = v.strip()
    if meta_dict.get("steps") != "720" or meta_dict.get("dt") != "30.0":
        print(
            "ERROR: expA run_meta.txt records steps=%s, dt=%s, but export "
            "requires steps=720, dt=30.0."
            % (meta_dict.get("steps"), meta_dict.get("dt")),
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Experiment C H_peak arrays to text."
    )
    parser.add_argument(
        "--expa",
        required=True,
        help="Experiment A output dir (cached hpeak + run_meta.txt).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory (created if needed).",
    )
    args = parser.parse_args()

    expa_dir = Path(args.expa).resolve()
    out_dir = Path(args.out).resolve()
    _check_expa_cache(expa_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_default_config()
    seeds = list(cfg.seeds.env_experiment_seeds())

    total_bytes = 0
    count = 0
    for i, seed in enumerate(seeds):
        for scenario in SCENARIO_NAMES:
            src = expa_dir / ("hpeak_seed%d_%s.npy" % (seed, scenario))
            if not src.exists():
                print("ERROR: missing %s" % src, file=sys.stderr)
                return 1
            arr = np.load(src)
            name = hpeak_filename(i, scenario)
            dst = out_dir / name
            write_hpeak_text(dst, arr)
            n = dst.stat().st_size
            total_bytes += n
            count += 1
            print("wrote %s (%d bytes)" % (name, n))

    manifest_path = out_dir / "manifest.txt"
    write_manifest(manifest_path, seeds)
    n = manifest_path.stat().st_size
    total_bytes += n
    count += 1
    print("wrote manifest.txt (%d bytes)" % n)
    print("total files: %d, total bytes: %d" % (count, total_bytes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
