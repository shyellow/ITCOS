"""Reproduce the 10 m RT-WFP, RUSLE and class-span MFO experiment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from rasterio import open as open_raster
from rasterio.crs import CRS


ROOT = Path(__file__).resolve().parent
TIN = ROOT / "data/tin/rongcha10.shp"
RKCP = ROOT / "data/RKCP.tif"
SEED = 20261019


def check_inputs(tin, rkcp):
    """Check the spatial assumptions before tracing a potentially large mesh."""
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        if not tin.with_suffix(suffix).is_file():
            raise FileNotFoundError(tin.with_suffix(suffix))
    crs = CRS.from_wkt(tin.with_suffix(".prj").read_text(encoding="utf-8-sig"))
    if not crs.is_projected or abs(crs.linear_units_factor[1] - 1) > 1e-12:
        raise ValueError("Projected XY coordinates in metres are required.")
    with open_raster(rkcp) as raster:
        if raster.crs != crs or raster.count != 1:
            raise ValueError("RKCP must have one band and use the same CRS as the TIN.")
    # The file records horizontal units. The caller supplies elevations in metres.


def run(script, *arguments):
    """Use the active interpreter for each stage and stop at the first failure."""
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, "-B", "-u", str(ROOT / script),
                    *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tin", type=Path, default=TIN, help="Raw PolygonZ triangle mesh")
    parser.add_argument("--rkcp", type=Path, default=RKCP, help="Single-band RKCP GeoTIFF")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs",
                        help="New or empty directory for generated results")
    parser.add_argument("--seed", type=int, default=SEED, help="Base seed for local MFO runs")
    args = parser.parse_args()
    tin, raster, output = (p.resolve() for p in (args.tin, args.rkcp, args.output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("The output directory must be new or empty. Supply --output to start another run.")
    bundled = (tin == TIN and raster == RKCP)
    if bundled:
        run("verify_example.py", "--inputs-only")
    check_inputs(tin, raster)
    output.mkdir(parents=True, exist_ok=True)
    terrain = output / "01_terrain"
    before = output / "02_erosion_before"
    after = output / "03_optimized"
    terrain_mesh = terrain / f"{tin.stem}_RT_WFP_LS.shp"
    before_mesh = before / f"{tin.stem}_RT_WFP_RUSLE_A.shp"

    run("src/terrain/compute_ls.py", "--source", tin, "--output", terrain)
    run("src/erosion/calculate_erosion.py", "--input", terrain_mesh,
        "--raster", raster, "--output", before)
    run("src/mfo/optimize_mfo.py", "--input", before_mesh, "--raster", raster,
        "--output", after, "--weight-count", "0.5", "--weight-span", "0.5",
        "--max-sweeps", 1, "--seed", args.seed)
    if bundled and args.seed == SEED:
        run("verify_example.py", "--output", output)
    print(json.dumps({"terrain": str(terrain_mesh),
                      "optimized": str(after / f"{before_mesh.stem}_MFO_SPAN.shp")},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
