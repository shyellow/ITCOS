# ITCOS

## Overview of the repository
[Input data and units](data/README.md)

This release provides the original 5, 10, 15 and 30 m terrain-tolerance meshes from the study area and one shared RKCP raster. The 10 m mesh is the default reproduction example. This process employs the RT-WFP algorithm to calculate the slope length, classify the erosion at the centroid of each original surface, and apply the Moth Flame Optimization (MFO) once, with the number of categories and the category span as the targets.

## Install the environment

Extract the whole package to a writable directory and open a terminal there. The reference platform is 64-bit Windows with CPython 3.10.11. Installing the pinned Python and GIS dependencies needs a package index or a local wheel cache. The computation then runs locally.

**Windows PowerShell**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux or macOS shell**

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Select the TIN tolerance

Run the line for the desired tolerance. Each command sets the original mesh with `--tin` and writes to its own directory with `--output`. All four commands automatically read the same `data/RKCP.tif`, on its native 30 m grid.

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --tin data/tin/rongcha5m.shp --output outputs_5m
.\.venv\Scripts\python.exe run_pipeline.py --tin data/tin/rongcha10.shp --output outputs_10m
.\.venv\Scripts\python.exe run_pipeline.py --tin data/tin/rongcha15m.shp --output outputs_15m
.\.venv\Scripts\python.exe run_pipeline.py --tin data/tin/rongcha30.shp --output outputs_30m
```

**Linux or macOS shell**

```sh
.venv/bin/python run_pipeline.py --tin data/tin/rongcha5m.shp --output outputs_5m
.venv/bin/python run_pipeline.py --tin data/tin/rongcha10.shp --output outputs_10m
.venv/bin/python run_pipeline.py --tin data/tin/rongcha15m.shp --output outputs_15m
.venv/bin/python run_pipeline.py --tin data/tin/rongcha30.shp --output outputs_30m
```

With no arguments, `python run_pipeline.py` runs the 10 m dataset in `outputs/`. To repeat a run, choose a new or empty output directory. In commands written as `python`, use `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux or macOS.

## Reproduce and check the 10 m reference run

1. Open a terminal in the extracted directory containing `run_pipeline.py` and `requirements.txt`. Use the virtual environment created above. In the commands below, `python` stands for `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux and macOS.
2. Check the installed dependencies and the six archived input files before starting the search.

   ```sh
   python -m pip check
   python verify_example.py --inputs-only
   ```

3. Run the three stages with the bundled terrain, raster and study seed. If you have already completed the 10 m run, continue with step 4 using that output directory.

   ```sh
   python run_pipeline.py --tin data/tin/rongcha10.shp --output outputs_10m --seed 20261019
   ```

4. Open `outputs_10m/reproduction_check.json`. A match has `"passed": true` and an empty `mismatches` list. The check covers 15 GIS component files, 29 numerical arrays and two optimization logs. To repeat the comparison on existing results, run `python verify_example.py --output outputs_10m`. If you used the no-argument run, use `outputs` in place of `outputs_10m`.

If a comparison fails, the report names each mismatching component. Check the input hashes, the interpreter and package versions, the seed and the input feature order before comparing vertex positions. Use a fresh output directory when repeating the computation.

## What to open

The paths below illustrate the default 10 m run. For a selected tolerance, use the output directory from its command and the stem of its input file. For example, the 15 m result is `outputs_15m/03_optimized/rongcha15m_RT_WFP_RUSLE_A_MFO_SPAN.shp`. The precomputed results in `reference/` are for 10 m.

| Item | Location | Contents |
|---|---|---|
| Raw inputs | `data/tin/`, `data/RKCP.tif` | Four raw PolygonZ meshes and the shared fixed native-grid RKCP product |
| Slope length and LS | `outputs/01_terrain/rongcha10_RT_WFP_LS.shp` | Per-face `Slope_Deg`, `SLen_m`, `L_Factor`, `S_Factor`, `LS` and route termination |
| Traced paths | `outputs/01_terrain/rtwfp_paths_and_results.npz` | Full-precision routes, segment-to-face mapping, slope and length arrays |
| Before optimization | `outputs/02_erosion_before/rongcha10_RT_WFP_RUSLE_A.shp` | Original geometry, centroid product `RUSLE_A` and grade `A_CLASS` |
| After optimization | `outputs/03_optimized/rongcha10_RT_WFP_RUSLE_A_MFO_SPAN.shp` | Optimized 3D geometry, objective fields and final mapped erosion grade `EROS_CLASS` |
| Run details | `outputs/03_optimized/optimization_report.json` | Seed, weights, score distributions, moved vertices and valid sample counts |
| Ready-to-inspect reference results | `reference/terrain/`, `reference/optimized/` | Historical slope-length and LS layer, per-face CSV, path archive and optimized layer |
| Release verification | `reference/reproduction_record.json`, `SHA256SUMS` | Clean-environment run record and SHA-256 manifest of the distributed files |

Keep each shapefile together with its SHX, DBF, PRJ and CPG files. The QGIS style alongside the optimized layer uses the six study colours and displays invalid centroid samples in grey. The stage reports, per-face CSVs and per-move logs are generated on each run.

## How the calculation works

1. **Terrain.** Each XYZ triangle defines a plane. The slope angle is `atan(sqrt(gx² + gy²))`, where `gx` and `gy` are the plane gradients. Starting at the centroid, RT-WFP traces uphill over faces and shared edges. At vertices it chooses an uphill continuation using the incident terrain, with deterministic tie-breaking. The sum of the horizontal XY lengths of all segments is `SLen_m` (metres). A flat starting face gives zero length and LS. A path ending at the mesh boundary contributes its in-mesh length.
2. **LS.** For horizontal length λ in metres, `L = (λ / 22.13)^m`. Use `m = 0.2, 0.3, 0.4, 0.5` for slope angles in `[0°,1°), [1°,3°), [3°,5°), [5°,90°)`, respectively. The steepness factor is `10.8 sin θ + 0.03` for θ < 5°, `16.8 sin θ − 0.5` for 5° ≤ θ < 10°, and `21.9 sin θ − 0.96` for θ ≥ 10°. The comparison uses degrees and the sine uses radians. `LS = L × S` is dimensionless.
3. **Original erosion grade.** Sample the native RKCP pixel containing the triangle centroid, then multiply by the face LS. The cut points are 500, 2,500, 5,000, 8,000 and 15,000 on the study's erosion-modulus scale. Equality enters the higher grade. Valid zero is grade 1. Out-of-bounds, masked, negative or non-finite samples yield an invalid grade of −1.
4. **Class-count and span objective.** Sample the face centroid and the three centroids of its equal-area subtriangles, using the original LS of that face throughout the search. For the four grades, K is the number of distinct classes and D is the maximum minus the minimum. Minimize `H = 0.5 (K − 1) / 3 + 0.5 D / 5` per face in the local sum. The equivalent integer score is `5 (K − 1) + 3 D = 30 H`. For grades (1, 3, 5, 5), K = 3, D = 4 and the integer score is 22. A candidate move must reduce the summed score of the faces incident to the shared vertex. Boundary vertices and vertices incident to initially invalid faces are fixed.
5. **Optimized map.** For each new centroid, query the original spatial LS field at that position and multiply by the RKCP in the same pixel. `CENT_LS`, `CENT_RKCP`, `RKCPLS` and `EROS_CLASS` record the resulting map values. `C0_NEW` stores the grade used by the objective at that centroid and can differ because the LS source face can change. Inherited `SLen_m` and `LS` describe the original RT-WFP computation. Moved vertex elevations are interpolated on the original terrain surface.

The study settings are one whole-mesh sweep in input record order, 50 moths, two coordinates, 100 iterations per local search, equal weights and base seed 20261019. The integer score describes the four sampling points. Independent assessments of whole-face uniformity or field prediction accuracy require their own reference and evaluation.

## Verify or run individual stages

```sh
python verify_example.py --inputs-only
python src/mfo/verify_small_cases.py
python verify_example.py --output outputs
```

For the bundled 10 m TIN and shared raster, the pipeline checks the input hashes. With the default seed it also writes `reproduction_check.json` in the selected output directory. For 5, 15 and 30 m, inspect the corresponding stage reports and GIS outputs. The historical output verifier and its example commands above are specific to 10 m. Use `--output outputs_10m` when checking the explicitly selected 10 m run.

The verifier checks the original 10 m input components and uses historical fingerprints for three GIS layers and the numerical arrays in both NPZ archives. It compares every DBF field and record, omitting the three file-header bytes used for the last-update date. The array checks include shape, dtype and every value. The verifier also compares the ordered local search and accepted-move logs, including the convergence curves and vertex positions. Runtime, log paths and ZIP timestamps can vary. These are strict checks for the tested numerical environment. Differences on another platform appear by file or array name in the report.

| Checkpoint for the 10 m example | Expected |
|---|---:|
| Faces / vertices | 26,861 / 14,180 |
| Traced segments / flat starting faces | 134,347 / 27 |
| Maximum horizontal length | 1,407.120386 m |
| Maximum LS | 93.537635 |
| Faces with four valid objective samples | 26,252 |
| Four-point mixed faces, before → after | 8,947 → 4,634 |
| Sum of grade spans, before → after | 17,177 → 7,299 |
| Sum of integer objective scores, before → after | 109,231 → 48,387 |

To run only the terrain stage, use `python src/terrain/compute_ls.py --source data/tin/rongcha10.shp --output terrain_only`. Full-precision per-face values are in `terrain_faces.csv`. In the NPZ archive, the route for zero-based record i is `points[point_offsets[i]:point_offsets[i+1]]`; use `segment_offsets` with `faces` and `kinds` for the corresponding segments.

For another compatible raw mesh and RKCP, run `python run_pipeline.py --tin path/to/raw.shp --rkcp path/to/RKCP.tif --output run_other`. The bundled historical output checkpoints apply to the 10 m example and default seed. See the [input specification](data/README.md) before substituting data.

## Methods and source provenance

The study bibliography identifies the following methodological sources.

- Lin, Z., Xie, J., Tang, Y., Ran, J. and Tan, Y. (2024). A reverse tracing of the water flow path algorithm for slope length extraction based on triangulated irregular network. *Computers & Geosciences*, 193, 105737. [DOI](https://doi.org/10.1016/j.cageo.2024.105737)
- Liu, B., Bi, X., Fu, S. et al. (2010). *Beijing Soil Loss Equation*. Science Press, Beijing. ISBN 9787030273857.
- Liu, B. Y., Nearing, M. A. and Risse, L. M. (1994). Slope gradient effects on soil loss for steep slopes. *Transactions of the ASAE*, 37, 1835–1840. [DOI](https://doi.org/10.13031/2013.28273)
- Mirjalili, S. (2015). Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm. *Knowledge-Based Systems*, 89, 228–249. [DOI](https://doi.org/10.1016/j.knosys.2015.07.006)


