"""Run one sweep of four-point, class-count and grade-span MFO on a PolygonZ TIN."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path
import numpy as np
import rasterio
import shapefile
import shapely
from shapely import STRtree
from mesh_geometry import RasterSampler, classify_mesh
from mfo_mesh_optimizer import optimize_mesh
from objective import (build_objective_spec, spans_from_classes, scores_from_metrics,
                       h_from_scores, validate_score_capacity)
from tin_io import (file_hash, geometry_metrics, interpolate_original_surface,
                    json_default, load_mesh, restore_inherited_dbf_bytes, source_identifiers)

HERE = Path(__file__).resolve().parent
THRESHOLDS = np.array([500,2500,5000,8000,15000], dtype=float)
COLORS = ["#38A800","#79C900","#CEED00","#FFCC00","#FF6600","#FF0000"]
LABELS = ["Very slight", "Slight", "Moderate", "Severe", "Very severe", "Extreme"]


def centroid_erosion(xy0, xy, triangles, ls, objectids, sampler):
    """Query the original piecewise-constant LS at the new face centroid and multiply by its RKCP."""
    centers = xy[triangles].mean(axis=1)
    source_polygons = shapely.polygons(xy0[triangles])
    tree = STRtree(source_polygons)
    points = shapely.points(centers)
    pairs = tree.query(points, predicate="intersects")
    hits = np.bincount(pairs[0], minlength=len(triangles))
    missing = np.flatnonzero(hits == 0)
    fallback_count = 0
    if len(missing):
        # Limit floating-point fallback to 1e-7 metres at polygon boundaries.
        fallback = tree.query(points[missing], predicate="dwithin", distance=1e-7)
        if fallback.shape[1]:
            fallback[0] = missing[fallback[0]]
            pairs = np.hstack((pairs, fallback))
            fallback_count = len(np.unique(fallback[0]))
            hits = np.bincount(pairs[0], minlength=len(triangles))
    selected = np.full(len(triangles), -1, dtype=int)
    if pairs.shape[1]:
        same_id = objectids[pairs[0]] == objectids[pairs[1]]
        # Break ties by matching source ID, then by the smallest source ID.
        order = np.lexsort((objectids[pairs[1]], ~same_id, pairs[0]))
        ordered = pairs[:, order]
        first = np.r_[True, ordered[0, 1:] != ordered[0, :-1]]
        selected[ordered[0, first]] = ordered[1, first]
    sampled_ls = np.full(len(triangles), np.nan)
    source_ids = np.full(len(triangles), -1, dtype=int)
    located = selected >= 0
    sampled_ls[located] = ls[selected[located]]
    source_ids[located] = objectids[selected[located]]
    rkcp = sampler.sample_xy(centers)
    valid = (located & np.isfinite(sampled_ls) & (sampled_ls >= 0)
             & np.isfinite(rkcp) & (rkcp >= 0))
    product = np.where(valid, sampled_ls * rkcp, np.nan)
    classes = np.full(len(triangles), -1, dtype=int)
    classes[valid] = np.searchsorted(THRESHOLDS, product[valid], side="right") + 1
    return dict(centers=centers, ls=sampled_ls, rkcp=rkcp, product=product,
                classes=classes, valid=valid, source_ids=source_ids, ties=hits,
                tolerance_fallback_count=fallback_count)


def write_qgis_style(path):
    """Write a categorical QGIS fill style for the final centroid erosion class."""
    ranges = ["[0, 500)", "[500, 2500)", "[2500, 5000)", "[5000, 8000)",
              "[8000, 15000)", "≥15000"]
    colors = COLORS + ["#B0B0B0"]
    labels = [f"{label}  {value}" for label, value in zip(LABELS, ranges)] + ["No valid centroid sample"]
    categories, symbols = [], []
    for i, (color, label) in enumerate(zip(colors, labels)):
        value = i + 1 if i < 6 else -1
        rgb = ",".join(str(int(color[j:j + 2], 16)) for j in [1, 3, 5]) + ",255"
        label = label.replace("<", "&lt;")
        categories.append(f'<category value="{value}" label="{label}" symbol="{i}" render="true"/>')
        symbols.append(f'<symbol name="{i}" type="fill"><layer class="SimpleFill" enabled="1">'
                       f'<Option type="Map"><Option name="color" value="{rgb}" type="QString"/>'
                       '<Option name="outline_style" value="no" type="QString"/></Option></layer></symbol>')
    path.write_text('<qgis version="3.34" styleCategories="Symbology">'
                    '<renderer-v2 type="categorizedSymbol" attr="EROS_CLASS">'
                    '<categories>' + ''.join(categories) + '</categories><symbols>' + ''.join(symbols) +
                    '</symbols></renderer-v2></qgis>', encoding="utf-8")


def log(event):
    """Print a machine-readable progress record."""
    print(json.dumps(event, ensure_ascii=False, default=json_default), flush=True)


def prepare_export(mesh, result, center):
    """Recheck the cached objective by resampling all faces before preparing export fields."""
    xy0, z0, triangles, ls, fields, records, column, ids = mesh
    counts, classes, values = classify_mesh(result["xy"], triangles, ls, result["sampler"])
    spans = spans_from_classes(classes)
    scores = scores_from_metrics(counts, spans, result["objective_spec"])
    for actual, key in ((counts, "counts_after"), (spans, "spans_after"), (scores, "scores_after")):
        if not np.array_equal(actual, result[key]):
            raise RuntimeError(f"Cache mismatch before export: {key}.")
    area, perimeter, slope, aspect = geometry_metrics(result["xy"], result["z"], triangles)
    changed = np.any(result["xy"][triangles] != xy0[triangles], axis=(1, 2))
    before, after = result["counts_before"], result["counts_after"]
    span0, span = result["spans_before"], result["spans_after"]
    score0, score = result["scores_before"], result["scores_after"]
    valid = before >= 1
    # Track improvement of the joint score separately from its count and span components.
    integers = dict(TRI_ID=ids, NCLASS0=before, NCLASS=after, SPAN0=span0, SPAN=span,
                    SCORE0=score0, SCORE=score, SAMPLE_OK=(classes > 0).sum(axis=1),
                    OPT_FLAG=changed, IMPROVED=valid & (score < score0),
                    N_IMPROVE=valid & (after < before), D_IMPROVE=valid & (span < span0),
                    C0_NEW=classes[:, 0], C1_NEW=classes[:, 1], C2_NEW=classes[:, 2],
                    C3_NEW=classes[:, 3], EROS_CLASS=center["classes"],
                    LS_SRC_ID=center["source_ids"], CEN_VALID=center["valid"], LS_TIES=center["ties"])
    floats = dict(OBJ0=h_from_scores(score0, result["objective_spec"]),
                  OBJ=h_from_scores(score, result["objective_spec"]),
                  A0_NEW=values[:, 0], A1_NEW=values[:, 1], A2_NEW=values[:, 2], A3_NEW=values[:, 3],
                  G_SLP_PCT=slope, G_ASPECT=aspect, CENT_X=center["centers"][:, 0],
                  CENT_Y=center["centers"][:, 1], CENT_LS=center["ls"],
                  CENT_RKCP=center["rkcp"], RKCPLS=center["product"])
    collisions = set(integers).union(floats).intersection(column)
    if collisions:
        raise ValueError(f"Input already contains result fields: {sorted(collisions)}.")
    return dict(integers=integers, floats=floats, area=area, perimeter=perimeter,
                changed=changed, classes=classes, sample_values=values)


def export_shapefile(path, indices, source, mesh, result, prepared):
    """Write PolygonZ for selected faces and preserve inherited DBF bytes by source row."""
    xy0, z0, triangles, ls, fields, records, column, ids = mesh
    indices = np.asarray(indices, dtype=np.int64)
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGONZ, encoding="utf-8") as writer:
        for field in fields:
            writer.field(field.name, str(field.field_type), field.size, field.decimal)
        for name in prepared["integers"]:
            writer.field(name, "N", 20 if name in {"SCORE0", "SCORE"} else 10, 0)
        for name in prepared["floats"]:
            writer.field(name, "F", 26, 12)
        for row_id in indices:
            ring = [[float(result["xy"][v, 0]), float(result["xy"][v, 1]), float(result["z"][v])]
                    for v in triangles[row_id]]
            writer.polyz([ring + [ring[0].copy()]])
            row = records[row_id].copy()
            row[column["Shape_Area"]] = float(prepared["area"][row_id])
            row[column["Shape_Leng"]] = float(prepared["perimeter"][row_id])
            row += [int(array[row_id]) for array in prepared["integers"].values()]
            # Write invalid float samples as DBF nulls and preserve numerical zero.
            row += [float(array[row_id]) if np.isfinite(array[row_id]) else None
                    for array in prepared["floats"].values()]
            writer.record(*row)
    restore_inherited_dbf_bytes(source.with_suffix(".dbf"), path.with_suffix(".dbf"), fields, indices)
    shutil.copyfile(source.with_suffix(".prj"), path.with_suffix(".prj"))
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    write_qgis_style(path.with_suffix(".qml"))


def write_audit(output, mesh, result, prepared):
    """Export four-point classification, accepted moves and the per-call search history."""
    xy0, z0, triangles, ls, fields, records, column, ids = mesh
    source_ids = source_identifiers(records, column)
    columns = {"SOURCE_ID": source_ids, **prepared["integers"], **prepared["floats"]}
    with (output / "triangle_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row_id in range(len(triangles)):
            row = [int(source_ids[row_id])]
            row += [int(array[row_id]) for array in prepared["integers"].values()]
            row += [float(array[row_id]) if np.isfinite(array[row_id]) else ""
                    for array in prepared["floats"].values()]
            writer.writerow(row)
    with (output / "moved_vertices.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["VERTEX_ID", "X_OLD", "Y_OLD", "Z_OLD", "X_NEW", "Y_NEW", "Z_NEW", "MOVE_M"])
        distance = np.linalg.norm(result["xy"] - xy0, axis=1)
        for vertex in np.flatnonzero(distance > 0):
            writer.writerow([vertex, *xy0[vertex], z0[vertex], *result["xy"][vertex],
                             result["z"][vertex], distance[vertex]])
    for filename, key in (("accepted_moves.jsonl", "movements"), ("mfo_runs.jsonl", "mfo_calls")):
        with (output / filename).open("w", encoding="utf-8") as stream:
            for row in result[key]:
                stream.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def summarize(mesh, result, center, prepared):
    """Count each valid face once in count, span, integer score and normalized heterogeneity totals."""
    xy0, z0, triangles, ls, fields, records, column, ids = mesh
    before, after = result["counts_before"], result["counts_after"]
    valid = before >= 1
    stats = dict(triangles=len(triangles), vertices=len(xy0), valid_objective=int(valid.sum()),
                 invalid_objective=int((~valid).sum()), mixed_before=int((before > 1).sum()),
                 mixed_after=int((after > 1).sum()), moved_vertices=int(np.any(result["xy"] != xy0, axis=1).sum()),
                 changed_triangles=int(prepared["changed"].sum()), valid_centroids=int(center["valid"].sum()),
                 centroid_class_counts=dict(Counter(center["classes"].tolist())))
    for label, key in (("count", "counts"), ("span", "spans"), ("score", "scores")):
        for stage in ("before", "after"):
            values = result[f"{key}_{stage}"]
            stats[f"{label}_sum_{stage}"] = int(values[valid].sum())
            stats[f"{label}_distribution_{stage}"] = dict(Counter(values.tolist()))
    scale = result["objective_spec"]["scale"]
    for stage in ("before", "after"):
        stats[f"objective_{stage}"] = stats[f"score_sum_{stage}"] / scale
        stats[f"mean_span_{stage}"] = stats[f"span_sum_{stage}"] / int(valid.sum())
        stats[f"large_span_faces_{stage}"] = int((result[f"spans_{stage}"][valid] >= 3).sum())
    stats["score_improved_faces"] = int(prepared["integers"]["IMPROVED"].sum())
    stats["score_worsened_faces"] = int((result["scores_after"][valid] > result["scores_before"][valid]).sum())
    return stats


def input_hashes(source, raster):
    """Hash the actual shapefile components and RKCP raster before and after the run."""
    return {**{suffix: file_hash(source.with_suffix(suffix))
               for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")
               if source.with_suffix(suffix).is_file()}, "RKCP": file_hash(raster)}


def write_run_readme(output, report):
    """Write the run parameters, field definitions and global score change alongside the result."""
    spec, stats = report["objective_spec"], report["statistics"]
    content = f"""Count-and-span MFO
Result: {report['output_shp']}
Four points: the centroid G and the centroids of the three edge-G subtriangles.
H = wN*(K-1)/3 + wD*SPAN/5. Weights: {spec['weight_count']}, {spec['weight_span']}.
S = {spec['a']}*(K-1) + {spec['b']}*SPAN. H = S/{spec['scale']}.
Total S: {stats['score_sum_before']} -> {stats['score_sum_after']}.
One sweep, population 50, 2 coordinates, 100 local iterations, seed {report['config']['seed']}.

NCLASS0/NCLASS and SPAN0/SPAN give the four-point class count and range before/after.
SCORE0/SCORE and OBJ0/OBJ give integer S and normalized H. Invalid scores are -1 or null.
The optimizer accepts a strict decrease of summed incident-face S. Individual scores may rise.
C0_NEW through C3_NEW use each face's original LS. RKCPLS and EROS_CLASS use the
original spatial LS field and RKCP at the final centroid. LS and SLen_m retain original values.
Moved elevations are interpolated on the original TIN. Areas and perimeters use the new XY.
"""
    (output / "RUN_NOTES.txt").write_text(content, encoding="utf-8-sig")


def main():
    """Run the optimizer, interpolate moved elevations and write the optimized mesh."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "raster", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--weight-count", default="0.5", help="Positive rational weight for the normalized class count.")
    parser.add_argument("--weight-span", default="0.5", help="Positive rational span weight; the two weights must sum to one.")
    parser.add_argument("--max-sweeps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if args.max_sweeps != 1:
        parser.error("This workflow requires exactly one sweep.")
    spec = build_objective_spec(args.weight_count, args.weight_span)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("The output directory must be new or empty.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before_hashes = input_hashes(args.input, args.raster)
    mesh = load_mesh(args.input)
    xy0, z0, triangles, ls, fields, records, column, ids = mesh
    if {"Shape_Area", "Shape_Leng"}.difference(column):
        raise ValueError("The input must contain Shape_Area and Shape_Leng for geometry updates.")
    validate_score_capacity(len(triangles), spec)
    with rasterio.open(args.raster) as raster:
        crs = rasterio.crs.CRS.from_wkt(args.input.with_suffix(".prj").read_text(encoding="utf-8-sig"))
        if crs != raster.crs:
            raise ValueError("The TIN and RKCP raster must use the same projected CRS.")
        sampler = RasterSampler(raster.read(1, masked=True), raster.transform, raster.nodata)
    config = dict(pop=50, max_iter=100, max_sweeps=1, seed=args.seed,
                  convergence_patience=2, weight_count=args.weight_count, weight_span=args.weight_span,
                  objective_spec=spec)
    log(dict(stage="input_loaded", triangles=len(triangles), vertices=len(xy0), objective_spec=spec))
    result = optimize_mesh(xy0, triangles, ls, sampler, config, callback=log)
    if not np.any(result["counts_before"] >= 1):
        raise ValueError("No triangle has four valid samples.")
    result["sampler"] = sampler
    log(dict(stage="interpolate_z_and_centroid_sampling"))
    result["z"] = interpolate_original_surface(xy0, z0, triangles, result["xy"])
    center = centroid_erosion(xy0, result["xy"], triangles, ls,
                             source_identifiers(records, column), sampler)
    prepared = prepare_export(mesh, result, center)
    prefix = args.input.stem + "_MFO_SPAN"
    groups = [(prefix, np.arange(len(triangles))),
              (prefix + "_changed", np.flatnonzero(prepared["changed"])),
              (prefix + "_unresolved", np.flatnonzero((result["counts_after"] > 1)
                                                       | (result["counts_after"] < 0)))]
    for name, indices in groups:
        log(dict(stage="export", name=name, count=len(indices)))
        export_shapefile(output / (name + ".shp"), indices, args.input, mesh, result, prepared)
    write_audit(output, mesh, result, prepared)
    stats = summarize(mesh, result, center, prepared)
    if input_hashes(args.input, args.raster) != before_hashes:
        raise RuntimeError("The source files changed during optimization.")
    report = dict(input=args.input.name, raster=args.raster.name,
                  output_shp=prefix + ".shp", config=config, objective_spec=result["objective_spec"],
                  statistics=stats, pass_stats=result["pass_stats"], input_hashes=before_hashes,
                  hashes={"input_shp": before_hashes[".shp"], "input_dbf": before_hashes[".dbf"],
                          "rkcp": before_hashes["RKCP"], "MFO_original": file_hash(HERE / "MFO_original.py")},
                  input_unchanged=True, sample_point_order=["G", "S1", "S2", "S3"],
                  objective_ls="fixed_original_LS_per_input_face",
                  map_ls="spatial_original_LS_field_at_final_triangle_centroid",
                  ls_source_id_field=next((name for name in ("OBJECTID", "ORIG_FID") if name in column),
                                          "SOURCE_ROW_1_BASED"),
                  thresholds=THRESHOLDS, colors=COLORS,
                  runtime_seconds=time.perf_counter()-started)
    (output / "optimization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    np.savez_compressed(output / "mesh_result.npz", xy0=xy0, z0=z0, xy=result["xy"], z=result["z"],
                        triangles=triangles, ls=ls, ids=ids,
                        counts_before=result["counts_before"], counts_after=result["counts_after"],
                        spans_before=result["spans_before"], spans_after=result["spans_after"],
                        scores_before=result["scores_before"], scores_after=result["scores_after"],
                        classes_after=prepared["classes"], sample_values_after=prepared["sample_values"],
                        centroid_ls=center["ls"], centroid_rkcp=center["rkcp"],
                        rkcp_ls=center["product"], erosion_class=center["classes"])
    write_run_readme(output, report)
    log(dict(stage="complete", statistics=stats, runtime_seconds=report["runtime_seconds"]))


if __name__ == "__main__":
    main()
