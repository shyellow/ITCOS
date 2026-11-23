"""Compute RT-WFP slope length and LS for a triangular PolygonZ shapefile.

Supply a projected metre-based TIN with elevations in metres. Input records
must have a unique Tri_Index, a Slope_Deg attribute and UTF-8 text fields.
The result retains the input XY, Z, record order and coordinate system.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import mmap
import shutil
import struct
import time
from collections import Counter

import numpy as np
import shapefile
from rtwfp_core import Terrain, factors

# This default refers to the caller's working directory.
OUTPUT = Path("output") / "terrain"


def digest(path):
    """Stream a SHA-256 check of a shapefile component."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_source(source):
    """Read one closed triangle per row and preserve the source row-to-ID mapping."""
    with shapefile.Reader(str(source), encoding="utf-8") as reader:
        fields = reader.fields[1:]
        records = reader.records()
        xyz = np.empty((len(reader), 3, 3))
        for index, shape in enumerate(reader.iterShapes()):
            if shape.shapeType != shapefile.POLYGONZ or len(shape.points) != 4 or len(shape.parts) != 1:
                raise ValueError("Each shape must be a single closed PolygonZ triangle.")
            if shape.points[0] != shape.points[-1] or shape.z[0] != shape.z[-1]:
                raise ValueError("Triangle ring is open in XY or Z.")
            xyz[index, :, :2] = shape.points[:3]
            xyz[index, :, 2] = shape.z[:3]
    if not np.isfinite(xyz).all():
        raise ValueError("Coordinates contain non-finite values.")
    ids = np.array([record["Tri_Index"] for record in records])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Tri_Index must be unique.")
    return xyz, fields, records, ids


def run_paths(terrain, output, limit=None):
    """Trace each centroid and archive its path vertices, segment types and stop reason.

    A pilot limits the number of origins while allowing paths to cross every face.
    """
    count = terrain.n if limit is None else min(limit, terrain.n)
    lengths = np.empty(count)
    statuses, path_points, path_faces, path_kinds = [], [], [], []
    point_offsets, segment_offsets = [0], [0]
    start = time.perf_counter()
    acceptable = {"BOUNDARY", "LOCAL_MAX", "FLAT_START", "FLAT", "LEVEL_RIDGE"}
    for face in range(count):
        trace = terrain.trace(face)
        if trace.status not in acceptable:
            raise RuntimeError(f"Trace failed at row {face}, Tri_Index={terrain.tri_ids[face]}, status={trace.status}.")
        lengths[face] = trace.length
        statuses.append(trace.status)
        points = np.asarray(trace.points)
        # Restore projected coordinates on a copy to preserve cached local coordinates.
        points[:, :2] += terrain.origin
        path_points.append(points)
        path_faces.append(np.asarray(trace.faces, dtype=np.int32))
        path_kinds.append(np.asarray(trace.kinds, dtype=np.uint8))
        point_offsets.append(point_offsets[-1] + len(points))
        segment_offsets.append(segment_offsets[-1] + len(trace.faces))
        if (face + 1) % 5000 == 0 or face + 1 == count:
            print(json.dumps(dict(processed=face + 1, total=count, seconds=round(time.perf_counter() - start, 1),
                                  segments=segment_offsets[-1], statuses=Counter(statuses)), ensure_ascii=False), flush=True)
    arrays = dict(points=np.vstack(path_points), faces=np.concatenate(path_faces), kinds=np.concatenate(path_kinds),
                  point_offsets=np.asarray(point_offsets), segment_offsets=np.asarray(segment_offsets),
                  lengths=lengths, statuses=np.asarray(statuses), tri_ids=terrain.tri_ids[:count],
                  slope_degrees=terrain.slope_deg[:count], gradient=terrain.gradient[:count])
    # Offsets delimit each origin's path in the concatenated point and segment arrays.
    # The segment face column is a zero-based row; tri_ids stores the external ID.
    np.savez_compressed(output / "rtwfp_paths_and_results.npz", **arrays)
    return arrays


def restore_original_field_bytes(source_dbf, target_dbf):
    """Restore unchanged DBF fields byte for byte, including scientific notation.

    Slope_Deg has a new width in the output, so each table needs its own offsets.
    """
    def layout(path):
        with path.open("rb") as stream:
            header = stream.read(32)
            count = struct.unpack_from("<I", header, 4)[0]
            head_size, row_size = struct.unpack_from("<HH", header, 8)
            columns, offset = {}, 1  # The first byte marks deleted records.
            while stream.tell() < head_size - 1:
                field = stream.read(32)
                name = field[:11].split(b"\0")[0].decode("ascii")
                width = field[16]
                columns[name] = (offset, width)
                offset += width
        return count, head_size, row_size, columns

    count, source_header, source_row, source_columns = layout(source_dbf)
    target_count, target_header, target_row, target_columns = layout(target_dbf)
    if count != target_count:
        raise ValueError("Source and target DBF row counts differ.")
    with source_dbf.open("rb") as source_handle, target_dbf.open("r+b") as target_handle:
        with mmap.mmap(source_handle.fileno(), 0, access=mmap.ACCESS_READ) as src:
            with mmap.mmap(target_handle.fileno(), 0, access=mmap.ACCESS_WRITE) as dst:
                for name, (old_offset, width) in source_columns.items():
                    if name == "Slope_Deg":
                        continue
                    new_offset, new_width = target_columns[name]
                    if width != new_width:
                        raise ValueError(f"Input field width changed for {name}.")
                    for i in range(count):
                        a = source_header + source_row * i + old_offset
                        b = target_header + target_row * i + new_offset
                        dst[b:b + width] = src[a:a + width]
                dst.flush()


def save_shapefile(source, output, fields, records, terrain, arrays, computed):
    """Write computed attributes, preserving the original SHP and SHX geometry."""
    target = output / f"{source.stem}_RT_WFP_LS.shp"
    additions = [("Slope_Pct", "F", 20, 10), ("SLen_m", "F", 20, 8),
                 ("L_Exp", "F", 5, 1), ("L_Factor", "F", 20, 10),
                 ("S_Factor", "F", 20, 10), ("LS", "F", 20, 10),
                 ("RT_Status", "C", 20, 0), ("RT_Steps", "N", 10, 0), ("ORIG_FID", "N", 10, 0),
                 ("End_X", "F", 20, 6), ("End_Y", "F", 20, 6), ("End_Z", "F", 20, 6)]
    existing = {field[0] for field in fields}
    if any(field[0] in existing for field in additions):
        raise ValueError("A computed field already exists in the raw input.")
    with shapefile.Writer(dbf=str(target.with_suffix(".dbf")), encoding="utf-8") as writer:
        for field in fields:
            if field[0] == "Slope_Deg":
                writer.field("Slope_Deg", "F", 20, 10)
            else:
                writer.field(*field)
        for field in additions:
            writer.field(*field)
        slope_index = [field[0] for field in fields].index("Slope_Deg")
        for i, record in enumerate(records):
            values = list(record)
            values[slope_index] = float(terrain.slope_deg[i])
            endpoint = arrays["points"][arrays["point_offsets"][i + 1] - 1]
            steps = arrays["segment_offsets"][i + 1] - arrays["segment_offsets"][i]
            # Append slope percent, horizontal length, (m, L, S, LS) and routing metadata.
            values.extend([float(terrain.gradient_norm[i] * 100), float(arrays["lengths"][i]),
                           *[float(value[i]) for value in computed], str(arrays["statuses"][i]),
                           int(steps), i, *endpoint.tolist()])
            writer.record(*values)
    restore_original_field_bytes(source.with_suffix(".dbf"), target.with_suffix(".dbf"))
    for suffix in (".shp", ".shx", ".prj"):
        shutil.copyfile(source.with_suffix(suffix), target.with_suffix(suffix))
    target.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    if digest(source) != digest(target) or digest(source.with_suffix(".shx")) != digest(target.with_suffix(".shx")):
        raise RuntimeError("Output geometry differs from the input.")
    return target


def main():
    """Build the full terrain and export its routes, factors and diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Single-ring PolygonZ triangles with XY and Z in metres.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pilot", type=int, help="Trace the first N origins using full topology; write the path archive only.")
    args = parser.parse_args()
    if args.pilot is not None and args.pilot < 1:
        parser.error("--pilot must be a positive integer.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be an empty or new directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {suffix: digest(args.source.with_suffix(suffix)) for suffix in (".shp", ".shx", ".dbf", ".prj")}
    xyz, fields, records, ids = load_source(args.source)
    terrain = Terrain(xyz, ids)
    print(json.dumps(dict(triangles=terrain.n, nodes=len(terrain.nodes), flat_faces=int((terrain.slope_deg == 0).sum())),
                     ensure_ascii=False), flush=True)
    arrays = run_paths(terrain, args.output, args.pilot)
    if args.pilot is not None:
        return
    computed = factors(terrain.slope_deg, arrays["lengths"])
    if not all(np.isfinite(value).all() and (value >= 0).all() for value in computed):
        raise RuntimeError("Factor calculation returned a non-finite or negative value.")
    target = save_shapefile(args.source, args.output, fields, records, terrain, arrays, computed)
    with (args.output / "terrain_faces.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ORIG_FID", "Tri_Index", "Slope_Deg", "Slope_Pct", "SLen_m", "L_Exp", "L_Factor", "S_Factor", "LS", "RT_Status", "RT_Steps"])
        for i in range(terrain.n):
            writer.writerow([i, ids[i], terrain.slope_deg[i], terrain.gradient_norm[i] * 100, arrays["lengths"][i],
                             *[value[i] for value in computed], arrays["statuses"][i],
                             arrays["segment_offsets"][i + 1] - arrays["segment_offsets"][i]])
    current_hashes = {suffix: digest(args.source.with_suffix(suffix)) for suffix in hashes}
    if current_hashes != hashes:
        raise RuntimeError("Input components changed during calculation.")
    original_slope = np.array([record["Slope_Deg"] for record in records])
    report = dict(source=args.source.name, output=target.name, triangle_count=terrain.n,
                  node_count=len(terrain.nodes), source_hashes=hashes, source_unchanged=True,
                  geometry_identical=True,
                  crs_wkt=args.source.with_suffix(".prj").read_text(encoding="utf-8-sig"),
                  xy_unit="m (input requirement; not reprojected)", z_unit="m (input requirement; caller must confirm)",
                  length_unit="horizontal m", slope_unit="degree", factors_unit="dimensionless",
                  trace_direction="uphill, reverse of downslope water flow", start="triangle centroid",
                  xy_tolerance_m=terrain.xy_eps, status_counts=dict(Counter(arrays["statuses"].tolist())),
                  path_segment_count=len(arrays["faces"]),
                  slope_range_deg=[float(terrain.slope_deg.min()), float(terrain.slope_deg.max())],
                  slope_original_max_abs_difference_deg=float(np.max(np.abs(original_slope - terrain.slope_deg))),
                  length_range_m=[float(arrays["lengths"].min()), float(arrays["lengths"].max())],
                  length_quantiles_m=np.quantile(arrays["lengths"], [0, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).tolist(),
                  ls_range=[float(computed[3].min()), float(computed[3].max())],
                  ten_degree_boundary="theta>=10 belongs to S=21.9*sin(theta)-0.96",
                  flat_policy="Flat origin: SLen_m=0, LS=0, RT_Status=FLAT_START; no artificial direction",
                  nodata_policy="Unexpected numeric/path errors abort; do not silently fill zero")
    (args.output / "calculation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
