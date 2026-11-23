"""Sample RKCP at each original triangle centroid and classify LS times RKCP."""

import argparse
import csv
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import rasterio
import shapefile


BREAKS = np.array([500.0, 2500.0, 5000.0, 8000.0, 15000.0])
ADDED_FIELDS = ("RKCP_VAL", "RUSLE_A", "SAMP_STAT", "A_CLASS")


def read_terrain(path):
    """Read the serialized LS field, exactly as the optimizer will receive it."""
    with shapefile.Reader(str(path), encoding="utf-8") as reader:
        if reader.shapeType != shapefile.POLYGONZ:
            raise ValueError("Expected a triangular PolygonZ mesh.")
        fields = list(reader.fields[1:])
        names = [field.name for field in fields]
        if not {"LS", "ORIG_FID", "Tri_Index"}.issubset(names):
            raise ValueError("The terrain layer must contain LS, ORIG_FID and Tri_Index.")
        if set(ADDED_FIELDS).intersection(names):
            raise ValueError("The terrain layer already contains erosion attributes.")
        records = list(reader.iterRecords())
        if len(records) != len(reader):
            raise ValueError("Deleted DBF records cannot be paired with geometry.")

        # The ring repeats its first vertex at the end. Average the three corners.
        centers = np.empty((len(records), 2), dtype=float)
        for i, shape in enumerate(reader.iterShapes()):
            if len(shape.parts) != 1 or len(shape.points) != 4 or shape.points[0] != shape.points[-1]:
                raise ValueError(f"Record {i} is not a closed, single-part triangle.")
            centers[i] = np.mean(shape.points[:3], axis=0)

    ls = np.array([record["LS"] for record in records], dtype=float)
    if not (np.isfinite(ls) & (ls >= 0)).all():
        raise ValueError("LS must be finite and nonnegative.")
    return fields, records, centers, ls


def sample_rkcp(path, centers, crs):
    """Select the containing native pixel. Status 1=valid, 2=outside, 3=invalid."""
    with rasterio.open(path) as src:
        if src.crs != crs:
            raise ValueError("The TIN and RKCP raster must have the same CRS.")
        transform = src.transform
        # Invert the full affine basis so that rotated rasters also map correctly.
        basis = np.array([[transform.a, transform.b], [transform.d, transform.e]])
        pixels = (centers - [transform.c, transform.f]) @ np.linalg.inv(basis).T
        col, row = pixels.T
        inside = np.isfinite(pixels).all(axis=1) & (col >= 0) & (col < src.width)
        inside &= (row >= 0) & (row < src.height)
        indices = np.flatnonzero(inside)
        cols, rows = np.floor(pixels[indices]).astype(np.int64).T
        sampled = src.read(1, masked=True)[rows, cols]
        valid = ~np.ma.getmaskarray(sampled) & np.isfinite(sampled.data) & (sampled.data >= 0)

    values = np.full(len(centers), np.nan)
    status = np.full(len(centers), 2, dtype=np.int8)
    status[indices] = 3
    status[indices[valid]] = 1
    values[indices[valid]] = sampled.data[valid]  # A valid zero is retained.
    return values, status


def classify(ls, rkcp, status):
    """Assign classes 1 through 6. A value on a break belongs to the higher class."""
    product = ls * rkcp
    valid = status == 1
    if not np.isfinite(product[valid]).all():
        raise ValueError("A valid centroid has a non-finite erosion value.")
    classes = np.full(len(ls), -1, dtype=np.int8)
    classes[valid] = np.searchsorted(BREAKS, product[valid], side="right") + 1
    return product, classes


def dbf_records(path, mode):
    """Map just the fixed-width record section of a dBASE attribute table."""
    with path.open("rb") as stream:
        header = stream.read(32)
    count = struct.unpack_from("<I", header, 4)[0]
    offset, width = struct.unpack_from("<HH", header, 8)
    return np.memmap(path, mode=mode, dtype=np.uint8, offset=offset, shape=(count, width))


def write_tables(source, target, fields, records, centers, ls, rkcp, product, status, classes):
    """Append four DBF attributes and write a human-readable per-face table."""
    with shapefile.Writer(dbf=str(target.with_suffix(".dbf")), encoding="utf-8") as writer:
        for field in fields:
            writer.field(field.name, str(field.field_type), field.size, field.decimal)
        writer.field("RKCP_VAL", "F", 26, 12)
        writer.field("RUSLE_A", "F", 26, 12)
        writer.field("SAMP_STAT", "N", 3, 0)
        writer.field("A_CLASS", "N", 3, 0)
        with (target.parent / "erosion_before.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            csv_writer = csv.writer(stream)
            csv_writer.writerow(["ORIG_FID", "Tri_Index", "LS", "CENT_X", "CENT_Y", *ADDED_FIELDS])
            for i, record in enumerate(records):
                valid = status[i] == 1
                values = (float(rkcp[i]) if valid else None, float(product[i]) if valid else None,
                          int(status[i]), int(classes[i]))
                writer.record(*record, *values)
                csv_writer.writerow([record["ORIG_FID"], record["Tri_Index"], ls[i], *centers[i], *values])

    # The schema prefix is unchanged. Restore the original bytes to avoid rounding
    # inherited numeric fields when extending the DBF with the centroid results.
    original = dbf_records(source.with_suffix(".dbf"), "r")
    extended = dbf_records(target.with_suffix(".dbf"), "r+")
    if len(original) != len(extended) or original.shape[1] >= extended.shape[1]:
        raise ValueError("Unexpected DBF layout while appending fields.")
    extended[:, :original.shape[1]] = original
    extended.flush()
    del original, extended
    for suffix in (".shp", ".shx", ".prj", ".cpg"):
        shutil.copyfile(source.with_suffix(suffix), target.with_suffix(suffix))


def run(source, raster, output):
    """Calculate the original-mesh centroid classification and return its SHP path."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty erosion output directory: {output}")
    fields, records, centers, ls = read_terrain(source)
    crs = rasterio.crs.CRS.from_wkt(source.with_suffix(".prj").read_text(encoding="utf-8-sig"))
    rkcp, status = sample_rkcp(raster, centers, crs)
    product, classes = classify(ls, rkcp, status)
    output.mkdir(parents=True, exist_ok=True)
    stem = source.stem.removesuffix("_RT_WFP_LS") + "_RT_WFP_RUSLE_A"
    target = output / (stem + ".shp")
    write_tables(source, target, fields, records, centers, ls, rkcp, product, status, classes)
    valid = status == 1
    report = {
        "input": source.name, "raster": raster.name, "output": target.name,
        "triangles": len(records), "crs": str(crs), "thresholds": BREAKS.tolist(),
        "class_counts": {str(k): int((classes == k).sum()) for k in (-1, 1, 2, 3, 4, 5, 6)},
        "status_counts": {str(k): int((status == k).sum()) for k in (1, 2, 3)},
        "valid_centroids": int(valid.sum()),
        "product_range": [float(product[valid].min()), float(product[valid].max())] if valid.any() else None,
    }
    (output / "erosion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="PolygonZ shapefile with an LS attribute")
    parser.add_argument("--raster", type=Path, required=True, help="RKCP raster in the same projected CRS")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args()
    print(run(args.input, args.raster, args.output))


if __name__ == "__main__":
    main()
