"""Read PolygonZ triangles, interpolate moved elevations and preserve inherited DBF fields."""

from pathlib import Path
import hashlib
import struct

import numpy as np
import shapefile
import shapely
from shapely import STRtree


def json_default(value):
    """Serialize NumPy scalars, arrays and paths in audit records."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def file_hash(path):
    """Stream the file into a SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field_name(field):
    """Get the field name from a PyShp descriptor."""
    return field.name if hasattr(field, "name") else field[0]


def _field_width(field):
    """Get the stored DBF field width in bytes."""
    return field.size if hasattr(field, "size") else field[2]


def _integer_identifiers(values, name):
    """Accept unique exact integer IDs, including zero and nonconsecutive values."""
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains missing or nonnumeric IDs.") from exc
    if (not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all()
            or np.any(np.abs(numeric) >= 2 ** 53)):
        raise ValueError(f"{name} must contain exactly representable finite integer IDs.")
    identifiers = numeric.astype(np.int64)
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError(f"{name} contains duplicate IDs.")
    return identifiers


def source_identifiers(records, column):
    """Identify original LS polygons by OBJECTID, ORIG_FID or one-based row number."""
    for name in ("OBJECTID", "ORIG_FID"):
        if name in column:
            return _integer_identifiers([row[column[name]] for row in records], name)
    return np.arange(1, len(records) + 1, dtype=np.int64)


def load_mesh(path):
    """Build the shared XYZ mesh and face attributes in source order from closed PolygonZ triangles."""
    with shapefile.Reader(str(Path(path)), encoding="utf-8") as reader:
        if reader.shapeType != shapefile.POLYGONZ:
            raise ValueError("Expected a PolygonZ shapefile.")
        fields = list(reader.fields[1:])
        records = [list(record) for record in reader.iterRecords()]
        if len(records) != len(reader):
            raise ValueError("Deleted DBF records prevent one-to-one matching of faces and rows.")
        if not records:
            raise ValueError("The input mesh contains no triangles.")
        coordinates = np.empty((len(records), 3, 2), dtype=np.float64)
        elevations = np.empty((len(records), 3), dtype=np.float64)
        for i, shape in enumerate(reader.iterShapes()):
            if list(shape.parts) != [0] or len(shape.points) != 4 or len(shape.z) != 4:
                raise ValueError(f"Feature {i + 1} must be a single-part closed PolygonZ triangle.")
            if tuple(shape.points[0]) != tuple(shape.points[-1]) or shape.z[0] != shape.z[-1]:
                raise ValueError(f"Feature {i + 1} has an unclosed XYZ ring.")
            coordinates[i] = np.asarray(shape.points[:3])
            elevations[i] = np.asarray(shape.z[:3])
    if not np.isfinite(coordinates).all() or not np.isfinite(elevations).all():
        raise ValueError("Mesh coordinates or elevations are not finite.")

    # Faces meeting at identical XY coordinates use one vertex index.
    xy, first, inverse = np.unique(coordinates.reshape(-1, 2), axis=0,
                                   return_index=True, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    z = elevations.ravel()[first]
    if not np.allclose(z[inverse], elevations.ravel(), atol=1e-9, rtol=0):
        raise ValueError("Vertices with identical XY have inconsistent Z elevations.")
    column = {_field_name(field): i for i, field in enumerate(fields)}
    missing = {"LS", "Tri_Index"}.difference(column)
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(sorted(missing))}.")
    ls = np.asarray([row[column["LS"]] for row in records], dtype=float)
    if not np.all(np.isfinite(ls) & (ls >= 0)):
        raise ValueError("LS must be finite and nonnegative.")
    # Use IDs for identification only; topology arrays always use source row order.
    ids = _integer_identifiers([row[column["Tri_Index"]] for row in records], "Tri_Index")
    return xy, z, triangles, ls, fields, records, column, ids


def geometry_metrics(xy, z, triangles):
    """Return planar area, perimeter, percent slope and downslope aspect (-1 for flat faces)."""
    points = xy[triangles]
    ab, ac = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    determinant = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    if np.any(np.abs(determinant) <= 1e-12):
        raise ValueError("Cannot compute the slope of a zero-area triangle.")
    area = np.abs(determinant) / 2
    perimeter = np.linalg.norm(points - np.roll(points, -1, axis=1), axis=2).sum(axis=1)
    dzb = z[triangles[:, 1]] - z[triangles[:, 0]]
    dzc = z[triangles[:, 2]] - z[triangles[:, 0]]
    gx = (dzb * ac[:, 1] - dzc * ab[:, 1]) / determinant
    gy = (ab[:, 0] * dzc - ac[:, 0] * dzb) / determinant
    slope = 100 * np.hypot(gx, gy)
    aspect = np.degrees(np.arctan2(-gx, -gy)) % 360
    aspect[slope < 1e-10] = -1
    return area, perimeter, slope, aspect


def interpolate_original_surface(xy0, z0, triangles, xy_new):
    """Use barycentric interpolation on the original TIN to assign elevations to moved vertices."""
    z_new = z0.copy()
    # Displacements of at most 1e-9 metres retain their original elevation.
    moved = np.flatnonzero(np.linalg.norm(xy_new - xy0, axis=1) > 1e-9)
    if not len(moved):
        return z_new
    polygons = shapely.polygons(xy0[triangles])
    tree = STRtree(polygons)
    pairs = tree.query(shapely.points(xy_new[moved]), predicate="intersects")
    solved = np.zeros(len(moved), dtype=bool)
    for local_id, face in pairs.T:
        if solved[local_id]:
            continue
        vertex = moved[local_id]
        a, b, c = xy0[triangles[face]]
        ab, ac, ap = b - a, c - a, xy_new[vertex] - a
        det = ab[0] * ac[1] - ab[1] * ac[0]
        wb = (ap[0] * ac[1] - ap[1] * ac[0]) / det
        wc = (ab[0] * ap[1] - ab[1] * ap[0]) / det
        weights = np.array([1 - wb - wc, wb, wc])
        if weights.min() >= -1e-8:
            z_new[vertex] = weights @ z0[triangles[face]]
            solved[local_id] = True
    if not solved.all():
        raise ValueError(f"Failed to locate {(~solved).sum()} moved vertices on the original surface.")
    return z_new


def restore_inherited_dbf_bytes(source_path, output_path, fields, indices):
    """Copy original field bytes by source row; area and perimeter use the new geometry."""
    source_path, output_path = Path(source_path), Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Output DBF must differ from the input DBF.")
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return

    def map_records(path, mode):
        """Map fixed-width records using the record count and lengths in the DBF header."""
        with path.open("rb") as handle:
            header = handle.read(32)
        count = struct.unpack_from("<I", header, 4)[0]
        header_size, record_size = struct.unpack_from("<HH", header, 8)
        return np.memmap(path, mode=mode, dtype=np.uint8, offset=header_size,
                         shape=(count, record_size))

    source = map_records(source_path, "r")
    target = map_records(output_path, "r+")
    if len(target) != len(indices):
        raise ValueError("DBF row count does not match the export index.")
    if indices.min() < 0 or indices.max() >= len(source):
        raise ValueError("Source index exceeds the DBF record range.")
    inherited_end = 1 + sum(_field_width(field) for field in fields)
    if inherited_end != source.shape[1] or target.shape[1] < inherited_end:
        raise ValueError("Source field widths disagree with the DBF layout.")
    offset = 1  # The first byte of each row stores its deletion flag.
    for field in fields:
        end = offset + _field_width(field)
        # Preserve historical attributes; the re-sampled centroid has dedicated fields.
        if _field_name(field) not in {"Shape_Area", "Shape_Leng"}:
            target[:, offset:end] = source[indices, offset:end]
        offset = end
    target.flush()
    del target, source
