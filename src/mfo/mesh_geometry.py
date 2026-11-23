"""Sample RKCP at native pixels, classify four points and build shared-vertex topology."""

import numpy as np

_THRESHOLDS = np.array([500.0, 2500.0, 5000.0, 8000.0, 15000.0])

class RasterSampler:
    """Sample the raster cell containing each projected (x, y) coordinate."""

    def __init__(self, array, transform, nodata):
        """Retain the source validity mask and invert the full affine transform."""
        source = np.asanyarray(array)
        if source.ndim != 2:
            raise ValueError("Expected a two-dimensional raster.")
        self.array = np.asarray(np.ma.getdata(source))
        self.mask = np.ma.getmaskarray(source) if np.ma.isMaskedArray(source) else None
        self.nodata = nodata
        self.height, self.width = self.array.shape

        # Accept a rasterio Affine or its six coefficients.
        if hasattr(transform, "a"):
            a, b, c, d, e, f = (
                transform.a, transform.b, transform.c,
                transform.d, transform.e, transform.f,
            )
        else:
            a, b, c, d, e, f = tuple(transform)[:6]
        self.matrix = np.array([[a, b], [d, e]], dtype=float)
        self.offset = np.array([c, f], dtype=float)
        if not np.isfinite(self.matrix).all() or not np.isfinite(self.offset).all():
            raise ValueError("Raster transform must be finite.")
        self.inverse = np.linalg.inv(self.matrix)

    def world_to_pixel(self, xy):
        """Convert world coordinates to fractional (column, row) coordinates."""
        points = np.asarray(xy, dtype=float)
        if points.shape[-1:] != (2,):
            raise ValueError("The final coordinate axis must have length two.")
        return (points - self.offset) @ self.inverse.T

    def pixel_to_world(self, xy):
        """Apply the forward raster transform to fractional (column, row) coordinates."""
        return np.asarray(xy, dtype=float) @ self.matrix.T + self.offset

    def sample_xy(self, xy):
        """Return a cell value per point; invalid or out-of-bounds samples are NaN."""
        pixels = self.world_to_pixel(xy)
        output_shape = pixels.shape[:-1]
        flat = pixels.reshape(-1, 2)
        values = np.full(len(flat), np.nan, dtype=float)
        finite = np.isfinite(flat).all(axis=1)
        # Check fractional bounds before flooring or converting NaN to integer.
        inside = (
            finite & (flat[:, 0] >= 0) & (flat[:, 0] < self.width)
            & (flat[:, 1] >= 0) & (flat[:, 1] < self.height)
        )
        indices = np.flatnonzero(inside)
        if len(indices):
            col_row = np.floor(flat[indices]).astype(np.int64)
            cols, rows = col_row[:, 0], col_row[:, 1]
            sampled = np.asarray(self.array[rows, cols], dtype=float)
            valid = np.isfinite(sampled)
            if self.nodata is not None:
                valid &= sampled != self.nodata
            if self.mask is not None:
                valid &= ~self.mask[rows, cols]
            values[indices[valid]] = sampled[valid]
        return values.reshape(output_shape)

def _classify_vertices(vertices, ls, sampler):
    """Return distinct grade counts, four grade labels and four erosion values per face."""
    center = vertices.mean(axis=-2, keepdims=True)
    # Sample G and the centroids of (V0,V1,G), (V1,V2,G) and (V2,V0,G).
    subcenters = (vertices + np.roll(vertices, -1, axis=-2) + center) / 3.0
    sampling_points = np.concatenate((center, subcenters), axis=-2)
    samples = sampler.sample_xy(sampling_points)
    ls = np.asarray(ls, dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        erosion = samples * ls[..., None]
    # For the objective, all samples use their own face's original LS.
    valid = (np.isfinite(samples) & (samples >= 0) & np.isfinite(erosion)
             & (erosion >= 0) & np.isfinite(ls[..., None]) & (ls[..., None] >= 0))
    erosion = np.where(valid, erosion, np.nan)
    classes = np.full(erosion.shape, -1, dtype=np.int8)
    # Values on a class break enter the higher class; zero belongs to grade 1.
    classes[valid] = np.searchsorted(_THRESHOLDS, erosion[valid], side="right") + 1

    # A change between sorted neighboring labels adds one distinct class.
    ordered = np.sort(classes, axis=-1)
    count = 1 + np.count_nonzero(ordered[..., 1:] != ordered[..., :-1], axis=-1)
    # Each face requires four valid samples; -1 marks incomplete coverage.
    count = np.where(valid.all(axis=-1), count, -1).astype(np.int8)
    return count, classes, erosion

def classify_mesh(xy, triangles, ls, sampler):
    """Gather vertices by face and classify the four samples in G, S1, S2, S3 order."""
    vertices = np.asarray(xy, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    return _classify_vertices(vertices, np.asarray(ls, dtype=float), sampler)

def _cross2(a, b):
    """Compute the 2D cross product on the last axis."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def _signed_area2(vertices):
    """Return twice the signed area; a sign change detects a flipped face."""
    return _cross2(vertices[..., 1, :] - vertices[..., 0, :],
                   vertices[..., 2, :] - vertices[..., 0, :])

def _minimum_angles(vertices):
    """Return the smallest interior angle in degrees."""
    first = np.roll(vertices, -1, axis=-2) - vertices
    second = np.roll(vertices, 1, axis=-2) - vertices
    dot = np.sum(first * second, axis=-1)
    angles = np.degrees(np.arctan2(np.abs(_cross2(first, second)), dot))
    return angles.min(axis=-1)

def _topology(triangles, xy0):
    """Build vertex-to-face incidence, identify boundary/nonmanifold nodes and shortest edges."""
    vertex_count = len(xy0)
    flat = triangles.ravel()
    order = np.argsort(flat, kind="stable")
    degrees = np.bincount(flat, minlength=vertex_count)
    # Incident entries for vertex v occupy [pointers[v], pointers[v+1]).
    pointers = np.r_[0, np.cumsum(degrees)]
    face_ids = np.repeat(np.arange(len(triangles)), 3)[order]
    corners = np.tile(np.arange(3), len(triangles))[order]

    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                            triangles[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    edges, uses = np.unique(edges, axis=0, return_counts=True)
    boundary = np.zeros(vertex_count, dtype=bool)
    boundary[edges[uses == 1].ravel()] = True
    nonmanifold = np.zeros(vertex_count, dtype=bool)
    nonmanifold[edges[uses > 2].ravel()] = True
    lengths = np.linalg.norm(xy0[edges[:, 1]] - xy0[edges[:, 0]], axis=1)
    shortest = np.full(vertex_count, np.inf)
    np.minimum.at(shortest, edges[:, 0], lengths)
    np.minimum.at(shortest, edges[:, 1], lengths)
    return pointers, face_ids, corners, degrees, boundary, nonmanifold, shortest
