"""Trace reverse water-flow paths (RT-WFP) on a piecewise-planar terrain.

All coordinates are in metres. Trace the uphill gradient of each face from
its centroid. Only horizontal distances contribute to slope length.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Trace:
    """Store local XY and original elevations for a path.

    Each segment has a zero-based face row and a kind (1: in-face, 2: ridge).
    """
    points: list
    faces: list
    kinds: list
    status: str

    @property
    def length(self):
        """Sum horizontal segment lengths in metres."""
        points = np.asarray(self.points)
        return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


class Terrain:
    """Index shared XY vertices and edges in the original triangle row order."""
    def __init__(self, xyz, tri_ids=None):
        """Fit a plane to each triangle and compute its unit uphill XY direction."""
        self.xyz = np.asarray(xyz, dtype=float)
        self.n = len(self.xyz)
        self.tri_ids = np.arange(self.n) if tri_ids is None else np.asarray(tri_ids)
        # Translate XY to a local origin for intersections at large projected coordinates.
        self.origin = self.xyz[:, :, :2].min(axis=(0, 1))
        xy = self.xyz[:, :, :2] - self.origin
        self.nodes, inverse = np.unique(xy.reshape(-1, 2), axis=0, return_inverse=True)
        self.faces = inverse.reshape(-1, 3)
        zflat = self.xyz[:, :, 2].ravel()
        zmin = np.full(len(self.nodes), np.inf)
        zmax = np.full(len(self.nodes), -np.inf)
        # A shared vertex must have one elevation within numerical tolerance.
        np.minimum.at(zmin, inverse, zflat)
        np.maximum.at(zmax, inverse, zflat)
        if np.max(zmax - zmin) > 1e-8:
            raise ValueError("Shared XY vertices have inconsistent elevations.")
        self.z = zmin
        self.xy = xy
        self.node_faces = [[] for _ in self.nodes]
        self.edge_faces = {}
        for face, vertices in enumerate(self.faces):
            for node in vertices:
                self.node_faces[node].append(face)
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = tuple(sorted((int(vertices[a]), int(vertices[b]))))
                self.edge_faces.setdefault(key, []).append(face)
        if any(len(faces) > 2 for faces in self.edge_faces.values()):
            raise ValueError("A nonmanifold edge has more than two incident faces.")
        # For edge matrix A, inv(A) maps local XY offsets to barycentric (u, v).
        matrix = np.stack((xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]), axis=2)
        determinant = np.linalg.det(matrix)
        if np.any(np.abs(determinant) < 1e-10):
            raise ValueError("Input contains a degenerate or zero-area triangle.")
        self.area = np.abs(determinant) / 2
        self.inverse = np.linalg.inv(matrix)
        dz = self.xyz[:, 1:, 2] - self.xyz[:, :1, 2]
        self.gradient = np.einsum("ni,nij->nj", dz, self.inverse)
        # The plane gradient is the elevation increment per horizontal metre.
        self.gradient_norm = np.linalg.norm(self.gradient, axis=1)
        self.up = np.zeros((self.n, 2))
        sloped = self.gradient_norm > 1e-14
        self.up[sloped] = self.gradient[sloped] / self.gradient_norm[sloped, None]
        self.slope_deg = np.degrees(np.arctan(self.gradient_norm))
        self.face_zsum = self.xyz[:, :, 2].sum(axis=1)
        # These tolerances handle floating-point edge/vertex hits.
        self.xy_eps = 1e-7
        self.z_eps = 1e-10
        self.node_choice = np.full(len(self.nodes), -2, dtype=np.int32)
        # Node choices: -2 = not visited, -1 = local maximum, otherwise a face row.
        self.node_tail = {}

    def barycentric(self, face, point):
        """Return barycentric weights in face-vertex order for a local XY point."""
        uv = self.inverse[face] @ (point - self.xy[face, 0])
        return np.array([1 - uv.sum(), uv[0], uv[1]])

    def z_at(self, face, point):
        """Interpolate elevation on the current triangle's plane."""
        return float(self.barycentric(face, point) @ self.xyz[face, :, 2])

    def choose_node_face(self, node):
        """Stop at a local maximum or route towards the highest adjacent vertices.

        Prefer a face with two tied highest neighbours. Otherwise use the
        incident face with the highest sum of vertex elevations among those
        touching a highest neighbour. Break ties by Tri_Index, then row.
        """
        if self.node_choice[node] != -2:
            return int(self.node_choice[node])
        incident = self.node_faces[node]
        neighbors = np.unique(self.faces[incident].ravel())
        # The neighbourhood contains the current vertex, so a plateau can terminate here.
        highest_z = float(self.z[neighbors].max())
        if self.z[node] >= highest_z - self.z_eps:
            chosen = -1
        else:
            highest = set(neighbors[self.z[neighbors] >= highest_z - self.z_eps].tolist())
            candidates = [face for face in incident if sum(int(n) in highest for n in self.faces[face]) >= 2]
            if candidates:
                chosen = min(candidates, key=lambda face: (self.tri_ids[face], face))
            else:
                candidates = [face for face in incident if any(int(n) in highest for n in self.faces[face])]
                chosen = min(candidates, key=lambda face: (-self.face_zsum[face], self.tri_ids[face], face))
        self.node_choice[node] = chosen
        return chosen

    def next_event(self, face, point):
        """Intersect the uphill ray with the nearest forward edge or vertex.

        At an outward-facing boundary, follow the current edge to its higher
        endpoint. Return the intersection and segment kind, or a stop reason.
        """
        bary = self.barycentric(face, point)
        # Convert the horizontal tolerance to a face-specific barycentric tolerance.
        bary_eps = max(1e-12, self.xy_eps * np.linalg.norm(self.inverse[face], ord=np.inf) * 2)
        boundary = np.flatnonzero(np.abs(bary) <= bary_eps)
        if bary.min() < -bary_eps * 4:
            return None, "OUTSIDE_FACE"
        if self.gradient_norm[face] <= 1e-14:
            return None, "FLAT"
        uv_derivative = self.inverse[face] @ self.up[face]
        derivative = np.array([-uv_derivative.sum(), *uv_derivative])
        # Barycentric weights vary linearly along the unit uphill direction.
        # The first decreasing weight to reach zero identifies the next edge.
        outward = any(derivative[k] < -1e-13 for k in boundary)
        if not outward:
            eligible = np.flatnonzero(derivative < -1e-13)
            distances = -bary[eligible] / derivative[eligible]
            forward = distances > self.xy_eps
            if np.any(forward):
                distance = float(distances[forward].min())
                hit = point + distance * self.up[face]
                # Snap near-corner hits to the exact vertex to trigger node routing.
                nearby = np.linalg.norm(self.xy[face] - hit, axis=1)
                corner = int(nearby.argmin())
                if nearby[corner] <= self.xy_eps * 4:
                    node = int(self.faces[face, corner])
                    return (self.nodes[node].copy(), node, None, 1), None
                zero = int(eligible[forward][int(np.argmin(distances[forward]))])
                edge = tuple(sorted(int(n) for i, n in enumerate(self.faces[face]) if i != zero))
                return (hit, None, edge, 1), None
        # An outward ray may continue along a ridge edge to its higher endpoint.
        candidates = []
        for zero in boundary:
            nodes = [int(n) for i, n in enumerate(self.faces[face]) if i != zero]
            a, b = nodes
            if abs(self.z[a] - self.z[b]) <= self.z_eps:
                continue
            high = a if self.z[a] > self.z[b] else b
            distance = float(np.linalg.norm(self.nodes[high] - point))
            if distance > self.xy_eps and self.z[high] > self.z_at(face, point) + self.z_eps:
                candidates.append(high)
        if candidates:
            # At a vertex, break elevation ties by the stable node index.
            node = min(set(candidates), key=lambda n: (-self.z[n], n))
            return (self.nodes[node].copy(), node, None, 2), None
        return None, "LEVEL_RIDGE" if len(boundary) else "NO_INTERSECTION"

    def trace(self, initial_face, use_cache=True):
        """Trace from a centroid to a local maximum, boundary, flat, or level ridge.

        Record the last in-domain segment before stopping at a boundary edge.
        A hit on a vertex uses the node routing rule, including boundary nodes.
        Shared vertex tails can be reused because routing is deterministic.
        """
        face = int(initial_face)
        point = self.xy[face].mean(axis=0)
        points = [np.r_[point, self.xyz[face, :, 2].mean()]]
        faces, kinds = [], []
        node = None
        node_starts, visited = {}, set()
        status = "MAX_STEPS"
        for _ in range(max(1000, self.n * 4)):
            if node is not None:
                if node in node_starts:
                    status = "CYCLE"
                    break
                node_starts[node] = len(points) - 1
                if use_cache and node in self.node_tail:
                    tail = self.node_tail[node]
                    points.extend([p.copy() for p in tail.points[1:]])
                    faces.extend(tail.faces)
                    kinds.extend(tail.kinds)
                    status = tail.status
                    break
                face = self.choose_node_face(node)
                if face == -1:
                    status = "LOCAL_MAX"
                    break
            state = (face, round(float(point[0]), 7), round(float(point[1]), 7))
            # Round only the cycle-detection key; coordinates retain full precision.
            if state in visited:
                status = "CYCLE"
                break
            visited.add(state)
            event, stop = self.next_event(face, point)
            if event is None:
                status = "FLAT_START" if stop == "FLAT" and len(points) == 1 else stop
                break
            hit, hit_node, hit_edge, kind = event
            height = self.z[hit_node] if hit_node is not None else self.z_at(face, hit)
            # Uphill segments must not lose elevation beyond round-off.
            if height < points[-1][2] - 1e-7:
                status = "DOWNHILL_ERROR"
                break
            points.append(np.r_[hit, height])
            faces.append(face)
            kinds.append(kind)
            point, node = hit, hit_node
            if hit_edge is not None:
                adjacent = self.edge_faces[hit_edge]
                if len(adjacent) == 1:
                    status = "BOUNDARY"
                    break
                face = adjacent[0] if adjacent[1] == face else adjacent[1]
        result = Trace(points, faces, kinds, status)
        valid_status = {"BOUNDARY", "LOCAL_MAX", "FLAT_START", "FLAT", "LEVEL_RIDGE"}
        if use_cache and status in valid_status:
            for node, start in node_starts.items():
                if node not in self.node_tail:
                    self.node_tail[node] = Trace(points[start:], faces[start:], kinds[start:], status)
        return result


def factors(slope_degrees, length_m):
    """Return the slope-length exponent m and the dimensionless L, S and LS.

    Select intervals in degrees. Convert to radians only for the sine term.
    The reference slope length in L is 22.13 horizontal metres; exactly
    10 degrees belongs to the steepest S branch.
    """
    theta = np.asarray(slope_degrees, dtype=float)
    length = np.asarray(length_m, dtype=float)
    exponent = np.select([theta < 1, theta < 3, theta < 5], [0.2, 0.3, 0.4], default=0.5)
    sine = np.sin(np.radians(theta))
    steepness = np.select([theta < 5, theta < 10], [10.8 * sine + 0.03, 16.8 * sine - 0.5],
                          default=21.9 * sine - 0.96)
    length_factor = np.power(length / 22.13, exponent)
    return exponent, length_factor, steepness, length_factor * steepness
