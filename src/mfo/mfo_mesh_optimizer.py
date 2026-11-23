"""Optimize shared mesh vertices by moth-flame searches using four-point count and span."""

from __future__ import annotations

import random
import time

import numpy as np
import shapely
from matplotlib.path import Path

import MFO_original as MFO
from mesh_geometry import (RasterSampler, classify_mesh, _classify_vertices,
                           _cross2, _signed_area2, _topology)


from objective import (build_objective_spec, validate_objective_spec,
                       validate_score_capacity, spans_from_classes,
                       scores_from_metrics, h_from_scores)


_AREA_EPSILON = 1e-8  # Threshold on twice the oriented area in square metres.
_SOURCE_CALCULATE_FITNESS = MFO.CalculateFitness


def _batch_calculate_fitness(positions, fun):
    """Evaluate fitness in population order without changing the sequence of random draws."""
    if hasattr(fun, "evaluate_batch"):
        return np.asarray(fun.evaluate_batch(positions), dtype=float).reshape(-1, 1)
    return _SOURCE_CALCULATE_FITNESS(positions, fun)


def run_original_mfo(fun, lb, ub, seed, batch=True):
    """Seed one serial local MFO search and restore the caller's RNG state on exit."""
    previous_calculator = MFO.CalculateFitness
    previous_random_state = random.getstate()
    try:
        random.seed(int(seed))
        MFO.CalculateFitness = _batch_calculate_fitness if batch else _SOURCE_CALCULATE_FITNESS
        return MFO.MFO(50, 2, np.asarray(lb, dtype=float), np.asarray(ub, dtype=float), 100, fun)
    finally:
        MFO.CalculateFitness = previous_calculator
        random.setstate(previous_random_state)


class LocalObjective:
    """Evaluate moves of one shared vertex on all its incident faces, retaining each face's original LS."""

    def __init__(self, vertices, corners, ls, sampler, polygon_xy, area_sign,
                 objective_spec=None):
        self.vertices = np.asarray(vertices, dtype=float).copy()
        self.corners = np.asarray(corners, dtype=np.int64)
        self.ls = np.asarray(ls, dtype=float)
        self.sampler = sampler
        self.area_sign = np.asarray(area_sign, dtype=float)
        self.local_faces = np.arange(len(vertices))
        self.objective_spec = validate_objective_spec(objective_spec)
        # For m incident faces, (m+1)*scale exceeds the worst valid sum m*scale.
        self.penalty = validate_score_capacity(len(vertices), self.objective_spec)
        polygon_xy = np.asarray(polygon_xy, dtype=float)
        self.path = Path(np.vstack((polygon_xy, polygon_xy[0])))
        self.evaluation_count = 0
        self.valid_evaluations = 0

    def _evaluate(self, positions):
        """Return joint scores and per-face metrics; reject the entire move if any face is invalid."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        scores = np.full(len(positions), float(self.penalty))
        shape = (len(positions), len(self.vertices))
        all_counts = np.full(shape, -1, dtype=np.int8)
        all_spans = np.full(shape, -1, dtype=np.int64)
        all_face_scores = np.full(shape, -1, dtype=np.int64)
        in_polygon = np.isfinite(positions).all(axis=1) & self.path.contains_points(positions)
        indices = np.flatnonzero(in_polygon)
        if not len(indices):
            return scores, all_counts, all_spans, all_face_scores
        moved = np.broadcast_to(self.vertices, (len(indices),) + self.vertices.shape).copy()
        # Replace the shared corner in every incident face of this proposal.
        moved[:, self.local_faces, self.corners, :] = positions[indices, None, :]
        oriented = _signed_area2(moved) * self.area_sign
        geometric_valid = (oriented > _AREA_EPSILON).all(axis=1)
        valid_indices = indices[geometric_valid]
        if len(valid_indices):
            counts, classes, _ = _classify_vertices(moved[geometric_valid], self.ls, self.sampler)
            spans = spans_from_classes(classes)
            face_scores = scores_from_metrics(counts, spans, self.objective_spec)
            all_counts[valid_indices] = counts
            all_spans[valid_indices] = spans
            all_face_scores[valid_indices] = face_scores
            complete = (face_scores >= 0).all(axis=1)
            scores[valid_indices[complete]] = face_scores[complete].sum(axis=1)
        return scores, all_counts, all_spans, all_face_scores

    def evaluate_batch(self, positions):
        """Batch interface used by the search; evaluation counts include invalid proposals."""
        scores, _, _, _ = self._evaluate(positions)
        self.evaluation_count += len(scores)
        self.valid_evaluations += int(np.count_nonzero(scores < self.penalty))
        return scores

    def __call__(self, position):
        """Evaluate one candidate with the same arithmetic as the population interface."""
        return float(self.evaluate_batch(np.asarray(position).reshape(1, 2))[0])

    def inspect_metrics(self, position):
        """Recompute the total and per-face metrics without advancing evaluation counters."""
        scores, counts, spans, face_scores = self._evaluate(np.asarray(position).reshape(1, 2))
        return int(scores[0]), counts[0], spans[0], face_scores[0]

    def inspect(self, position):
        """Return a candidate's summed integer score and per-face class counts."""
        score, counts, _, _ = self.inspect_metrics(position)
        return float(score), counts



def _convex_neighborhood(vertex_id, neighbor_ids, xy):
    """Order neighbors around the vertex and require a simple convex polygon strictly enclosing it."""
    neighbors = xy[neighbor_ids]
    if len(neighbors) < 3 or len(np.unique(neighbors, axis=0)) != len(neighbors):
        return None
    relative = neighbors - xy[vertex_id]
    order = np.argsort(np.arctan2(relative[:, 1], relative[:, 0]), kind="stable")
    neighbors = neighbors[order]
    edges = np.roll(neighbors, -1, axis=0) - neighbors
    tolerance = 1e-10 * max(1.0, float(np.max(np.sum(edges ** 2, axis=1))))
    # The tolerance allows nearly collinear turns at the scale of the one-ring.
    turns = _cross2(edges, np.roll(edges, -1, axis=0))
    if not ((turns >= -tolerance).all() and (turns > tolerance).any()):
        return None
    if not (_cross2(edges, xy[vertex_id] - neighbors) > tolerance).all():
        return None
    polygon = shapely.Polygon(neighbors)
    if not polygon.is_valid or polygon.area <= tolerance:
        return None
    return neighbors


def _select_vertex(face_id, xy, triangles, fixed, pointers, face_ids, corner_ids,
                   neighbor_cache):
    """Choose the eligible corner with the fewest neighbors, breaking ties in face vertex order."""
    selected = None
    minimum_neighbors = np.inf
    for value in triangles[face_id]:
        vertex_id = int(value)
        if fixed[vertex_id]:
            continue
        adjacency = slice(pointers[vertex_id], pointers[vertex_id + 1])
        faces, corners = face_ids[adjacency], corner_ids[adjacency]
        if vertex_id not in neighbor_cache:
            neighbors = np.unique(triangles[faces])
            neighbor_cache[vertex_id] = neighbors[neighbors != vertex_id]
        neighbor_ids = neighbor_cache[vertex_id]
        if len(neighbor_ids) >= minimum_neighbors:
            continue
        polygon_xy = _convex_neighborhood(vertex_id, neighbor_ids, xy)
        if polygon_xy is not None:
            minimum_neighbors = len(neighbor_ids)
            selected = vertex_id, faces, corners, polygon_xy
    return selected


def optimize_mesh(xy0, triangles, ls, sampler, config, callback=None):
    """Visit initially mixed faces in order. Accept only strict decreases in their incident score sum."""
    started = time.perf_counter()
    original = np.asarray(xy0, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    ls = np.asarray(ls, dtype=float)
    if original.ndim != 2 or original.shape[1] != 2 or not np.isfinite(original).all():
        raise ValueError("xy0 must be a finite (vertex_count, 2) array.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or ls.shape != (len(triangles),):
        raise ValueError("Expected triangles shaped (n, 3) and one LS value per face.")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(original)):
        raise ValueError("Triangle vertex index out of bounds.")
    if np.any((triangles[:, 0] == triangles[:, 1])
              | (triangles[:, 1] == triangles[:, 2])
              | (triangles[:, 2] == triangles[:, 0])):
        raise ValueError("A triangle contains a repeated vertex index.")
    supplied = config or {}
    settings = dict(max_sweeps=1, pop=50, max_iter=100, seed=20261019,
                    batch_fitness=True, convergence_patience=2,
                    weight_count="0.5", weight_span="0.5")
    settings.update(supplied)
    forbidden = {"max_move_ratio", "min_area_ratio", "max_area_ratio", "min_angle_deg"}
    if forbidden.intersection(supplied):
        raise ValueError("Unexpected movement, area-ratio or minimum-angle constraint.")
    if int(settings["pop"]) != 50 or int(settings["max_iter"]) != 100:
        raise ValueError("This MFO uses population 50, dimension 2 and 100 iterations.")
    if supplied.get("objective_spec") is not None:
        spec = validate_objective_spec(supplied["objective_spec"])
        # Check the weights against the specification if the caller supplies both.
        if "weight_count" in supplied or "weight_span" in supplied:
            explicit = build_objective_spec(supplied.get("weight_count", spec["weight_count"]),
                                            supplied.get("weight_span", spec["weight_span"]))
            if any(explicit[key] != spec[key] for key in ("a", "b", "scale")):
                raise ValueError("Config weights disagree with the objective specification.")
    else:
        spec = build_objective_spec(settings["weight_count"], settings["weight_span"])
    validate_score_capacity(len(triangles), spec)
    settings.update(objective_spec=spec, weight_count=spec["weight_count"],
                    weight_span=spec["weight_span"])
    max_sweeps = int(settings["max_sweeps"])
    patience = int(settings["convergence_patience"])
    base_seed = int(settings["seed"])
    if max_sweeps < 0 or patience < 1:
        raise ValueError("Sweep count must be nonnegative and convergence patience must be positive.")

    def totals(k, d, s):
        valid = k >= 1
        score = int(s[valid].sum())
        return dict(K=int(k[valid].sum()), D=int(d[valid].sum()),
                    S=score, H=score / spec["scale"])

    def local_fields(before, after):
        return {f"local_{key}_{when}": value[key] if value is not None else None
                for when, value in (("before", before), ("after", after))
                for key in ("K", "D", "S", "H")}

    def progress(event, **values):
        if callback is not None:
            payload = dict(event=event, objective_metric="integer_S")
            payload.update(values)
            callback(payload)

    xy = original.copy()
    counts_before, initial_classes, _ = classify_mesh(xy, triangles, ls, sampler)
    spans_before = spans_from_classes(initial_classes)
    scores_before = scores_from_metrics(counts_before, spans_before, spec)
    counts, spans, scores = counts_before.copy(), spans_before.copy(), scores_before.copy()
    if not np.array_equal(counts >= 1, scores >= 0):
        raise RuntimeError("Inconsistent initial validity of count, span and score.")
    (pointers, face_ids, corner_ids, degrees, boundary, nonmanifold,
     shortest) = _topology(triangles, original)
    signed_initial = _signed_area2(original[triangles])
    area_sign = np.sign(signed_initial)
    fixed = boundary | nonmanifold | (degrees < 3) | ~np.isfinite(shortest) | (shortest <= 0)
    invalid_faces = (counts_before < 0) | (np.abs(signed_initial) <= _AREA_EPSILON)
    fixed[triangles[invalid_faces].ravel()] = True
    neighbor_cache = {}
    movements, pass_stats, call_logs = [], [], []
    stale_sweeps = 0
    initial_totals = totals(counts, spans, scores)
    progress("start", algorithm="original_MFO_four_point_span", triangles=len(triangles),
             vertices=len(xy), mixed=int(np.count_nonzero(counts > 1)),
             invalid=int(np.count_nonzero(counts < 0)), fixed_vertices=int(fixed.sum()),
             seed=base_seed, pop=50, max_iter=100, objective_spec=spec, **initial_totals)

    for sweep in range(1, max_sweeps + 1):
        sweep_started = time.perf_counter()
        # Freeze the queue at sweep start; skip any trigger resolved by an earlier move.
        anomalies = np.flatnonzero(counts > 1)
        if not len(anomalies):
            break
        before_mixed = len(anomalies)
        before_totals = totals(counts, spans, scores)
        accepted = calls = no_eligible_vertex = 0
        progress("sweep_start", sweep=sweep, mixed=before_mixed, **before_totals)
        for processed, face_id in enumerate(anomalies, start=1):
            if counts[face_id] > 1:
                selected = _select_vertex(int(face_id), xy, triangles, fixed, pointers,
                                          face_ids, corner_ids, neighbor_cache)
                if selected is None:
                    no_eligible_vertex += 1
                else:
                    vertex_id, faces, corners, polygon_xy = selected
                    objective = LocalObjective(xy[triangles[faces]], corners, ls[faces],
                                               sampler, polygon_xy, area_sign[faces], spec)
                    old_counts, old_spans, old_scores = (counts[faces].copy(),
                                                        spans[faces].copy(), scores[faces].copy())
                    old = totals(old_counts, old_spans, old_scores)
                    # Use the same four-point score to check the old position before evaluating a move.
                    original_score, original_k, original_d, original_s = objective.inspect_metrics(xy[vertex_id])
                    if (original_score != old["S"] or not np.array_equal(original_k, old_counts)
                            or not np.array_equal(original_d, old_spans)
                            or not np.array_equal(original_s, old_scores)):
                        raise RuntimeError("Cached scores disagree with resampling the current vertex.")
                    call_index = len(call_logs)
                    call_seed = (base_seed + 104729 * call_index) % (2 ** 32)
                    call_started = time.perf_counter()
                    best_fitness, best_position, curve = run_original_mfo(
                        objective, polygon_xy.min(axis=0), polygon_xy.max(axis=0),
                        call_seed, batch=bool(settings["batch_fitness"]))
                    calls += 1
                    position = np.asarray(best_position, dtype=float).reshape(2)
                    returned_score = float(np.asarray(best_fitness).reshape(-1)[0])
                    score, new_counts, new_spans, new_scores = objective.inspect_metrics(position)
                    if score != returned_score:
                        raise RuntimeError("The MFO score does not match its returned position.")
                    candidate_valid = score < objective.penalty and (new_scores >= 0).all()
                    new = totals(new_counts, new_spans, new_scores) if candidate_valid else None
                    improvement = bool(candidate_valid and new["S"] < old["S"])
                    entry = dict(
                        call_index=call_index, sweep=sweep, trigger_face=int(face_id),
                        vertex_id=vertex_id, seed=int(call_seed), pop=50, dim=2, max_iter=100,
                        lb=polygon_xy.min(axis=0).tolist(), ub=polygon_xy.max(axis=0).tolist(),
                        old_sum=old["S"], best_fitness=score, fitness_metric="integer_S",
                        original_position_fitness=original_score, penalty=objective.penalty,
                        candidate_valid=bool(candidate_valid), accepted=improvement,
                        position=position.tolist(), fitness_curve=np.asarray(curve).reshape(-1).tolist(),
                        evaluations=objective.evaluation_count,
                        valid_evaluations=objective.valid_evaluations,
                        seconds=round(time.perf_counter() - call_started, 6),
                        **local_fields(old, new))
                    call_logs.append(entry)
                    if improvement:
                        # Accept a smaller joint sum even when a component or an individual face worsens.
                        if (new_counts < 1).any() or (new_spans < 0).any() or new["S"] >= old["S"]:
                            raise RuntimeError("Invalid candidate or local integer score failed to decrease.")
                        movements.append(dict(
                            vertex_id=vertex_id, before_xy=xy[vertex_id].tolist(),
                            after_xy=position.tolist(), affected_faces=faces.tolist(),
                            old_counts=old_counts.tolist(), new_counts=new_counts.tolist(),
                            old_spans=old_spans.tolist(), new_spans=new_spans.tolist(),
                            old_scores=old_scores.tolist(), new_scores=new_scores.tolist(),
                            mfo_call_index=call_index, seed=int(call_seed), sweep=sweep,
                            objective_before=old["S"], objective_after=new["S"],
                            objective_metric="integer_S", **local_fields(old, new)))
                        xy[vertex_id] = position
                        counts[faces], spans[faces], scores[faces] = new_counts, new_spans, new_scores
                        accepted += 1
            if processed % 200 == 0:
                progress("progress", sweep=sweep, processed=processed, total=len(anomalies),
                         accepted=accepted, mfo_calls=calls,
                         mixed=int(np.count_nonzero(counts > 1)), **totals(counts, spans, scores))
        after_totals = totals(counts, spans, scores)
        statistic = dict(
            sweep=sweep, processed=len(anomalies), mfo_calls=calls,
            accepted_moves=accepted, no_eligible_vertex=no_eligible_vertex,
            mixed_before=before_mixed, mixed_after=int(np.count_nonzero(counts > 1)),
            objective_before=before_totals["S"], objective_after=after_totals["S"],
            objective_metric="integer_S",
            seconds=round(time.perf_counter() - sweep_started, 3),
            **{f"{key}_{when}": value[key]
               for when, value in (("before", before_totals), ("after", after_totals))
               for key in ("K", "D", "S", "H")})
        pass_stats.append(statistic)
        progress("sweep_end", **statistic)
        stale_sweeps = stale_sweeps + 1 if accepted == 0 else 0
        if stale_sweeps >= patience:
            break

    counts_after, final_classes, _ = classify_mesh(xy, triangles, ls, sampler)
    spans_after = spans_from_classes(final_classes)
    scores_after = scores_from_metrics(counts_after, spans_after, spec)
    if not (np.array_equal(counts_after, counts) and np.array_equal(spans_after, spans)
            and np.array_equal(scores_after, scores)):
        raise RuntimeError("Final four-point samples disagree with the cached count, span or score.")
    if not np.array_equal(xy[fixed], original[fixed]):
        raise RuntimeError("A boundary vertex or a vertex of an invalid face moved.")
    valid = counts_before >= 1
    if ((counts_after[valid] < 1).any() or (spans_after[valid] < 0).any()
            or (scores_after[valid] < 0).any()
            or not np.all(counts_after[~valid] == -1)
            or not np.all(spans_after[~valid] == -1)
            or not np.all(scores_after[~valid] == -1)):
        raise RuntimeError("The set of valid objective faces changed during optimization.")
    if scores_after[valid].sum() > scores_before[valid].sum():
        raise RuntimeError("The global integer objective increased.")
    progress("complete", mixed=int(np.count_nonzero(counts_after > 1)),
             accepted_moves=len(movements), mfo_calls=len(call_logs),
             seconds=round(time.perf_counter() - started, 3),
             **totals(counts_after, spans_after, scores_after))
    return dict(xy=xy, counts_before=counts_before, counts_after=counts_after,
                spans_before=spans_before, spans_after=spans_after,
                scores_before=scores_before, scores_after=scores_after,
                objective_spec=spec, movements=movements, pass_stats=pass_stats,
                boundary_mask=boundary, fixed_mask=fixed, config=settings, mfo_calls=call_logs)

