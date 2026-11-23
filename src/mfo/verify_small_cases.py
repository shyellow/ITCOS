"""Run small, deterministic checks of the four-point count-and-span optimizer."""
import random
import unittest

import numpy as np

from mesh_geometry import RasterSampler, classify_mesh
from mfo_mesh_optimizer import optimize_mesh, run_original_mfo
from objective import build_objective_spec, h_from_scores, scores_from_metrics, spans_from_classes


def square_fan():
    """Make four triangles sharing a movable center in a square with a fixed boundary."""
    xy = np.array([[10., 10.], [2., 2.], [18., 2.], [18., 18.], [2., 18.]])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
    return xy, triangles


class TestOptimizer(unittest.TestCase):
    def test_count_and_span(self):
        # The first two rows have three classes each, but their spans differ.
        classes = [[1, 3, 5, 1], [1, 2, 3, 1], [4, 4, 4, 4], [-1, 2, 3, 5]]
        spec = build_objective_spec()
        span = spans_from_classes(classes)
        score = scores_from_metrics([3, 3, 1, -1], span, spec)
        np.testing.assert_array_equal(span, [4, 2, 0, -1])
        np.testing.assert_array_equal(score, [22, 16, 0, -1])
        np.testing.assert_allclose(h_from_scores(score, spec), [22/30, 16/30, 0, np.nan])

    def test_grade_breaks_and_sampling_order(self):
        xy = np.array([[0., 0.], [9., 0.], [0., 9.]])
        tri = [[0, 1, 2]]
        # The expected (column,row) sequence is G=(3,3), then (4,1), (4,4), (1,4).
        cells = np.zeros((10, 10))
        locations = ([3, 1, 4, 4], [3, 4, 4, 1])
        for values, grades in (([0., 500., 2500., 5000.], [1, 2, 3, 4]),
                               ([-9999., 8000., 15000., 0.], [-1, 5, 6, 1])):
            cells[locations] = values
            sampler = RasterSampler(cells, (1, 0, 0, 0, 1, 0), -9999.)
            count, actual, _ = classify_mesh(xy, tri, [1.], sampler)
            np.testing.assert_array_equal(actual[0], grades)
            self.assertEqual(int(count[0]), -1 if -1 in grades else 4)

    def test_actual_mfo(self):
        # The same seeded search must produce exactly the same trajectory and restore caller RNG state.
        def squared_distance(p):
            return float(p @ p)

        saved_state = random.getstate()
        scalar = run_original_mfo(squared_distance, [-1, -1], [1, 1], seed=17, batch=False)
        repeated = run_original_mfo(squared_distance, [-1, -1], [1, 1], seed=17, batch=False)
        self.assertEqual(random.getstate(), saved_state)
        for left, right in zip(scalar, repeated):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(scalar[2].shape, (100, 1))

    def test_shared_vertex_and_local_scores(self):
        xy, triangles = square_fan()
        bands = np.array([0., 600., 3000., 6000., 10000., 20000.])
        cell_values = np.broadcast_to(bands[np.arange(24) // 4], (24, 24))
        sampler = RasterSampler(cell_values, (1, 0, 0, 0, 1, 0), None)
        result = optimize_mesh(xy, triangles, np.ones(4), sampler, {"seed": 20261019})
        self.assertGreater(len(result["mfo_calls"]), 0)
        self.assertGreater(len(result["movements"]), 0)
        self.assertTrue(all(move["objective_after"] < move["objective_before"]
                            for move in result["movements"]))
        np.testing.assert_array_equal(result["xy"][1:], xy[1:])
        np.testing.assert_array_equal(result["fixed_mask"], [False, True, True, True, True])
        self.assertLess(result["scores_after"].sum(), result["scores_before"].sum())
        # Freeze all corners of a face when its objective includes a NoData pixel.
        masked = np.full((24, 24), np.nan)
        frozen = optimize_mesh(xy, triangles, np.ones(4), RasterSampler(masked, (1,0,0,0,1,0), None), {})
        np.testing.assert_array_equal(frozen["xy"], xy)
        np.testing.assert_array_equal(frozen["scores_after"], [-1, -1, -1, -1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
