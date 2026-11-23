"""Exact integer scores for four-point erosion class count and span."""
from fractions import Fraction
from math import lcm

import numpy as np

MAX_WEIGHT_DENOMINATOR = 1_000_000
MAX_EXACT_FLOAT_INTEGER = 2 ** 53 - 1


def build_objective_spec(weight_count="0.5", weight_span="0.5"):
    """Convert normalized rational weights into equivalent integer coefficients."""
    weights = []
    for name, value in (("weight_count", weight_count), ("weight_span", weight_span)):
        try:
            fraction = Fraction(str(value))
        except (ValueError, ZeroDivisionError, TypeError) as exc:
            raise ValueError(f"{name} must be a finite positive rational number.") from exc
        if fraction <= 0:
            raise ValueError(f"{name} must be positive.")
        if fraction.denominator > MAX_WEIGHT_DENOMINATOR:
            raise ValueError(f"{name} has a reduced denominator above {MAX_WEIGHT_DENOMINATOR}.")
        weights.append(fraction)
    wn, wd = weights
    if wn + wd != 1:
        raise ValueError("The count and span weights must sum to exactly one.")
    coefficient_count, coefficient_span = wn / 3, wd / 5
    scale = lcm(coefficient_count.denominator, coefficient_span.denominator)
    a, b = int(coefficient_count * scale), int(coefficient_span * scale)
    return dict(name="four_point_count_span", version=1,
                weight_count=str(wn), weight_span=str(wd),
                weight_count_value=float(wn), weight_span_value=float(wd),
                a=a, b=b, scale=scale, max_triangle_score=scale,
                formula_S="a*(K-1)+b*D", formula_H="S/scale",
                sample_order=["G", "S1", "S2", "S3"], sample_count=4,
                class_range=[1, 6], count_range=[1, 4], span_range=[0, 5],
                max_weight_denominator=MAX_WEIGHT_DENOMINATOR,
                max_exact_float_integer=MAX_EXACT_FLOAT_INTEGER)


def validate_objective_spec(spec):
    """Rebuild the score scale from the weights and check all supplied coefficients."""
    if spec is None:
        return build_objective_spec()
    rebuilt = build_objective_spec(spec["weight_count"], spec["weight_span"])
    for key in ("a", "b", "scale", "max_triangle_score", "sample_count"):
        if key in spec and spec[key] != rebuilt[key]:
            raise ValueError(f"Inconsistent objective coefficient: {key}.")
    return rebuilt


def validate_score_capacity(triangle_count, spec):
    """Return an exactly representable penalty larger than any valid sum on these faces."""
    count = int(triangle_count)
    if count < 0 or count != triangle_count:
        raise ValueError("triangle_count must be a nonnegative integer.")
    maximum = (count + 1) * int(spec["scale"])
    if maximum > MAX_EXACT_FLOAT_INTEGER:
        raise ValueError("The maximum penalty exceeds exact float64 integers. Use simpler weights.")
    return maximum


def spans_from_classes(classes):
    """Return the largest minus the smallest class, or -1 if any of the four samples is invalid."""
    values = np.asarray(classes)
    if values.shape[-1:] != (4,):
        raise ValueError("classes must have four entries on its final axis in G, S1, S2, S3 order.")
    valid = (np.isfinite(values) & (values >= 1) & (values <= 6)
             & (values == np.rint(values))).all(axis=-1)
    return np.where(valid, values.max(axis=-1) - values.min(axis=-1), -1).astype(np.int64)


def scores_from_metrics(counts, spans, spec):
    """Score each face; invalid count/span combinations receive -1. A uniform face scores zero."""
    k, d = np.broadcast_arrays(np.asarray(counts), np.asarray(spans))
    valid = (np.isfinite(k) & np.isfinite(d) & (k == np.rint(k)) & (d == np.rint(d))
             & (k >= 1) & (k <= 4) & (d >= 0) & (d <= 5)
             & (d >= k - 1) & ((k != 1) | (d == 0)))
    scores = np.full(k.shape, -1, dtype=np.int64)
    scores[valid] = (int(spec["a"]) * (k[valid].astype(np.int64) - 1)
                     + int(spec["b"]) * d[valid].astype(np.int64))
    return scores


def h_from_scores(scores, spec):
    """Convert integer score S to normalized heterogeneity H, with NaN for invalid scores."""
    values = np.asarray(scores)
    return np.where(values >= 0, values / int(spec["scale"]), np.nan)
