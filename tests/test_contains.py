import pytest

# Skip the test if numpy or pycuda are missing
np = pytest.importorskip("numpy")
pytest.importorskip("pycuda")
import polycheck


def _side(ep1, ep2, point):
    diff = ep2 - ep1
    return diff[0] * (point[1] - ep1[1]) - (point[0] - ep1[0]) * diff[1]


def _local_contains(polygon, point):
    wn = 0
    poly = np.asarray(polygon, dtype=np.float32)
    point = np.asarray(point, dtype=np.float32)
    for idx in range(len(poly)):
        ep1 = poly[idx]
        ep2 = poly[(idx + 1) % len(poly)]
        if ep1[1] <= point[1]:
            if ep2[1] > point[1] and _side(ep1, ep2, point) > 0:
                wn += 1
        else:
            if ep2[1] <= point[1] and _side(ep1, ep2, point) < 0:
                wn -= 1
    return wn != 0


def _point_to_segment_distance(point, ep1, ep2):
    point = np.asarray(point, dtype=np.float64)
    ep1 = np.asarray(ep1, dtype=np.float64)
    ep2 = np.asarray(ep2, dtype=np.float64)
    segment = ep2 - ep1
    seg_norm_sq = np.dot(segment, segment)
    if seg_norm_sq <= 0.0:
        return np.linalg.norm(point - ep1)
    t = np.dot(point - ep1, segment) / seg_norm_sq
    t = np.clip(t, 0.0, 1.0)
    projection = ep1 + t * segment
    return np.linalg.norm(point - projection)


def _min_distance_to_polygon_edges(point, polygon):
    poly = np.asarray(polygon, dtype=np.float64)
    d_min = np.inf
    for idx in range(len(poly)):
        ep1 = poly[idx]
        ep2 = poly[(idx + 1) % len(poly)]
        d_min = min(d_min, _point_to_segment_distance(point, ep1, ep2))
    return d_min


def reference_contains(polygon, points):
    return np.array([_local_contains(polygon, p) for p in points], dtype=bool)


def build_polygon_and_points():
    polygon = [
        [5.0, 5.0],
        [0.0, 0.5],
        [5.0, -5.0],
        [0.5, -0.5],
        [-5.0, -5.0],
        [0.0, -0.5],
        [-5.0, 5.0],
        [-0.5, 0.5],
    ]

    dots = np.linspace(-8, 8, 200)
    xs, ys = np.meshgrid(dots, dots, indexing="xy")
    points = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return np.array(polygon, dtype=np.float64), points.astype(np.float64)


def test_contains_matches_reference():
    polygon, points = build_polygon_and_points()
    gpu_res = polycheck.contains(polygon, points)
    ref_res = reference_contains(polygon, points)
    gpu_bool = gpu_res.astype(bool)

    if np.array_equal(gpu_bool, ref_res):
        return

    mismatch_idx = np.flatnonzero(gpu_bool != ref_res)
    boundary_tol = 1e-3
    far_mismatch_idx = [
        idx
        for idx in mismatch_idx
        if _min_distance_to_polygon_edges(points[idx], polygon) > boundary_tol
    ]

    assert not far_mismatch_idx, (
        f"Found {len(far_mismatch_idx)} non-boundary mismatches "
        f"(total mismatches={len(mismatch_idx)}, boundary_tol={boundary_tol}). "
        f"Example index={far_mismatch_idx[0]} point={points[far_mismatch_idx[0]]}"
    )
