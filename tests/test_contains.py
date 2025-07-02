import importlib
import pytest

# Skip the test if numpy or pycuda are missing
np = pytest.importorskip("numpy")
pytest.importorskip("pycuda")
import polycheck

# try to use shapely if available, else fallback to local implementation
shapely_spec = importlib.util.find_spec("shapely")
if shapely_spec is not None:
    from shapely.geometry import Polygon, Point

    def reference_contains(polygon, points):
        poly = Polygon(polygon)
        return np.array([poly.contains(Point(p)) for p in points], dtype=bool)
else:

    def _side(ep1, ep2, point):
        diff = ep2 - ep1
        return diff[0] * (point[1] - ep1[1]) - (point[0] - ep1[0]) * diff[1]

    def _local_contains(polygon, point):
        wn = 0
        poly = np.asarray(polygon)
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
    # polycheck returns float32 0/1 values
    gpu_bool = gpu_res.astype(bool)
    assert np.array_equal(gpu_bool, ref_res)
