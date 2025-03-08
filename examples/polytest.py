import polycheck as poly
from shapely.geometry import Polygon, Point

import numpy as np
import time
import matplotlib.pyplot as plt


# isLeft( ep1, ep2, point ): tests if a point is Left|On|Right of an infinite line.
#
# @param: ep1: end point 1 of the line
# @param: ep2: end point 2 of the line
# @param: point: the point to check
# @return: +ve if point is left of ep1,ep2, 0 if on the line, -ve otherwise
#
def side(ep1, ep2, point):
    diff = ep2 - ep1
    return (diff[:, 0]) * (point[1] - ep1[:, 1]) - (point[0] - ep1[:, 0]) * (diff[:, 1])


# contains( poly, point ): check if a point can be found within a polygon.  Based on an
# implementation found at:
#
#   https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html
#
# @param: poly: an numpy array of polygon points
# @param: point: the point location to check
#
def contains(poly, point):
    wn = 0

    poly = np.array(poly)
    point = np.array(point)

    # loop through all edges of the polygon
    e1 = poly
    e2 = np.roll(poly, 1, axis=0)
    sides = side(e1, e2, point)

    for ep1, ep2, s in zip(e1, e2, sides):
        if ep1[1] <= point[1]:
            if ep2[1] > point[1]:
                if s > 0:
                    wn += 1
        else:
            if ep2[1] <= point[1]:
                if s < 0:
                    wn -= 1

    return wn != 0


poly = [
    [5.0, 5.0],
    [0, 0.5],
    [5.0, -5.0],
    [0.5, -0.5],
    [-5.0, -5.0],
    [0.0, -0.5],
    [-5.0, 5.0],
    [-0.5, 0.5],
]

dots = np.linspace(-8, 8, 200)
xs, ys = np.meshgrid(dots, dots, indexing="xy")
pts = [[x, y] for x, y in zip(xs.flatten(), ys.flatten())]

t0 = time.time()
res = []
sh_poly = Polygon(poly)
for pt in pts:
    pt = Point(pt)
    res.append(sh_poly.contains(pt))
print(f"Shapely total time: {time.time()-t0}")
plt.figure()
plt.imshow(np.array(res).reshape(xs.shape))
plt.title("Shapely")

t0 = time.time()
pc_poly = np.array(poly).astype(np.float64)
pc_pts = np.array(pts).astype(np.float64)
res = poly.contains(pc_poly, pc_pts)
print(f"Polycheck total time: {time.time()-t0}")
plt.figure()
plt.imshow(np.array(res).reshape(xs.shape))
plt.title("Polycheck")

t0 = time.time()
res = []
for pt in pts:
    res.append(contains(poly, pt))
print(f"Local check total time: {time.time()-t0}")
plt.figure()
plt.imshow(np.array(res).reshape(xs.shape))
plt.title("Local check")

#
plt.show(block=True)

print("done")
