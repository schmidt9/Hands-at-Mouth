import cv2
import numpy
from scipy.spatial import ConvexHull


def get_hull_points(points):
    hull = ConvexHull(points)
    hull_points = []

    for vertice in hull.vertices:
        point = points[vertice]
        hull_points.append(point)

    return hull_points


def plot_polylines(img, points):
    hull_points_array = numpy.array(points).reshape((-1, 1, 2))
    cv2.polylines(img, [hull_points_array], True, (255, 0, 0), thickness=2)
