import numpy as np
import math
import copy


def normalize(point):
    """
    :param point: [num,2]
    :return:
    """
    origin_point = copy.deepcopy(point)
    padding = np.ones(point.shape[0], dtype=np.float)
    c = np.mean(point, axis=0)
    point[:, :2] = point[:, :2] - c[:2]
    squre = np.square(point)
    sum = np.sum(squre, axis=1)
    mean = np.mean(np.sqrt(sum))
    scale = math.sqrt(2) / mean
    t = np.array([[scale, 0, -scale * c[0]],
                  [0, scale, -scale * c[1]],
                  [0, 0, 1]], dtype=np.float)
    origin_point = np.column_stack((origin_point, padding))
    new_point = t.dot(origin_point.T)
    return new_point, t
