import time

import numpy as np
import math

from util.gpu_svd import gpu_svd
from util.normalize import normalize

"""验证通过"""


def vgg_get_nonhomg(x):
    """
    :param x: 特征点：[num_points,3]
    :return: 特征点[num_points-1,2]
    """
    col = x.shape[1] - 1
    a = np.ones((col, 1))
    b = np.expand_dims(x[:, -1], 0)
    scale = a.dot(b)
    # print(a.shape)
    # print(b.shape)
    x = x[:, :2] / scale.T
    return x


"""验证通过"""


def vgg_condition_from_pts(src_point):
    src_point = src_point.T
    col = src_point.shape[1]
    pts = vgg_get_nonhomg(src_point)
    pts = pts[:, :col - 1]
    # todo 计算x与y的均值
    mean = np.mean(pts, axis=0)
    # todo 计算x与y的标准偏差
    s = np.std(pts, axis=0, ddof=1)
    s = s + (s == 0)
    # todo 根据均值来计算转换矩阵
    a = np.diag(math.sqrt(2) / s)
    b = np.diag(math.sqrt(2) / s).dot(mean.T)
    T = np.zeros((3, 3), dtype=np.float)
    T[:2, :2] = a[:, :]
    T[:2, 2] = -b
    T[2, :] = 0
    T[2, 2] = 1
    return T


"""验证通过"""
def vgg_condition_2d(src_point, C):
    """
    :param src_point: [num_points,3]
    :param C1: 变换矩阵
    :return: 经过condition后的坐标
    """
    xs = C.dot(src_point)
    return xs


def vgg_H_from_lin(src_point, dst_point):
    """
    :param src_point: 已经经过normalise的原图特征点，【x,y,1】
    :param dst_point: 已经经过normalise的目标图特征点，【x,y,1】
    :return:
    """
    A = np.zeros([2 * src_point.shape[1], 9], dtype=np.float)
    assert (src_point.shape == dst_point.shape), 'src_shape {} is different to dst_shape {}'. \
        format(src_point.shape, dst_point.shape)
    if src_point.shape[1] == 2:
        padding = np.ones(len(src_point), dtype=np.float)
        src_point = np.column_stack((src_point, padding))
        dst_point = np.column_stack((dst_point, padding))
    C1 = vgg_condition_from_pts(src_point)
    C2 = vgg_condition_from_pts(dst_point)
    # TODO 这里的src_point1，dst_point1已经转换为了【3，n】的形状
    src_point1 = vgg_condition_2d(src_point, C1)
    dst_point1 = vgg_condition_2d(dst_point, C2)
    for i in range(src_point.shape[1]):
        p1 = src_point1[:, i]
        p2 = dst_point1[:, i]
        A[2 * i, :] = [p1[0] * p2[2], p1[1] * p2[2], p1[2] * p2[2], 0, 0, 0, -p1[0] * p2[0], -p1[1] * p2[0],
                       -p1[2] * p2[0]]
        A[2 * i + 1, :] = [0, 0, 0, p1[0] * p2[2], p1[1] * p2[2], p1[2] * p2[2], -p1[0] * p2[1], -p1[1] * p2[1],
                           -p1[2] * p2[1]]
    [u, s, v] = np.linalg.svd(A)
    H = v[8, :].reshape(3, 3)
    H = np.linalg.inv(C2).dot(H).dot(C1)
    H = H / H[2, 2]
    return H, A, C1, C2


def homography(src_point, dst_point):
    H, A, C1, C2 = vgg_H_from_lin(src_point, dst_point)
    return H, A, C1, C2


def  get_final_homography(src_point, dst_point):
    (src_normalize, T1) = normalize(src_point)
    (dst_normalize, T2) = normalize(dst_point)
    H1, A1, C1, C2 = homography(src_normalize, dst_normalize)
    H1 = np.linalg.inv(T2).dot(H1.dot(T1))
    return H1, A1, C1, C2, T1, T2


def get_weight_homography(weight, A, C1, C2, T1, T2):
    A = weight.dot(A)
    v = gpu_svd(A)
    H = v[8, :].reshape(3, 3)
    H = np.linalg.inv(C2).dot(H).dot(C1)
    H = np.linalg.inv(T2).dot(H).dot(T1)
    return H



