import numpy as np
import math

from util.normalize import normalize
from util.ransac import ransac, get_ransac_group


#############################################################################
##todo 找出分块特征点,并求出每个分块特征点的相似性变换，保留其中旋转最小的部分 #####
#############################################################################
def ransac_group(src_point, dst_point):
    """
    :param src_point: 源图点
    :param dst_point: 目标图点
    :return: 相似性矩阵
    """
    (src_normalize, T1) = normalize(src_point)
    (dst_normalize, T2) = normalize(dst_point)
    src_point = src_normalize.T[:, :2]
    dst_point = dst_normalize.T[:, :2]
    group = []
    final_theta = math.pi
    while src_point.shape[0] > 100:
        # print(src_point.shape[0])
        final_point, final_src_point, final_dst_point = ransac(1000, 5, 100, src_point, dst_point)
        # todo 记录该组数据
        group.append(final_point)
        # todo 将该组数据删除
        src_point = np.delete(src_point, final_point, axis=0)
        dst_point = np.delete(dst_point, final_point, axis=0)
        similar_matrix, theta = calculate_similar_matrix(final_src_point, final_dst_point)
        if theta < final_theta:
            final_theta = theta
            final_similar_matrix = similar_matrix
            small_src = final_src_point
            small_dst = final_dst_point
    final_similar_matrix = np.linalg.inv(T2).dot(final_similar_matrix.dot(T1))
    return group, final_similar_matrix, final_theta, small_src, small_dst


def calculate_similar_matrix(src_point, dst_point):
    a = np.zeros([2 * src_point.shape[0], 4], dtype=np.float)
    b = np.zeros([2 * src_point.shape[0], 1], dtype=np.float)
    for i in range(src_point.shape[0]):
        a[2 * i, :] = np.array([src_point[i, 0], -src_point[i, 1], 1, 0], dtype=np.float)
        a[2 * i + 1, :] = np.array([src_point[i, 1], src_point[i, 0], 0, 1], dtype=np.float)
        b[2 * i, :] = np.array(dst_point[i, 0], dtype=np.float)
        b[2 * i + 1, :] = np.array(dst_point[i, 1], dtype=np.float)
    beta = np.linalg.pinv(a).dot(b)
    similar_matrix = np.array([[beta[0, 0], -beta[1, 0], beta[2, 0]], [beta[1, 0], beta[0, 0], beta[3, 0]], [0, 0, 1]])
    # todo 这里求的是弧度
    theta = math.atan(beta[1] / beta[0])
    return similar_matrix, theta
