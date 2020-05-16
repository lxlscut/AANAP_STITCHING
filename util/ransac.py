import numpy as np
import copy

from util.gpu_svd import gpu_svd


def ransac(n_iter, threshold, n, src_point, dst_point):
    """
    :param n: 内点个数
    :param threshold: 阈值，超过该阈值则认为其为外点
    :param n_iter: 循环的次数
    :param src_point: 原图的点
    :param dst_point: 目标图的点，顺序已经对应
    :return:
    """
    origin_src = copy.deepcopy(src_point)
    origin_dst = copy.deepcopy(dst_point)

    final_num = 0
    # todo [x,y] --> [x,y,1]
    padding = np.ones(len(src_point), dtype=np.float32)
    src_point = np.column_stack((src_point, padding))
    # todo 构建全局的大的A矩阵
    A, A1, A2 = construct_A(src_point, dst_point)
    for i in range(n_iter):
        while True:
            # todo 随机挑选出特征点对，每4对特征点进行一次运算
            # todo 这里要求四点不能共线,共线问题以后再处理
            chose_point = np.arange(len(src_point))
            np.random.shuffle(chose_point)
            new_src_array = src_point[chose_point[:4], :]
            new_dst_array = dst_point[chose_point[:4], :]
            # todo 构建相关的A矩阵，来计算最终的单应性矩阵
            H = calculate_homography(new_src_array, new_dst_array)
            result1 = A1.dot(H)
            result2 = A2.dot(H)
            result = result1 * result1 + result2 * result2
            result = np.sqrt(result)
            ok = np.where(result < threshold)
            num_ok = len(ok[0])
            if num_ok > n:
                break
        if final_num < num_ok:
            final_num = num_ok
            final_point = ok[0]
    final_src_point = origin_src[final_point, :]
    final_dst_point = origin_dst[final_point, :]
    return final_point, final_src_point, final_dst_point


def get_ransac_group(n_iter, threshold, n, src_point, dst_point):
    """
    :param n: 内循环次数
    :param threshold: 阈值，超过该阈值则认为其为外点
    :param n_iter: 循环的次数
    :param src_point: 原图的点
    :param dst_point: 目标图的点，顺序已经对应
    :return:
    """
    origin_src = copy.deepcopy(src_point)
    origin_dst = copy.deepcopy(dst_point)

    final_num = 0
    # todo [x,y] --> [x,y,1]
    padding = np.ones(len(src_point), dtype=np.float32)
    src_point = np.column_stack((src_point, padding))
    # todo 构建全局的大的A矩阵
    A, A1, A2 = construct_A(src_point, dst_point)
    for i in range(n_iter):
        while n > 0:
            n = n - 1
            # todo 随机挑选出特征点对，每4对特征点进行一次运算
            # todo 这里要求四点不能共线,共线问题以后再处理
            chose_point = np.arange(len(src_point))
            np.random.shuffle(chose_point)
            new_src_array = src_point[chose_point[:4], :]
            new_dst_array = dst_point[chose_point[:4], :]
            # todo 构建相关的A矩阵，来计算最终的单应性矩阵
            H = calculate_homography(new_src_array, new_dst_array)
            result1 = A1.dot(H)
            result2 = A2.dot(H)
            result = result1 * result1 + result2 * result2
            result = np.sqrt(result)
            ok = np.where(result < threshold)
            num_ok = len(ok[0])
        if final_num < num_ok:
            final_num = num_ok
            final_point = ok[0]
    final_src_point = origin_src[final_point, :]
    final_dst_point = origin_dst[final_point, :]
    return final_point, final_src_point, final_dst_point


def construct_A(src_points, dst_points):
    # todo 构建初始的A矩阵
    A1 = np.zeros([(src_points.shape[0]), 9])
    A2 = np.zeros([(src_points.shape[0]), 9])
    # todo 对A矩阵进行赋值
    for i in range(len(src_points) - 1):
        A1[i, :] = [-src_points[i, 0], -src_points[i, 1], -1, 0, 0, 0, src_points[i, 0] * dst_points[i, 0],
                    src_points[i, 1] * dst_points[i, 0], dst_points[i, 0]]
        A2[i, :] = [0, 0, 0, -src_points[i, 0], -src_points[i, 1], -1, src_points[i, 0] * dst_points[i, 1],
                    dst_points[i, 0] * dst_points[i, 1], dst_points[i, 1]]
    # todo 最终将两部分堆叠在一起即可
    A = np.row_stack([A1, A2])
    # todo 对A矩阵进行svd分解
    return A, A1, A2


def calculate_homography(src_points, dst_points):
    # todo 构建初始的A矩阵
    A, A1, A2 = construct_A(src_points, dst_points)
    # todo 对A矩阵进行svd分解
    v = gpu_svd(A)
    H = v[-1, :]
    H = H / H[-1]
    return H
