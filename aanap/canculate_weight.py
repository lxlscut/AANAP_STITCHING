import numpy as np


def get_weight(center_point, K_min, K_max):
    """
    :param center_point: 所要计算的权重的点
    :param K_min: 对应的k值
    :param K_max:
    :return:µ h (i) = h −−−−→ κ m p(i), −−−−→ κ m κ M i/| −−−−→ κ m κ M |,
    """
    y = center_point[1]
    x = center_point[0]
    # TODO 当小于最小值时，一般认为其值小于零，故取值为0
    if x < K_min[1]:
        w1 = 0
        w2 = 1
    # todo 当大于最大值时，其值显然大于1，故取值为零
    elif x > K_max[1]:
        w1 = 1
        w2 = 0
    # todo 当取值在中间时，取值为中间值
    else:
        a = (x - K_min[0]) * (K_max[0] - K_min[0])
        b = (y - K_min[1]) * (K_max[1] - K_min[1])
        c = (K_max[0] - K_min[0]) ** 2 + (K_max[1] - K_min[1]) ** 2
        w1 = abs(a + b) / c
        w2 = 1 - w1
    return w1, w2


def get_center_line(H, src, dst):
    OR = np.array([dst.shape[1] / 2, dst.shape[0] / 2])
    OT = np.array([src.shape[1] / 2, src.shape[0] / 2, 1])
    OT = np.linalg.inv(H).dot(OT.T)
    OT = OT / OT[2]
    k = (OT[1] - OR[1]) / (OT[0] - OR[0])
    b = OR[1] - k * OR[0]
    return k, b


def get_k(k, b, img, image_info):
    k_minx = min(image_info.left_top[0], image_info.left_button[0], image_info.right_top[0], image_info.right_button[0])
    k_maxx = max(image_info.left_top[0], image_info.left_button[0], image_info.right_top[0], image_info.right_button[0])
    k_miny = k * k_minx + b
    k_maxy = k * k_maxx + b
    k_1x = img.shape[1]
    k_2x = k_1x + (k_maxx - k_1x) / 2
    k_1y = k_1x * k + b
    k_2y = k_2x * k + b

    kmin = np.array([k_minx, k_miny], dtype=np.float)
    kmax = np.array([k_maxx, k_maxy], dtype=np.float)
    k_1 = np.array([k_1x, k_1y], dtype=np.float)
    k_2 = np.array([k_2x, k_2y], dtype=np.float)
    return kmin, kmax, k_1, k_2


def calculate_weight(target_point, feature_point, gama, sigma):
    """
    :param target_point: 所要计算的目标点
    :param feature_point: 特征图检测到的特征点
    :param gama: 超参数，控制最小权重
    :param sigma: 超参数，尺度因子
    :return: 所有距离参数
    """
    feature_point = feature_point[:, [1, 0]]
    value = feature_point - target_point
    value = (value[:, 0]) ** 2 + (value[:, 1]) ** 2
    value = np.sqrt(value)
    w = np.exp(-value / (sigma ** 2))
    w[w < gama] = gama
    w_final = np.repeat(w, 2)
    # # todo 通过w来计算最终的权重矩阵weight
    weight = np.diag(w_final)
    return weight, w
