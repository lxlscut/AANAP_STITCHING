import numpy as np


def tylor_series(H, anchor_points):
    h1 = H[0, 0]
    h2 = H[0, 1]
    h3 = H[0, 2]
    h4 = H[1, 0]
    h5 = H[1, 1]
    h6 = H[1, 2]
    h7 = H[2, 0]
    h8 = H[2, 1]
    h9 = H[2, 2]
    x = anchor_points[0]
    y = anchor_points[1]

    # print(h1,h2,h3,h4,h5,h6,h7,h8,h9)
    # print(x,y)

    dxx = h1 / (h9 + h7 * x + h8 * y) - (h7 * (h3 + h1 * x + h2 * y)) / (h9 + h7 * x + h8 * y) ** 2
    dxy = h2 / (h9 + h7 * x + h8 * y) - (h8 * (h3 + h1 * x + h2 * y)) / (h9 + h7 * x + h8 * y) ** 2
    dyx = h4 / (h9 + h7 * x + h8 * y) - (h7 * (h6 + h4 * x + h5 * y)) / (h9 + h7 * x + h8 * y) ** 2
    dyy = h5 / (h9 + h7 * x + h8 * y) - (h8 * (h6 + h4 * x + h5 * y)) / (h9 + h7 * x + h8 * y) ** 2

    target_x = (h3 + h1 * x + h2 * y) / (h9 + h7 * x + h8 * y)
    target_y = (h6 + h4 * x + h5 * y) / (h9 + h7 * x + h8 * y)
    # print(dxx, dxy, target_x - x * dxx - y * dxy,dyx, dyy, target_y - x * dyx - y * dyy,0, 0, 1)
    A = np.array([[dxx, dxy, target_x - x * dxx - y * dxy],
                  [dyx, dyy, target_y - x * dyx - y * dyy],
                  [0, 0, 1]], dtype=np.float)

    return A


def get_linear_homography(H, point, anchor_points):
    """
    :param H: 单应性矩阵
    :param point: 像素点
    :param anchor_points: 锚点
    :return:
    """
    vega = 5
    # todo alpha来控制每个锚点位置单应性矩阵的权重
    distance = anchor_points - point
    distance = distance[:, 0] ** 2 + distance[:, 1] ** 2
    alpha = np.power((1 + distance / vega), -(vega + 1) / 2)
    alpha = alpha / np.sum(alpha)
    H_output = np.zeros([3, 3], dtype=np.float)
    for i in range(alpha.shape[0]):
        a = tylor_series(H, anchor_points[i])
        H_output = H_output+alpha[i]*a
    return H_output
