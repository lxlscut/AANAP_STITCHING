import numpy as np
import cv2


def get_anchor(img, anchor_num):
    """
    :param img: 将要划锚点的图片
    :param anchor_num: 每条边画锚点的个数
    :return: 锚点的坐标,坐标形式为【x,y】
    """
    # todo x方向的个数
    hx = np.linspace(0, img.shape[1], anchor_num)
    # todo y方向的个数
    hy = np.linspace(0, img.shape[0], anchor_num)
    padding0 = np.zeros_like(hx)
    padding1 = np.zeros_like(hy)
    padding2 = np.ones_like(hx) * img.shape[0]
    padding3 = np.ones_like(hy) * img.shape[1]

    # todo 上边界锚点
    up_anchor = np.column_stack([hx, padding0])
    # todo 下边界锚点
    down_anchor = np.column_stack([hx, padding2])
    # todo 左边界锚点
    left_anchor = np.column_stack([padding1, hy])
    # todo 右边界锚点
    right_anchor = np.column_stack([padding3, hy])

    anchor = np.row_stack([left_anchor, right_anchor, up_anchor, down_anchor])

    return anchor
