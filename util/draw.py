import cv2
import random


def draw(src, src_point):
    """
    :param img: 图片
    :param points: 图片中需要上色的点
    :return: 返回已绘制点的图片
    """
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    for i in range(src_point.shape[0]):
        cv2.circle(src, (int(src_point[i, 0]), int(src_point[i, 1])), 5, color=(b, g, r))
    return src
