import cv2
import random
import numpy as np

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


def draw_match(src, dst, src_point, dst_point):
    src_point = src_point.astype(np.int)
    dst_point = dst_point.astype(np.int)

    pic = np.zeros([max(src.shape[0],dst.shape[0]), src.shape[1]+dst.shape[1], 3], dtype=np.uint8)
    pic[:src.shape[0], :src.shape[1], :] = src[:, :, :]
    pic[:dst.shape[0], src.shape[1]:, :] = dst[:, :, :]
    dst_point[:, 1] = dst_point[:, 1] + src.shape[1]
    for i in range(0, src_point.shape[0], 15):
        cv2.line(pic, (src_point[i, 1], src_point[i, 0]), (dst_point[i, 1], dst_point[i, 0]), (0, 0, 255), 1)
    return pic
