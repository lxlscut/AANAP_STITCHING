import cv2
import numpy as np
import copy
from aanap.anchor import get_anchor
from aanap.canculate_weight import calculate_weight, get_center_line, get_k, get_weight
from aanap.homography_linearization import get_linear_homography
from aanap.similar_transform import ransac_group
from util.blending import blending_average
from util.calculate_homography import get_final_homography, get_weight_homography
from util.draw import draw, draw_match
from util.image_info import Image_info
from util.ransac import ransac
from util.vlfeat import Match
from util.warp import warp_global, warp_local_homography_point

if __name__ == '__main__':
    src = cv2.imread("image/DSC00318.JPG")
    dst = cv2.imread("image/DSC00319.JPG")
    # todo 提取特征点
    match = Match(src=src, dst=dst)
    match.getInitialFeaturePairs()
    src_point = match.src_match
    dst_point = match.dst_match

    src_show = draw(src=src, src_point=src_point)
    dst_show = draw(src=dst, src_point=dst_point)


    mmmmmm = draw_match(src, dst, src_point, dst_point)
    # todo 利用ransac来对特征点进行筛选
    final_num, final_src_point, final_dst_point = ransac(src_point=src_point, dst_point=dst_point, n=200, threshold=30,
                                                         n_iter=2000)



    # todo 显示处理后的特征点位置
    dst_point_show = draw(dst, final_dst_point)
    src_point_show = draw(src, final_src_point)
    cv2.imshow("dst_point_show", dst_point_show)
    cv2.imshow("src_point_show", src_point_show)



    print("特征点选择完毕。。。")
    # todo 获取全局矩阵，由于运算后坐标会发生改变，因此将点提前备份
    cal_src, cal_dst = copy.deepcopy(final_src_point), copy.deepcopy(final_dst_point)

    cv2.imshow("mmm",mmmmmm)
    H1, A1, C1, C2, T1, T2 = get_final_homography(cal_src, cal_dst)
    print(H1)
    print("全局单应性矩阵计算完毕。。。")
    # todo 获取投影后最终图片的大小
    image_info = Image_info()
    image_info.get_final_size(src, dst, H1)

    # todo 获取锚点,坐标形式为【x,y】
    anchor = get_anchor(dst, 20)

    # # todo 全局矩阵拼接的效果
    # z = warp_global(src, image_info, H1)
    # z[image_info.offset_y:image_info.offset_y + dst.shape[0], image_info.offset_x:image_info.offset_x + dst.shape[1],
    # :] = dst[:, :, :]
    # z = draw(z, anchor)
    # cv2.imshow("final", z)
    #
    # cv2.waitKey(0)


    # todo 计算全局相似性矩阵
    group, final_similar_matrix, final_theta, small_src, small_dst = ransac_group(final_src_point, final_dst_point)
    print(final_similar_matrix)

    # todo 在目标图上进行方格的划分
    height_num = 100
    width_num = 100
    grid_center_point = np.meshgrid(np.linspace(1, image_info.height, height_num),
                                    np.linspace(1, image_info.width, width_num))
    grid_center_point = np.stack([grid_center_point[0], grid_center_point[1]], axis=-1)
    warp_point = copy.deepcopy(grid_center_point)
    grid_center_point = grid_center_point[:, :, :2] - np.array([image_info.offset_x, image_info.offset_y])
    print("已完成背景图分割。。。")
    # todo grid_center_point[y,x]坐标
    # print(grid_center_point[15, 99])
    # todo 获取得到权重需要的KM,Km等值
    k, b = get_center_line(H1, src, dst)
    kmin, kmax, k_1, k_2 = get_k(k, b, src, image_info)
    print("已获取kmin,kmax,k_1,k_2参数。。。")

    weight_kkk = np.zeros([height_num * width_num, final_dst_point.shape[0]], dtype=np.float)

    # todo 用来装a,b的值
    ab = np.zeros([height_num * width_num, 2], dtype=np.float)

    # todo 计算每个网格中心对应的单应性矩阵
    mmm = np.zeros([height_num, width_num])
    local_H = np.zeros([height_num, width_num, 3, 3])
    local_H_r = np.zeros([height_num, width_num, 3, 3])
    local_H_or = np.zeros([height_num, width_num, 3, 3])
    for i in range(grid_center_point.shape[0]):
        for j in range(grid_center_point.shape[1]):
            weight, w = calculate_weight(grid_center_point[i, j, :], final_dst_point, 0.01, 12.5)
            weight_kkk[i * width_num + j, :] = w[:]
            h = get_weight_homography(weight, A1, C1, C2, T1, T2)
            h_temp = np.linalg.inv(h)
            [a, b] = get_weight(grid_center_point[i, j, :], kmin, kmax)
            # todo 查看a,b的值，在调试时使用
            ab[i * width_num + j, :] = np.array([a, b])
            [c, d] = get_weight(grid_center_point[i, j, :], k_1, k_2)
            # todo 在重叠区域外采用单应性矩阵线性化
            if grid_center_point[i, j, 0] >= src.shape[1] or grid_center_point[i, j, 1] >= src.shape[0] or \
                    grid_center_point[i, j, 0] <= 0 or grid_center_point[i, j, 1] <= 0:
                h_linear = get_linear_homography(h_temp, grid_center_point[i, j, :], anchor)
                h = c * h_linear + d * h_temp
            else:
                h = h_temp
            # # todo 对所有的矩阵全都得做相似性转换
            h_target = b * np.linalg.inv(final_similar_matrix) + a * h
            h_target = h_target / h_target[2, 2]
            su = np.sum(h_target)
            mmm[i, j] = su
            local_H[i, j, :] = h_target
            h_r = np.linalg.inv(h_temp).dot(h_target)
            local_H_r[i, j, :] = h_r / h_r[2, 2]
    print("所有区域矩阵计算完毕。。。")

    # local_H_watch = local_H_or.reshape((-1, 9))

    # todo 测试APAP算法是否可行
    # mmm = np.sum(local_H, axis=-3)
    my_warp1 = warp_local_homography_point(image_info, local_H, src, warp_point)
    result = np.zeros_like(my_warp1)
    result[:dst.shape[0], :dst.shape[1], :] = dst[:, :, :]

    my_warp2 = warp_local_homography_point(image_info, local_H_r, dst, warp_point)

    my_warp = blending_average(my_warp1, my_warp2)

    cv2.imshow("my_warp1", my_warp1)
    cv2.imshow("my_warp2", my_warp2)
    cv2.imshow("my_warp", my_warp)

    cv2.waitKey(0)
