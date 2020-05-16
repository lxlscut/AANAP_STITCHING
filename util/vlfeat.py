import cv2
import numpy as np
import cyvlfeat as vl

class Match:
    def __init__(self, src, dst):
        self.src_img = src
        self.dst_img = dst
        self.src_match = None
        self.dst_match = None

    def getInitialFeaturePairs(self):
        """这里加断言"""
        src_point = FeaturePoint(self.src_img)
        dst_point = FeaturePoint(self.dst_img)
        src_point.detect()
        dst_point.detect()

        feature_descriptors_src = src_point.descriptor
        feature_descriptors_dst = dst_point.descriptor

        # todo  对特征进行匹配
        matcher = cv2.DescriptorMatcher.create("BruteForce")
        rawMatcher = matcher.knnMatch(feature_descriptors_src, feature_descriptors_dst, 2)
        match = []
        for (m, n) in rawMatcher:
            if m.distance < n.distance * 0.5:
                match.append((m.trainIdx, m.queryIdx))
        if len(match) > 4:
            psta = np.array([src_point.feature_point[i] for (_, i) in match])
            pstb = np.array([dst_point.feature_point[i] for (i, _) in match])

        self.src_match = psta[:, :2]
        self.src_match = psta[:, [1, 0]]
        self.dst_match = pstb[:, :2]
        self.dst_match = pstb[:, [1, 0]]
        return

class FeaturePoint:
    def __init__(self, image):
        self.image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        self.feature_point = None
        self.descriptor = None

    def detect(self):
        (self.feature_point, self.descriptor) = vl.sift.sift(self.image, compute_descriptor=True)