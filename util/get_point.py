import cv2
import numpy as np

def find_match_points_surf(src, dst):
    detector = cv2.xfeatures2d_SURF.create()
    # detector = cv2.ORB_create()
    (ksp1, descriptor1) = detector.detectAndCompute(src, None)
    (ksp2, descriptor2) = detector.detectAndCompute(dst, None)
    ksp1 = np.float32([ksp.pt for ksp in ksp1])
    ksp2 = np.float32([ksp.pt for ksp in ksp2])
    matcher = cv2.DescriptorMatcher.create("BruteForce")
    rawMatcher = matcher.knnMatch(descriptor1, descriptor2, 2)
    match = []
    for (m, n) in rawMatcher:
        if m.distance < n.distance * 0.5:
            match.append((m.trainIdx, m.queryIdx))
    if len(match) > 4:
        psta = np.float32([ksp1[i] for (_, i) in match])
        pstb = np.float32([ksp2[i] for (i, _) in match])
        return psta, pstb
    else:
        print("there are not enough point")
    return


def find_match_points_sift(src, dst):
    detector = cv2.xfeatures2d_SIFT.create()
    (ksp1, descriptor1) = detector.detectAndCompute(src, None)
    (ksp2, descriptor2) = detector.detectAndCompute(dst, None)
    ksp1 = np.float32([ksp.pt for ksp in ksp1])
    ksp2 = np.float32([ksp.pt for ksp in ksp2])
    matcher = cv2.DescriptorMatcher.create("BruteForce")
    rawMatcher = matcher.knnMatch(descriptor1, descriptor2, 2)
    match = []
    for (m, n) in rawMatcher:
        if m.distance < n.distance * 0.7:
            match.append((m.trainIdx, m.queryIdx))
    if len(match) > 4:
        psta = np.float32([ksp1[i] for (_, i) in match])
        pstb = np.float32([ksp2[i] for (i, _) in match])
        return psta, pstb
    else:
        print("there are not enough point")
    return