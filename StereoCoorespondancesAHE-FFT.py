
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_point_correspondences(img_left, img_right, visualize=True):
    # 0. Grayscale
    gray_left  = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq_left = clahe.apply(gray_left)
    eq_right= clahe.apply(gray_right)

    # 2. Phase correlation for coarse translation
    #    (returns shift to align eq_left → eq_right)
    shift, response = cv2.phaseCorrelate(
        np.float32(eq_left), 
        np.float32(eq_right)
    )
    dx, dy = shift
    # print(f"Coarse shift: dx={dx:.1f}, dy={dy:.1f}, response={response:.3f}")

    # 3. Warp right image (both color and equalized)
    h, w = eq_right.shape
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    eq_right_warp = cv2.warpAffine(eq_right, M, (w, h))
    img_right_warp= cv2.warpAffine(img_right, M, (w, h))

    # 4. SIFT on eq_left & eq_right_warp
    sift = cv2.SIFT_create()
    kpL, desL = sift.detectAndCompute(eq_left,      None)
    kpR, desR = sift.detectAndCompute(eq_right_warp,None)

    # 5. FLANN + Lowe’s ratio test
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), 
        dict(checks=50)
    )
    raw_matches = flann.knnMatch(desL, desR, k=2)

    good = []
    for m,n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return np.empty((0,2)), np.empty((0,2))  # not enough matches

    ptsL = np.float32([kpL[m.queryIdx].pt for m in good])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in good])

    # 6. RANSAC to filter outliers
    F, mask = cv2.findFundamentalMat(
        ptsL, ptsR, 
        cv2.FM_RANSAC, 
        1.0, 0.99
    )
    inliers = mask.ravel() == 1
    ptsL_in = ptsL[inliers]
    ptsR_in = ptsR[inliers]

    if visualize and ptsL_in.shape[0] > 0:
        inlier_matches = [good[i] for i in range(len(good)) if inliers[i]]
        img_matches = cv2.drawMatches(
            img_left,  kpL,
            img_right_warp, kpR,
            inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(12,6))
        plt.title("Coarse‐aligned Inlier Matches")
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return ptsL_in, ptsR_in


if __name__ == "__main__":
    imgL = cv2.imread("/home/nicole/pythonfiles/SPIN/testdata/vessel1.png")
    imgR = cv2.imread("/home/nicole/pythonfiles/SPIN/testdata/vessel2.png")

    left_pts, right_pts = find_point_correspondences(imgL, imgR)
    print(f"Found {len(left_pts)} robust correspondences.")
