import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_point_correspondences(img_left, img_right, visualize=True):
    # Convert to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_left = clahe.apply(gray_left)
    eq_right = clahe.apply(gray_right)

    # 1. Detect SIFT features and compute descriptors on equalized images
    sift = cv2.SIFT_create(nfeatures=10000)  # Set a higher number of features
    keypoints_left, descriptors_left = sift.detectAndCompute(eq_left, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(eq_right, None)

    # 2. Match features using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

    # 3. Lowe's ratio test to filter ambiguous matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 4. Extract matched points
    pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches])
    pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches])

    # 5. Filter matches using RANSAC to remove outliers
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, 1.5, 0.99)
    pts_left_inliers = pts_left[mask.ravel() == 1]
    pts_right_inliers = pts_right[mask.ravel() == 1]

    if visualize:
        # draw only inlier matches
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask.ravel()[i]]
        img_matches = cv2.drawMatches(
            img_left, keypoints_left,
            img_right, keypoints_right,
            inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(12, 6))
        plt.title("Inlier Matches After RANSAC")
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return pts_left_inliers, pts_right_inliers

if __name__ == "__main__":
    img_left = cv2.imread("/home/nicole/pythonfiles/SPIN/testdata/vessel2-1.png")
    img_right = cv2.imread("/home/nicole/pythonfiles/SPIN/testdata/vessel2-2.png")

    left_pts, right_pts = find_point_correspondences(img_left, img_right)
    print(f"Found {len(left_pts)} robust correspondences.")
