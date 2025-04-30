import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_radius_boundary(image, border_radius):
    """
    Visualizes a circle with a specific radius (border_radius) on the image.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Draw the circle with the specified radius
    img_with_boundary = image.copy()
    cv2.circle(img_with_boundary, center, border_radius, (0, 255, 0), 2)  # Green circle with thickness 2

    return img_with_boundary

def remove_points_outside_boundary(pts, image_shape, border_radius):
    """
    Removes points that are outside the circular boundary defined by border_radius.
    """
    height, width = image_shape[:2]
    center = np.array([width / 2, height / 2])
    valid_pts = []
    
    for pt in pts:
        distance = np.linalg.norm(pt - center)
        if distance <= border_radius:  # Keep points inside the circular region
            valid_pts.append(pt)
    
    return np.array(valid_pts)

def find_point_correspondences(img_left, img_right, visualize=True, border_radius=0):
    # Convert to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq_left = clahe.apply(gray_left)
    eq_right = clahe.apply(gray_right)

    # 1. Detect SIFT features and compute descriptors on equalized images
    sift = cv2.SIFT_create()  # Set a higher number of features
    keypoints_left_sift, descriptors_left_sift = sift.detectAndCompute(eq_left, None)
    keypoints_right_sift, descriptors_right_sift = sift.detectAndCompute(eq_right, None)

    # 2. Detect ORB features and compute descriptors on equalized images
    orb = cv2.ORB_create()  # Set a higher number of features
    keypoints_left_orb, descriptors_left_orb = orb.detectAndCompute(eq_left, None)
    keypoints_right_orb, descriptors_right_orb = orb.detectAndCompute(eq_right, None)

    # 3. Pad ORB descriptors to match the size of SIFT descriptors (128 dimensions)
    if descriptors_left_orb is not None:
        descriptors_left_orb_padded = np.pad(descriptors_left_orb, ((0, 0), (0, 128 - descriptors_left_orb.shape[1])), 'constant')
    else:
        descriptors_left_orb_padded = np.zeros((0, 128), dtype=np.float32)
        
    if descriptors_right_orb is not None:
        descriptors_right_orb_padded = np.pad(descriptors_right_orb, ((0, 0), (0, 128 - descriptors_right_orb.shape[1])), 'constant')
    else:
        descriptors_right_orb_padded = np.zeros((0, 128), dtype=np.float32)

    # 4. Combine SIFT and padded ORB descriptors
    if descriptors_left_sift is not None and descriptors_left_orb_padded is not None:
        descriptors_left = np.vstack((descriptors_left_sift, descriptors_left_orb_padded))
    else:
        descriptors_left = descriptors_left_sift if descriptors_left_sift is not None else descriptors_left_orb_padded
        
    if descriptors_right_sift is not None and descriptors_right_orb_padded is not None:
        descriptors_right = np.vstack((descriptors_right_sift, descriptors_right_orb_padded))
    else:
        descriptors_right = descriptors_right_sift if descriptors_right_sift is not None else descriptors_right_orb_padded

    # 5. Match features using separate FLANN matchers for SIFT and ORB
    # FLANN for SIFT (floating-point descriptors)
    FLANN_INDEX_KDTREE = 1
    index_params_sift = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params_sift = dict(checks=100)
    flann_sift = cv2.FlannBasedMatcher(index_params_sift, search_params_sift)

    # FLANN for ORB (binary descriptors)
    FLANN_INDEX_LSH = 6
    index_params_orb = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params_orb = dict(checks=50)
    flann_orb = cv2.FlannBasedMatcher(index_params_orb, search_params_orb)

    # Match descriptors for SIFT and ORB separately
    matches_sift = flann_sift.knnMatch(descriptors_left_sift, descriptors_right_sift, k=2)
    matches_orb = flann_orb.knnMatch(descriptors_left_orb_padded, descriptors_right_orb_padded, k=2)

    # 6. Lowe's ratio test to filter ambiguous matches for both SIFT and ORB
    good_matches_sift = []
    for m, n in matches_sift:
        if m.distance < 0.8 * n.distance:  # Lowe's ratio test
            good_matches_sift.append(m)

    good_matches_orb = []
    for m, n in matches_orb:
        if m.distance < 0.8 * n.distance:  # Lowe's ratio test
            good_matches_orb.append(m)

    # 7. Combine good matches from both SIFT and ORB
    good_matches = good_matches_sift + good_matches_orb

    # 8. Extract matched points
    pts_left = np.float32([keypoints_left_sift[m.queryIdx].pt for m in good_matches if m.queryIdx < len(keypoints_left_sift)] +
                         [keypoints_left_orb[m.queryIdx - len(keypoints_left_sift)].pt for m in good_matches if m.queryIdx >= len(keypoints_left_sift)])
    pts_right = np.float32([keypoints_right_sift[m.trainIdx].pt for m in good_matches if m.trainIdx < len(keypoints_right_sift)] +
                          [keypoints_right_orb[m.trainIdx - len(keypoints_right_sift)].pt for m in good_matches if m.trainIdx >= len(keypoints_right_sift)])

    # 9. Remove points outside the boundary in both images
    valid_pts_left = remove_points_outside_boundary(pts_left, img_left.shape, border_radius)
    valid_pts_right = remove_points_outside_boundary(pts_right, img_right.shape, border_radius)

    # 10. Remove the corresponding pairs where either point is outside the boundary
    final_pts_left = []
    final_pts_right = []
    for i in range(len(valid_pts_left)):
        # Check if the pair of points from left and right images are both inside the boundary
        if valid_pts_left[i] in pts_left and valid_pts_right[i] in pts_right:
            final_pts_left.append(valid_pts_left[i])
            final_pts_right.append(valid_pts_right[i])

    # Convert final points to numpy arrays
    final_pts_left = np.array(final_pts_left)
    final_pts_right = np.array(final_pts_right)

    # 11. Filter matches using RANSAC to remove outliers
    F, mask = cv2.findFundamentalMat(final_pts_left, final_pts_right, cv2.FM_RANSAC, 3, 0.1)  # Adjust threshold
    pts_left_inliers = final_pts_left[mask.ravel() == 1]
    pts_right_inliers = final_pts_right[mask.ravel() == 1]

    if visualize:
        # Visualize the circle with border_radius on the images
        img_left_with_radius = visualize_radius_boundary(img_left, border_radius)
        img_right_with_radius = visualize_radius_boundary(img_right, border_radius)

        # Draw only inlier matches
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask.ravel()[i]]
        img_matches = cv2.drawMatches(
            img_left_with_radius, keypoints_left_sift + keypoints_left_orb,
            img_right_with_radius, keypoints_right_sift + keypoints_right_orb,
            inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(12, 6))
        plt.title("Inlier Matches After RANSAC with Radius Boundary")
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return pts_left_inliers, pts_right_inliers

if __name__ == "__main__":
    img_left = cv2.imread("/home/nicole/pythonfiles/SPIN/testdata/vessel2-1masked.png")
    img_right = cv2.imread("/home/nicole/pythonfiles/SPIN/testdata/vessel2-2masked.png")

    left_pts, right_pts = find_point_correspondences(img_left, img_right)
    print(f"Found {len(left_pts)} robust correspondences.")
