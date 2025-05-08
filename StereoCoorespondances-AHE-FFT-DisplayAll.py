import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_to_smallest(img1, img2):
    """
    Crops the larger of two images to the size of the smaller one.
    Assumes both images are grayscale and 2D.
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    h_crop = min(h1, h2)
    w_crop = min(w1, w2)

    img1_cropped = img1[0:h_crop, 0:w_crop]
    img2_cropped = img2[0:h_crop, 0:w_crop]

    return img1_cropped, img2_cropped

def fft_magnitude(img):
    """ Compute log magnitude of FFT (for visualization) """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)  # +1 to avoid log(0)
    return magnitude

def find_point_correspondences(img_left, img_right, visualize=True):
    # 0. Grayscale
    gray_left  = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE (AHE version)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq_left = clahe.apply(gray_left)
    eq_right= clahe.apply(gray_right)

    # ---- Crop to smallest size BEFORE phaseCorrelate ----
    eq_left, eq_right = crop_to_smallest(eq_left, eq_right)

    # 2. Phase correlation for coarse translation
    shift, response = cv2.phaseCorrelate(
        np.float32(eq_left), 
        np.float32(eq_right)
    )
    dx, dy = shift

    # 3. Warp right image (both color and equalized)
    h, w = eq_right.shape
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    eq_right_warp = cv2.warpAffine(eq_right, M, (w, h))
    img_right_warp = cv2.warpAffine(img_right, M, (img_right.shape[1], img_right.shape[0]))

    # 4. SIFT on eq_left & eq_right_warp
    sift = cv2.SIFT_create()
    kpL, desL = sift.detectAndCompute(eq_left, None)
    kpR, desR = sift.detectAndCompute(eq_right_warp, None)

    # 5. FLANN + Lowe’s ratio test
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), 
        dict(checks=50)
    )
    raw_matches = flann.knnMatch(desL, desR, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 1:
        return np.empty((0, 2)), np.empty((0, 2))  # Not enough matches

    ptsL = np.float32([kpL[m.queryIdx].pt for m in good])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in good])

    # 6. RANSAC to filter outliers
    F, mask = cv2.findFundamentalMat(
    ptsL, ptsR,
    cv2.FM_RANSAC,
    ransacReprojThreshold=0.1,  # Tighter threshold
    confidence=0.999
)
    if mask is None:
        return np.empty((0, 2)), np.empty((0, 2))

    inliers = mask.ravel() == 1
    ptsL_in = ptsL[inliers]
    ptsR_in = ptsR[inliers]

    if visualize and ptsL_in.shape[0] > 0:
        inlier_matches = [good[i] for i in range(len(good)) if inliers[i]]

        # ----- FFT magnitude images -----
        fft_left = fft_magnitude(eq_left)
        fft_right = fft_magnitude(eq_right_warp)

       # plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.title("FFT (CLAHE Left)")
        plt.imshow(fft_left, cmap='gray')
        plt.axis("off")

        #plt.subplot(1, 2, 2)
        plt.title("FFT (CLAHE Right Warp)")
        plt.imshow(fft_right, cmap='gray')
        plt.axis("off")
        plt.show()

        # ----- AHE (CLAHE) images -----
       # plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.title("AHE Left (CLAHE)")
        plt.imshow(eq_left, cmap='gray')
        plt.axis("off")

        #plt.subplot(1, 2, 2)
        plt.title("AHE Right Warp (CLAHE)")
        plt.imshow(eq_right_warp, cmap='gray')
        plt.axis("off")
        plt.show()

        # ----- Display matches on original images -----
        img_matches = cv2.drawMatches(
            img_left, kpL,
            img_right_warp, kpR,
            inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(12, 6))
        plt.title("Coarse‐aligned Inlier Matches (Original Images)")
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return ptsL_in, ptsR_in


if __name__ == "__main__":
    imgL = cv2.imread(r"C:\Users\nmgun\Pictures\Screenshots\throat1.png")
    imgR = cv2.imread(r"C:\Users\nmgun\Pictures\Screenshots\throat2.png")

    left_pts, right_pts = find_point_correspondences(imgL, imgR)
    print(f"Found {len(left_pts)} robust correspondences.")
