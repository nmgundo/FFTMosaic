import os
import cv2
import numpy as np
import sys


def get_darkest_near_center(img, x_thresh=100, margin=20):
    """
    Find the darkest pixel within a certain distance from the image center.
    Assumes img is a 2D grayscale or 3-channel image (uses brightness if 3-channel).
    """
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Crop to margin
    roi = gray[margin:h-margin, margin:w-margin]
    yy, xx = np.meshgrid(np.arange(margin, h - margin), np.arange(margin, w - margin), indexing='ij')
    
    # Distance mask
    dist_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    mask = dist_sq <= x_thresh ** 2

    # Apply mask to ROI
    masked_roi = np.where(mask, roi, 255 + 1)  # Set out-of-range to max+1

    # Find min pixel location in masked area
    min_val = np.min(masked_roi)
    min_idx = np.argwhere(masked_roi == min_val)[0]
    y_dark, x_dark = min_idx[0] + margin, min_idx[1] + margin

    return (x_dark, y_dark)


def compute_radii(center, img_shape, r_inner_fixed=30):
    h, w = img_shape[:2]
    x, y = center
    r_outer = int(min(x, w - x, y, h - y))
    r_inner = int(r_inner_fixed)
    return r_inner, r_outer


def unwrap_cylinder(img, center, r_inner, r_outer, out_width=360):
    flags = cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    polar = cv2.warpPolar(img, (out_width, r_outer), center, r_outer, flags)
    unwrapped = polar[r_inner:r_outer, :]
    unwrapped = cv2.rotate(unwrapped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return unwrapped


def annotate_original(img, center, r_inner, r_outer, color=(0, 255, 0), thickness=2):
    annotated = img.copy()
    cv2.circle(annotated, center, r_inner, color, thickness)
    cv2.circle(annotated, center, r_outer, color, thickness)
    return annotated


def restart_script():
    cv2.destroyAllWindows()
    sys.exit()


# --- User-specified input path ---------------------------------------------
IMAGE_PATH = r"/home/nicole/pythonfiles/SPIN/testdata/vessel1.png"
DEPTH_PATH = None  # set to a depth map path if available
RINNER = 10        # inner radius
OUT_WIDTH = 100    # angular resolution
X_THRESH = 30      # maximum distance from true center to search for darkest pixel

while True:
    tmp = cv2.imread(IMAGE_PATH)
    if tmp is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    img = tmp

    # compute center
    if DEPTH_PATH:
        depth = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
        center = get_depth_center(depth)
    else:
        center = tuple(map(int, get_darkest_near_center(img, x_thresh=X_THRESH)))


    # compute radii
    r_inner, r_outer = compute_radii(center, img.shape, RINNER)

    # unwrap
    unwrapped = unwrap_cylinder(img, center, r_inner, r_outer, OUT_WIDTH)

    # annotate original
    annotated = annotate_original(img, center, r_inner, r_outer)

    # show result
    cv2.imshow('Annotated Image', annotated)
    cv2.imshow('Unwrapped Image', unwrapped)
    
    while True:
        key = cv2.waitKey(100)

        # If ESC key is pressed, exit
        if key == 27:
            print("ESC pressed. Exiting.")
            cv2.destroyAllWindows()
            sys.exit()

        # If any other key is pressed, restart
        elif key != -1:
            print("Key pressed. Restarting script.")
            cv2.destroyAllWindows()
            restart_script()

        # If both windows are closed, restart
        if (cv2.getWindowProperty('Annotated Image', cv2.WND_PROP_VISIBLE) < 1 and
            cv2.getWindowProperty('Unwrapped Image', cv2.WND_PROP_VISIBLE) < 1):
            print("restarting")
            restart_script()
