import numpy as np
import cv2
import matplotlib.pyplot as plt

# ─── USER PARAMETERS ────────────────────────────────────────────────
VIDEO_PATH      = "/home/nicole/pythonfiles/getReal/testvideos/normalvessel.mp4"
FPS_TO_EXTRACT  = 1
WINDOW_WIDTH    = 1200
WINDOW_HEIGHT   = 400
DPI             = 100
DISPLAY_PAUSE   = 4
# ──────────────────────────────────────────────────

def extract_frames_from_video(video_path, frames_per_second):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        raise ValueError(f"Cannot read FPS from video: {video_path}")
    frame_interval = int(video_fps / frames_per_second)
    frames = []
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def clean_frame(frame_bgr, spec_vmin=120, spec_smax=100):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    spec_mask = cv2.inRange(hsv, (0, 0, spec_vmin), (179, spec_smax, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    spec_mask = cv2.dilate(spec_mask, kernel, iterations=2)
    inpainted = cv2.inpaint(frame_bgr, spec_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    return gray

def estimate_rigid_transform_logpolar(img1_gray, img2_gray):
    f1 = np.fft.fft2(img1_gray)
    f2 = np.fft.fft2(img2_gray)
    f1_shift = np.fft.fftshift(f1)
    f2_shift = np.fft.fftshift(f2)
    mag1 = np.abs(f1_shift)
    mag2 = np.abs(f2_shift)

    center = (mag1.shape[1]//2, mag1.shape[0]//2)
    logpolar1 = cv2.logPolar(mag1, center, 40, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    logpolar2 = cv2.logPolar(mag2, center, 40, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    (sx, sy), _ = cv2.phaseCorrelate(np.float32(logpolar1), np.float32(logpolar2))
    num_rows, num_cols = logpolar1.shape
    rotation_angle = 360.0 * sx / num_cols
    scale_factor = np.exp(sy / 40)

    M = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0/scale_factor)
    img2_warped = cv2.warpAffine(img2_gray, M, (img2_gray.shape[1], img2_gray.shape[0]))

    (dx, dy), _ = cv2.phaseCorrelate(np.float32(img1_gray), np.float32(img2_warped))

    return rotation_angle, scale_factor, dx, dy

def blend_images(canvas_accum, weight_map, new_img, top_left):
    """
    Blend by accumulating pixel values and weights separately.
    """
    h_img, w_img = new_img.shape
    y0, x0 = top_left
    h_canvas, w_canvas = canvas_accum.shape

    y1 = min(y0 + h_img, h_canvas)
    x1 = min(x0 + w_img, w_canvas)
    new_img_clipped = new_img[0:(y1 - y0), 0:(x1 - x0)]

    roi_canvas = canvas_accum[y0:y1, x0:x1]
    roi_weight = weight_map[y0:y1, x0:x1]

    mask_new = (new_img_clipped > 0).astype(np.float32)

    roi_canvas += new_img_clipped.astype(np.float32) * mask_new
    roi_weight += mask_new

    canvas_accum[y0:y1, x0:x1] = roi_canvas
    weight_map[y0:y1, x0:x1] = roi_weight

    return canvas_accum, weight_map


def stitch_frames_logpolar(frames_gray):
    h, w = frames_gray[0].shape
    origins = [(0, 0)]
    transforms = [(0.0, 1.0, 0.0, 0.0)]

    for idx in range(1, len(frames_gray)):
        img1 = frames_gray[idx-1]
        img2 = frames_gray[idx]
        rotation, scale, dx, dy = estimate_rigid_transform_logpolar(img1, img2)
        transforms.append((rotation, scale, dx, dy))

    xs, ys = [0], [0]
    curr_x, curr_y = 0, 0
    for rot, sca, dx, dy in transforms[1:]:
        curr_x += dx
        curr_y += dy
        xs.append(curr_x)
        ys.append(curr_y)

    min_x, min_y = int(np.floor(min(xs))), int(np.floor(min(ys)))
    max_x, max_y = int(np.ceil(max(xs))) + w, int(np.ceil(max(ys))) + h

    canvas_w = max_x - min_x + w
    canvas_h = max_y - min_y + h
    offset_x, offset_y = -min_x, -min_y

    # ✅ Correct initialization of canvas and weight maps
    canvas_accum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weight_map   = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    curr_origin = np.array([offset_y, offset_x])

    for idx, frame in enumerate(frames_gray):
        if idx == 0:
            canvas_accum, weight_map = blend_images(canvas_accum, weight_map, frame, tuple(curr_origin))
            continue

        rotation, scale, dx, dy = transforms[idx]
        center = (frame.shape[1]//2, frame.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, -rotation, 1.0/scale)
        frame_warped = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        curr_origin = curr_origin + np.array([dy, dx]).astype(int)

        canvas_accum, weight_map = blend_images(canvas_accum, weight_map, frame_warped, tuple(curr_origin))

    #  Normalize properly at the end
    weight_map[weight_map == 0] = 1  # Avoid divide-by-zero
    stitched_final = (canvas_accum / weight_map).clip(0, 255).astype(np.uint8)

    return stitched_final


if __name__ == "__main__":
    frames = extract_frames_from_video(VIDEO_PATH, FPS_TO_EXTRACT)
    frames_gray = [clean_frame(frame) for frame in frames]
    stitched = stitch_frames_logpolar(frames_gray)

    plt.figure(figsize=(WINDOW_WIDTH/DPI, WINDOW_HEIGHT/DPI), dpi=DPI)
    plt.imshow(stitched, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('Log-Polar Phase-Correlation Stitched Panorama')
    plt.tight_layout()
    plt.show()
