import cv2
import numpy as np

def color_correction(frame):
    b, g, r = cv2.split(frame)

    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)

    corrected = cv2.merge((b, g, r))
    return corrected

def color_correction_v2(frame):
    # Convert to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Normalize only the L channel (lightness)
    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    
    # Keep color channels (a, b) unchanged
    corrected_lab = cv2.merge([l, a, b])
    corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    return corrected_bgr

def white_balance_simple(frame):
    """Simple gray world white balance"""
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * 0.9)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * 0.9)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    enhanced_bgr_image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_bgr_image


VIDEO_PATH = "dataset.mp4"   
HORIZONTAL_FOV_DEG = 100   


LOWER_RED_1 = np.array([0, 80, 40])
UPPER_RED_1 = np.array([10, 255, 255])

LOWER_RED_2 = np.array([165, 80, 40])
UPPER_RED_2 = np.array([180, 255, 255])

LOWER_YELLOW = np.array([17, 100, 100])
UPPER_YELLOW = np.array([45, 255, 255])

LOWER_BLUE = np.array([90, 150, 100])
UPPER_BLUE = np.array([105, 255, 255])


capture = cv2.VideoCapture(VIDEO_PATH)
if not capture.isOpened():
    print("Error opening video")
    exit()


while True:
    ret, frame = capture.read() # Frame is taken in BGR by cv2
    if not ret:
        break

    cv2.imshow("Before color correction", frame)

    frame_normalized = color_correction(frame)
    # frame_normalized = white_balance_simple(frame)
    cv2.imshow("After color correction", frame_normalized)

    frame_normalized_blur = cv2.medianBlur(frame_normalized, 3)
    cv2.imshow("After color correction blur", frame_normalized_blur)

    frame_normalized_blur_clahe = apply_clahe(frame_normalized_blur)
    cv2.imshow("After color correction blur clahe", frame_normalized_blur_clahe)
    hsv_image = cv2.cvtColor(frame_normalized_blur_clahe, cv2.COLOR_BGR2HSV)


    # RED MASK
    mask_1 = cv2.inRange(hsv_image, LOWER_RED_1, UPPER_RED_1)
    mask_2 = cv2.inRange(hsv_image, LOWER_RED_2, UPPER_RED_2)
    mask_red = cv2.bitwise_or(mask_1, mask_2)

    kernel_red = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    # kernel_red = np.ones((5, 5), np.uint8)
    # mask_red = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_red)
    for i in range(3):
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_red)
    cv2.imshow("Red Mask", mask_red)


    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_red, y_red, w_red, h_red = 0, 0, 0, 0
    frame_bbox = frame_normalized.copy()
    img_h, img_w = frame_bbox.shape[:2]
    pole_candidates_red = []
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area < 300:
            continue  # noise

        x, y, w, h = cv2.boundingRect(contour)

        # Geometry filters
        aspect_ratio = h / w if w != 0 else 0

        if aspect_ratio < 3:
            continue  # not pole like

        # if y + h < 0.6 * img_h: 
        #     continue  # not touching floor

        # If it passes all conditions, then it might be a pole
        pole_candidates_red.append((contour, x, y, w, h))

    if len(pole_candidates_red) > 0:
        best = max(pole_candidates_red, key=lambda item: item[4])
        contour, x_red, y_red, w_red, h_red = best
        cv2.rectangle(frame_bbox, (x_red, y_red), (x_red + w_red, y_red + h_red), (0, 255, 0), 2)
        cv2.putText(frame_bbox, "RED POLE", (x_red, y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Calculating the yaw angle
        img_center_x = img_w // 2
        red_pole_center_x = x_red + (w_red // 2)
        x_error = red_pole_center_x - img_center_x  
        yaw_angle = (x_error / img_center_x) * (HORIZONTAL_FOV_DEG / 2)
        cv2.line(frame_bbox, (img_center_x, 0), (img_center_x, img_h), (255, 0, 0), 2)
        cv2.circle(frame_bbox, (red_pole_center_x, y_red), 6, (0, 0, 255), -1)
        cv2.putText(frame_bbox, f"Yaw: {yaw_angle:.2f} deg", (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 255), 2)
    cv2.imshow("Detected Red Pole", frame_bbox)





    # YELLOW MASK
    # kernel_yellow = np.ones((3, 3), np.uint8)
    kernel_yellow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    yellow_mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW) 
    yellow_mask = cv2.dilate(yellow_mask, kernel_yellow, iterations=1)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_yellow)
    cv2.imshow("Yellow Mask", yellow_mask)

    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = frame_bbox.shape[:2]
    pole_candidates_yellow = []
    for cnt in contours_yellow:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue  # noise

        x, y, w, h = cv2.boundingRect(cnt)

        # Geometry filters
        aspect_ratio = h / w if w != 0 else 0

        if aspect_ratio < 3:
            continue  # not pole-like

        if y + h < 0.6 * img_h: 
            continue  # not touching floor
        
        # If it passes all conditions, then it might be a pole
        pole_candidates_yellow.append((cnt, x, y, w, h))



    if len(pole_candidates_yellow) > 0:
        best = max(pole_candidates_yellow, key=lambda item: item[4])
        cnt, x, y, w, h = best
        cv2.rectangle(frame_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_bbox, "YELLOW POLE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Detected Yellow Pole", frame_bbox)






















    # key = cv2.waitKey(0) & 0xFF   # wait indefinitely
    # if key == ord('q'):
    #     break

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()












# def gamma_correction(frame, gamma=1.2):
#     inv_gamma = 1.0 / gamma
#     table = np.array([
#         ((i / 255.0) ** inv_gamma) * 255
#         for i in range(256)
#     ]).astype("uint8")

#     return cv2.LUT(frame, table)