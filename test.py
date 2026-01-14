import cv2
import numpy as np

# ==========================================
# --- 1. ROBUST UNDERWATER ENHANCEMENT ---
# ==========================================

def apply_enhanced_processing(frame):
    """
    1. Gray World: Fixes blue color cast.
    2. Gamma: Brightens dark areas (poles).
    3. CLAHE (L-channel): Fixes 'breaks' by boosting local contrast.
    """
    
    # --- Step 1: Robust Gray World (Fixes Blue Cast) ---
    # We balance the B and R channels based on the Green channel (which is usually best underwater)
    b, g, r = cv2.split(frame)
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    
    # Avoid division by zero
    if r_mean == 0: r_mean = 1
    if b_mean == 0: b_mean = 1
    
    # Scale Blue and Red to match Green's brightness
    # (We don't simply stretch to 255, which causes bleeding)
    k_r = g_mean / r_mean
    k_b = g_mean / b_mean
    
    r_balanced = cv2.multiply(r, k_r)
    b_balanced = cv2.multiply(b, k_b)
    
    # Clip to valid range
    r_balanced = np.clip(r_balanced, 0, 255).astype(np.uint8)
    b_balanced = np.clip(b_balanced, 0, 255).astype(np.uint8)
    
    balanced_frame = cv2.merge([b_balanced, g, r_balanced])

    # --- Step 2: Gamma Correction (Fixes Dark Breaks) ---
    # Slight brightness boost to see the full length of the pole
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_frame = cv2.LUT(balanced_frame, table)

    # --- Step 3: CLAHE on Luminance ONLY (Fixes Breaks without Bleeding) ---
    # Converting to LAB allows us to boost DETAILS (L) without messing up COLORS (A, B)
    lab = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to Lightness channel only
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    final_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return final_frame


# ==========================================
# --- 2. DETECTION HELPERS (Strict & Nuclear) ---
# ==========================================

def get_red_mask_strict(frame):
    """ Strict Red: Checks if Red > Green AND Red > Blue to kill floor noise. """
    b, g, r = cv2.split(frame)
    
    # 1. Red must be dominant over Green (Kills Sand/Floor)
    diff_r_g = cv2.subtract(r, g)
    
    # 2. Red must be dominant over Blue (Kills Water/Glare)
    diff_r_b = cv2.subtract(r, b)
    
    # Intersection
    mask_strict = cv2.bitwise_and(diff_r_g, diff_r_b)
    
    # Threshold
    _, mask = cv2.threshold(mask_strict, 45, 255, cv2.THRESH_BINARY)
    
    # Clean up
    mask = cv2.medianBlur(mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Removes tiny noise
    
    return mask

def merge_broken_contours(contours, max_x_dist=50, max_y_gap=150):
    """ Smart Merger: Connects broken pole segments vertically. """
    if not contours: return []
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 30: continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, w, h])

    if not boxes: return []
    boxes.sort(key=lambda b: b[1]) # Sort Top-to-Bottom

    merged_boxes = []
    current_box = boxes[0]

    for i in range(1, len(boxes)):
        next_box = boxes[i]
        curr_x, curr_y, curr_w, curr_h = current_box
        next_x, next_y, next_w, next_h = next_box
        
        curr_center_x = curr_x + curr_w // 2
        next_center_x = next_x + next_w // 2
        
        # Logic: Are they aligned vertically?
        x_aligned = abs(curr_center_x - next_center_x) < max_x_dist
        gap = next_y - (curr_y + curr_h)
        y_close = gap < max_y_gap
        
        if x_aligned and y_close:
            # Merge
            min_x = min(curr_x, next_x)
            min_y = min(curr_y, next_y)
            max_x = max(curr_x + curr_w, next_x + next_w)
            max_y = max(curr_y + curr_h, next_y + next_h)
            current_box = [min_x, min_y, max_x - min_x, max_y - min_y]
        else:
            merged_boxes.append(current_box)
            current_box = next_box

    merged_boxes.append(current_box)
    return merged_boxes


# ==========================================
# --- 3. MAIN LOOP ---
# ==========================================

VIDEO_PATH = "dataset.mp4"   
HORIZONTAL_FOV_DEG = 100 

# HSV Ranges (Only for Yellow)
LOWER_YELLOW = np.array([15, 80, 80])  # Slightly wider range
UPPER_YELLOW = np.array([45, 255, 255])

capture = cv2.VideoCapture(VIDEO_PATH)
if not capture.isOpened():
    print("Error in opening the video")
    exit()

while True:
    ret, frame = capture.read()
    if not ret: break

    # Show Original
    cv2.imshow("Original", frame)

    # --- 1. APPLY NEW ENHANCEMENT ---
    frame_enhanced = apply_enhanced_processing(frame)
    cv2.imshow("Enhanced", frame_enhanced)
    
    # Blur slightly for masking
    frame_blur = cv2.medianBlur(frame_enhanced, 3)
    hsv_frame = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    # Output Frame
    frame_bbox = frame_enhanced.copy()
    img_h, img_w = frame_bbox.shape[:2]
    cv2.line(frame_bbox, (img_w // 2, 0), (img_w // 2, img_h), (255, 0, 0), 1)

    # -----------------------------------------------------
    # RED POLE (Strict Mask + Merger)
    # -----------------------------------------------------
    mask_red = get_red_mask_strict(frame_blur)
    cv2.imshow("Red Mask", mask_red)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_red = merge_broken_contours(contours_red)
    
    best_red_box = None
    max_h_red = 0

    for box in merged_red:
        x, y, w, h = box
        if w * h < 300: continue
        if h / w < 3: continue 
        
        if h > max_h_red:
            max_h_red = h
            best_red_box = box

    if best_red_box:
        x, y, w, h = best_red_box
        cv2.rectangle(frame_bbox, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame_bbox, "RED POLE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        yaw = ((x + w//2 - img_w//2) / (img_w//2)) * (HORIZONTAL_FOV_DEG / 2)
        cv2.putText(frame_bbox, f"Yaw: {yaw:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # -----------------------------------------------------
    # YELLOW POLE (Nuclear Combine-All)
    # -----------------------------------------------------
    mask_yellow = cv2.inRange(hsv_frame, LOWER_YELLOW, UPPER_YELLOW)
    
    # Tiny morph to clean up but keep dots
    kernel_yellow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_yellow)
    cv2.imshow("Yellow Mask", mask_yellow)

    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gather ALL yellow points
    valid_yellow_contours = [c for c in contours_yellow if cv2.contourArea(c) > 10]

    if valid_yellow_contours:
        all_points = np.vstack(valid_yellow_contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Check aspect ratio to ensure it's a pole-like shape
        if h > 50 and (h/w > 1.5):
            cv2.rectangle(frame_bbox, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame_bbox, "YELLOW POLE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            yaw = ((x + w//2 - img_w//2) / (img_w//2)) * (HORIZONTAL_FOV_DEG / 2)
            cv2.putText(frame_bbox, f"Yaw: {yaw:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # -----------------------------------------------------
    
    cv2.imshow("Final Result", frame_bbox)

    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()