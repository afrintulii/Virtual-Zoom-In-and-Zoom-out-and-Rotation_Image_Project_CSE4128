import numpy as np
import cv2

background = None
frames_elapsed = 0
FRAME_HEIGHT = 600
FRAME_WIDTH = 600
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18
startDistance = None
startAngle = None
handshake_threshold = 5
distance_buffer = []
angle_buffer = []
BUFFER_SIZE = 10

region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH / 2)
region_right = FRAME_WIDTH
region_width = region_right - region_left

ZOOM_MODE = 1
ROTATION_MODE = 2
current_mode = ZOOM_MODE

class HandData:
    def __init__(self):
        self.top_points = []

    def update(self, top_points):
        self.top_points = top_points

def write_on_image(frame, hand):
    if hand and hand.top_points:
        for i, (x, y) in enumerate(hand.top_points):
            x += region_left
            y += region_top
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for i in range(len(hand.top_points) - 1):
            x1, y1 = hand.top_points[i]
            x2, y2 = hand.top_points[i + 1]
            x1 += region_left
            y1 += region_top
            x2 += region_left
            y2 += region_top
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)

def get_region(frame):
    region = frame[region_top:region_bottom, region_left:region_right]
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region_blur = cv2.GaussianBlur(region_gray, (5, 5), 0)
    return region, region_blur

def get_average(region_blur):
    global background
    if background is None:
        background = region_blur.copy().astype("float")
        return
    cv2.accumulateWeighted(region_blur, background, BG_WEIGHT)

def segment(region, region_blur):
    global hand
    diff = cv2.absdiff(background.astype(np.uint8), region_blur)
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    thresholded_region = cv2.morphologyEx(thresholded_region, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresholded_region = cv2.morphologyEx(thresholded_region, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour

def get_skin_mask(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

    lower_hsv = np.array([0, 48, 80], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    skin_mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    skin_mask = cv2.bitwise_or(skin_mask_ycrcb, skin_mask_hsv)

    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    skin_mask = cv2.medianBlur(skin_mask, 5)

    return skin_mask

def combine_masks(background_mask, skin_mask):
    combined_mask = cv2.bitwise_and(background_mask, skin_mask)
    return combined_mask

def get_hand_data(segmented_image, hand):
    convexHull = cv2.convexHull(segmented_image)
    hand.top_points = []

    finger_tips = {}
    for contour_point in convexHull:
        x, y = contour_point[0]
        finger_index = x // (region_width // 5)
        if finger_index not in finger_tips:
            finger_tips[finger_index] = []
        finger_tips[finger_index].append((x, y))

    for points in finger_tips.values():
        topmost_point = min(points, key=lambda point: point[1])
        hand.top_points.append(topmost_point)

def calculate_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return np.degrees(np.arctan2(dy, dx))

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def main():
    global frames_elapsed, startDistance, startAngle, distance_buffer, angle_buffer, OBJ_THRESHOLD, current_mode
    capture = cv2.VideoCapture(0)
    hand = HandData()

    img = cv2.imread("test.jpg")
    img = cv2.resize(img, (150, 150))
    img_orig = img.copy()
    rotation_angle = 0
    scale_factor = 1.0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)
        
        img_h, img_w = img.shape[:2]
        frame[0:img_h, 0:img_w] = img

        region, region_blur = get_region(frame)
        cv2.imshow("Region", region)
        cv2.imshow("Region Blur", region_blur)

        if frames_elapsed < CALIBRATION_TIME:
            get_average(region_blur)
        else:
            hand_contour = segment(region, region_blur)
            if hand_contour is not None:
                skin_mask = get_skin_mask(region)
                cv2.imshow("Skin Mask", skin_mask)

                background_mask = np.zeros_like(skin_mask)
                cv2.drawContours(background_mask, [hand_contour], -1, 255, thickness=cv2.FILLED)
                cv2.imshow("Background Mask", background_mask)

                combined_mask = combine_masks(background_mask, skin_mask)
                cv2.imshow("Combined Mask", combined_mask)
                
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                cv2.imshow("Combined Mask after Morphology", combined_mask)
                
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    hand_contour = max(contours, key=cv2.contourArea)
                    
                    x, y, w, h = cv2.boundingRect(hand_contour)
                    aspect_ratio = w / float(h)
                    area = cv2.contourArea(hand_contour)
                    rect_area = w * h
                    extent = area / float(rect_area)

                    if 0.2 < aspect_ratio < 1.8 and 0.3 < extent < 0.9:
                        cv2.drawContours(region, [hand_contour], -1, (255, 255, 255))
                        cv2.imshow("Segmented Image", region)
                        get_hand_data(hand_contour, hand)

        write_on_image(frame, hand)

        if len(hand.top_points) >= 2:
            thumb_tip = min(hand.top_points, key=lambda p: p[0])  
            index_tip = min([p for p in hand.top_points if p != thumb_tip], key=lambda p: p[0])  

            d_thumb_index = int(np.sqrt((index_tip[0] - thumb_tip[0]) ** 2 + (index_tip[1] - thumb_tip[1]) ** 2))
            current_angle = calculate_angle(thumb_tip, index_tip)

            if startDistance is None:
                startDistance = d_thumb_index
                distance_buffer = [d_thumb_index] * BUFFER_SIZE
                startAngle = current_angle
                angle_buffer = [current_angle] * BUFFER_SIZE
            else:
                distance_buffer.append(d_thumb_index)
                if len(distance_buffer) > BUFFER_SIZE:
                    distance_buffer.pop(0)

                median_distance = int(np.median(distance_buffer))

                if current_mode == ZOOM_MODE:
                    if abs(median_distance - startDistance) > handshake_threshold:
                        scale = ((median_distance - startDistance) // 2)
                        h1, w1 = img_orig.shape[:2]
                        newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2

                        newH = min(max(1, newH), FRAME_HEIGHT)
                        newW = min(max(1, newW), FRAME_WIDTH)

                        scale_factor = newH / h1
                        img = cv2.resize(img_orig, (newW, newH))
                        cv2.imshow("Zoomed Image", img)

                if current_mode == ROTATION_MODE:
                    angle_buffer.append(current_angle)
                    if len(angle_buffer) > BUFFER_SIZE:
                        angle_buffer.pop(0)

                    median_angle = np.median(angle_buffer)
                    rotation_angle = median_angle - startAngle
                    img = rotate_image(img_orig, rotation_angle)
                    cv2.imshow("Rotated Image", img)

        else:
            startDistance = None
            startAngle = None

        cv2.imshow("Camera Input", frame)
        frames_elapsed += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            break
        elif key == ord('u'):
            OBJ_THRESHOLD += 1
            print(f"Increasing OBJ_THRESHOLD to {OBJ_THRESHOLD}")
        elif key == ord('d'):
            OBJ_THRESHOLD -= 1
            print(f"Decreasing OBJ_THRESHOLD to {OBJ_THRESHOLD}")
        elif key == ord('z'):
            current_mode = ZOOM_MODE
            print("Switched to Zoom Mode")
        elif key == ord('r'):
            current_mode = ROTATION_MODE
            print("Switched to Rotation Mode")

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
