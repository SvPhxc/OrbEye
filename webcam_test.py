import cv2
import numpy as np

# --- Default HSV range for black ---
DEFAULT_LOWER_HSV = [0, 0, 0]
DEFAULT_UPPER_HSV = [180, 255, 50]

# --- Auto-expand HSV when target is selected ---
def update_hsv_range_from_blob(frame, blob, tolerance=(10, 50, 50)):
    x, y, w, h = blob
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0).astype(int)

    h, s, v = avg_hsv
    dh, ds, dv = tolerance

    lower = np.clip([h - dh, s - ds, v - dv], [0, 0, 0], [180, 255, 255])
    upper = np.clip([h + dh, s + ds, v + dv], [0, 0, 0], [180, 255, 255])

    # Update GUI sliders
    cv2.setTrackbarPos("Lower H", "Controls", lower[0])
    cv2.setTrackbarPos("Lower S", "Controls", lower[1])
    cv2.setTrackbarPos("Lower V", "Controls", lower[2])
    cv2.setTrackbarPos("Upper H", "Controls", upper[0])
    cv2.setTrackbarPos("Upper S", "Controls", upper[1])
    cv2.setTrackbarPos("Upper V", "Controls", upper[2])

    print(f"HSV adjusted to target: Lower={lower}, Upper={upper}")

# --- Mouse Callback Function ---
def select_blob(event, x, y, flags, param):
    shared_data = param["shared_data"]
    tracked_blobs = param["tracked_blobs"]
    frame = param["frame"]
    for (bx, by, bw, bh) in tracked_blobs:
        if bx <= x <= bx + bw and by <= y <= by + bh:
            shared_data["selected_blob"] = (bx, by, bw, bh)
            shared_data["target"] = (bx + bw // 2, by + bh // 2)
            update_hsv_range_from_blob(frame, (bx, by, bw, bh))
            print(f"Locked onto blob at ({bx}, {by})")
            break

# --- Main Tracking Function ---
def run_tracking(shared_data):
    lock_threshold = 2000

    cv2.namedWindow("Controls")
    cv2.resizeWindow("Controls", 500, 300)

    # Create HSV trackbars with default black
    cv2.createTrackbar("Lower H", "Controls", DEFAULT_LOWER_HSV[0], 180, lambda x: None)
    cv2.createTrackbar("Lower S", "Controls", DEFAULT_LOWER_HSV[1], 255, lambda x: None)
    cv2.createTrackbar("Lower V", "Controls", DEFAULT_LOWER_HSV[2], 255, lambda x: None)
    cv2.createTrackbar("Upper H", "Controls", DEFAULT_UPPER_HSV[0], 180, lambda x: None)
    cv2.createTrackbar("Upper S", "Controls", DEFAULT_UPPER_HSV[1], 255, lambda x: None)
    cv2.createTrackbar("Upper V", "Controls", DEFAULT_UPPER_HSV[2], 255, lambda x: None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    shared_data["selected_blob"] = None
    shared_data["target"] = None

    cv2.namedWindow("Webcam")
    # Keep a mutable wrapper to hold dynamic params
    mouse_params = {
        "shared_data": shared_data,
        "tracked_blobs": [],
        "frame": None
    }
    cv2.setMouseCallback("Webcam", select_blob, param=mouse_params)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read HSV values from trackbars
        lh = cv2.getTrackbarPos("Lower H", "Controls")
        ls = cv2.getTrackbarPos("Lower S", "Controls")
        lv = cv2.getTrackbarPos("Lower V", "Controls")
        uh = cv2.getTrackbarPos("Upper H", "Controls")
        us = cv2.getTrackbarPos("Upper S", "Controls")
        uv = cv2.getTrackbarPos("Upper V", "Controls")

        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tracked_blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                tracked_blobs.append((x, y, w, h))

        selected_blob = shared_data["selected_blob"]
        selected_center = shared_data["target"]

        # Update selected blob if locked
        if selected_center and tracked_blobs:
            closest_blob = None
            min_dist = float("inf")
            for (x, y, w, h) in tracked_blobs:
                cx, cy = x + w // 2, y + h // 2
                dist = np.hypot(cx - selected_center[0], cy - selected_center[1])
                if dist < min_dist and dist < lock_threshold:
                    min_dist = dist
                    closest_blob = (x, y, w, h)

            if closest_blob:
                shared_data["selected_blob"] = closest_blob
                shared_data["target"] = (closest_blob[0] + closest_blob[2] // 2,
                                         closest_blob[1] + closest_blob[3] // 2)
            else:
                print("Lost blob")
                shared_data["selected_blob"] = None
                shared_data["target"] = None

        # Draw
        if shared_data["selected_blob"]:
            x, y, w, h = shared_data["selected_blob"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Locked", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            for (x, y, w, h) in tracked_blobs:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show windows
        cv2.imshow("Webcam", frame)
        cv2.imshow("Mask", mask)

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            shared_data["selected_blob"] = None
            shared_data["target"] = None

            # Reset HSV sliders to black
            cv2.setTrackbarPos("Lower H", "Controls", DEFAULT_LOWER_HSV[0])
            cv2.setTrackbarPos("Lower S", "Controls", DEFAULT_LOWER_HSV[1])
            cv2.setTrackbarPos("Lower V", "Controls", DEFAULT_LOWER_HSV[2])
            cv2.setTrackbarPos("Upper H", "Controls", DEFAULT_UPPER_HSV[0])
            cv2.setTrackbarPos("Upper S", "Controls", DEFAULT_UPPER_HSV[1])
            cv2.setTrackbarPos("Upper V", "Controls", DEFAULT_UPPER_HSV[2])

            print("Reset: tracking all blobs with default black HSV range")

        # Update mouse callback
        mouse_params["tracked_blobs"] = tracked_blobs
        mouse_params["frame"] = frame.copy()

    cap.release()
    cv2.destroyAllWindows()
