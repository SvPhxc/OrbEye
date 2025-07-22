import cv2
import numpy as np

# --- Global Variables ---
tracked_blobs = []
selected_blob = None
selected_center = None
lock_threshold = 100  # max distance (pixels) to keep tracking

# --- Mouse Callback Function ---
def select_blob(event, x, y, flags, param):
    global selected_blob, selected_center
    if event == cv2.EVENT_LBUTTONDOWN:
        for (bx, by, bw, bh) in tracked_blobs:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                selected_blob = (bx, by, bw, bh)
                selected_center = (bx + bw // 2, by + bh // 2)
                print(f"✅ Locked onto blob at ({bx}, {by})")
                break

# --- Trackbar Callback Stub ---
def nothing(x):
    pass

# --- Create GUI Controls ---
cv2.namedWindow("Controls")
cv2.resizeWindow("Controls", 500, 300)

cv2.createTrackbar("Lower H", "Controls", 0, 180, nothing)
cv2.createTrackbar("Lower S", "Controls", 0, 255, nothing)
cv2.createTrackbar("Lower V", "Controls", 0, 255, nothing)
cv2.createTrackbar("Upper H", "Controls", 180, 180, nothing)
cv2.createTrackbar("Upper S", "Controls", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Controls", 50, 255, nothing)

# --- Start Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", select_blob)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame. Exiting ...")
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV range from trackbars
    lh = cv2.getTrackbarPos("Lower H", "Controls")
    ls = cv2.getTrackbarPos("Lower S", "Controls")
    lv = cv2.getTrackbarPos("Lower V", "Controls")
    uh = cv2.getTrackbarPos("Upper H", "Controls")
    us = cv2.getTrackbarPos("Upper S", "Controls")
    uv = cv2.getTrackbarPos("Upper V", "Controls")

    lower_hsv = np.array([lh, ls, lv])
    upper_hsv = np.array([uh, us, uv])

    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tracked_blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            tracked_blobs.append((x, y, w, h))

    # --- Update locked blob by nearest center ---
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
            selected_blob = closest_blob
            selected_center = (closest_blob[0] + closest_blob[2] // 2,
                               closest_blob[1] + closest_blob[3] // 2)
        else:
            print("⚠️ Lost blob - no match within range")
            selected_blob = None
            selected_center = None

    # Draw blobs
    if selected_blob:
        x, y, w, h = selected_blob
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Locked", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        for (x, y, w, h) in tracked_blobs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display frames
    cv2.imshow("Webcam", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        selected_blob = None
        selected_center = None
        print("🔁 Reset: tracking all blobs again")

cap.release()
cv2.destroyAllWindows()
