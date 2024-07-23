import cv2
import dlib
import time
import os
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

video_path = 'demo.mp4'
video_cap = cv2.VideoCapture(video_path)

fps_video = video_cap.get(cv2.CAP_PROP_FPS)
fps = max(30, fps_video)
frame_duration = 1.0 / fps

start_time = time.time()

os.makedirs("video_frames", exist_ok=True)
os.makedirs("camera_frames", exist_ok=True)

frame_count = 0

left_pupil_positions = []
right_pupil_positions = []


def detect_pupil(eye_region, frame, gray):
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    eye = eye[min_y:max_y, min_x:max_x]

    eye_blur = cv2.GaussianBlur(eye, (7, 7), 0)  # Apply Gaussian blur to reduce noise
    _, threshold_eye = cv2.threshold(eye_blur, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = threshold_eye.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Only consider large enough contours to be the pupil
                cv2.drawContours(mask, [contour], -1, 255, -1)

        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx + min_x, cy + min_y

    return None


def calculate_movement(pupil_positions, current_pupil):
    if pupil_positions:
        prev_pupil = pupil_positions[-1]
        movement = np.sqrt((current_pupil[0] - prev_pupil[0]) ** 2 + (current_pupil[1] - prev_pupil[1]) ** 2)
        return movement
    return 0


def calculate_range(pupil_positions):
    if len(pupil_positions) > 1:
        x_positions = [p[0] for p in pupil_positions]
        y_positions = [p[1] for p in pupil_positions]
        x_range = max(x_positions) - min(x_positions)
        y_range = max(y_positions) - min(y_positions)
        return x_range, y_range
    return 0, 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time() - start_time

    ret_video, video_frame = video_cap.read()
    if not ret_video:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a bilateral filter to reduce noise and preserve edges (helps with glasses)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                     (landmarks.part(43).x, landmarks.part(43).y),
                                     (landmarks.part(44).x, landmarks.part(44).y),
                                     (landmarks.part(45).x, landmarks.part(45).y),
                                     (landmarks.part(46).x, landmarks.part(46).y),
                                     (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        left_pupil = detect_pupil(left_eye_region, frame, gray)
        right_pupil = detect_pupil(right_eye_region, frame, gray)

        left_movement = calculate_movement(left_pupil_positions, left_pupil) if left_pupil else 0
        right_movement = calculate_movement(right_pupil_positions, right_pupil) if right_pupil else 0

        if left_pupil:
            left_pupil_positions.append(left_pupil)
        if right_pupil:
            right_pupil_positions.append(right_pupil)

        left_range = calculate_range(left_pupil_positions)
        right_range = calculate_range(right_pupil_positions)

        cv2.putText(frame, f'Left Eye Movement: {left_movement:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(frame, f'Right Eye Movement: {right_movement:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(frame, f'Left Eye Range: {left_range[0]:.2f}, {left_range[1]:.2f}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Right Eye Range: {right_range[0]:.2f}, {right_range[1]:.2f}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(f"video_frames/frame_{frame_count:04d}.jpg", video_frame)
    cv2.imwrite(f"camera_frames/frame_{frame_count:04d}.jpg", frame)

    frame_count += 1

    cv2.imshow("Video Frame", video_frame)
    cv2.imshow("Camera Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    while time.time() - start_time < frame_count * frame_duration:
        time.sleep(0.001)

cap.release()
video_cap.release()
cv2.destroyAllWindows()
