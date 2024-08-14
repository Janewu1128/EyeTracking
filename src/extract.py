import cv2
import os

def detect_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eye_regions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + int(h/2), x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(30, 30))
        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            if len(sorted_eyes) > 2:
                sorted_eyes = sorted_eyes[:2]  # Ensure only two eyes are considered

            # Calculate the bounding box for both eyes with added buffer
            buffer = 20  # Additional pixels added to each side of the eye region
            start_x = min(sorted_eyes, key=lambda e: e[0])[0] + x - buffer
            end_x = max(sorted_eyes, key=lambda e: e[0])[0] + x + max(sorted_eyes, key=lambda e: e[2])[2] + buffer
            start_y = min(sorted_eyes, key=lambda e: e[1])[1] + y - buffer // 2  # Smaller buffer for y to avoid including too much forehead or cheeks
            end_y = max(sorted_eyes, key=lambda e: e[1])[1] + y + max(sorted_eyes, key=lambda e: e[3])[3] + buffer // 2

            eye_regions.append((start_x, start_y, end_x - start_x, end_y - start_y))
    return eye_regions

def extract_eyes_from_video(input_video_path, output_folder):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    frame_count = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        eye_regions = detect_eyes(frame, face_cascade, eye_cascade)
        for i, (ex, ey, ew, eh) in enumerate(eye_regions):
            eye_region = frame[ey:ey+eh, ex:ex+ew]
            resized_eye_region = cv2.resize(eye_region, (384, 128))  # Resize the eye region to 384x128
            cv2.imwrite(f"{output_folder}/frame_{frame_count}_eyes.png", resized_eye_region)
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imwrite(f"{output_folder}/annotated_frame_{frame_count}.png", frame)
        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames.")

# Example usage
input_video_path = 'input/camera_output23.22.mp4'  # Path to the input video
output_folder = 'output/eye_extract'  # Folder to store output images
extract_eyes_from_video(input_video_path, output_folder)
