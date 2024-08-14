import cv2
import time
import numpy as np

# Desired FPS
desired_fps = 30
frame_duration = 1.0 / desired_fps

# Load OpenCV's pre-trained face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

# Load a video file
video_path = 'dummy.mp4'
video_cap = cv2.VideoCapture(video_path)

# Get default frame size for the camera and video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object for camera output in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_camera = cv2.VideoWriter('camera_output1.mp4', fourcc, desired_fps, (frame_width, frame_height))

# Define VideoWriter for video output in MP4 format
out_video = cv2.VideoWriter('video_output1.mp4', fourcc, desired_fps, (video_width, video_height))

frame_count = 0

# Start the loop timer
start_time_loop = time.time()

# Lists to store eye positions for movement tracking
left_eye_positions = []
right_eye_positions = []

def detect_eye_movement(eye_region, prev_positions):
    x, y, w, h = eye_region
    center = (x + w // 2, y + h // 2)

    if prev_positions:
        prev_center = prev_positions[-1]
        movement = np.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
    else:
        movement = 0

    prev_positions.append(center)
    return movement, center

while True:
    # Capture frame-by-frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Capture frame-by-frame from the video
    ret_video, video_frame = video_cap.read()
    if not ret_video:
        break

    # Convert the frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Assuming there are 2 eyes max
            eye_region = (x + ex, y + ey, ew, eh)
            cv2.rectangle(frame, (eye_region[0], eye_region[1]), (eye_region[0] + ew, eye_region[1] + eh), (0, 255, 0), 2)

            if i == 0:  # Left eye
                left_movement, left_center = detect_eye_movement(eye_region, left_eye_positions)
                cv2.circle(frame, left_center, 3, (0, 255, 0), -1)
                cv2.putText(frame, f'Left Eye Movement: {left_movement:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif i == 1:  # Right eye
                right_movement, right_center = detect_eye_movement(eye_region, right_eye_positions)
                cv2.circle(frame, right_center, 3, (0, 255, 0), -1)
                cv2.putText(frame, f'Right Eye Movement: {right_movement:.2f}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write the frames to the video files
    out_camera.write(frame)
    out_video.write(video_frame)

    # Display the frames
    cv2.imshow("Camera Frame", frame)
    cv2.imshow("Video Frame", video_frame)

    frame_count += 1

    # Break the loop if any key is pressed
    if cv2.waitKey(1) & 0xFF != 255:
        break

    # Ensure at least the desired FPS
    elapsed_time = time.time() - start_time_loop
    expected_time = frame_count * frame_duration
    if elapsed_time < expected_time:
        time.sleep(expected_time - elapsed_time)

# End the loop timer
end_time_loop = time.time()

# Calculate and write FPS and duration to a file
total_time = end_time_loop - start_time_loop
fps_achieved = frame_count / total_time
duration = total_time

with open("capture_info.txt", "w") as f:
    f.write(f"Capture Duration: {duration:.2f} seconds\n")
    f.write(f"Frames per second achieved: {fps_achieved:.2f}\n")

# Print to console (optional)
print(f"Capture Duration: {duration:.2f} seconds")
print(f"Frames per second achieved: {fps_achieved:.2f}")

# Release the VideoWriter objects and the captures
out_camera.release()
out_video.release()
cap.release()
video_cap.release()
cv2.destroyAllWindows()
