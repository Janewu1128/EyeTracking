import cv2
import time

# Desired FPS
desired_fps = 30
frame_duration = 1.0 / desired_fps

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
out_camera = cv2.VideoWriter('camera_output.mp4', fourcc, desired_fps, (frame_width, frame_height))

# Define VideoWriter for video output in MP4 format
out_video = cv2.VideoWriter('video_output.mp4', fourcc, desired_fps, (video_width, video_height))

frame_count = 0

# Start the loop timer
start_time_loop = time.time()

while True:
    # Capture frame-by-frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Capture frame-by-frame from the video
    ret_video, video_frame = video_cap.read()
    if not ret_video:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

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
