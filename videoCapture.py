import cv2
import time
import os  # Import the os module for renaming

# Desired FPS
desired_fps = 30
frame_duration = 1.0 / desired_fps

# Load OpenCV libraries
cap = cv2.VideoCapture(0)  # Capture from camera
video_cap = cv2.VideoCapture('dummy.mp4')  # Replace with your video path

# Get frame size for camera and video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter objects for video output (MP4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_camera_temp = cv2.VideoWriter('temp_camera_output.mp4', fourcc, desired_fps, (frame_width, frame_height))
out_video = cv2.VideoWriter('video_output.mp4', fourcc, desired_fps, (video_width, video_height))

frame_count = 0

# Start the loop timer
start_time_loop = time.time()

while True:
    # Capture frames from camera and video
    ret, frame = cap.read()
    ret_video, video_frame = video_cap.read()

    # Check if frames were captured successfully
    if not ret or not ret_video:
        break

    # Write frames to video files
    out_camera_temp.write(frame)
    out_video.write(video_frame)

    # Display frames
    cv2.imshow("Camera Frame", frame)
    cv2.imshow("Video Frame", video_frame)

    frame_count += 1

    # Break the loop if a key is pressed
    if cv2.waitKey(1) & 0xFF != 255:
        break

    # Ensure at least the desired FPS
    elapsed_time = time.time() - start_time_loop
    expected_time = frame_count * frame_duration
    if elapsed_time < expected_time:
        time.sleep(expected_time - elapsed_time)

# End loop timer and calculate FPS and duration
end_time_loop = time.time()

total_time = end_time_loop - start_time_loop
fps_achieved = frame_count / total_time
duration = total_time

# Define the camera output filename with the achieved FPS
camera_output_filename = f"camera_output{fps_achieved:.2f}.mp4"

# Release the temporary file before renaming
out_camera_temp.release()

# Rename the temporary file to the new filename with FPS
os.rename('temp_camera_output.mp4', camera_output_filename)

# Write capture info to a file
with open("capture_info.txt", "w") as f:
    f.write(f"Capture Duration: {duration:.2f} seconds\n")
    f.write(f"Frames per second achieved: {fps_achieved:.2f}\n")
    f.write(f"Camera output saved as: {camera_output_filename}\n")

# Print info to console (optional)
print(f"Capture Duration: {duration:.2f} seconds")
print(f"Frames per second achieved: {fps_achieved:.2f}")
print(f"Camera output saved as: {camera_output_filename}")

# Release resources
out_video.release()
cap.release()
video_cap.release()
cv2.destroyAllWindows()
