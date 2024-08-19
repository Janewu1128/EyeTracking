import cv2
import pyautogui
import numpy as np
import keyboard

# Initialize the camera
camera = cv2.VideoCapture(0)

# Get the full resolution of the camera
camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera.get(cv2.CAP_PROP_FRAME_WIDTH))
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Retrieve the actual resolution
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_resolution = (frame_width, frame_height)

# Get the screen size
screen_size = pyautogui.size()

# Define codec and create VideoWriter object for screen recording (MP4)
fourcc_screen = cv2.VideoWriter_fourcc(*'mp4v')
out_screen = cv2.VideoWriter('output_screen.mp4', fourcc_screen, 20.0, screen_size)

# Define codec and create VideoWriter object for camera recording (MP4)
fourcc_cam = cv2.VideoWriter_fourcc(*'mp4v')
out_cam = cv2.VideoWriter('output_camera.mp4', fourcc_cam, 20.0, camera_resolution)

# Initialize frame counters
screen_frame_count = 0
camera_frame_count = 0

while True:
    # Capture screen
    screen_img = pyautogui.screenshot()
    screen_frame = np.array(screen_img)
    screen_frame = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2RGB)

    # Capture camera
    ret, cam_frame = camera.read()

    # Write the frames to the video files
    out_screen.write(screen_frame)
    screen_frame_count += 1

    if ret:
        out_cam.write(cam_frame)
        camera_frame_count += 1

    # Check for 'q' key press to stop recording
    if keyboard.is_pressed('q'):
        break

# Release everything
camera.release()
out_screen.release()
out_cam.release()
cv2.destroyAllWindows()

# Output the number of frames captured
print(f"Total screen frames captured: {screen_frame_count}")
print(f"Total camera frames captured: {camera_frame_count}")
