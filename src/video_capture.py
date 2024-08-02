import cv2
from datetime import datetime
import os
from capture_screen import capture_screen
from gaze_detection import detect_gaze_direction
import pyautogui

def capture_video(participant_id, video_name):
    # Initialize video capture
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    # Initialize face and eye cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"{output_dir}/{participant_id}_{video_name}_{timestamp}.mp4"
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    # Set up screen capture output directory
    screen_folder = f"{output_dir}/screens_{timestamp}"
    os.makedirs(screen_folder, exist_ok=True)

    frame_count = 0  # Initialize frame count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform face and eye detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        gaze_x, gaze_y = -1, -1  # Default values if no gaze is detected
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            # Assuming gaze detection uses the first detected face
            if gaze_x == -1 and gaze_y == -1:  # Only detect gaze once per frame for the first face
                gaze_x, gaze_y = detect_gaze_direction(frame, face_cascade, eye_cascade)

        # Capture the screen with gaze overlay if gaze is detected
        if gaze_x != -1 and gaze_y != -1:
            capture_screen(frame_count, timestamp, screen_folder, gaze_x, gaze_y)

        # Write the frame into the file
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('Video Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1  # Increment frame count

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
capture_video('Participant01', 'Video01')
