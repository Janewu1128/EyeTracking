# capture camera feed/desktop screen and store as png, csv output
import cv2
from datetime import datetime
import os
import csv
from gaze_detection import detect_gaze_direction
from capture_screen import capture_screen

def capture_video(participant_id, video_name):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_folder = f"{output_dir}/images_{timestamp}"
    screen_folder = f"{output_dir}/screens_{timestamp}"
    os.makedirs(image_folder)
    os.makedirs(screen_folder)

    output_filename = f"{output_dir}/output_{timestamp}.csv"
    fields = ['Participant ID', 'Video Name', 'Timestamp', 'Eye', 'X', 'Y', 'Width', 'Height']
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))  # Draw rectangle around each face
                left_eye, right_eye = detect_gaze_direction(frame, face_cascade, eye_cascade)

                for eye, color, label in [(left_eye, (0, 255, 0), 'Left'), (right_eye, (255, 0, 0), 'Right')]:
                    if eye['x'] != -1:
                        eye_x = eye['x']
                        eye_y = eye['y']
                        eye_w = eye['w']
                        eye_h = eye['h']
                        cv2.rectangle(frame, (eye_x - eye_w // 2, eye_y - eye_h // 2),
                                    (eye_x + eye_w // 2, eye_y + eye_h // 2), color, 2)  # Draw rectangle for the eye
                        csvwriter.writerow([participant_id, video_name, frame_timestamp, label, eye_x, eye_y, eye_w, eye_h])
                        capture_screen(frame_count, frame_timestamp, screen_folder, eye_x, eye_y)  # Save screenshot with gaze

            # Save the frame after all rectangles have been drawn
            cv2.imwrite(f"{image_folder}/frame_{frame_count}_{frame_timestamp}.png", frame)

            cv2.imshow('Webcam Feed', frame)  # Display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

# Example usage
capture_video('Participant01', 'Video01')