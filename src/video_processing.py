# capture camera feed and store as png output
import cv2
from datetime import datetime
import os
import csv

def capture_video(participant_id, video_name):
    cap = cv2.VideoCapture(1)  # Ensure the correct camera index
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_folder = f"{output_dir}/images_{timestamp}"
    os.makedirs(image_folder)

    output_filename = f"{output_dir}/output_{timestamp}.csv"
    fields = ['Participant ID', 'Video Name', 'Timestamp', 'X', 'Y', 'Width', 'Height']
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around the face

                # Adjusting the eye detection area to the upper half of the face
                roi_gray = gray[y:y + int(h/2), x:x + w]
                roi_color = frame[y:y + int(h/2), x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)  # Draw rectangle around the eyes
                    csvwriter.writerow([participant_id, video_name, datetime.now().strftime("%Y%m%d_%H%M%S%f"), x+ex, y+ey, ew, eh])

            # Save frame with drawn rectangles for face and eyes
            frame_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            cv2.imwrite(f"{image_folder}/frame_{frame_count}_{frame_timestamp}.png", frame)

            cv2.imshow('Webcam Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

# Example usage
capture_video('Participant01', 'Video01')



# # capture camera feed and desktop screen
# import time
# import cv2
# from datetime import datetime
# import os
# import csv
# import pyautogui 
# from capture_screen import capture_screen
# from gaze_detection import detect_gaze_direction

# def capture_video(participant_id, video_name):
#     cap = cv2.VideoCapture(1)
#     cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS from the webcam
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#     output_dir = 'output'
#     os.makedirs(output_dir, exist_ok=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")  # More precise timestamp
#     screen_folder = os.path.join(output_dir, f"screens_{timestamp}")  # Unique screen folder per session
#     os.makedirs(screen_folder, exist_ok=True)
#     image_folder = os.path.join(output_dir, f"images_{timestamp}")  # Unique image folder per session
#     os.makedirs(image_folder, exist_ok=True)

#     output_filename = f"{output_dir}/output_{timestamp}.csv"
#     fields = ['Participant ID', 'Video Name', 'Timestamp', 'X', 'Y', 'Width', 'Height']
#     with open(output_filename, 'w', newline='') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(fields)

#         frame_count = 0
#         start_time = time.time()

#         while True:
#             if time.time() - start_time < 1./30:
#                 time.sleep(max(0, 1./30 - (time.time() - start_time)))  # Ensures exactly 1/30 second per loop
#                 continue

#             ret, frame = cap.read()
#             if not ret:
#                 break

#             current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")  # Refresh timestamp for each frame
#             gaze_x, gaze_y = detect_gaze_direction(frame, face_cascade, eye_cascade)  # Get gaze direction
#             capture_screen(frame_count, current_timestamp, screen_folder, gaze_x, gaze_y)  # Pass gaze coordinates to screen capture

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#                 roi_gray = gray[y:y + int(h/2), x:x + w]
#                 # roi_color = frame[y:y + int(h/2), x:x + w]
#                 eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, minNeighbors=5, minSize=(30, 30))

#                 for (ex, ey, ew, eh) in eyes:
#                     cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)
#                     csvwriter.writerow([participant_id, video_name, current_timestamp, x+ex, y+ey, ew, eh])

#             frame_path = f"{image_folder}/frame_{frame_count}_{current_timestamp}.png"
#             cv2.imwrite(frame_path, frame)  # Save each frame with a unique timestamp

#             cv2.imshow('Webcam Feed', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#             frame_count += 1
#             start_time = time.time()  # Reset the timer after the frame is processed

#         cap.release()
#         cv2.destroyAllWindows()

# # Example usage
# capture_video('Participant02', 'Video02')