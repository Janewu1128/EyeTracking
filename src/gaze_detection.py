import cv2

def detect_gaze_direction(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Initialize variables to store gaze information for each eye
    left_eye = {'x': -1, 'y': -1, 'w': -1, 'h': -1}
    right_eye = {'x': -1, 'y': -1, 'w': -1, 'h': -1}

    for (x, y, w, h) in faces:
        # Focusing on the upper half of the face to improve eye detection
        roi_gray = gray[y:y + int(h/2), x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(30, 30))

        if len(eyes) >= 2:
            # Sort the detected eyes based on the x coordinate to identify left and right eyes
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            left_eye_details, right_eye_details = sorted_eyes[0], sorted_eyes[1]

            # Left eye coordinates
            left_eye['x'] = x + left_eye_details[0] + left_eye_details[2] // 2
            left_eye['y'] = y + left_eye_details[1] + left_eye_details[3] // 2
            left_eye['w'] = left_eye_details[2]
            left_eye['h'] = left_eye_details[3]

            # Right eye coordinates
            right_eye['x'] = x + right_eye_details[0] + right_eye_details[2] // 2
            right_eye['y'] = y + right_eye_details[1] + right_eye_details[3] // 2
            right_eye['w'] = right_eye_details[2]
            right_eye['h'] = right_eye_details[3]

            break  # Only process the first detected face for simplicity

    return left_eye, right_eye