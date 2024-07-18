import cv2

def detect_gaze_direction(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Placeholder values in case no eyes are detected
    gaze_x, gaze_y = -1, -1

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        if len(eyes) >= 2:
            # Assuming the first two detected areas are the eyes
            eye_1 = eyes[0]
            eye_2 = eyes[1]

            # Calculate the center point of each eye's bounding box
            eye_1_center = (x + eye_1[0] + eye_1[2] // 2, y + eye_1[1] + eye_1[3] // 2)
            eye_2_center = (x + eye_2[0] + eye_2[2] // 2, y + eye_2[1] + eye_2[3] // 2)

            # Average the center points to estimate gaze point
            gaze_x = (eye_1_center[0] + eye_2_center[0]) // 2
            gaze_y = (eye_1_center[1] + eye_2_center[1]) // 2
            break  # Assuming we only deal with one face for simplicity

    return gaze_x, gaze_y
