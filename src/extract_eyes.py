import cv2
import os

def detect_eyes(image, face_cascade, eye_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eye_regions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + int(h/2), x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(30, 30))

        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            eye_details = []
            for eye in sorted_eyes[:2]:  # Only consider the first two eyes
                ex, ey, ew, eh = eye
                eye_details.append((x + ex, y + ey, ew, eh))
            eye_regions.append(eye_details)
    
    return eye_regions

def extract_eyes(input_image_path, output_folder):
    # Load the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print("Could not read the image.")
        return

    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect eyes in the image
    all_eye_regions = detect_eyes(image, face_cascade, eye_cascade)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each face's eyes
    for face_index, eyes in enumerate(all_eye_regions):
        if len(eyes) == 2:
            # Compute the bounding box covering both eyes
            left_eye, right_eye = eyes
            start_x = min(left_eye[0], right_eye[0])
            end_x = max(left_eye[0] + left_eye[2], right_eye[0] + right_eye[2])
            start_y = min(left_eye[1], right_eye[1])
            end_y = max(left_eye[1] + left_eye[3], right_eye[1] + right_eye[3])

            # Extract and save the region containing both eyes
            eye_region = image[start_y:end_y, start_x:end_x]
            cv2.imwrite(f"{output_folder}/face_{face_index}_eyes.png", eye_region)

            # Draw rectangle around the combined eye region for visualization
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Optionally save the annotated input image
    cv2.imwrite(f"{output_folder}/annotated_image.png", image)
    print(f"Extracted eyes for {len(all_eye_regions)} faces.")

# Example usage
input_image_path = 'output/test1.png'  # Path to the input image
output_folder = 'output/eye_extract'  # Folder to store output images
extract_eyes(input_image_path, output_folder)
