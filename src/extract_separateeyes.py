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
            for eye in sorted_eyes[:2]:  # Only consider the first two eyes
                ex, ey, ew, eh = eye
                eye_regions.append((x + ex, y + ey, ew, eh))
    
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
    eyes = detect_eyes(image, face_cascade, eye_cascade)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the eyes images
    for i, (ex, ey, ew, eh) in enumerate(eyes):
        eye_region = image[ey:ey+eh, ex:ex+ew]
        cv2.imwrite(f"{output_folder}/eye_{i}.png", eye_region)
        cv2.rectangle(image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # # Optionally save the annotated input image
    # cv2.imwrite(f"{output_folder}/annotated_image.png", image)
    # print(f"Extracted {len(eyes)} eyes.")


# Example usage
input_image_path = 'output/test1.png'  # Path to the input image
output_folder = 'output/extract'  # Folder to store output images
extract_eyes(input_image_path, output_folder)