import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.engine import data_adapter


def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)


data_adapter._is_distributed_dataset = _is_distributed_dataset
images = []
labels = []

root_dir = 'D://MPIIGaze//Data//Original'

for participant in os.listdir(root_dir):
    # Stop if participant is
    if participant == 'p04':
        break
    participant_dir = os.path.join(root_dir, participant)
    if os.path.isdir(participant_dir):
        csv_file = os.path.join(participant_dir, f'{participant}_normalized.csv')
        if os.path.exists(csv_file):
            # print(f"Processing participant: {participant}")
            df = pd.read_csv(csv_file, header=None)

            # Iterate through each day of the participant
            for day_folder in os.listdir(participant_dir):
                day_dir = os.path.join(participant_dir, day_folder)
                if os.path.isdir(day_dir):
                    # print(f"Processing day folder: {day_folder}")
                    # Iterate through each image file of the day
                    for img_file in os.listdir(day_dir):
                        if img_file.endswith('_revised.jpg'):
                            img_path = os.path.join(day_dir, img_file)
                            original_img_name = f"{day_folder}/{img_file.replace('_revised.jpg', '.jpg')}"  # Match format dayXX/XXXX.jpg

                            row = df[df[0] == original_img_name]
                            if not row.empty:

                                img = cv2.imread(img_path)
                                if img is None:
                                    print(f"Failed to load image: {img_path}")
                                    continue

                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                h, w, _ = img.shape
                                cropped_img = img[(h - 128) // 2:(h + 128) // 2, (w - 384) // 2:(w + 384) // 2]

                                # Check if the cropped image contains mostly non-black pixels
                                if np.mean(cropped_img) > 0:  # Adjust the threshold if necessary
                                    # Normalize the cropped image to [0, 1] range
                                    cropped_img = cropped_img / 255.0
                                    # Add the cropped image to the images list
                                    images.append(cropped_img)
                                # images.append(img)
                                coords = row.iloc[0, 1:3].values.astype(float)
                                labels.append(coords)
                            # else:
                            # print(f"Coordinates not found for {original_img_name}")

if len(images) == 0 or len(labels) == 0:
    print("No images or labels were loaded. Please check the directory structure or CSV files.")
else:
    min_length = min(len(images), len(labels))
    # Trim both lists to the smaller length to ensure they are consistent
    images = images[:min_length]
    labels = labels[:min_length]
    images = np.array(images)
    labels = np.array(labels)
    print(f"Loaded {len(images)} images and {len(labels)} labels")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Add dropout to prevent overfitting
    Dense(128, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print("Model defined and compiled.")
print("Training the model...")

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

model.fit(train_dataset, epochs=10, validation_data=test_dataset)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f'Test Accuracy: {test_accuracy:.4f}')