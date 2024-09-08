import os
import csv


# Function to normalize coordinates based on screen width and height
def normalize_coordinates(coords, screen_width, screen_height):
    normalized_coords = []
    for i in range(0, len(coords), 2):
        x = coords[i] / screen_width
        y = coords[i + 1] / screen_height
        normalized_coords.extend([x, y])
    return normalized_coords

input_file = "D:\\MPIIGaze\\Annotation Subset\\p00.txt"
output_file = "D:\\MPIIGaze\\Annotation Subset\\p00_normalized.csv"

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    header = ['Image'] + [f'Coord_{i + 1}' for i in range(16)]
    csv_writer.writerow(header)

    for line in infile:
        parts = line.strip().split()
        image_path = parts[0]
        coordinates = list(map(int, parts[1:]))
        screen_width, screen_height = (1280, 800)
        normalized_coords = normalize_coordinates(coordinates, screen_width, screen_height)
        csv_writer.writerow([image_path] + normalized_coords)

print("Normalization complete. Results saved to", output_file)
