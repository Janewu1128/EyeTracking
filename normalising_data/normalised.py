import os
import scipy.io

# Function to normalize the coordinates
def normalize_coordinates(x, y, screen_width, screen_height):
    x_norm = x / screen_width  # Normalize X coordinate
    y_norm = y / screen_height  # Normalize Y coordinate
    return x_norm, y_norm

# Function to process the annotation file and write to CSV
def process_annotation_file_to_csv(input_file, output_file, screen_width, screen_height):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as out_f:
        # Write the header row with only x_left_eye, y_right_eye
        out_f.write("File_name,x_left_eye,y_right_eye\n")

        for line in lines:
            # Split the line by spaces
            parts = line.strip().split()
            
            # The first part is the image file name
            image_file = parts[0]
            
            # Convert the relevant coordinates (2nd and 3rd columns)
            x_left_eye = int(parts[1])  # X-coordinate of the left eye
            y_right_eye = int(parts[2])  # Y-coordinate of the right eye
            
            # Normalize these coordinates
            x_left_norm, y_right_norm = normalize_coordinates(x_left_eye, y_right_eye, screen_width, screen_height)
            
            # Write the image file and normalized coordinates to the output CSV file
            out_f.write(f"{image_file},{x_left_norm:.6f},{y_right_norm:.6f}\n")

# Function to load screen size from the .mat file
def load_screen_size(mat_file_path):
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Extract the screen width and height in pixels
    if 'width_pixel' in mat_data and 'height_pixel' in mat_data:
        screen_width = mat_data['width_pixel'][0][0]  # Get the width in pixels
        screen_height = mat_data['height_pixel'][0][0]  # Get the height in pixels
        return int(screen_width), int(screen_height)
    else:
        raise KeyError("Required keys not found in the .mat file.")

# Example usage
mat_file = 'screenSize14.mat'  # Path to the screenSize.mat file
screen_width, screen_height = load_screen_size(mat_file)

# Input and output file paths
input_file = 'p14.txt'  # Replace with your actual file path
output_file = 'p14_normalized_file.csv'  # Output CSV file for normalized data

# Process the file and generate the CSV
process_annotation_file_to_csv(input_file, output_file, screen_width, screen_height)

print(f"Normalized coordinates have been written to {output_file}")
