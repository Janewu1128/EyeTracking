# import pyautogui

# def capture_screen(frame_count, timestamp, screen_folder):
#     screen = pyautogui.screenshot()
#     screen_path = f"{screen_folder}/screen_{frame_count}_{timestamp}.png"
#     screen.save(screen_path)

import pyautogui
from PIL import Image, ImageDraw

def capture_screen(frame_count, timestamp, screen_folder, gaze_x, gaze_y):
    try:
        screen = pyautogui.screenshot()
        draw = ImageDraw.Draw(screen)

        # Ensure gaze coordinates are within the screen bounds
        screen_width, screen_height = screen.size
        gaze_x = max(0, min(gaze_x, screen_width - 1))
        gaze_y = max(0, min(gaze_y, screen_height - 1))

        # Draw a rectangle or point on the gaze area
        box_size = 10  # Size of the box to draw around gaze point
        draw.rectangle([gaze_x - box_size, gaze_y - box_size, gaze_x + box_size, gaze_y + box_size], outline='red')

        screen_path = f"{screen_folder}/screen_{frame_count}_{timestamp}.png"
        screen.save(screen_path)
    except Exception as e:
        print(f"Error capturing or drawing on screen: {e}")
