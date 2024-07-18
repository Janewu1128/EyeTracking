import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from video_processing import capture_video

if __name__ == "__main__":
    capture_video()



# $python scripts/run_capture_video.py