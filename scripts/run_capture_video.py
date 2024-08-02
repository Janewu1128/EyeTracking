# capture camera feed and store as png output
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from video_processing import capture_video

if __name__ == "__main__":
    capture_video('Participant01', 'TestVideo')

# # capture camera feed and store as video type but can't open
# import sys
# import os

# # Add the src directory to the Python path to ensure imports work correctly
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# from video_capture import capture_video

# def main():
#     try:
#         # Assuming you want to capture video for a specific participant and video name
#         participant_id = 'Participant01'
#         video_name = 'VideoTest'
#         capture_video(participant_id, video_name)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         print("Cleanup and exit.")

# if __name__ == "__main__":
#     main()

# # multithread to handle
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# from threaded_video_processing import start_threads

# def main():
#     thread_capture, thread_process, shutdown_event = start_threads()
#     try:
#         while not shutdown_event.is_set():
#             # Placeholder for any main thread activity
#             pass
#     except KeyboardInterrupt:
#         print("Received keyboard interrupt.")
#         shutdown_event.set()
#     finally:
#         thread_capture.join()
#         thread_process.join()
#         print("All threads have been closed.")

# if __name__ == "__main__":
#     main()

# $python scripts/run_capture_video.py