import cv2
import threading
import queue
from datetime import datetime

def capture_frames(frame_queue, shutdown_event):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to open the camera.")
        shutdown_event.set()
        return

    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        frame_queue.put(frame)
    
    cap.release()
    frame_queue.put(None)  # Signal the end of the stream
    print("Camera capture has stopped.")

def process_frames(frame_queue, shutdown_event):
    while not shutdown_event.is_set():
        frame = frame_queue.get()
        if frame is None:
            break
        # Processing can be done here
        print("Processing a frame...")
    
    print("Processing has stopped.")

def start_threads():
    frame_queue = queue.Queue(maxsize=10)
    shutdown_event = threading.Event()
    thread_capture = threading.Thread(target=capture_frames, args=(frame_queue, shutdown_event))
    thread_process = threading.Thread(target=process_frames, args=(frame_queue, shutdown_event))

    thread_capture.start()
    thread_process.start()
    return thread_capture, thread_process, shutdown_event
