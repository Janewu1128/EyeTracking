import cv2

def test_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

test_camera()