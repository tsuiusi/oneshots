import cv2

def mirror(frame, cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2

    frame[:, :width] = cv2.flip(frame[:, width:], 1)

    return frame






