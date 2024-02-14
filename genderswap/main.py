import cv2
import mediapipe as mp
import numpy as np
from function import add_filter

mphol = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

image_path = 'crown.png'

def detection(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False # this step just saves compute
    results = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, results

def draw_styled_keypoints(image, results):
    mp_drawing.draw_landmarks(
            image, results.face_landmarks, mphol.FACEMESH_TESSELATION, 
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), 
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )

def cv2draw(image, results):
    n = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = image.shape[:2]
    if results.face_landmarks:
        for i in results.face_landmarks.landmark:
            x = int(i.x * width)
            y = int(i.y * height)
            cv2.circle(image, (x, y), 4, (255, 0, 0), 3)
            cv2.putText(image, str(n), (x, y), font, 0.5, (255, 0, 0))
            n += 1

def extract_keypoints(results):
    return np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

def get_coords(image, landmark):
    height, width = image.shape[:2]
    x = int(landmark.x * width)
    y = int(landmark.y * height) 
    return x, y

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open video device")
    exit()

# Set properties, if needed (width, height, and so forth)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with mphol.Holistic(min_detection_confidence=0.8, min_tracking_confidence = 0.8) as holistic:
    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
          
            img, results = detection(frame, holistic)
            # draw_styled_keypoints(img, results)
            cv2draw(img, results)
            
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
           
            try:
                print(results.face_landmarks.landmark[64])
                lcoord = get_coords(frame, results.face_landmarks.landmark[64])
                rcoord = get_coords(frame, results.face_landmarks.landmark[284]) 
                
                #try:
                cv2.imshow('frame', add_filter(image, img, lcoord, rcoord))
                #except:
                    #cv2.imshow('frame', img)
            except:
                cv2.imshow('frame', img)
            

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        # Handle any cleanup here if the loop is exited via keyboard interrupt
        pass
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
