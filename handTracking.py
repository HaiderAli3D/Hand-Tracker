import cv2
import mediapipe as mp
import pyautogui as pag
import numpy as np
import time

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

mp_drawing = mp.solutions.drawing_utils

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error")
    exit()

#get screen rez
screenWidth, screenHeight = pag.size()

mouseDown = False
handEnabled = False
timer = 0


while True:
    timer += 1
    # Capture frame by frame from the camera
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame color from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    #frame rez
    frameHeight, frameWidth, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            midpointX = (index_finger_tip.x + thumb_tip.x) /2
            midpointY = (index_finger_tip.y + thumb_tip.y) /2
            
            Index_Thumb_distance = np.sqrt((index_finger_tip.x - thumb_tip.x )**2 + (index_finger_tip.y - thumb_tip.y )**2)
            Middle_Thumb_distance = np.sqrt((middle_finger_tip.x - thumb_tip.x )**2 + (middle_finger_tip.y - thumb_tip.y )**2)
            Ring_Thumb_distance = np.sqrt((ring_finger_tip.x - thumb_tip.x )**2 + (ring_finger_tip.y - thumb_tip.y )**2)
            
            if Ring_Thumb_distance < 0.05 and timer > 3:
                handEnabled = not handEnabled
                timer = 0
                
            if handEnabled:
            
                if Index_Thumb_distance < 0.05 and mouseDown == False:
                    pag.mouseDown()
                    mouseDown = True
                
                if Index_Thumb_distance > 0.1 and mouseDown == True:
                    pag.mouseUp()
                    mouseDown = False

                if Middle_Thumb_distance < 0.05:
                    pag.rightClick()
                    mouseDown = True

            
                if mouseDown:
                    cv2.circle(frame, (int(thumb_tip.x*frameWidth), int(thumb_tip.y * frameHeight)), 10, (255, 50, 0), -1)
                else:
                    cv2.circle(frame, (int(thumb_tip.x*frameWidth), int(thumb_tip.y * frameHeight)), 10, (255, 50, 0), 4)
                
                MappedX = np.interp(midpointX, (0,1), (0, screenWidth))
                MappedY = np.interp(midpointY, (0,1), (0, screenHeight))
                
                pag.moveTo(MappedX, MappedY, duration = 0.1)
            
            
            

    # Display the resulting frame
    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
