import cv2
import mediapipe as mp
import pyautogui as pag
import numpy as np
import time
import mediapipe_cheats as mpc
import threading
import queue
from pynput.mouse import Button, Controller


mouse = Controller()

pag.FAILSAFE = False

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.75
)

mp_drawing = mp.solutions.drawing_utils

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error")
    exit()

# Get screen resolution
screenWidth, screenHeight = pag.size()

# Track states for both hands
right_hand_states = {
    "mouseDown": False,
    "rightMouseDown": False,
    "middleMouseDown": False,
}

left_hand_states = {
    "shift": False
}

CONTROL_AREA_PERCENTAGE = 80

# Calculate the margins (how far from the edges the control area should be)
margin_percentage = (100 - CONTROL_AREA_PERCENTAGE) / 2
margin = margin_percentage / 100

# Define the control area boundaries
control_area = {
    "x_min": margin,          # e.g., 0.15 for 70% width
    "x_max": 1 - margin,      # e.g., 0.85 for 70% width
    "y_min": margin,          # e.g., 0.15 for 70% height
    "y_max": 1 - margin       # e.g., 0.85 for 70% height
}

global_hand_state = False
scroll_enable = False
two_hand_gesture = False
previous_hand_distance = None

timer = 0
timer2 = 0

def draw_hand_connection(frame, left_landmarks, right_landmarks, is_pinching=False):
    if left_landmarks and right_landmarks:
        # Get center points of each hand
        left_x, left_y = calculate_hand_midpoint(left_landmarks)
        right_x, right_y = calculate_hand_midpoint(right_landmarks)
        
        # Convert the normalized (0-1) coordinates to actual pixel positions
        height, width, _ = frame.shape
        start_point = (int(left_x * width), int(left_y * height))
        end_point = (int(right_x * width), int(right_y * height))
        
        # Set line color based on gesture state
        # OpenCV uses BGR color format
        color = (255, 0, 0) if is_pinching else (0, 255, 255)  # Blue if pinching, Yellow if not
        
        # Draw the connection line
        cv2.line(frame, start_point, end_point, color, 2)

def calculate_finger_distances(hand_landmarks):
    """Calculate distances between thumb and other fingers"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    return {
        "index": np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2),
        "middle": np.sqrt((middle_tip.x - thumb_tip.x)**2 + (middle_tip.y - thumb_tip.y)**2),
        "ring": np.sqrt((ring_tip.x - thumb_tip.x)**2 + (ring_tip.y - thumb_tip.y)**2),
        "pinky": np.sqrt((pinky_tip.x - thumb_tip.x)**2 + (pinky_tip.y - thumb_tip.y)**2)
    }

def calculate_hand_midpoint(hand_landmarks):
    """Calculate the midpoint of the hand"""
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    midpointX = (index_base.x + ring_base.x + thumb_base.x + wrist.x) / 4
    midpointY = (index_base.y + ring_base.y + thumb_base.y + wrist.y) / 4
    
    return midpointX, midpointY

def map_to_screen_coordinates(x, y):
    """
    Map coordinates from the control area to full screen coordinates
    Uses the CONTROL_AREA_PERCENTAGE to determine the active control zone
    """
    # Ensure the point is within the control area
    x = max(control_area["x_min"], min(control_area["x_max"], x))
    y = max(control_area["y_min"], min(control_area["y_max"], y))
    
    # Map from control area to screen coordinates
    x_normalized = (x - control_area["x_min"]) / (control_area["x_max"] - control_area["x_min"])
    y_normalized = (y - control_area["y_min"]) / (control_area["y_max"] - control_area["y_min"])
    
    # Convert to screen coordinates
    screen_x = x_normalized * screenWidth
    screen_y = y_normalized * screenHeight
    
    return screen_x, screen_y

def process_right_hand(hand_landmarks, frame, frameWidth, frameHeight):
    global timer
    global scroll_enable
    global global_hand_state
    global two_hand_gesture
    
    """Handle right hand mouse control functions"""
    distances = calculate_finger_distances(hand_landmarks)
    midpointX, midpointY = mpc.get_hand_center(hand_landmarks)
    
    # Draw control area rectangle
    x1 = int(control_area["x_min"] * frameWidth)
    y1 = int(control_area["y_min"] * frameHeight)
    x2 = int(control_area["x_max"] * frameWidth)
    y2 = int(control_area["y_max"] * frameHeight)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Toggle hand enabled state
    if mpc.get_hand_gesture(hand_landmarks) == "FIST" and timer > 10:
        global_hand_state = not global_hand_state
        timer = 0
    
    if two_hand_gesture:
        return
    
    if global_hand_state:
        # Left click
        if distances["index"] < 0.045 and not right_hand_states["mouseDown"] and not right_hand_states["rightMouseDown"] and not right_hand_states["middleMouseDown"]:
            #pag.mouseDown()
            mouse.click(Button.left, 1)
            
        if distances["ring"] < 0.065 and not right_hand_states["mouseDown"] and not right_hand_states["rightMouseDown"] and not right_hand_states["middleMouseDown"]:
            mouse.press(Button.left)
            right_hand_states["mouseDown"] = True
        if distances["ring"] > 0.09 and right_hand_states["mouseDown"]:
            mouse.release(Button.left)
            right_hand_states["mouseDown"] = False

        # Right click
        if distances["middle"] < 0.065 and not right_hand_states["rightMouseDown"] and not right_hand_states["middleMouseDown"]:
            #pag.mouseDown(button="right")
            mouse.press(Button.right)
            right_hand_states["rightMouseDown"] = True
        if distances["middle"] > 0.09 and right_hand_states["rightMouseDown"]:
            #pag.mouseUp(button="right")
            mouse.release(Button.right)
            right_hand_states["rightMouseDown"] = False
        
        # Middle click
        if (distances["index"] < 0.045 and  
            distances["middle"] < 0.055 and 
            not right_hand_states["middleMouseDown"]):
            #pag.mouseUp()
            mouse.release(Button.left)
            right_hand_states["mouseDown"] = False
            #pag.mouseUp(button="right")
            mouse.release(Button.right)
            right_hand_states["rightMouseDown"] = False
            #pag.mouseDown(button="middle")
            mouse.press(Button.middle)
            right_hand_states["middleMouseDown"] = True
        if ((distances["index"] > 0.06 or  
            distances["middle"] > 0.08) and 
            right_hand_states["middleMouseDown"]):            
            pag.mouseUp(button="middle")
            mouse.release(Button.middle)
            right_hand_states["middleMouseDown"] = False

        if distances["pinky"] < 0.055 and timer > 6:
            scroll_enable = not scroll_enable
            timer = 0
                 
        # Draw cursor indicator
        cursor_x = int(midpointX * frameWidth)
        cursor_y = int(midpointY * frameHeight)
        if right_hand_states["mouseDown"] or right_hand_states["middleMouseDown"]:
            cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 50, 0), -1)
        else:
            cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 50, 0), 4)
        
        # Map hand position to screen coordinates
        screen_x, screen_y = map_to_screen_coordinates(midpointX, midpointY)
        #pag.moveTo(screen_x, screen_y)
        #pag.dragTo(screen_x, screen_y, mouseDownUp = False)
        mouse.position = (screen_x, screen_y)

def process_left_hand(hand_landmarks, frame, frameWidth, frameHeight):
    global global_hand_state
    global scroll_enable
    global timer2
    global two_hand_gesture

    if two_hand_gesture:
        return

    """Handle left hand functions"""
    distances = calculate_finger_distances(hand_landmarks)
    midpointX, midpointY = mpc.get_hand_center(hand_landmarks)
    
    # Toggle left hand enabled state
        
    if global_hand_state:
            
        # Draw different colored cursor for left hand
        cursor_x = int(midpointX * frameWidth)
        cursor_y = int(midpointY * frameHeight)
        cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), 4)    
    
        # Get hand direction
        direction, angle = mpc.detect_hand_direction(hand_landmarks)
        
        # # shift
        # if distances["middle"] < 0.065 and timer2 > 3:
        #     if left_hand_states["shift"] == True:
        #         pag.keyUp("shift")
        #         left_hand_states["shift"] = False
        #     else:
        #         pag.keyDown("shift")
        #         left_hand_states["shift"] = True
        #     timer2 = 0                
            
        
        # Draw direction indicator on frame
        cv2.putText(frame, 
                    f"Direction: {direction}", 
                    (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (200, 50, 0), 
                    2)
        
        # Draw angle indicator (optional, helpful for debugging)
        cv2.putText(frame, 
                    f"Angle: {angle:.1f}", 
                    (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (200, 50, 0), 
                    2)

        # show whether scroll is enabled
        cv2.putText(frame, 
                    str(scroll_enable), 
                    (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (200, 50, 0), 
                    2)
        
        # if left_hand_states["shift"]:
        #     cv2.putText(frame, 
        #                 "shift", 
        #                 (10, 190), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 
        #                 1, 
        #                 (200, 50, 0), 
        #                 2)
            
        
        
        # You can now use the direction for specific controls
        if global_hand_state:
            if scroll_enable:
                if direction == "RIGHT":
                    #pag.scroll(60)
                    mouse.scroll(0, 1)
                elif direction == "LEFT":
                    #pag.scroll(-60)
                    mouse.scroll(0, -1)

def process_two_handed_gestures(left_landmarks, right_landmarks):
    """
    Handles two-handed zoom gesture, but only when tracking is enabled.
    Think of this like needing to turn on your computer before you can use any programs.
    """
    global global_hand_state
    global previous_hand_distance
    global two_hand_gesture
    
    if not global_hand_state:
        return


    # Now we can process the gesture since tracking is enabled
    left_distances = calculate_finger_distances(left_landmarks)
    right_distances = calculate_finger_distances(right_landmarks)
    
    # Check if both hands are pinching
    if left_distances["index"] < 0.055 and right_distances["index"] < 0.055:
        two_hand_gesture = True
        #pag.mouseUp()
        mouse.release(Button.left)

        right_hand_states["mouseDown"] = False
        #pag.mouseUp(button="right")
        mouse.release(Button.right)
        right_hand_states["rightMouseDown"] = False
        #pag.mouseUp(button="middle")
        mouse.release(Button.middle)
        right_hand_states["middleMouseDown"] = False
        

        # Calculate distance between hands for zoom
        left_x, left_y = mpc.get_hand_center(left_landmarks)
        right_x, right_y = mpc.get_hand_center(right_landmarks)
        current_distance = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
        
        if previous_hand_distance == None:
            previous_hand_distance = current_distance
            amount_to_scroll = 0
        
        # Calculate and apply zoom
        distance_change = current_distance - previous_hand_distance
        amount_to_scroll = int(distance_change * 50)
        #pag.scroll(amount_to_scroll)
        mouse.scroll(0, amount_to_scroll)
        
        previous_hand_distance = current_distance
    
    elif left_distances["index"] > 0.1 or right_distances["index"] < 1:
        previous_hand_distance = None
        two_hand_gesture = False

while True:
    timer += 1
    timer2 += 1
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 2)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    frameHeight, frameWidth, _ = frame.shape

    if results.multi_hand_landmarks:
        # Sort hands into left and right
        left_hand = None
        right_hand = None
        
        # First pass: identify hands
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            if handedness == "Right":
                right_hand = hand_landmarks
            else:
                left_hand = hand_landmarks
            
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Check for two-handed gestures first
        if left_hand and right_hand:
            process_two_handed_gestures(left_hand, right_hand)
        
        # Process individual hands
        if right_hand:
            process_right_hand(right_hand, frame, frameWidth, frameHeight)
        if left_hand:
            process_left_hand(left_hand, frame, frameWidth, frameHeight)
        
        if not two_hand_gesture:
            draw_hand_connection(frame, left_hand, right_hand, False)
        else:
            draw_hand_connection(frame, left_hand, right_hand, True)

    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()