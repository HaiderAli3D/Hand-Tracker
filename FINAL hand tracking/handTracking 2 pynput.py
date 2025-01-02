import cv2
import mediapipe as mp
import pyautogui as pag
import numpy as np
import time
import mediapipe_cheats as mpc
import threading
import queue
from pynput.mouse import Button 
from pynput.mouse import  Controller as Controller
from pynput.keyboard import Controller as KeyController

mouse = Controller()
keyboard = KeyController()

pag.FAILSAFE = False

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.85,
    min_tracking_confidence = 0.95,
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
    "down": False,
    "ShiftMiddleMouseDown": False
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
scroll_enable = True
two_hand_gesture = False
previous_hand_distance = None
PrevY = None

timer = 0
timer2 = 0
timer3 = 0

last_s_time = time.time()
last_r_time = time.time()
last_g_time = time.time()

last_gesture_time = time.time()
last_scroll_time = time.time()
last_click_time = time.time()

movement_smoothing = 0.9
previous_positions = []
smoothing_window = 2

def get_elapsed_time(last_time):
    """
    Calculate elapsed time since a given timestamp in seconds
    
    Args:
        last_time: The timestamp to measure from
        
    Returns:
        float: Number of seconds elapsed
    """
    return time.time() - last_time

def draw_hand_connection(frame, left_landmarks, right_landmarks, is_pinching=False):
    if left_landmarks and right_landmarks:
        # Get center points of each hand
        left_x, left_y = mpc.get_hand_center(left_landmarks)
        right_x, right_y = mpc.get_hand_center(right_landmarks)
        
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
    
    finalX, finalY = smooth_position(screen_x, screen_y)
    
    return round(finalX,3), round(finalY, 3)
    return screen_x, screen_y

def process_right_hand(hand_landmarks, frame, frameWidth, frameHeight):
    global last_gesture_time
    global last_scroll_time
    global last_click_time
    global timer
    global scroll_enable
    global global_hand_state
    global two_hand_gesture
    global timer3
    
    """Handle right hand mouse control functions"""
    distances = calculate_finger_distances(hand_landmarks)
    midpointX, midpointY = mpc.get_hand_center(hand_landmarks)

    cv2.putText(frame, 
                f"R-Hand Cords: {round(midpointX, 2)}, {round(midpointY, 2)}", 
                (10, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (200, 50, 0), 
                2)

    current_gesture_fun = mpc.get_hand_gesture(hand_landmarks)
    cv2.putText(frame, current_gesture_fun, (10, 425), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 0), 2)



    # Draw control area rectangle
    x1 = int(control_area["x_min"] * frameWidth)
    y1 = int(control_area["y_min"] * frameHeight)
    x2 = int(control_area["x_max"] * frameWidth)
    y2 = int(control_area["y_max"] * frameHeight)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if mpc.detect_thumbs_gesture(hand_landmarks, "Right", 7) == "UP" and get_elapsed_time(last_gesture_time) > 1.5:
        global_hand_state = not global_hand_state
        last_gesture_time = time.time()
    
    if two_hand_gesture:
        return
    
    if scroll_enable and not global_hand_state:
        pinch_scroll(hand_landmarks)

    if not global_hand_state:
        return 

    # Shift Middle click Blender
    if (distances["index"] < 0.05 and  
        distances["middle"] < 0.07 and 
        distances["ring"] < 0.08 and
        distances["pinky"] < 0.08 and
        not right_hand_states["ShiftMiddleMouseDown"]):
        right_hand_states["down"] = True
        right_hand_states["middleMouseDown"] = False
        right_hand_states["ShiftMiddleMouseDown"] = True
        mouse.release(Button.middle)
        #mouse.release(Button.left)
        right_hand_states["mouseDown"] = False
        #mouse.release(Button.right)
        right_hand_states["rightMouseDown"] = False
        mouse.press(Button.x2)
    elif ((
        distances["index"] > 0.06 or  
        distances["middle"] > 0.08 or
        distances["ring"] > 0.08 or
        distances["pinky"] > 0.08
        ) and right_hand_states["ShiftMiddleMouseDown"]):            
        mouse.release(Button.x2)
        right_hand_states["ShiftMiddleMouseDown"] = False
        right_hand_states["down"] = False

    # Middle click
    if (distances["index"] < 0.04 and  
        distances["middle"] < 0.06 and 
        not right_hand_states["middleMouseDown"]):
        right_hand_states["down"] = True
        right_hand_states["middleMouseDown"] = True
        #mouse.release(Button.left)
        right_hand_states["mouseDown"] = False
        #mouse.release(Button.right)
        right_hand_states["rightMouseDown"] = False
        mouse.press(Button.middle)
    elif ((distances["index"] > 0.06 or  
        distances["middle"] > 0.08) and 
        right_hand_states["middleMouseDown"]):            
        mouse.release(Button.middle)
        right_hand_states["middleMouseDown"] = False
        right_hand_states["down"] = False


    # Left click
    if distances["index"] < 0.045 and not right_hand_states["down"] and get_elapsed_time(last_click_time) > 0.5:
        mouse.click(Button.left, 1)
        timer3 = 0
        last_click_time = time.time()
        
    if distances["ring"] < 0.065 and not right_hand_states["down"]:
        mouse.press(Button.left)
        right_hand_states["mouseDown"] = True
        right_hand_states["down"] = True
    elif distances["ring"] > 0.12 and right_hand_states["mouseDown"]:
        mouse.release(Button.left)
        right_hand_states["mouseDown"] = False
        right_hand_states["down"] = False

    # Right click
    if distances["middle"] < 0.04 and not right_hand_states["down"] and not right_hand_states["middleMouseDown"]:
        mouse.press(Button.right)
        right_hand_states["rightMouseDown"] = True
        right_hand_states["down"] = True
    elif distances["middle"] > 0.065 and right_hand_states["rightMouseDown"]:
        mouse.release(Button.right)
        right_hand_states["rightMouseDown"] = False
        right_hand_states["down"] = False

    if distances["pinky"] < 0.055 and get_elapsed_time(last_scroll_time) > 0.6:
        scroll_enable = not scroll_enable
        timer = 0
        last_scroll_time = time.time()
                
    # Draw cursor indicator
    cursor_x = int(midpointX * frameWidth)
    cursor_y = int(midpointY * frameHeight)

    if right_hand_states["mouseDown"] or right_hand_states["middleMouseDown"]:
        cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 50, 0), -1)
    else:
        cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 50, 0), 4)
    
    # Map hand position to screen coordinates
    screen_x, screen_y = map_to_screen_coordinates(midpointX, midpointY)

    mouse.position = (screen_x, screen_y)
    

def pinch_scroll(hand_landmarks):
    global PrevY

    distances = calculate_finger_distances(hand_landmarks)

    if (distances["index"] < 0.03 and  
        distances["middle"] < 0.05 and 
        distances["ring"] < 0.05 and
        distances["pinky"] > 0.13):
        X, Y =  mpc.get_hand_center(hand_landmarks)

        if PrevY == None:
            PrevY = Y
        
        currentDist = Y - PrevY

        amount_to_scroll = currentDist * 60

        mouse.scroll(0, round(amount_to_scroll, 2))

        PrevY = Y

    elif (
        distances["index"] > 0.06 or  
        distances["middle"] > 0.08 or
        distances["ring"] > 0.08 or
        distances["pinky"] < 0.08):

        PrevY = None

def process_left_hand(hand_landmarks, frame, frameWidth, frameHeight):
    global global_hand_state
    global scroll_enable
    global timer2
    global two_hand_gesture
    global keyboard
    global last_s_time
    global last_r_time
    global last_g_time

    if two_hand_gesture:
        return

    ## OLD ALWAYS SCROLL THUMBS 
    # if scroll_enable:
    #     if mpc.detect_thumbs_gesture(hand_landmarks, "Left", 55) == "UP":
    #         #print("UP")
    #         mouse.scroll(0, 0.5)
    #     elif mpc.detect_thumbs_gesture(hand_landmarks, "Left", 70) == "DOWN":
    #         #print("DOWN")
    #         mouse.scroll(0, -0.5)

    """Handle left hand functions"""
    distances = calculate_finger_distances(hand_landmarks)
    midpointX, midpointY = mpc.get_hand_center(hand_landmarks)
    
    direction, angle = mpc.detect_hand_direction(hand_landmarks) 

    # Draw direction indicator on frame
    cv2.putText(frame, f"Direction: {direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 0), 2)
    
    # Draw angle indicator (optional, helpful for debugging)
    cv2.putText(frame, f"Angle: {angle:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 0), 2)

    if not global_hand_state:
        return
            
    # Draw different colored cursor for left hand
    cursor_x = int(midpointX * frameWidth)
    cursor_y = int(midpointY * frameHeight)
    cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), 4)    

    # Get hand direction
    
    if distances["middle"] < 0.065 and get_elapsed_time(last_g_time) > 1.2:
        keyboard.press("g")
        keyboard.release("g")
        last_g_time = time.time()

    if distances["ring"] < 0.06 and get_elapsed_time(last_r_time) > 1.2:
        keyboard.press("r")
        keyboard.release("r")
        last_r_time = time.time()

    if distances["pinky"] < 0.065 and get_elapsed_time(last_s_time) > 1.2:
        keyboard.press("s")
        keyboard.release("s")
        last_s_time = time.time()


    # You can now use the direction for specific controls
    if scroll_enable:
        if direction == "RIGHT":
            mouse.scroll(0, 0.4)
        elif direction == "LEFT":
            mouse.scroll(0, -0.4)



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
        #mouse.release(Button.left)
        right_hand_states["mouseDown"] = False
        #mouse.release(Button.right)
        right_hand_states["rightMouseDown"] = False
        #mouse.release(Button.middle)
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
        amount_to_scroll = distance_change * 60
        mouse.scroll(0, amount_to_scroll)
        
        previous_hand_distance = current_distance
    
    elif left_distances["index"] > 0.08 or right_distances["index"] > 0.08:
        previous_hand_distance = None
        two_hand_gesture = False

def smooth_position(newX, newY):
    """
    this function will smooth out the hand tracking input before mapping it to the mouse.
    It does this by averaging the movement over the past few calculated frames. The exact amount and properties can be configured in the _init_

    basicaly I just work out the mean potition of my hand over the last few frames and return the result. It"s not that complex.
    """
    #add the most recent hand potition to a last 
    if smoothing_window == 0:
        return newX, newY
    
    new_position = {"x": newX, "y": newY}
    previous_positions.append(new_position)
    
    # checks if the length of the list is longer than the frame window set, if it is then it deletes the oldest item
    if len(previous_positions) > smoothing_window:
        previous_positions.pop(0)
    
    # created a new dicitonary with blank X and Y cords
    smoothed = {"x": 0, "y": 0}
    # I then add up all the X and Y coordinated from the previous positions
    for pos in previous_positions:
        smoothed["x"] += pos["x"]
        smoothed["y"] += pos["y"]
    
    # then I divide the sum by the number of items added
    smoothed["x"] /= len(previous_positions)
    smoothed["y"] /= len(previous_positions)
    
    #basicaly these last few lines just calculated the mean potion of my hand over the last few frames
    
    # these last lines just apply the movement_smoothing amount variable set at initialisation. This vairable just says how much of the smoothing to actualy apply.
    final_x = (smoothed["x"] * movement_smoothing + 
            new_position["x"] * (1 - movement_smoothing))
    final_y = (smoothed["y"] * movement_smoothing + 
            new_position["y"] * (1 - movement_smoothing))
    
    # return the smoothed coordinates
    return final_x, final_y


while True:
    timer += 1
    timer2 += 1
    timer3 += 1
    success, frame = cap.read()
    if not success:
        print("CANOT INTIATE CAMERA")
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
        
        # show whether scroll is enabled
        cv2.putText(frame, str(scroll_enable), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 0), 2)

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