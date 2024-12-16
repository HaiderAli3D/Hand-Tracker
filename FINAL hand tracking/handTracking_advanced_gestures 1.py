import cv2
import mediapipe as mp
import pyautogui as pag
import numpy as np
import time
import mediapipe_cheats as mpc
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from GestureStates import GestureState

""" 
I use openCV (cv2) to get the camera and display the feed and do some camera proccessing
mediapipe is the main module i'm using to do the handtracking - it was made by google
i use pyautogui to control the mouse and keyboard, it"s the itnerface between the hadn tracking and the input
nump is for maths
time is obvious
mediapipe_cheats is a set of functions I created with claude that simplifies the use of the mediapipe library
threading and queue are used for parraelel proccessing to speed up the program. - I removed some of this functionality so queue is no longer used
ThreadPoopExecutor is a more advances tool for managing threads that helps a ton when needing to manage a bunch of threads. 

Gesture states is code I"ve writen that"s just in another py file. It contains a data structure to hold the current gesture of each hand.
It also hold functions to manage hand gestures and resolve conflicts
"""



class HandTrackingSystem:
    def __init__(self):
        # here i just initialise the mediapipe hand tracking system and set it"s properties. Pretty self explanatory just read the documentation.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils

        #open up the camera video feed using openCV
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        self.screenWidth, self.screenHeight = pag.size()
        
        # set the boundaries on the camera for screen control
        self.CONTROL_AREA_PERCENTAGE = 70
        margin_percentage = (100 - self.CONTROL_AREA_PERCENTAGE) / 2
        margin = margin_percentage / 100
        self.control_area = {
            "x_min": margin,
            "x_max": 1 - margin,
            "y_min": margin,
            "y_max": 1 - margin
        }

        # set the variables for the state of each hand. Done using dictionaries to make things easier too read and contain
        self.right_hand_states = {
            "mouseDown": False,
            "rightMouseDown": False,
            "middleMouseDown": False,
        }
        self.left_hand_states = {}
        
        self.global_hand_state = False # is the hand control enabled?
        self.scroll_enable = False # is scroll enabled?
        self.timer = 0 # random bugy timer lol i'll imrpove this later
        self.clickTimer = 0
        
        self.draw_debug_info = True # should draw debug info?

        # intitiate the ThreadPoolExecutor and set it"s parameters
        self.thread_pool = ThreadPoolExecutor(max_workers=2) 
        self.processing_lock = threading.Lock()

        # set parameters for the motion control. Sets things like motion smoothing and relative control.
        self.motion_config = {
            "use_smoothing": True,      # Enable/disable motion smoothing
            "use_relative": False,       # Enable/disable relative motion
            "smoothing_amount": 0.9,    # How much smoothing to apply (0-1)
            "smoothing_window": 3       # How many positions to consider for smoothing
        }
        # motion smoothing makes the hand tracking and mouse control less jittery  but increases latency
        # relative motion swaps the hand-mouse input from directly mapping a camera co-ordinate to screen co-ordinate to instead mapping the movement of our hand to the movement of the mouse.

        # configures the variables for the motion smoothing and relative motion
        self.movement_smoothing = self.motion_config["smoothing_amount"]
        self.previous_positions = []
        self.smoothing_window = self.motion_config["smoothing_window"]
        self.use_relative_motion = self.motion_config["use_relative"]
        self.previous_hand_position = None
        self.cursor_position = {"x": self.screenWidth/2, "y": self.screenHeight/2}

        self.use_relative_motion = True
        self.previous_hand_position = None
        self.cursor_position = {"x": self.screenWidth/2, "y": self.screenHeight/2}  
        
        # more mouse control parameters if needed
        self.boost_factor = 1.2
        self.base_scaleX = 1400
        self.base_scaleY = 1800  
        
        self.gesture_state = GestureState()
        

    def set_motion_mode(self, use_smoothing=True, use_relative=False):
        
        """ 
        This function will allow us to change the motion control mode while the code is running and will correctly configure the motion control
        By default I"ve set motion smoothing to True and use relative to False
        """
        
        self.motion_config["use_smoothing"] = use_smoothing
        self.motion_config["use_relative"] = use_relative
        
        self.use_relative_motion = use_relative
        
        self.previous_hand_position = None
        self.cursor_position = {"x": self.screenWidth/2, "y": self.screenHeight/2}
        self.previous_positions = []

    def smooth_position(self, new_position):
        """
        this function will smooth out the hand tracking input before mapping it to the mouse.
        It does this by averaging the movement over the past few calculated frames. The exact amount and properties can be configured in the _init_

        basicaly I just work out the mean potition of my hand over the last few frames and return the result. It"s not that complex.
        """
        #add the most recent hand potition to a last 
        self.previous_positions.append(new_position)
        
        # checks if the length of the list is longer than the frame window set, if it is then it deletes the oldest item
        if len(self.previous_positions) > self.smoothing_window:
            self.previous_positions.pop(0)
        
        # created a new dicitonary with blank X and Y cords
        smoothed = {"x": 0, "y": 0}
        # I then add up all the X and Y coordinated from the previous positions
        for pos in self.previous_positions:
            smoothed["x"] += pos["x"]
            smoothed["y"] += pos["y"]
        
        # then I divide the sum by the number of items added
        smoothed["x"] /= len(self.previous_positions)
        smoothed["y"] /= len(self.previous_positions)
        
        #basicaly these last few lines just calculated the mean potion of my hand over the last few frames
        
        # these last lines just apply the movement_smoothing amount variable set at initialisation. This vairable just says how much of the smoothing to actualy apply.
        final_x = (smoothed["x"] * self.movement_smoothing + 
                new_position["x"] * (1 - self.movement_smoothing))
        final_y = (smoothed["y"] * self.movement_smoothing + 
                new_position["y"] * (1 - self.movement_smoothing))
        
        # return the smoothed coordinates
        return {"x": final_x, "y": final_y}

    def process_relative_motion(self, hand_position):
        """
        This is an interesting alternative way to mapping hte hand mvoement to mouse movement.
        
        Normaly the hand movement is mapped to mouse movement by getting hte co-ordinate of the hand on the camera frame and mapping that to the equivalent coordinate on the screen. I called this direct mapping.
        Direct mapping is an easy and simple way to map movement that"s intuitive to use and quick to compute. However it has some inherent flaws that can make it anoying to use for everyday tasks.
        This is why I"ve added the ability to use relative motion.
        Relative motion instead tracks the movement of my hand to movement of the mouse. So it doesn"t actualy matter where on the camera frame the movement happends, the same movement will move the mouse in the same way anywhere.
        This can sometimes make the mouse easier to control, however sometimes it can be anoying and it takes extra compute power. Hence why I"ts still disables by default. Still cool tho.
        
        This function is what proccesses the relative motion. It takes an input of the hand position and will output the new cursor position.
        """
        
        # this is the edge case of it being the first frame. The function needs to know the delta between the last and most recent frame, but  there is no last frame because this is the first frame.
        # So here i just say if it"s the first frame then set the last frame as the current frame and skip proccessing for this frame :)
        if self.previous_hand_position is None:
            self.previous_hand_position = hand_position
            return self.cursor_position
        
        # here i get the delta between the previous hand position and current hand position. I calculate this delta individualy for the X and Y 
        dx = hand_position["x"] - self.previous_hand_position["x"]
        dy = hand_position["y"] - self.previous_hand_position["y"]
        
        # here I calculate the center of the defined control area that was set at initialisation
        control_center_x = (self.control_area["x_max"] + self.control_area["x_min"]) / 2
        control_center_y = (self.control_area["y_max"] + self.control_area["y_min"]) / 2
        
        #this is a really cool thing that basicaly calculates how far on a scale of 0-1 the hand is from the center of the camera frame. This is really usefull data.
        edge_factor_x = abs(hand_position["x"] - control_center_x) / (self.control_area["x_max"] - control_center_x)
        edge_factor_y = abs(hand_position["y"] - control_center_y) / (self.control_area["y_max"] - control_center_y)
        
        # this line creates a new variable used later to boost the mouse movement if the hand is close to the edge of the frame.
        # We do this to ensure that the edge of the computer screen can always be reached before the hand moves out of the tracking range.
        edge_multiplier_x = 1.0 + (edge_factor_x * self.boost_factor)
        edge_multiplier_y = 1.0 + (edge_factor_y * self.boost_factor)
        
        #this scales the delta acording to the screen dimensions. So the code works regardless of screen size, resolution and aspect ratio.
        x_scale = self.base_scaleX * (self.screenWidth / self.screenHeight) * edge_multiplier_x
        y_scale = self.base_scaleY * (self.screenWidth / self.screenHeight) * edge_multiplier_y
        
        # times the deltaX and deltaY by the scale
        dx *= x_scale
        dy *= y_scale
        
        # i just then have to add the deltaX and deltaY to the mouse position x and y
        new_x = self.cursor_position["x"] + dx
        new_y = self.cursor_position["y"] + dy
        
        # ensure the mouse position isn"t out of bounds
        new_x = max(0, min(self.screenWidth, new_x))
        new_y = max(0, min(self.screenHeight, new_y))
        
        #update the internal cursor position and previous hand position variables
        self.cursor_position = {"x": new_x, "y": new_y}
        self.previous_hand_position = hand_position
        
        #return the new cursor position
        return self.cursor_position

    def calculate_finger_distances(self, hand_landmarks):
        """
        This is one of the most usefull functions to map hand gestures and movement to input.
        It will calculate the distance between each finger tip and the tip of the thumb.
        This easily allows me to map actions to pinching gestures.
        """
        
        # define what each tip is on the hand tracker - refer to the mediapipe documentation
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # a little messy code here but basicaly use pythag to calculate the distance between each tip and thumb then return it as a dictionary.
        return {
            "index": np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2),
            "middle": np.sqrt((middle_tip.x - thumb_tip.x)**2 + (middle_tip.y - thumb_tip.y)**2),
            "ring": np.sqrt((ring_tip.x - thumb_tip.x)**2 + (ring_tip.y - thumb_tip.y)**2),
            "pinky": np.sqrt((pinky_tip.x - thumb_tip.x)**2 + (pinky_tip.y - thumb_tip.y)**2)
        }

    def map_to_screen_coordinates(self, x, y):
        """
        This function maps the hand movement to mouse movement
        It takes into acount whether smoothing and relative motion are enabled/disabled
        """
        
        # makes sure the x and y cords are valid and not outside bounds, if htey are it just snaps them to the closest allowed location.
        x = max(self.control_area["x_min"], min(self.control_area["x_max"], x))
        y = max(self.control_area["y_min"], min(self.control_area["y_max"], y))
        position = {"x": x, "y": y}
        
        # if motion smoothing is enables then call the function to proccess it and redifine the position.
        if self.motion_config["use_smoothing"]:
            position = self.smooth_position(position)
        
        # if relative motion is enabled call the function to proccess it and set the mouse screen position to be the result. Then return the X and Y of this result as the new mouse cords.
        if self.motion_config["use_relative"]:
            screen_pos = self.process_relative_motion(position)
            return (screen_pos["x"], screen_pos["y"])
        # if use relative motion is not enables then don"t call the function and use the old simpler way of mappging to the mouse cords.
        else:
            # think of the normalized varibles as just a percentage along the screen
            x_normalized = (position["x"] - self.control_area["x_min"]) / (self.control_area["x_max"] - self.control_area["x_min"])
            y_normalized = (position["y"] - self.control_area["y_min"]) / (self.control_area["y_max"] - self.control_area["y_min"])
            # then return that same percentage but across the display now
            return (x_normalized * self.screenWidth, y_normalized * self.screenHeight)

    def draw_hand_info(self, frame, direction, angle):
        """
        just a function for debuging. Will draw some information like hand angle and direction onto the camera video feed.
        """
        cv2.putText(frame, 
                    f"Direction: {direction}", 
                    (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 50, 0), 
                    3)
        
        cv2.putText(frame, 
                    f"Angle: {angle:.1f}", 
                    (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 50, 0), 
                    3)
                    
        cv2.putText(frame, 
                    str(self.scroll_enable), 
                    (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 50, 0), 
                    3)

    def process_mouse_actions(self, distances, midpointX, midpointY, frame, frameWidth, frameHeight):
        """
        This is one of the most important functions. It is what actualy detects the hand gestures and gives the mouse input.
        I use the result of the distances function alot here.
        This function requires inputs of:
        the hand distances to detect gestures:
        the calculated hand center X and Y to draw the virtual circle pointer
        the video frame
        the width and height of the frame 
        """
        
        # this will detect a pinch with the ring finger. if it"s pinched then put down the left mouse button. When it"s unpinched bring the mouse button up. A quick pinch is the equivalent of a click.
        # it detects a pinch by checking seeing if the distance between the thumb tip and finger tip is bellow a certain threshold. If it is then it puts the mouse button down.
        # once it"s above a certain threshold I release the mouse button
        if distances["ring"] < 0.06 and not self.right_hand_states["mouseDown"] and not self.right_hand_states["middleMouseDown"]:
            pag.mouseDown()
            self.right_hand_states["mouseDown"] = True
            self.gesture_state.start_gesture("left_mouse_down")
        elif distances["ring"] > 0.08 and self.right_hand_states["mouseDown"]:
            pag.mouseUp()
            self.right_hand_states["mouseDown"] = False
            self.gesture_state.end_gesture("left_mouse_down")

        # use index finger to click
        if distances["index"] < 0.06 and not self.right_hand_states["mouseDown"] and not self.right_hand_states["middleMouseDown"] and not self.right_hand_states["rightMouseDown"] and self.clickTimer > 3:
        #if distances["index"] < 0.06 and self.gesture_state.states["single_hand"]["right"]["current_gesture"] == None and self.clickTimer < 3:
            pag.leftClick()
            self.clickTimer = 0
            

        # same as before, detect a pinch between middle and thumb, if it"s pinched then use the right mouse button.
        if distances["middle"] < 0.08 and not self.right_hand_states["rightMouseDown"] and not self.right_hand_states["middleMouseDown"]:
            pag.mouseDown(button="right")
            self.right_hand_states["rightMouseDown"] = True
            self.gesture_state.start_gesture("right_mouse_down")
        elif distances["middle"] > 0.8 and self.right_hand_states["rightMouseDown"]:
            pag.mouseUp(button="right")
            self.right_hand_states["rightMouseDown"] = False
            self.gesture_state.end_gesture("right_mouse_down")
        
        # this is a little more complicated. I wanted to hold my index, middle and thumb together to click with the middle mouse button. THis is usefull in 3D software to navigate.
        # All I do is check if the distance between both the index finger and middle finger and thumb is bellow a certain threshold. If it is then I make sure all other mouse buttons are released and put down the middle mouse button.
        if (distances["index"] < 0.1 and  
            distances["middle"] < 0.075 and 
            not self.right_hand_states["middleMouseDown"]):
            pag.mouseUp()
            self.right_hand_states["mouseDown"] = False
            pag.mouseUp(button="right")
            self.right_hand_states["rightMouseDown"] = False
            pag.mouseDown(button="middle")
            self.right_hand_states["middleMouseDown"] = True
            self.gesture_state.start_gesture("middle_mouse_down")
        # then release the middle mouse button once the fingers are raised again
        elif ((distances["index"] > 0.1 or  
            distances["middle"] > 0.1) and 
            self.right_hand_states["middleMouseDown"]):            
            pag.mouseUp(button="middle")
            self.right_hand_states["middleMouseDown"] = False
            self.gesture_state.end_gesture("middle_mouse_down")

        # detects a pinch with the pinky. If pinched it toggles the ability to scroll with the other hand.
        if distances["pinky"] < 0.055 and self.timer > 2:
            self.scroll_enable = not self.scroll_enable
            self.timer = 0
                 
        cursor_x = int(midpointX * frameWidth)
        cursor_y = int(midpointY * frameHeight)

        # draws the virtual cursor
        if self.draw_debug_info:
            if self.right_hand_states["mouseDown"] or self.right_hand_states["middleMouseDown"]:
                cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 50, 0), -1)
            else:
                cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 50, 0), 4)
        
        # calls the function to map the hand movement to mouse input and then actualy moves the mouse (finaly lol)
        screen_x, screen_y = self.map_to_screen_coordinates(midpointX, midpointY)
        pag.dragTo(screen_x, screen_y, mouseDownUp=False)

    def process_scroll_actions(self, direction):
        """
        pretty self explanatory just asks for a mysterious direction parameter and if it"s right scroll up if it"s left scroll down :)
        """
        if direction == "RIGHT":
            pag.scroll(60)
        elif direction == "LEFT":
            pag.scroll(-60)

    def process_hand_thread(self, hand_data):
        # this is the functuon that gets the threads to handle the right and left hand seperately. Massively improves performance.
        hand_landmarks, handedness, frame, frameWidth, frameHeight = hand_data
        
        if handedness == "Right":
            self.process_right_hand(hand_landmarks, frame, frameWidth, frameHeight)
        else:
            self.process_left_hand(hand_landmarks, frame, frameWidth, frameHeight)
        
        return frame

    def process_right_hand(self, hand_landmarks, frame, frameWidth, frameHeight):
        """
        Process right hand gestures with better state management.
        We want to maintain tracking regardless of gesture state.
        """
        with self.processing_lock:
            distances = self.calculate_finger_distances(hand_landmarks)
            midpointX, midpointY = mpc.get_hand_center(hand_landmarks)

            # First, always handle mouse movement if hand control is enabled
            if self.global_hand_state:
                # This should happen regardless of gesture state
                screen_x, screen_y = self.map_to_screen_coordinates(midpointX, midpointY)
                pag.dragTo(screen_x, screen_y, mouseDownUp=False)
                
                # Now handle gestures and additional mouse actions
                current_gesture = (self.gesture_state.states['two_handed']['current_gesture'] or 
                                self.gesture_state.states['single_hand']['right']['current_gesture'])
                
                if not current_gesture:  # Only process additional mouse actions if no gesture is active
                    self.process_mouse_actions(distances, midpointX, midpointY, frame, frameWidth, frameHeight)

            # Check for fist gesture to toggle tracking
            if mpc.get_hand_gesture(hand_landmarks) == "FIST" and self.timer > 8:
                self.global_hand_state = not self.global_hand_state
                self.timer = 0

    def process_left_hand(self, hand_landmarks, frame, frameWidth, frameHeight):
        """
        same as before, just a function to proccess the left hand
        """
        with self.processing_lock:
            if self.global_hand_state:
                # calculate important data
                midpointX, midpointY = mpc.get_hand_center(hand_landmarks)
                
                cursor_x = int(midpointX * frameWidth)
                cursor_y = int(midpointY * frameHeight)
                
                # draw debug stuff
                if self.draw_debug_info:
                    cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), 4)
                
                # calculate directions using an mediapipe_cheats function
                direction, angle = mpc.detect_hand_direction(hand_landmarks)
                self.draw_hand_info(frame, direction, angle)
                
                # if the scrolling is enables then call the function to proccess the scroll action
                if self.scroll_enable:
                    self.process_scroll_actions(direction)

    def process_two_handed_gestures(self, left_landmarks, right_landmarks, frame, frame_width, frame_height):
        """Process all gestures that require both hands to work together."""
        processed_frame = frame.copy()
                
        # Step 1: Detect the gesture
        left_distances = self.calculate_finger_distances(left_landmarks)
        right_distances = self.calculate_finger_distances(right_landmarks)
        hand_distance = mpc.calculate_hand_distance(left_landmarks, right_landmarks)
        
        # Step 2: Check for gesture transitions
        if left_distances["index"] < 0.08 and right_distances["index"] < 0.08:
            if self.gesture_state.states["two_handed"]["current_gesture"] != "two_hand_scroll":
                self.gesture_state.start_gesture("two_hand_scroll", True)
        else:
            if self.gesture_state.states["two_handed"]["current_gesture"] == "two_hand_scroll":
                self.gesture_state.end_gesture("two_handed")
        
        # Step 3: Process active gestures
        if self.gesture_state.states["two_handed"]["current_gesture"] == "two_hand_scroll":
            prev_distance = self.gesture_state.states["two_handed"]["gesture_data"].get("previous_distance")
            
            if prev_distance is not None:
                distance_change = hand_distance - prev_distance
                if abs(distance_change) > 0.01:
                    scroll_amount = int(distance_change * 1000)
                    pag.scroll(scroll_amount)
            
            self.gesture_state.states["two_handed"]["gesture_data"]["previous_distance"] = hand_distance
        
        # Always return the processed frame
        return processed_frame

    # a function to cleanup the threads and close the camera feed window after the program ends.
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.thread_pool.shutdown()

    def run(self):
        timer2 = 0
        while True:
            self.timer += 1
            self.clickTimer += 1
            
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 2)
            results = mpc.process_frame(self.hands, frame)
            frameHeight, frameWidth, _ = frame.shape

            # Initialize hand landmarks outside the loop
            left_landmarks = None
            right_landmarks = None

            if results.multi_hand_landmarks:
                # First pass: Sort hands into left and right
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = mpc.get_hand_type(results.multi_handedness[idx])
                    if handedness == "Right":
                        right_landmarks = hand_landmarks
                    else:
                        left_landmarks = hand_landmarks
                    mpc.draw_landmarks(frame, hand_landmarks)

                # Check for two-handed gestures BEFORE starting individual processing
                if left_landmarks and right_landmarks:
                    # Process two-handed gestures
                    if self.draw_debug_info:
                        cv2.putText(frame, "Two hands detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Make sure we always have a valid frame
                    processed_frame = self.process_two_handed_gestures(
                        left_landmarks,
                        right_landmarks,
                        frame,
                        frameWidth,
                        frameHeight
                    )
                    if processed_frame is not None:
                        frame = processed_frame

                # Then process individual hands if needed
                futures = []
                if right_landmarks and not self.gesture_state.states["two_handed"]["current_gesture"]:
                    future = self.thread_pool.submit(
                        self.process_hand_thread,
                        (right_landmarks, "Right", frame.copy(), frameWidth, frameHeight)
                    )
                    futures.append(future)
                    
                if left_landmarks and not self.gesture_state.states["two_handed"]["current_gesture"]:
                    future = self.thread_pool.submit(
                        self.process_hand_thread,
                        (left_landmarks, "Left", frame.copy(), frameWidth, frameHeight)
                    )
                    futures.append(future)

                for future in futures:
                    processed_frame = future.result()
                    frame = cv2.addWeighted(frame, 0.5, processed_frame, 0.5, 0)

            # Draw gesture state debug info if enabled
            # if self.draw_debug_info:
            #     self.draw_gesture_state_info(frame)
            if timer2 > 15:
                print("")
                print("")
                print("")
                print("")
                print("")
                print("")
                print("Right Hand Gesture = " + str(self.gesture_state.states["single_hand"]["right"]["current_gesture"]))
                print("Left Hand Gesture = " + str(self.gesture_state.states["single_hand"]["left"]["current_gesture"]))
                print("Both Hand Gesture = " + str(self.gesture_state.states["two_handed"]["current_gesture"]))
                timer2 = 0
            
            timer2 = timer2 + 1
            
            
            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cleanup()
               
if __name__ == "__main__":
    # create an object of the class and run it :)
    pag.FAILSAFE = False
    tracker = HandTrackingSystem()
    tracker.run()