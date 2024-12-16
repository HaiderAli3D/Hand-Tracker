import time






class GestureState:
    def __init__(self):
        self.states = {
            'single_hand': {
                'right': {
                    'current_gesture': None,
                    'start_time': 0,
                    'duration': 0,
                    'previous_state': None
                },
                'left': {
                    'current_gesture': None,
                    'start_time': 0,
                    'duration': 0,
                    'previous_state': None
                }
            },
            'two_handed': {
                'current_gesture': None,
                'start_time': 0,
                'duration': 0,
                'previous_state': None,
                'gesture_data': {}  # For storing gesture-specific information
            }
        }
        self.transition_cooldown = 0.2  # Seconds to wait before allowing new gesture

    def start_gesture(self, gesture_type, is_two_handed=False, hand=None):
        """
        Start a gesture while maintaining tracking functionality.
        """
        current_time = time.time()
        
        if is_two_handed:
            self.states['two_handed']['current_gesture'] = gesture_type
            self.states['two_handed']['start_time'] = current_time
        else:
            if hand:
                self.states['single_hand'][hand]['current_gesture'] = gesture_type
                self.states['single_hand'][hand]['start_time'] = current_time

        # Important: We don't disable tracking, just change what actions are available
        self.states['tracking_enabled'] = True

    def end_gesture(self, gesture_type, hand=None):
        """
        Properly end a gesture and clean up its state.
        
        Args:
            gesture_type: Either 'two_handed' or 'single_hand'
            hand: If single_hand gesture, specify which hand ('left' or 'right')
        """
        current_time = time.time()
        
        if gesture_type == 'two_handed':
            # Store the previous state before clearing
            self.states['two_handed']['previous_state'] = {
                'gesture': self.states['two_handed']['current_gesture'],
                'duration': current_time - self.states['two_handed']['start_time']
            }
            # Reset the current state
            self.states['two_handed']['current_gesture'] = None
            self.states['two_handed']['start_time'] = 0
            self.states['two_handed']['gesture_data'] = {}
        
        elif hand:  # For single-hand gestures
            # Store the previous state
            self.states['single_hand'][hand]['previous_state'] = {
                'gesture': self.states['single_hand'][hand]['current_gesture'],
                'duration': current_time - self.states['single_hand'][hand]['start_time']
            }
            # Reset the current state
            self.states['single_hand'][hand]['current_gesture'] = None
            self.states['single_hand'][hand]['start_time'] = 0

    def end_all_single_hand_gestures(self):
        """Helper function to end all single hand gestures at once"""
        self.end_gesture('single_hand', 'left')
        self.end_gesture('single_hand', 'right')

    def resolve_gesture_conflicts(self):
        """
        Implement rules for gesture conflicts
        Think of this like a traffic light system - clear rules about who has right of way
        """
        current_time = time.time()
        
        # Rule 1: Two-handed gestures have highest priority
        if self.states['two_handed']['current_gesture']:
            return 'two_handed'
            
        # Rule 2: Don't start a new gesture during cooldown
        if (current_time - max(self.states['single_hand']['right']['start_time'],
                            self.states['single_hand']['left']['start_time']) 
            < self.transition_cooldown):
            return None
            
        # Rule 3: Prefer continuing existing gestures over starting new ones
        right_active = self.states['single_hand']['right']['current_gesture']
        left_active = self.states['single_hand']['left']['current_gesture']
        
        if right_active and not left_active:
            return ('single', 'right')
        elif left_active and not right_active:
            return ('single', 'left')
        
        # Rule 4: If both hands are doing gestures, prefer the longer-running one
        if right_active and left_active:
            right_duration = current_time - self.states['single_hand']['right']['start_time']
            left_duration = current_time - self.states['single_hand']['left']['start_time']
            return ('single', 'right' if right_duration > left_duration else 'left')