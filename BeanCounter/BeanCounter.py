import pyautogui
import time
import numpy as np
import cv2
import sys
import platform
import threading
import mss

# Configuration variables
# You may need to adjust these values based on your screen and game window
GAME_REGION = None  # Will be set by calibration
START_KEY = 'f8'    # Key to start/stop the bot
PAUSE_KEY = 'f10'   # Key to pause/unpause the bot
DEBUG_KEY = 'd'     # Key to toggle debug mode
QUIT_KEY = 'q'      # Key to quit program
DETECTION_SCALE = 4 # For speed

# Color detection thresholds (BGR format for OpenCV)
# These may need adjustment based on the game's colors on your screen
BEAN_BAG_COLOR_LOWER = np.array([100, 150, 190])  # Brown bean bags
BEAN_BAG_COLOR_UPPER = np.array([110, 160, 200])

FISH_COLOR_LOWER = np.array([80, 200, 240])      # Yellow fish
FISH_COLOR_UPPER = np.array([90, 210, 250])

ANVIL_COLOR_LOWER = np.array([45, 45, 45])         # Black anvils
ANVIL_COLOR_UPPER = np.array([55, 55, 55])

POT_COLOR_LOWER = np.array([195, 150, 235])     # Pink flower pots
POT_COLOR_UPPER = np.array([205, 160, 245])

ONEUP_COLOR_LOWER = np.array([140, 95, 35])     # Oneup
ONEUP_COLOR_UPPER = np.array([150, 105, 45])

EARNINGS_SCREEN_LOWER = np.array([195, 120, 50]) # Earnings screen (For resetting)
EARNINGS_SCREEN_UPPER = np.array([205, 130, 60])

# Global variables
running = False
paused = False
debug_mode = False
exit_program = False

def cross_platform_key_listener():
    """Platform-independent key listener implementation"""
    try:
        # Try to use keyboard library first (works well on Windows)
        import keyboard
        
        # Set up key handlers with keyboard library
        keyboard.on_press_key(START_KEY, lambda _: toggle_running())
        keyboard.on_press_key(PAUSE_KEY, lambda _: toggle_pause())
        keyboard.on_press_key(DEBUG_KEY, lambda _: toggle_debug())
        keyboard.on_press_key(QUIT_KEY, lambda _: trigger_exit())
        
        print(f"Using keyboard library for key detection")
        
        # This will keep running until program ends
        while not exit_program:
            time.sleep(0.1)
            
    except (ImportError, ValueError, AttributeError) as e:
        # Fallback to pynput for macOS and Linux
        print(f"Keyboard library failed: {e}")
        print("Falling back to pynput for key detection")
        
        try:
            from pynput import keyboard
            
            def on_press(key):
                try:
                    # Handle regular keys
                    if hasattr(key, 'char') and key.char:
                        if key.char.lower() == QUIT_KEY:
                            trigger_exit()
                        elif key.char.lower() == DEBUG_KEY:
                            toggle_debug()
                    # Handle function keys
                    elif hasattr(key, 'name'):
                        if key.name == START_KEY:
                            toggle_running()
                        elif key.name == PAUSE_KEY:
                            toggle_pause()
                except AttributeError:
                    # Special keys like function keys
                    key_name = str(key).replace('Key.', '')
                    if key_name == START_KEY:
                        toggle_running()
                    elif key_name == PAUSE_KEY:
                        toggle_pause()
                    elif key_name == 'q':
                        trigger_exit()
            
            # Start listener
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
            
            # Keep running until program ends
            while not exit_program:
                time.sleep(0.1)
                
            listener.stop()
            
        except ImportError:
            print("Neither keyboard nor pynput library available.")
            print("Please install one of them: pip install keyboard pynput")
            sys.exit(1)

def trigger_exit():
    """Signal the program to exit"""
    global exit_program
    exit_program = True
    print("Exiting program...")

def calibrate_game_region():
    """Ask user to specify the game region by clicking on the top-left and bottom-right corners."""
    print("Please position your mouse at the top-left corner of the game area and press Enter...")
    input()
    top_left = pyautogui.position()
    print(f"Top-left position captured: {top_left}")
    
    print("Now position your mouse at the bottom-right corner of the game area and press Enter...")
    input()
    bottom_right = pyautogui.position()
    print(f"Bottom-right position captured: {bottom_right}")
    
    region = (top_left.x, top_left.y, bottom_right.x - top_left.x, bottom_right.y - top_left.y)
    print(f"Game region set to: {region}")
    return region

def detect_objects(frame):
    """Detect bean bags, fish, anvils, and flower pots in the current frame."""
    # Create masks for each object type
    bean_bag_mask = cv2.inRange(frame, BEAN_BAG_COLOR_LOWER, BEAN_BAG_COLOR_UPPER)
    fish_mask = cv2.inRange(frame, FISH_COLOR_LOWER, FISH_COLOR_UPPER)
    anvil_mask = cv2.inRange(frame, ANVIL_COLOR_LOWER, ANVIL_COLOR_UPPER)
    pot_mask = cv2.inRange(frame, POT_COLOR_LOWER, POT_COLOR_UPPER)
    oneup_mask = cv2.inRange(frame, ONEUP_COLOR_LOWER, ONEUP_COLOR_UPPER)
    earnings_mask = cv2.inRange(frame, EARNINGS_SCREEN_LOWER, EARNINGS_SCREEN_UPPER)
    
    # Find contours for each object type
    bean_bags = find_objects(bean_bag_mask)
    fishes = find_objects(fish_mask)
    anvils = find_objects(anvil_mask)
    pots = find_objects(pot_mask)
    oneups = find_objects(oneup_mask)
    earnings_mask = find_objects(earnings_mask)

    if len(earnings_mask) > 0:
        if debug_mode: print("Restarting game")
        restart_game_sequence()
        time.sleep(0.1)
    
    # Draw contours if debug mode is on
    if debug_mode:
        debug_frame = frame.copy()
        cv2.drawContours(debug_frame, bean_bags, -1, (0, 255, 0), 2)
        cv2.drawContours(debug_frame, fishes, -1, (0, 255, 255), 2)
        cv2.drawContours(debug_frame, anvils, -1, (0, 0, 255), 2)
        cv2.drawContours(debug_frame, pots, -1, (255, 0, 255), 2)
        cv2.drawContours(debug_frame, oneups, -1, (255, 255, 0), 2)
        cv2.imshow('Debug View', debug_frame)
        cv2.waitKey(1)
    
    return bean_bags, fishes, anvils, pots, oneups

def find_objects(mask):
    """
    Find objects based on average position of pixels in mask.
    Returns a list containing a single tuple with (x, y, w, h) if objects are found,
    otherwise returns an empty list.
    """
    # Find all non-zero pixels (matching the color mask)
    non_zero_pixels = cv2.findNonZero(mask)
    
    # If no pixels match, return empty list
    if non_zero_pixels is None or len(non_zero_pixels) < 25:  # Minimum pixel threshold
        return []
    
    # Calculate the mean of matched pixels
    non_zero_pixels = non_zero_pixels.reshape(-1, 2)
    mean_x, mean_y = np.mean(non_zero_pixels, axis=0)
    
    # Create a bounding box around the mean position
    # This simulates the (x, y, w, h) format of cv2.boundingRect
    box_width = box_height = 10  # Fixed size for simplicity and speed
    x = int(mean_x - box_width / 2)
    y = int(mean_y - box_height / 2)
    
    # Create a "fake contour" that's compatible with the existing code
    # It's just a rectangle centered at the mean position
    contour = np.array([
        [[x, y]],
        [[x + box_width, y]],
        [[x + box_width, y + box_height]],
        [[x, y + box_height]]
    ], dtype=np.int32)
    
    return [contour]

def determine_action(bean_bags, fishes, anvils, pots, oneups, width): # ['left'|'middle'|'right', hazard:True/False]
    """Determine the best action based on detected objects."""
    
    # If there is an avil or pot, always go to the left. If there is a fish, always go to the middle.
    # Otherwise, if there is a bean bag on the left of the screen, go to the left. If there is a bean bag in the middle, go to the middle. If there is a bean bag to the right of the screen, go to the right.
    
    # Define the three lane regions
    left_region_righthand = width * 0.3
    middle_region_righthand = width * 0.7

    # Check for fish (always go to the middle to catch fish)
    if len(fishes) > 0:
        if debug_mode: print('Detected Fish')
        
        if len(pots) > 0: # Pots have not been checked yet, so we don't know if it's safe
            return ('right', True)
        else:
            return ('middle', True)

    if len(pots) > 0:
        if debug_mode: print('Detected Pot')
        return ('left', True)
    
    if len(anvils) > 0:
        if debug_mode: print('Detected Anvil')
        return ('left', True)
    
    # join oneups and bean bags
    desireables = bean_bags + oneups
    
    # If no hazards, collect bean bags or oneups based on their position
    if len(desireables) > 0:
        # Find the closest bean bag (lowest y-value)
        closest_bean_bag = None
        min_y = float('inf')
        
        for contour in bean_bags:
            x, y, w, h = cv2.boundingRect(contour)
            if y < min_y:
                min_y = y
                closest_bean_bag = (x, y, w, h)
        
        # If a closest bean bag was found, move toward it
        if closest_bean_bag:
            x, y, w, h = closest_bean_bag
            center_x = (x + w // 2) # scale back up
        
            # Determine which lane the bean bag is in
            if center_x < left_region_righthand:
                if debug_mode: print('Detected lefthand bean bag')
                return ('left', False)
            elif center_x < middle_region_righthand:
                if debug_mode: print('Detected middle bean bag')
                return ('middle', False)
            else:
                if debug_mode: print('Detected righthand bean bag')
                return ('right', False)
    
    # Default action if nothing is detected
    if debug_mode: print('Nothing Detected')
    return ('left', False)

def move_penguin(action, hazard, game_region):
    """Move the penguin based on the determined action."""
    # Calculate the three lane positions
    left_x = game_region[0] + game_region[2] * 0.25
    middle_x = game_region[0] + game_region[2] * 0.5
    right_x = game_region[0] + game_region[2] * 0.75
    y_position = game_region[1] + game_region[3] * 0.65  # Position the penguin near the bottom
    
    def deposit_bags():
        pyautogui.moveTo(left_x, y_position)
        pyautogui.click(clicks=4)
    
    # Move to the appropriate lane
    if action == 'left':
        deposit_bags()
    elif action == 'middle':
        if not hazard: deposit_bags()
        pyautogui.moveTo(middle_x, y_position)
    elif action == 'right':
        if not hazard: deposit_bags()
        pyautogui.moveTo(right_x, y_position)

def toggle_running():
    """Toggle the running state of the bot."""
    global running
    running = not running
    print(f"Bot {'started' if running else 'stopped'}")

def toggle_pause():
    """Toggle the paused state of the bot."""
    global paused
    paused = not paused
    print(f"Bot {'paused' if paused else 'resumed'}")

def toggle_debug():
    """Toggle debug mode to show object detection visualization."""
    global debug_mode
    debug_mode = not debug_mode
    print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")

def restart_game_sequence():
    game_region = GAME_REGION
    ExitButtonPos = (573/802, 200/502)
    CafeCoffeeBagsPos = (825/1215, 511/763)
    ConfirmPlayPos = (525/1215, 356/763)
    StartGamePos = (1006/1215, 580/763)
    
    #Exit Game
    pyautogui.moveTo(game_region[0] + game_region[2] * ExitButtonPos[0], game_region[1] + game_region[3] * ExitButtonPos[1])
    pyautogui.click()
    time.sleep(0.5)
    # Click on Coffee Bags
    pyautogui.moveTo(game_region[0] + game_region[2] * CafeCoffeeBagsPos[0], game_region[1] + game_region[3] * CafeCoffeeBagsPos[1])
    pyautogui.click()
    time.sleep(3)
    # Confirm Play
    pyautogui.moveTo(game_region[0] + game_region[2] * ConfirmPlayPos[0], game_region[1] + game_region[3] * ConfirmPlayPos[1])
    pyautogui.click()
    time.sleep(0.5)
    # Start Game
    pyautogui.moveTo(game_region[0] + game_region[2] * StartGamePos[0], game_region[1] + game_region[3] * StartGamePos[1])
    pyautogui.click()
    

def main():
    global running, paused, GAME_REGION, exit_program
    
    # Set pyautogui settings for faster movement
    pyautogui.PAUSE = 0.01
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.MINIMUM_SLEEP = 0
    
    print("Club Penguin Bean Counters Bot")
    print("-----------------------------")
    print(f"Press {START_KEY} to start/stop")
    print(f"Press {PAUSE_KEY} to pause/resume")
    print(f"Press {DEBUG_KEY} to toggle debug mode")
    print("Press 'q' to quit")
    
    # Start key listener in a separate thread
    listener_thread = threading.Thread(target=cross_platform_key_listener)
    listener_thread.daemon = True
    listener_thread.start()
    
    # Calibrate game region
    GAME_REGION = calibrate_game_region()
    sct = mss.mss()
    
    try:
        while not exit_program:
            if running and not paused:
                try:
                    # Capture the game screen
                    # Convert top_left_x, top_left_y, x_length, y_length to bottom_left_x, bottom_left_y, top_right_x, top_right_y
                    # bbox_screen = (
                    #     GAME_REGION[0], # bottom left x
                    #     GAME_REGION[1], # bottom left y
                    #     GAME_REGION[0] + GAME_REGION[2], # top right x
                    #     GAME_REGION[1] + GAME_REGION[3] # top right y
                    #     )
                    # bbox = (
                    #     bbox_screen[0] + int(GAME_REGION[2] * 0.2),
                    #     bbox_screen[1] + int(GAME_REGION[3] * 0.5), 
                    #     bbox_screen[2] - int(GAME_REGION[2] * 0.14), 
                    #     bbox_screen[3] - int(GAME_REGION[3] * 0.15),
                    #     )
                    monitor = {
                        "left": GAME_REGION[0] + int(GAME_REGION[2] * 0.2),
                        "top": GAME_REGION[1] + int(GAME_REGION[3] * 0.4),
                        "width": int(GAME_REGION[2] * 0.66),
                        "height": int(GAME_REGION[3] * 0.45)
                    }
                    screenshot = np.array(sct.grab(monitor))
                    # Convert from RGB to BGR (for OpenCV)
                    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                    frame = cv2.resize(frame, (frame.shape[1]//DETECTION_SCALE, frame.shape[0]//DETECTION_SCALE))
                    
                    # Detect objects
                    bean_bags, fishes, anvils, pots, oneups = detect_objects(frame)
                    
                    # Determine action
                    action, hazard = determine_action(bean_bags, fishes, anvils, pots, oneups, frame.shape[1])
                    
                    # Move the penguin
                    move_penguin(action, hazard, GAME_REGION)
                except Exception as e:
                    print(f"Error during gameplay: {e}")
                    time.sleep(1)  # Pause briefly on error
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Bot terminated by user")
    finally:
        # Clean up
        exit_program = True
        if debug_mode:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check for required libraries
    required_packages = ["pyautogui", "numpy", "opencv-python", "pillow"]
    missing_packages = []
    
    # Check each package
    try:
        import pyautogui
    except ImportError:
        missing_packages.append("pyautogui")
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")
    
    try:
        from PIL import ImageGrab
    except ImportError:
        missing_packages.append("pillow")
    
    # Check for keyboard libraries
    keyboard_lib_present = False
    try:
        import keyboard
        keyboard_lib_present = True
    except ImportError:
        try:
            from pynput import keyboard
            keyboard_lib_present = True
        except ImportError:
            missing_packages.append("keyboard or pynput")
    
    # Report missing packages
    if missing_packages:
        print("Missing required libraries:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install the required libraries using:")
        print(f"pip install {' '.join(required_packages)} {'keyboard pynput' if not keyboard_lib_present else ''}")
        sys.exit(1)
        
    # Check platform compatibility
    current_platform = platform.system()
    print(f"Running on {current_platform}")
    
    # Platform-specific adjustments
    if current_platform == "Darwin":  # macOS
        print("On macOS, you may need to grant accessibility permissions to your terminal/IDE")
        print("If keyboard detection doesn't work, try installing pynput: pip install pynput")
    elif current_platform == "Windows":
        print("On Windows, run this script with administrator privileges for best results")
    elif current_platform == "Linux":
        print("On Linux, you may need to install additional dependencies for screen capture")
    
    main()