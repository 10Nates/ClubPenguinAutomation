import pyautogui
import time
import numpy as np
import cv2
import sys
import platform
import threading
import mss

# Configuration variables
GAME_REGION = None  # Will be set by calibration
START_KEY = 'f8'    # Key to start/stop the bot
PAUSE_KEY = 'f10'   # Key to pause/unpause the bot
DEBUG_KEY = 'd'     # Key to toggle debug mode
QUIT_KEY = 'q'      # Key to quit program
DETECTION_SCALE = 4 # For speed

# Color detection thresholds (BGR format for OpenCV)
# Yellow turning indicators (adjust as needed based on your game's colors)
INDICATOR_COLOR_LOWER = np.array([60, 200, 240])
INDICATOR_COLOR_UPPER = np.array([80, 255, 255])

# Brightness detection for corners (adjust as needed)
CORNER_BRIGHTNESS_THRESHOLD = 110
GAME_FINISH_BRIGHTNESS_THRESHOLD = 180

# Global variables
running = False
paused = False
debug_mode = False
exit_program = False
current_trick = 0  # 0 for first trick, 1 for second trick
last_indicator = 'none'
last_indicator_time = 0
last_corner_time = 0
last_trick_time = 0

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

def detect_turn_indicators(frame):
    """Detect left and right turn indicators in the current frame."""
    # Create masks for indicators
    mask = cv2.inRange(frame, INDICATOR_COLOR_LOWER, INDICATOR_COLOR_UPPER)
    
    # Check if indicators are present in their respective regions
    # Assuming indicators appear on the left/right sides of the screen
    height, width = frame.shape[:2]
    
    # Left side region
    left_region = mask[:, :width//3]
    left_indicator = cv2.countNonZero(left_region) > 100  # Adjust threshold as needed
    if debug_mode and left_indicator:
        print("Detected left indicator")
    
    # Right side region
    right_region = mask[:, 2*width//3:]
    right_indicator = cv2.countNonZero(right_region) > 100  # Adjust threshold as needed
    if debug_mode and right_indicator:
        print("Detected right indicator")
    
    # Debug visualization
    if debug_mode:
        debug_frame = frame.copy()
        if left_indicator:
            cv2.rectangle(debug_frame, (0, 0), (width//3, height), (0, 0, 255), 2)
        if right_indicator:
            cv2.rectangle(debug_frame, (2*width//3, 0), (width, height), (0, 0, 255), 2)
        cv2.rectangle(debug_frame, (int(width*0.35), int(height*0.4)), (int(width*0.4), int(height*0.45)), (255, 0, 0), 2)
        cv2.imshow('Debug View', debug_frame)
        cv2.waitKey(1)
    
    return left_indicator, right_indicator

def detect_corner(frame):
    """Detect if we're approaching or in a corner (brighter area)."""
    # Convert to grayscale and check brightness
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame[int(height*0.4):int(height*0.45), int(width*0.35):int(width*0.4)], cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # If brightness is above threshold, we're likely in a corner
    is_corner = avg_brightness > CORNER_BRIGHTNESS_THRESHOLD
    
    if avg_brightness > GAME_FINISH_BRIGHTNESS_THRESHOLD:
        if debug_mode: 
            print(f"Detected game end: Brightness {avg_brightness}")
        restart_game_sequence()
    
    if debug_mode and is_corner:
        print(f"Corner detected: Brightness {avg_brightness}")
    
    return is_corner

def perform_tricks(left_indicator, right_indicator, is_corner):
    """Perform appropriate tricks or turns based on the game state."""
    global current_trick
    global last_indicator
    global last_indicator_time
    global last_corner_time
    global last_trick_time
    
    # Measurements of indicators and corners
    if left_indicator:
        last_indicator = 'left'
        last_indicator_time = time.time()
    
    if right_indicator:
        last_indicator = 'right'
        last_indicator_time = time.time()
        
    if is_corner:
        last_corner_time = time.time()
        
    if time.time() - last_indicator_time > 1:
        last_indicator = 'none'
    
    # Analysis of measurements
    if time.time() - last_corner_time < 0.8:
        pyautogui.keyDown('down')
        if last_indicator == 'left':
            pyautogui.keyDown('right')
            return
        elif last_indicator == 'right':
            pyautogui.keyDown('left')
            return
    elif last_indicator != 'none':
        pyautogui.keyUp('down')
        pyautogui.keyUp('right')
        pyautogui.keyUp('left')
        return
    
    # No indicators, perform alternating tricks
    if last_indicator == 'none':
        # Release any held keys first
        for key in ['down', 'left', 'right', 'space']:
            pyautogui.keyUp(key)
        
        # Perform the current trick
        if current_trick == 0 and time.time() - last_trick_time > 1:
            if debug_mode:
                print("Performing trick 1: down arrow -> space")
            
            pyautogui.keyDown('down')
            pyautogui.keyUp('down')
            pyautogui.keyDown('space')
            pyautogui.keyUp('space')
            
            current_trick = 1
            last_trick_time = time.time()
        elif current_trick == 1 and time.time() - last_trick_time > 1:
            if debug_mode:
                print("Performing trick 2: space -> right arrow")
            
            pyautogui.keyDown('space')
            pyautogui.keyDown('right')
            pyautogui.keyUp('space')
            pyautogui.keyUp('right')
            
            current_trick = 0
            last_trick_time = time.time()

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
    global exit_program
    
    def manage_exit_on_long_sleeps():
        if exit_program == True:
            raise KeyboardInterrupt
    
    game_region = GAME_REGION
    ExitButtonPos = (832/1186, 101/746)
    MinecartsPos = (993/1186, 239/746)
    ConfirmPlayPos = (508/1186, 343/746)
    StartGamePos = (998/1186, 591/746)
    
    #Wait for exit screen
    time.sleep(2)
    manage_exit_on_long_sleeps()
    
    #Exit Game
    pyautogui.moveTo(game_region[0] + game_region[2] * ExitButtonPos[0], game_region[1] + game_region[3] * ExitButtonPos[1])
    pyautogui.click()
    time.sleep(1)
    manage_exit_on_long_sleeps()
    # Click on Minecarts
    pyautogui.moveTo(game_region[0] + game_region[2] * MinecartsPos[0], game_region[1] + game_region[3] * MinecartsPos[1])
    pyautogui.click()
    time.sleep(4)
    manage_exit_on_long_sleeps()
    # Confirm Play
    pyautogui.moveTo(game_region[0] + game_region[2] * ConfirmPlayPos[0], game_region[1] + game_region[3] * ConfirmPlayPos[1])
    pyautogui.click()
    time.sleep(1)
    manage_exit_on_long_sleeps()
    # Start Game
    pyautogui.moveTo(game_region[0] + game_region[2] * StartGamePos[0], game_region[1] + game_region[3] * StartGamePos[1])
    pyautogui.click()

def main():
    global running, paused, GAME_REGION, exit_program
    
    # Set pyautogui settings for faster movement
    pyautogui.PAUSE = 0.01
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.MINIMUM_SLEEP = 0
    
    print("Club Penguin Cart Surfer Bot")
    print("----------------------------")
    print(f"Press {START_KEY} to start/stop")
    print(f"Press {PAUSE_KEY} to pause/resume")
    print(f"Press {DEBUG_KEY} to toggle debug mode")
    print(f"Press {QUIT_KEY} to quit")
    
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
                    monitor = {
                        "left": GAME_REGION[0],
                        "top": GAME_REGION[1],
                        "width": GAME_REGION[2],
                        "height": GAME_REGION[3]
                    }
                    screenshot = np.array(sct.grab(monitor))
                    
                    # Convert from RGB to BGR (for OpenCV)
                    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                    frame = cv2.resize(frame, (frame.shape[1]//DETECTION_SCALE, frame.shape[0]//DETECTION_SCALE))
                    
                    # Detect turn indicators and corners
                    left_indicator, right_indicator = detect_turn_indicators(frame)
                    is_corner = detect_corner(frame)
                    
                    # Perform appropriate actions
                    perform_tricks(left_indicator, right_indicator, is_corner)
                    
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
        # Release all pressed keys
        for key in ['down', 'left', 'right', 'space']:
            pyautogui.keyUp(key)
        if debug_mode:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check for required libraries
    required_packages = ["pyautogui", "numpy", "opencv-python", "pillow", "mss"]
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
        import mss
    except ImportError:
        missing_packages.append("mss")
    
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