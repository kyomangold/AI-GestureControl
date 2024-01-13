INSTRUCTIONS = """
------------------------------------------------------------------------------------------------------
-----------------------------------------INSTRUCTIONS-------------------------------------------------
------------------------------------------------------------------------------------------------------

Move your mouse pointer with your hand.
 
The pointer moves when your hand is doing the FIVE pose (open hand).
The pointer clicks when your hand is doing the FIST pose.
The pointer scrolls when your hand is doing the PEACE pose (or just index + middle finger together).

------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------
"""

print(INSTRUCTIONS)

import numpy as np
import time
import argparse
from screeninfo import get_monitors
from pynput.mouse import Button, Controller

from hand_pose_controller import HandController

# Initialize the parser
parser = argparse.ArgumentParser(description="Sample argument parser")
parser.add_argument('-r', '--enable-renderer', action='store_true', help='Enable renderer')

# Parse the arguments
args = parser.parse_args()

# Check if '-r' was used
if args.enable_renderer:
    enable_flag = True
else:
    enable_flag = False
    
# Control mouse
mouse = Controller()

# Get screen resolution 
monitor = get_monitors()[0] # Replace '0' by the index of your screen in case of multiscreen
print(monitor)

# Smoothing filter
class DoubleExponentialSmoothing:
    def __init__(self,smoothing=0.65, correction=1.0, prediction=0.85, jitter_radius=250., max_deviation_radius=540., out_int=False):
        self.smoothing = smoothing
        self.correction = correction
        self.prediction = prediction
        self.jitter_radius = jitter_radius
        self.max_deviation_radius = max_deviation_radius
        self.count = 0
        self.filtered_pos = 0
        self.trend = 0
        self.raw_pos = 0
        self.out_int = out_int
        self.enable_scrollbars = False
    
    def reset(self):
        self.count = 0
        self.filtered_pos = 0
        self.trend = 0
        self.raw_pos = 0
    
    def update(self, pos):
        raw_pos = np.asanyarray(pos)
        if self.count > 0:
            prev_filtered_pos = self.filtered_pos
            prev_trend = self.trend
            prev_raw_pos = self.raw_pos
        if self.count == 0:
            self.shape = raw_pos.shape
            filtered_pos = raw_pos
            trend = np.zeros(self.shape)
            self.count = 1
        elif self.count == 1:
            filtered_pos = (raw_pos + prev_raw_pos)/2
            diff = filtered_pos - prev_filtered_pos
            trend = diff*self.correction + prev_trend*(1-self.correction)
            self.count = 2
        else:
            # First apply jitter filter
            diff = raw_pos - prev_filtered_pos
            length_diff = np.linalg.norm(diff)
            if length_diff <= self.jitter_radius:
                alpha = pow(length_diff/self.jitter_radius,1.5)
                # alpha = length_diff/self.jitter_radius
                filtered_pos = raw_pos*alpha \
                                + prev_filtered_pos*(1-alpha)
            else:
                filtered_pos = raw_pos
            # Now the double exponential smoothing filter
            filtered_pos = filtered_pos*(1-self.smoothing) \
                        + self.smoothing*(prev_filtered_pos+prev_trend)
            diff = filtered_pos - prev_filtered_pos
            trend = self.correction*diff + (1-self.correction)*prev_trend
        # Predict into the future to reduce the latency
        predicted_pos = filtered_pos + self.prediction*trend
        # Check that we are not too far away from raw data
        diff = predicted_pos - raw_pos
        length_diff = np.linalg.norm(diff)
        if length_diff > self.max_deviation_radius:
            predicted_pos = predicted_pos*self.max_deviation_radius/length_diff \
                        + raw_pos*(1-self.max_deviation_radius/length_diff)
        # Save the data for this frame
        self.raw_pos = raw_pos
        self.filtered_pos = filtered_pos
        self.trend = trend
        # Output the data
        if self.out_int:
            return predicted_pos.astype(int)
        else:
            return predicted_pos

smooth = DoubleExponentialSmoothing(smoothing=0.3, prediction=0.1, jitter_radius=700, out_int=True)

# Camera image size with aspect ratio 16:9
cam_width = 1152
cam_height = 648


def move(event):
    # Use location of index
    x, y = event.hand.landmarks[8,:2]
    x /= cam_width
    x = 1 - x
    y /= cam_height
    e = 0.15
    p1 = monitor.width/(1-2*e)
    q1 = -p1*e
    mx = int(max(0, min(monitor.width-1, p1*x+q1)))
    et = 0.05
    eb= 0.4
    p2 = monitor.height/(1-et-eb)
    q2 = -p2*et
    my = int(max(0, min(monitor.height-1, p2*y+q2)))
    mx,my = smooth.update((mx,my))
    mouse.position = (mx+monitor.x, my+monitor.y)

def click(event):
    mouse.press(Button.left)
    mouse.release(Button.left)

last_y_position = 0

def scroll(event):
    global last_y_position
    
    # Use the Y location of the middle finger for scrolling
    _, current_y = event.hand.landmarks[12, :2]  # Assuming landmark 12 is the middle finger tip
    current_y /= cam_height

    # Calculate the change in Y position
    delta_y = last_y_position - current_y

    # Threshold for detecting a deliberate scroll (adjust as needed)
    scroll_threshold = 0.01  

    # Only trigger a scroll if the hand has moved a sufficient distance
    if abs(delta_y) > scroll_threshold:
        scroll_speed = int(delta_y * 500)  # Convert to an integer scroll value; adjust the multiplier as needed (higher number = faster scrolling)
        mouse.scroll(0, scroll_speed)  # Scrolling action, with horizontal scroll = 0

    # Update the last Y position
    last_y_position = current_y

    # Small delay to make continuous scrolling smoother
    time.sleep(0.01)
    
    
config = {
    'renderer' : {'enable': enable_flag},
    
    'pose_actions' : [

        {'name': 'MOVE', 'pose':'FIVE', 'callback': 'move', "trigger":"continuous", "first_trigger_delay":0.1,},
        {'name': 'CLICK', 'pose':'FIST', 'callback': 'click', "trigger":"enter_leave", "first_trigger_delay":0.1},
        {'name': 'SCROLL', 'pose':'PEACE', 'callback': 'scroll', "trigger":"continuous", "first_trigger_delay":0.1},
    ]
}

HandController(config).loop()