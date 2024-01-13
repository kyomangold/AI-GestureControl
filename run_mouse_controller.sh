#!/bin/bash

# Change to the specified directory
cd /home/bexl/vision_control

# Set the DISPLAY environment variable
export DISPLAY=:0

# Run the Python script with a timeout
timeout 300 python3 mouse_controller.py
