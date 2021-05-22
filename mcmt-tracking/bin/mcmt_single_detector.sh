#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/eloquent/setup.bash"

# colcon build ros2 packages
source "../mcmt-tracking/install/setup.bash"

# run single detector ROS2 launch
ros2 launch mcmt_bringup single_detector.launch.py