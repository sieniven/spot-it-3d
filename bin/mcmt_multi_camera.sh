#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/eloquent/setup.bash"

# colcon build ros2 packages
source "../mcmt-tracking/install/setup.bash"

# run multi camera ROS2 launch
ros2 launch mcmt_bringup multi_camera.launch.py