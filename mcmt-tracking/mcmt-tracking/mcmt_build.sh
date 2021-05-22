#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/eloquent/setup.bash"

# colcon build ros2 packages
colcon build --symlink-install