#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
	config = os.path.join(
        get_package_share_directory('mcmt_bringup'),
        'config',
        'config.yaml')

	return LaunchDescription([
		Node(
			package='mcmt_detect',
			node_executable='mcmt_processor',
			output='screen',
			emulate_tty=True,
			parameters=[config]),
	])