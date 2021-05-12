#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
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
			node_executable='mcmt_processor_1',
			name='ProcessorNode1',
			output='screen',
			emulate_tty=True,
			parameters=[config]),
		Node(
			package='mcmt_detect',
			node_executable='mcmt_processor_2',
			name='ProcessorNode2',
			output='screen',
			emulate_tty=True,
			parameters=[config]),
	])