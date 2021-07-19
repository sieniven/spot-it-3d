 /**
 * @file mcmt_single_detect_main.cpp
 * @author Dr Sutthiphong Srigrarom (Spot), spot.srigrarom@nus.edu.sg
 * @author Mr Niven Sie, sieniven@gmail.com
 * @author Mr Seah Shao Xuan, seahshaoxuan@gmail.com
 * @author Mr Lau Yan Han, sps08.lauyanhan@gmail.com
 * 
 * This code is conceptualised, created and published by the SPOT-IT 3D team
 * from the Department of Mechanical Engineering, Faculty of Engineering 
 * at the National University of Singapore. SPOT-IT 3D refers to the 
 * Simultaneous Positioning, Observing, Tracking, Identifying Targets in 3D.
 * This software utilizes a multi-camera surveillance system for real-time 
 * multiple target tracking capabilities. This software capability is highly
 * applicable for monitoring specific areas, and some use cases include monitoring 
 * airspaces, traffic junctions, etc.
 * 
 * This file is part of the SPOT-IT 3D repository and can be downloaded at:
 * https://github.com/sieniven/spot-it-3d
 * 
 * This file contains the main launch pipeline for the detector node that
 * can be found in mcmt_single_detect_node.cpp.
 */

// opencv header files
#include <opencv2/opencv.hpp>

// ROS2 header files
#include <rclcpp/rclcpp.hpp>

// local header files
#include <mcmt_detect/mcmt_single_detect_node.hpp>

// standard package imports
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <csignal>
#include <stdio.h>

using namespace std;
using namespace mcmt;

sig_atomic_t signalled = 0;

void signal_handler(int signal_num) {
	signalled = 1;
	
	printf("Single Camera Detector Node interrupted. Shutting down...\n");
	rclcpp::shutdown();
}

int main(int argc, char * argv[])
{
	printf("Launching Single Detector Node....\n");

	signal(SIGINT, signal_handler);

	rclcpp::init(argc, argv);

	// initialize detector node
	auto detect_node = make_shared<McmtSingleDetectNode>();

	while (signalled == false) {
		detect_node->start_record();
	}

	return 0;
}