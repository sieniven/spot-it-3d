/**
 * @file mcmt_multi_tracker_main.cpp
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
 * This file contains the main launch pipeline for the tracker node that
 * can be found in mcmt_multi_tracker_node.cpp.
 */

// opencv header files
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

// local header files
#include <mcmt_track/mcmt_multi_tracker_node.hpp>

// standard package imports
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <exception>
#include <signal.h>
#include <stdlib.h>

using namespace mcmt;

class InterruptException : public std::exception {
	public:
		InterruptException(int s) : S(s) {}
		int S;
};

void sig_to_exception(int s) {
	throw InterruptException(s);
}

int main(int argc, char * argv[]) {
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = sig_to_exception;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	printf("Launching Multi Tracker Node....\n");
		
	rclcpp::init(argc, argv);

	// initialize detector node
	auto track_node = std::make_shared<McmtMultiTrackerNode>();

	try {
		rclcpp::spin(track_node);
	} catch(InterruptException& e) {
		printf("Multi Camera Tracker Node interrupted. Saving video and shutting down...\n");
		track_node->recording_.release();
		cv::destroyAllWindows();
		rclcpp::shutdown();
		return 1;
	}
    return 0;
}