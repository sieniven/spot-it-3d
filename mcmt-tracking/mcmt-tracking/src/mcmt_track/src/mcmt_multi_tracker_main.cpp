/**
 * Author: Niven Sie, sieniven@gmail.com 
 * This code contains the main pipeline to launch our McmtMMultiTrackerNode.
 */

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <mcmt_track/mcmt_multi_tracker_node.hpp>

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

class InterruptException : public std::exception
{
public:
  InterruptException(int s) : S(s) {}
  int S;
};

void sig_to_exception(int s)
{
  throw InterruptException(s);
}

int main(int argc, char * argv[])
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = sig_to_exception;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	printf("Launching Multi Detector Node....\n");
		
	rclcpp::init(argc, argv);

	// initialize detector node
	auto track_node = std::make_shared<mcmt::McmtMultiTrackerNode>();

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