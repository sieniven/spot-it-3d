/**
 * Pipeline to launch Detect Nodes
 * Author: Niven Sie, sieniven@gmail.com 
 * This code contains the main pipeline to launch our Detector Node for 
 * camera 1 detection.
 */

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <mcmt_detect/mcmt_multi_detect_node.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <csignal>
#include <stdio.h>

std::sig_atomic_t signalled = 0;

void signal_handler(int signal_num)
{
	signalled = 1;
}

int main(int argc, char * argv[])
{
	printf("Launching Multi Detector Node....\n");

	std::signal(SIGINT, signal_handler);

	rclcpp::init(argc, argv);

	// initialize detector node
	auto detect_node = std::make_shared<mcmt::McmtMultiDetectNode>();

	while (signalled == false) {
		detect_node->start_record();
	}

	// catch SIGINT
	detect_node->stop_record();
	
	printf("Multi Camera Detector Node interrupted. Shutting down...\n");
	rclcpp::shutdown();
	return 0;
}