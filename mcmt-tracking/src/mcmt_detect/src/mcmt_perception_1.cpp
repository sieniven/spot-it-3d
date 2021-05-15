/**
 * Pipeline to launch Detect Nodes
 * Author: Niven Sie, sieniven@gmail.com 
 * This code contains the main pipeline to launch our Detector Node for 
 * camera 2 detection.
 */

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <mcmt_detect/mcmt_detect_node.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <stdio.h>

int main(int argc, char * argv[])
{
	printf("Launching Detector Node 1....\n");

	rclcpp::init(argc, argv);

	// initialize detector node
	auto detect_node = std::make_shared<mcmt::McmtDetectNode>("1");

	detect_node->start_record();
	detect_node->stop_record();
	
	printf("Detector Node 1 interrupted. Shutting down...\n");
	rclcpp::shutdown();
	return 0;
}