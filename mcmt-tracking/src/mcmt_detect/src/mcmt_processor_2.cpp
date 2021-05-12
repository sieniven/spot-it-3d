// Pipeline to launch UVCDrivers
// Author: Niven Sie, sieniven@gmail.com
// 
// This code contains the main pipeline to launch our UVCDriver.

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <mcmt_detect/mcmt_processor.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <stdio.h>

int main(int argc, char * argv[])
{
	printf("Launching Processor Node 2....\n");

	rclcpp::init(argc, argv);

	// initialize camera UVCDriver nodes
	std::string cam_index(argv[1]);
	auto processor_node = std::make_shared<mcmt::McmtProcessorNode>("2");

	rclcpp::spin(processor_node);

	printf("Processor Node 2 interrupted. Shutting down...\n");
	rclcpp::shutdown();
	return 0;
}