// Pipeline to launch UVCDrivers
// Author: Niven Sie, sieniven@gmail.com
// 
// This code contains the main pipeline to launch our UVCDriver.

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <mcmt_detect/mcmt_uvc_driver.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <stdio.h>

int main(int argc, char * argv[])
{
	printf("Launching UVCDriver Node 2....\n");

	rclcpp::init(argc, argv);

	// initialize camera UVCDriver node
	auto uvc_driver_node = std::make_shared<mcmt::McmtDetectNode>("2");

	uvc_driver_node->start_record();
	uvc_driver_node->stop_record();
	
	printf("UVCDriver Node 2 interrupted. Shutting down...\n");
	rclcpp::shutdown();
	return 0;
}