/** MCMT UVCDriver Node
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the UVCDriver node class that runs our camera, and publish the 
 * raw frames into our ROS2 DDS-RTPS ecosystem.
 */

#include <mcmt_detect/mcmt_uvc_driver.hpp>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <functional>

using mcmt::UVCDriver;

UVCDriver::UVCDriver(const std::string cam_index)
: Node("UVCDriverNode" + cam_index)
{
	RCLCPP_INFO(this->get_logger(), "Initializing UVCDriver" + cam_index);
	node_handle_ = std::shared_ptr<::rclcpp::Node>(this, [](::rclcpp::Node *) {});
	
	declare_parameters();
	get_parameters();

	if (is_realtime_ == true) {
		cap_ = cv::VideoCapture(video_input_);
	} else {
		cap_ = cv::VideoCapture(filename_);
	}
	
	// create raw image publisher with topic name "mcmt/raw_image_{cam_index}"
	topic_name_ = "mcmt/raw_image_" + cam_index);
	image_pub_ = this->create_publisher<sensor_msgs::msg::Image> (topic_name_, 10);

	if (!cap_.isOpened()) {
    std::cout << "Error: Cannot open camera " + cam_index + "! Please check!" << std::endl;
  }
	else {
		std::cout << "Camera " + cam_index + " opened successful!" << std::endl;
	}
	cap_.set(cv::CAP_PROP_FPS, 25);
}

// function to start video capture
void UVCDriver::start_record()
{
	frame_id_ = 1;
	while (1) {
		// get camera frame
		cap_ >> frame_;
		// check if getting frame was successful
		if (frame_.empty()) {
			std::cout << "Error: Video camera " + cam_index + " is disconnected!" << std::endl;
			break;
		}
		// publish raw image frames as ROS2 messages
		publish_image();
		//  spin the UVCDriver node once
		rclcpp::spin_some(node_handle_);

		frame_id_++;
	}
}

// function to stop video capture
void UVCDriver::stop_record()
{
	std::cout << "Stop capturing camera " + cam_index + " completed!" << std::endl;
	cap_.release();
}

// function to publish image
void UVCDriver::publish_image()
{
	// set message header info
	frame_id_str_ = std::to_string(frame_id_);
	rclcpp::Time timestamp_ = this->now();
	std_msgs::msg::Header header_;
	std::string encoding_;
	header.stamp = timestamp_;
	header.frame_id = frame_id_str_;

	// publish raw image frame
	encoding_ = mat_type2encoding(frame_.type());
	sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(
		header_, encoding_, frame_).toImageMsg();
	image_pub_->publish(*msg);
}

void UVCDriver::declare_parameters()
{
	// declare ROS2 video parameters
	this->declare_parameter("IS_REALTIME");
	this->declare_parameter("VIDEO_INPUT");
}

void UVCDriver::get_parameters()
{
	// get video parameters
	IS_REALTIME_param = this->get_parameter("IS_REALTIME");
	VIDEO_INPUT_param = this->get_parameter("VIDEO_INPUT");

	is_realtime_ = IS_REALTIME_param.as_bool();
	if (is_realtime_ == true) {
		video_input_ = VIDEO_INPUT_param.as_int();
	} else {
		filename_ = VIDEO_INPUT_param.as_string();
	}
}

std::string UVCDriver::mat_type2encoding(int mat_type)
{
	switch (mat_type) {
		case CV_8UC1:
			return "mono8";
		case CV_8UC3:
			return "bgr8";
		case CV_16SC1:
			return "mono16";
		case CV_8UC4:
			return "rgba8";
		default:
			throw std::runtime_error("Unsupported encoding type");
	}
}
