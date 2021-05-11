/** MCMT Processor Node
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the MCMT Processor node class that runs our camera image processing pipeline.
 * It subscribes to a camera's raw image topic channel that is published by our UVCDriver node.
 */

#include <mcmt_detect/mcmt_processor.hpp>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <functional>

using mcmt::ProcessorNode;

McmtProcessorNode::McmtProcessorNode(std::string cam_index)
: Node("ProcessorNode" + cam_index)
{
	cam_index_ = cam_index;
	RCLCPP_INFO(this->get_logger(), "Initializing MCMT Processor Node" + cam_index_);
	node_handle_ = std::shared_ptr<::rclcpp::Node>(this, [](::rclcpp::Node *) {});
	declare_parameters();
	get_parameters();

	if (is_realtime_ == true) {
		cap_ = cv::VideoCapture(video_input_);
	} else {
		cap_ = cv::VideoCapture(filename_);
	}

	// get camera frame width and height. Processor node has to be initialized before UVCDriver node
	frame_w_ = int(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
	frame_h_ = int(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
	cap_.release();
	
	// initialize Camera class
	camera_(params_, cam_index, frame_w_, frame_h_);

	// create detection info publisher with topic name "mcmt/detection_info_{cam_index}"
	topic_name_ = "mcmt/detection_info_" + cam_index);
	detection_pub_ = this->create_publisher<mcmt_msg::msg::DetectionInfo> (topic_name_, 10);

	// initialize raw image subscriber
	detection_callback();
}

/**
 * This function creates subcription to the raw image camera feed and processes the
 * images to detect tracked targets. This function is our main pipeline for our 
 * multi-target detection algorithm.
 */
void McmtProcessorNode::detection_callback()
{
	// set qos and topic name
	auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
	topic_name_ = "mcmt/raw_image_" + cam_index_);

	// create raw image subscriber that subscribes to topic name
	raw_img_sub->create_subscription<sensor_msgs::msg::Image> (
		topic_name_,
		qos,
		[this](const sensor_msgs::msg::Image::SharedPtr msg) -> void
		{
			// decode message and get camera frame and id
			camera_.frame_id_ = msg->header.frame_id;
			camera_.frame_ = cv::Mat(msg->height, msg->width, encoding2mat_type(msg->encoding),
															 const_cast<unsigned char *>(msg->data.data()), msg->step);
			if (msg->encoding == "rgb8") {
				cv::cvtColor(raw_image, raw_image, cv::COLOR_RGB2BGR);
			}

			// detect and track targets in the camera frame 
			camera_.detect_and_track();

			// publish camera information
			publish_info();
		}
	);
}

/**
 * This function declares our mcmt software parameters as ROS2 parameters.
 */
void McmtProcessorNode::declare_parameters()
{
	// declare ROS2 video parameters
	this->declare_parameter("IS_REALTIME");
	this->declare_parameter("VIDEO_INPUT");
	this->declare_parameter("FRAME_WIDTH");
	this->declare_parameter("FRAME_HEIGHT");
	this->declare_parameter("VIDEO_FPS");
	this->declare_parameter("MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES");
	
	// declare ROS2 filter parameters
	this->declare_parameter("VISIBILITY_RATIO");
	this->declare_parameter("VISIBILITY_THRESH");
	this->declare_parameter("CONSECUTIVE_THRESH");
	this->declare_parameter("AGE_THRESH");
	this->declare_parameter("SECONDARY_FILTER");
	this->declare_parameter("SEC_FILTER_DELAY");

	// declare ROS2 background subtractor parameters
	this->declare_parameter("FGBG_HISTORY");
	this->declare_parameter("BACKGROUND_RATIO");
	this->declare_parameter("NMIXTURES");
	this->declare_parameter("BRIGHTNESS_GAIN");
	this->declare_parameter("FGBG_LEARNING_RATE");
	this->declare_parameter("DILATION_ITER");
	this->declare_parameter("REMOVE_GROUND_ITER");
	this->declare_parameter("BACKGROUND_CONTOUR_CIRCULARITY");
}

/**
 * This function gets the mcmt parameters from the ROS2 parameters, and
 * stores them as in our McmtParams class.
 */
void McmtProcessorNode::get_parameters()
{
	// get video parameters
	IS_REALTIME_param = this->get_parameter("IS_REALTIME");
	VIDEO_INPUT_param = this->get_parameter("VIDEO_INPUT");
	FRAME_WIDTH_param = this->get_parameter("FRAME_WIDTH");
	FRAME_HEIGHT_param = this->get_parameter("FRAME_HEIGHT");
	VIDEO_FPS_param = this->get_parameter("VIDEO_FPS");
	MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_param = this->get_parameter("MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES");
	
	// get filter parameters
	VISIBILITY_RATIO_param = this->get_parameter("VISIBILITY_RATIO");
	VISIBILITY_THRESH_param = this->get_parameter("VISIBILITY_THRESH");
	CONSECUTIVE_THRESH_param = this->get_parameter("CONSECUTIVE_THRESH");
	AGE_THRESH_param = this->get_parameter("AGE_THRESH");
	SECONDARY_FILTER_param = this->get_parameter("SECONDARY_FILTER");
	SEC_FILTER_DELAY_param = this->get_parameter("SEC_FILTER_DELAY");

	// get background subtractor parameters
	FGBG_HISTORY_param = this->get_parameter("FGBG_HISTORY");
	BACKGROUND_RATIO_param = this->get_parameter("BACKGROUND_RATIO");
	NMIXTURES_param = this->get_parameter("NMIXTURES");
	BRIGHTNESS_GAIN_param = this->get_parameter("BRIGHTNESS_GAIN");
	FGBG_LEARNING_RATE_param = this->get_parameter("FGBG_LEARNING_RATE");
	DILATION_ITER_param = this->get_parameter("DILATION_ITER");
	REMOVE_GROUND_ITER_param = this->get_parameter("REMOVE_GROUND_ITER");
	BACKGROUND_CONTOUR_CIRCULARITY_param = this->get_parameter("BACKGROUND_CONTOUR_CIRCULARITY");

	// initialize McmtParams class to store the parameter values
	params_(VIDEO_INPUT_0_param.as_int(),
					VIDEO_INPUT_1_param.as_int(),
					FRAME_WIDTH_param.as_int(),
					FRAME_HEIGHT_param.as_int(), 
					VIDEO_FPS_param.as_int(),
					MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_param.as_int(),
					VISIBILITY_RATIO_param.as_double(),
					VISIBILITY_THRESH_param.as_double(),
					CONSECUTIVE_THRESH_param.as_double(),
					AGE_THRESH_param.as_double(),
					SECONDARY_FILTER_param.as_int(),
					SEC_FILTER_DELAY_param.as_double(),
					FGBG_HISTORY_param.as_int(),
					BACKGROUND_RATIO_param.as_double(),
					NMIXTURES_param.as_int(),
					BRIGHTNESS_GAIN_param.as_int(),
					FGBG_LEARNING_RATE_param.as_double(),
					DILATION_ITER_param.as_int(),
					REMOVE_GROUND_ITER_param.as_double(),
					BACKGROUND_CONTOUR_CIRCULARITY_param.as_double());

	// get video parameters
	is_realtime_ = IS_REALTIME_param.as_bool();
	if (is_realtime_ == true) {
		video_input_ = VIDEO_INPUT_param.as_int();
	} else {
		filename_ = VIDEO_INPUT_param.as_string();
	}
}

std::string McmtProcessorNode::mat_type2encoding(int mat_type)
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

int McmtProcessorNode::encoding2mat_type(const std::string & encoding)
{
	if (encoding == "mono8") {
		return CV_8UC1;
	} else if (encoding == "bgr8") {
		return CV_8UC3;
	} else if (encoding == "mono16") {
		return CV_16SC1;
	} else if (encoding == "rgba8") {
		return CV_8UC4;
	} else if (encoding == "bgra8") {
		return CV_8UC4;
	} else if (encoding == "32FC1") {
		return CV_32FC1;
	} else if (encoding == "rgb8") {
		return CV_8UC3;
	} else {
		throw std::runtime_error("Unsupported encoding type");
	}
}

void McmtProcessorNode::publish_info()
{
	rclcpp::Time timestamp = this->now();
	std_msgs::msg::Header header;
	std::string encoding;

	header.stamp = timestamp;
	header.frame_id = camera_.frame_id_;

	mcmt_msg::msg::DetectionInfo dect_info;

	// convert cv::Mat frame to ROS2 msg type frame
	encoding = mat_type2encoding(camera_.frame_.type());
	sensor_msgs::msg::Image::SharedPtr detect_img_msg = cv_bridge::CvImage(
		header, encoding, camera_.frame_).toImageMsg();

	// convert cv::Mat masked to ROS2 msg type masked
	encoding = mat_type2encoding(camera_.masked_.type());
	sensor_msgs::msg::Image::SharedPtr masked_msg = cv_bridge::CvImage(
		header, encoding, camera_.masked_).toImageMsg();

	std::vector<int16_t> goodtrack_id_list;
	std::vector<int16_t> goodtrack_x_list;
	std::vector<int16_t> goodtrack_y_list;
	std::vector<int16_t> deadtrack_id_list;

	// get good track's information
	for (auto & track : camera_.good_tracks_) {
		goodtrack_id_list.push_back(track.id);
		goodtrack_x_list.push_back(track.centroid_.x);
		goodtrack_y_list.push_back(track.centroid_.y);
	}

	// get gone track ids
	for (auto & deadtrack_index : camera_.dead_tracks_) {
		deadtrack_id_list.push_back(deadtrack_index);
	}

	// get DetectionInfo message
	dect_info.image = *detect_img_msg;
	dect_info.masked = *masked_msg;
	dect_info.goodtracks_id = goodtrack_id_list;
	dect_info.goodtracks_x = goodtrack_x_list;
	dect_info.goodtracks_y = goodtrack_y_list;
	dect_info.gonetracks_id = deadtrack_id_list;

	// publish detection info
	detection_pub_->publish(dect_info);
}