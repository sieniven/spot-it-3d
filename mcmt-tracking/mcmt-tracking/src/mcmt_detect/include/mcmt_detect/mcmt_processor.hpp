// MCMT Processor Node
// Author: Niven Sie, sieniven@gmail.com
// 
// This code contains the MCMT Processor node class to launch our multi camera detector,
// and it will launch 2 UVCDriver nodes.

#ifndef MCMT_PROCESSOR_HPP_
#define MCMT_PROCESSOR_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <mcmt_detect/mcmt_params.hpp>
#include <mcmt_detect/mcmt_detect_utils.hpp>

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt
{
class McmtProcessorNode : public rclcpp::Node {
	public:
		McmtProcessorNode(std::string cam_index);
		// declare ROS2 video parameters
		rclcpp::Parameter IS_REALTIME_param, VIDEO_INPUT_param, FRAME_WIDTH_param, FRAME_HEIGHT_param,
											VIDEO_FPS_param, MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_param;

		// declare ROS2 filter parameters
		rclcpp::Parameter VISIBILITY_RATIO_param, VISIBILITY_THRESH_param, CONSECUTIVE_THRESH_param,
											AGE_THRESH_param, SEC_FILTER_DELAY_param, SECONDARY_FILTER_param;

		// declare ROS2 background subtractor parameters
		rclcpp::Parameter FGBG_HISTORY_param, NMIXTURES_param, BRIGHTNESS_GAIN_param, DILATION_ITER_param,
											BACKGROUND_RATIO_param, FGBG_LEARNING_RATE_param, REMOVE_GROUND_ITER_param,
											BACKGROUND_CONTOUR_CIRCULARITY_param;
		
		// declare Params and Camera class variables
		McmtParams params_;
		Camera camera_;

		// declare video parameters
		int video_input_, frame_w_, frame_h_;
		std::string filename_;
		bool is_realtime_;
    cv::VideoCapture cap_;

	private:
		rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_img_sub_;
		rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detected_img_pub_;
		std::string cam_index_, topic_name_;

		std::string mat_type2encoding(int mat_type);
		int encoding2mat_type(const std::string & encoding);

		void declare_parameters();
		void get_parameters();
		void detection_callback();
		void publish_info();
};
}

#endif			// MCMT_PROCESSOR_HPP_