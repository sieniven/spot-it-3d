/**
 * @file mcmt_single_detect_node.hpp
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
 * This file contains the declarations and definitions of the functions 
 * primarily used in the detection pipeline. These functions interact with the 
 * key classes Camera and Track which are essential to the detection process.
 * These classes may be found at mcmt_single_detect_node.cpp. This code also 
 * contains the McmtMultiDetectNode class that runs our camera, and publishes the 
 * raw frames into our ROS2 DDS-RTPS ecosystem.
 */

#ifndef MCMT_SINGLE_DETECTOR_NODE_HPP_
#define MCMT_SINGLE_DETECTOR_NODE_HPP_

// opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xphoto.hpp>

// ros2 header files
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

// local header files
#include <mcmt_detect/mcmt_detect_utils.hpp>
#include <mcmt_msg/msg/single_detection_info.hpp>

// standard package imports
#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt {

	class McmtSingleDetectNode : public rclcpp::Node {
		
		public:
			McmtSingleDetectNode();
			virtual ~McmtSingleDetectNode() {}
				
			// declare node parameters
			rclcpp::Node::SharedPtr node_handle_;
			std::string topic_name_;

			// declare video parameters
			cv::VideoCapture cap_;
			cv::Mat frame_, frame_ec_, color_converted_, element_;
			std::array<cv::Mat, 2> masked_, removebg_;
			std::string video_input_;
			int frame_w_, frame_h_, fps_, frame_id_, next_id_;
			float scale_factor_, aspect_ratio_;
			bool is_realtime_, downsample_;

			// declare tracking variables
			std::vector<std::shared_ptr<mcmt::Track>> tracks_, good_tracks_;
			std::vector<int> dead_tracks_;

			// declare detection variables
			std::array<std::vector<float>,2> sizes_temp_;
			std::vector<float> sizes_;
			std::array<std::vector<cv::Point2f>,2> centroids_temp_;
			std::vector<cv::Point2f> centroids_;

			// declare blob detector and background subtractor
			cv::Ptr<cv::SimpleBlobDetector> detector_;
			std::array<cv::Ptr<cv::BackgroundSubtractorMOG2>, 2> fgbg_;

			// declare tracking variables
			std::vector<int> unassigned_tracks_, unassigned_detections_;
			std::vector<int> unassigned_tracks_kf_, unassigned_detections_kf_;
			std::vector<int> unassigned_tracks_dcf_, unassigned_detections_dcf_;
			
			// we store the matched track index and detection index in the assigments vector
			std::vector<std::vector<int>> assignments_;
			std::vector<std::vector<int>> assignments_kf_;
			std::vector<std::vector<int>> assignments_dcf_;	
			std::vector<int> tracks_to_be_removed_;

			// declare ROS2 video parameters
			rclcpp::Parameter IS_REALTIME_param, VIDEO_INPUT_param, FRAME_WIDTH_param, FRAME_HEIGHT_param,
												VIDEO_FPS_param, MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_param;

			// declare ROS2 filter parameters
			rclcpp::Parameter VISIBILITY_RATIO_param, VISIBILITY_THRESH_param, CONSECUTIVE_THRESH_param,
												AGE_THRESH_param, SEC_FILTER_DELAY_param, SECONDARY_FILTER_param;

			// declare ROS2 background subtractor parameters
			rclcpp::Parameter FGBG_HISTORY_param, BACKGROUND_RATIO_param, NMIXTURES_param, BRIGHTNESS_GAIN_param,
												FGBG_LEARNING_RATE_param, DILATION_ITER_param, REMOVE_GROUND_ITER_param, 
												BACKGROUND_CONTOUR_CIRCULARITY_param; 

			// declare video parameters
			int FRAME_WIDTH_, FRAME_HEIGHT_, VIDEO_FPS_, MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_;

			// declare filter parameters
			float VISIBILITY_RATIO_, VISIBILITY_THRESH_, CONSECUTIVE_THRESH_, AGE_THRESH_, SEC_FILTER_DELAY_;
			int SECONDARY_FILTER_;
			
			// declare background subtractor parameters
			int FGBG_HISTORY_, NMIXTURES_, BRIGHTNESS_GAIN_, DILATION_ITER_;
			float BACKGROUND_RATIO_, FGBG_LEARNING_RATE_, REMOVE_GROUND_ITER_, BACKGROUND_CONTOUR_CIRCULARITY_;

			// detector functions
			void start_record();
			void stop_record();

		private:
			rclcpp::Publisher<mcmt_msg::msg::SingleDetectionInfo>::SharedPtr detection_pub_;

			// declare node functions
			void declare_parameters();
			void get_parameters();
			void publish_info();

			// declare detection and tracking functions
			void apply_env_compensation();
			cv::Mat apply_bg_subtractions(int frame_id);
			void detect_objects();
			cv::Mat remove_ground(int masked_id);
			void remove_overlapped_detections();
			void predict_new_locations_of_tracks();
			void clear_track_variables();
			void detection_to_track_assignment_KF();
			void detection_to_track_assignment_DCF();
			void compare_cost_matrices();
			void update_assigned_tracks();
			void update_unassigned_tracks();
			void create_new_tracks();
			void delete_lost_tracks();
			std::vector<std::shared_ptr<mcmt::Track>> filter_tracks();

			// declare utility functions
			double euclideanDist(cv::Point2f & p, cv::Point2f & q);
			std::vector<int> apply_hungarian_algo(std::vector<std::vector<double>> & cost_matrix);
			int average_brightness(cv::ColorConversionCodes colortype, int channel);
			std::string mat_type2encoding(int mat_type);
			int encoding2mat_type(const std::string & encoding);
	};
}

#endif			// MCMT_SINGLE_DETECTOR_NODE_HPP_