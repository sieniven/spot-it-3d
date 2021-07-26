/**
 * @file mcmt_single_tracker_node.hpp
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
 * This file contains the declarations of the functions primarily used in the
 * tracking and re-identification pipeline. These functions interact with the 
 * key classes CameraTracks and TrackPlots which are essential to the tracking 
 * and re-identification process. These classes may be found at 
 * mcmt_track_utils.cpp.
 */

#ifndef MCMT_SINGLE_TRACKER_NODE_HPP_
#define MCMT_SINGLE_TRACKER_NODE_HPP_

// opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

// ros2 header files
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

// local header files
#include <mcmt_track/mcmt_track_utils.hpp>
#include <mcmt_msg/msg/single_detection_info.hpp>

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>
#include <fstream>

namespace mcmt {

	class McmtSingleTrackerNode : public rclcpp::Node {
		public:
			McmtSingleTrackerNode();
			virtual ~McmtSingleTrackerNode() {}

			// declare node parameters
			rclcpp::Node::SharedPtr node_handle_;
			rclcpp::Subscription<mcmt_msg::msg::SingleDetectionInfo>::SharedPtr detection_sub_;
			std::string topic_name_;

			// declare camera parameters
			cv::VideoCapture cap_;
			int frame_w_, frame_h_, fps_;
			double scale_factor_, aspect_ratio_;
			bool downsample_;

			// declare video parameters
			std::string video_input_;
			std::string output_vid_path_, output_csv_path_;
			bool is_realtime_;
			int FRAME_WIDTH_, FRAME_HEIGHT_;
			cv::VideoWriter recording_;

			// declare ROS2 parameters
			rclcpp::Parameter IS_REALTIME_param, VIDEO_INPUT_param, FRAME_WIDTH_param, 
												FRAME_HEIGHT_param, OUTPUT_VIDEO_PATH_param, OUTPUT_CSV_PATH_param;

			// define good tracks
			typedef struct GoodTrack {
				int id;
				int x;
				int y;
				int size;
			} GoodTrack;

			// declare tracking variables
			int frame_count_;
			std::shared_ptr<CameraTracks> cumulative_tracks_;

			// declare plotting parameters
			int plot_history_;
			std::vector<std::vector<int>> colors_;
			double font_scale_;
			std::vector<int> shown_indexes_;

			// declare colors
			std::vector<cv::Scalar> colors = {
				cv::Scalar(124, 104, 66), // 1
				cv::Scalar(20, 60, 96), // 2
				cv::Scalar(46, 188, 243), // 3
				cv::Scalar(143, 89, 255), // 4
				cv::Scalar(6, 39, 156), // 5
				cv::Scalar(92, 215, 206), // 6
				cv::Scalar(105, 139, 246), // 7
				cv::Scalar(84, 43, 0), // 8
				cv::Scalar(137, 171, 197), // 9
				cv::Scalar(147, 226, 255) // 10
			};

			// debugging
			std::vector<std::string> debug_messages;
			
			// tracker node methods
			void process_msg_info(mcmt_msg::msg::SingleDetectionInfo::SharedPtr msg,
					std::shared_ptr<cv::Mat> & frame,
					std::vector<std::shared_ptr<GoodTrack>>& good_tracks);

			void declare_parameters();
			void get_parameters();
			
			void process_detection_callback();
			void print_frame_summary(std::vector<std::shared_ptr<GoodTrack>>);
			void annotate_frames(std::shared_ptr<cv::Mat> frame_, std::shared_ptr<CameraTracks> cumulative_tracks_);
			void graphical_UI(std::shared_ptr<cv::Mat> frame_, std::shared_ptr<CameraTracks> cumulative_tracks_);
			void imshow_resized_dual(std::string & window_name, cv::Mat & img);
			int encoding2mat_type(const std::string & encoding);

			std::ofstream trackplot_file;
		
	};
}

#endif	// MCMT_SINGLE_TRACKER_NODE_HPP_