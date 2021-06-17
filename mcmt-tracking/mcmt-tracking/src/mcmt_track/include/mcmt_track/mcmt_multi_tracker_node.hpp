/**
 * @file mcmt_multi_tracker_node.hpp
 * @author Niven Sie, sieniven@gmail.com
 * @author Seah Shao Xuan
 * 
 * This code contains the McmtMultiTrackerNode class that runs our tracking and
 * re-identification process
 */

#ifndef MCMT_MULTI_TRACKER_NODE_HPP_
#define MCMT_MULTI_TRACKER_NODE_HPP_

// opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

// ros2 header files
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

// local header files
#include <mcmt_track/mcmt_track_utils.hpp>
#include <mcmt_msg/msg/multi_detection_info.hpp>

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt
{
class McmtMultiTrackerNode : public rclcpp::Node {
	public:
		McmtMultiTrackerNode();
		virtual ~McmtMultiTrackerNode() {}

		// declare node parameters
		rclcpp::Node::SharedPtr node_handle_;
		rclcpp::Subscription::<mcmt_msg::msg::MultiDetectionInfo>::SharedPtr detection_sub_;
		std::string topic_name_;

		// declare camera parameters
		cv::VideoCapture cap_;
		int frame_w_, frame_h_, fps_;
		float scale_factor_, aspect_ratio_;
		bool downsample_;

		// declare video parameters
		std::string video_input_1_, video_input_2_;
		std::string output_vid_path_, output_csv_path_1_, output_csv_path_2_;
		bool is_realtime_;
		int FRAME_WIDTH_, FRAME_HEIGHT_;
		cv::VideoWriter recording_;

		// declare ROS2 parameters
		rclcpp::Parameter IS_REALTIME_param, VIDEO_INPUT_1_param, VIDEO_INPUT_2_param, FRAME_WIDTH_param, 
											FRAME_HEIGHT_param, OUTPUT_VIDEO_PATH_param, OUTPUT_CSV_PATH_1_param,
											OUTPUT_CSV_PATH_2_param;

		// declare time variables
		std::chrono::time_point start_, end_;

		// define good tracks
		typedef struct GoodTracks {
			int id;
			int x;
			int y;
		} GoodTracks;

		// declare tracking variables
		std::vector<std::shared_ptr<cv::Mat>> frames_;
		std::vector<std::vector<std::shared_ptr<GoodTracks>>> good_tracks_, filter_good_tracks_;
		std::vector<std::vector<int>> dead_tracks_;
		int frame_count_;

		
		// tracker node methods
		void process_detection_callback();
		void declare_parameters();
		void get_parameters();
		void process_msg_info(mcmt_msg::msg::MultiDetectionInfo::SharedPtr msg);
		void update_cumulative_tracks();
		void prune_tracks();
		void verify_existing_tracks();
		void process_new_tracks();
		void get_total_number_of_tracks();
		void normalise_track_plot();
		void compute_matching_score();
		void geometric_similarity();
		void geometric_similarity_relative();
		void heading_error();
		void calculate_3D();
		void imshow_resized_dual();
		std::string mat_type2encoding(int mat_type)
};
}

#endif	// MCMT_MULTI_TRACKER_NODE_HPP_