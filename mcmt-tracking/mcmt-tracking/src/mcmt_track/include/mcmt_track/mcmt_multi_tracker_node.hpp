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
#include <cv_bridge/cv_bridge.h>

// local header files
#include <mcmt_track/mcmt_track_utils.hpp>
#include <mcmt_msg/msg/multi_detection_info.hpp>

#include <string>
#include <map>
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
		rclcpp::Subscription<mcmt_msg::msg::MultiDetectionInfo>::SharedPtr detection_sub_;
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

		// define good tracks
		typedef struct GoodTrack {
			int id;
			int x;
			int y;
		} GoodTrack;

		// declare tracking variables
		int frame_count_, next_id_;
		std::vector<int> origin_;
		std::array<std::shared_ptr<mcmt::CameraTracks>, 2> cumulative_tracks_;
		std::array<int, 2> total_tracks_;
		std::map<int, int> matching_dict;

		// declare plotting parameters
		int plot_history_;
		std::vector<std::vector<int>> colors_;
		float font_scale_;
		std::vector<int> shown_indexes_;
		
		// tracker node methods
		void process_msg_info(mcmt_msg::msg::MultiDetectionInfo::SharedPtr msg,
				std::array<std::shared_ptr<cv::Mat>, 2> & frames,
				std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & good_tracks,
				std::array<std::vector<int>, 2> & dead_tracks);

		void process_detection_callback();
		void declare_parameters();
		void get_parameters();
		void update_cumulative_tracks(
			int & index, 
			std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & good_tracks);
		// void prune_tracks();
		// void verify_existing_tracks();
		// void process_new_tracks();
		// void get_total_number_of_tracks();
		// void normalise_track_plot();
		// void compute_matching_score();
		// void geometric_similarity();
		// void geometric_similarity_relative();
		float heading_error(mcmt::TrackPlot & track_plot, mcmt::TrackPlot & alt_track_plot, int & history);
		void calculate_3D();
		void imshow_resized_dual(std::string & window_name, cv::Mat & img);
		int encoding2mat_type(const std::string & encoding);
};
}

#endif	// MCMT_MULTI_TRACKER_NODE_HPP_