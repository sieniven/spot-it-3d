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

namespace mcmt {

	class McmtSingleTrackerNode : public rclcpp::Node {
		public:
			McmtSingleTrackerNode();
			virtual ~McmtSingleTrackerNode() {}

			// declare node parameters
			rclcpp::Node::SharedPtr node_handle_;
			rclcpp::Subscription<mcmt_msg::msg::MultiDetectionInfo>::SharedPtr detection_sub_;
			string topic_name_;

			// declare camera parameters
			VideoCapture cap_;
			int frame_w_, frame_h_, fps_;
			double scale_factor_, aspect_ratio_;
			bool downsample_;

			// declare video parameters
			string video_input_1_, video_input_2_;
			string output_vid_path_, output_csv_path_1_, output_csv_path_2_;
			bool is_realtime_;
			int FRAME_WIDTH_, FRAME_HEIGHT_;
			VideoWriter recording_;

			// declare ROS2 parameters
			rclcpp::Parameter IS_REALTIME_param, VIDEO_INPUT_1_param, VIDEO_INPUT_2_param, FRAME_WIDTH_param, 
												FRAME_HEIGHT_param, OUTPUT_VIDEO_PATH_param, OUTPUT_CSV_PATH_1_param,
												OUTPUT_CSV_PATH_2_param;

			// define good tracks
			typedef struct GoodTrack {
				int id;
				int x;
				int y;
				int size;
			} GoodTrack;

			// declare tracking variables
			int frame_count_, next_id_;
			vector<int> origin_;
			array<shared_ptr<CameraTracks>, 2> cumulative_tracks_;
			array<int, 2> total_tracks_;
			array<map<int, int>, 2> matching_dict_;

			// declare plotting parameters
			int plot_history_;
			vector<vector<int>> colors_;
			double font_scale_;
			vector<int> shown_indexes_;

			// declare colors
			vector<Scalar> colors = {
				Scalar(124, 104, 66), // 1
				Scalar(20, 60, 96), // 2
				Scalar(46, 188, 243), // 3
				Scalar(143, 89, 255), // 4
				Scalar(6, 39, 156), // 5
				Scalar(92, 215, 206), // 6
				Scalar(105, 139, 246), // 7
				Scalar(84, 43, 0), // 8
				Scalar(137, 171, 197), // 9
				Scalar(147, 226, 255) // 10
			};
			
			// tracker node methods
			void process_msg_info(mcmt_msg::msg::MultiDetectionInfo::SharedPtr msg,
					array<shared_ptr<Mat>, 2> & frames,
					array<vector<shared_ptr<GoodTrack>>, 2> & good_tracks,
					array<vector<int>, 2> & dead_tracks);

			void declare_parameters();
			void get_parameters();
			
			void process_detection_callback();
			void update_cumulative_tracks(int index, array<vector<shared_ptr<GoodTrack>>, 2> & good_tracks);
			void prune_tracks(int index);
			void verify_existing_tracks();
			void process_new_tracks(int index, int alt,
				array<vector<shared_ptr<GoodTrack>>, 2> & good_tracks,
				array<vector<shared_ptr<GoodTrack>>, 2> & filter_good_tracks,
				array<vector<int>, 2> & dead_tracks);
			void get_total_number_of_tracks();
			vector<double> normalise_track_plot(shared_ptr<TrackPlot> track_plot);
			double crossCorrelation(vector<double> X, vector<double> Y);
			double compute_matching_score(shared_ptr<TrackPlot> track_plot,
				shared_ptr<TrackPlot> alt_track_plot, int index, int alt);
			double geometric_similarity(vector<shared_ptr<TrackPlot::OtherTrack>> & other_tracks_0,
				vector<shared_ptr<TrackPlot::OtherTrack>> & other_tracks_1);
			double heading_error(shared_ptr<TrackPlot> track_plot, 
				shared_ptr<TrackPlot> alt_track_plot, int history);
			void calculate_3D();
			void print_frame_summary();
			void annotate_frames(array<shared_ptr<Mat>, 2> frames_, array<shared_ptr<CameraTracks>, 2> cumulative_tracks_);
			void graphical_UI(Mat combined_frame, array<shared_ptr<CameraTracks>, 2> cumulative_tracks_);
			void imshow_resized_dual(string & window_name, Mat & img);
			int encoding2mat_type(const string & encoding);

			ofstream trackplot_file;
		
	};
}

#endif	// MCMT_SINGLE_TRACKER_NODE_HPP_