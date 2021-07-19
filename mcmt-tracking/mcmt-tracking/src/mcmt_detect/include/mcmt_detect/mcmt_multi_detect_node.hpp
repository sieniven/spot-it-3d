/**
 * @file mcmt_multi_detect_node.hpp
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
 * These classes may be found at mcmt_multi_detect_node.cpp. This code also 
 * contains the McmtMultiDetectNode class that runs our camera, and publishes the 
 * raw frames into our ROS2 DDS-RTPS ecosystem.
 */

#ifndef MCMT_MULTI_DETECTOR_NODE_HPP_
#define MCMT_MULTI_DETECTOR_NODE_HPP_

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
#include <mcmt_detect/mcmt_detect_utils.hpp>
#include <mcmt_msg/msg/multi_detection_info.hpp>

// standard package imports
#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

using namespace std;
using namespace cv;
using namespace mcmt;

namespace mcmt {

	class McmtMultiDetectNode : public rclcpp::Node {
		
		public:
			McmtMultiDetectNode();
			virtual ~McmtMultiDetectNode() {}
				
				// declare node parameters
			rclcpp::Node::SharedPtr node_handle_;
			string topic_name_;

			// declare Camera variables
			vector<shared_ptr<Camera>> cameras_;
			bool is_realtime_;
			int frame_id_;
			string video_input_1_, video_input_2_;
			Mat element_;

			// declare ROS2 video parameters
			rclcpp::Parameter IS_REALTIME_param, VIDEO_INPUT_1_param, VIDEO_INPUT_2_param, FRAME_WIDTH_param, 
						FRAME_HEIGHT_param, VIDEO_FPS_param, MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_param;

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

			// detector function
			void start_record();

		private:
			rclcpp::Publisher<mcmt_msg::msg::MultiDetectionInfo>::SharedPtr detection_pub_;

			// declare node functions
			void declare_parameters();
			void get_parameters();
			void publish_info();

			// declare detection and tracking functions
			void initialize_cameras();
			void sky_saturation(shared_ptr<Camera> & camera);
			void detect_objects(shared_ptr<Camera> & camera);
			Mat remove_ground(shared_ptr<Camera> & camera);
			Mat apply_bg_subtractions(shared_ptr<Camera> & camera);
			void predict_new_locations_of_tracks(shared_ptr<Camera> & camera);
			void clear_track_variables(shared_ptr<Camera> & camera);
			void detection_to_track_assignment_KF(shared_ptr<Camera> & camera);
			void detection_to_track_assignment_DCF(shared_ptr<Camera> & camera);
			void compare_cost_matrices(shared_ptr<Camera> & camera);
			void update_assigned_tracks(shared_ptr<Camera> & camera);
			void update_unassigned_tracks(shared_ptr<Camera> & camera);
			void create_new_tracks(shared_ptr<Camera> & camera);
			void delete_lost_tracks(shared_ptr<Camera> & camera);
			vector<shared_ptr<Track>> filter_tracks(shared_ptr<Camera> & camera);

			// declare utility functions
			double euclideanDist(Point2f & p, Point2f & q);
			vector<int> apply_hungarian_algo(vector<vector<double>> & cost_matrix);
			int average_brightness(shared_ptr<Camera> & camera);
			string mat_type2encoding(int mat_type);
			int encoding2mat_type(const string & encoding);
	};
}

#endif    			// MCMT_MULTI_DETECTOR_NODE_HPP_