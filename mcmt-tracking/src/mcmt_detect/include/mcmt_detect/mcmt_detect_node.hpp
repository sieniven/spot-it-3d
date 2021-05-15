/** MCMT UVCDriver Node
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the UVCDriver node class that runs our camera, and publish the 
 * raw frames into our ROS2 DDS-RTPS ecosystem.
 */

#ifndef MCMT_UVC_DRIVER_HPP_
#define MCMT_UVC_DRIVER_HPP_

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
#include <mcmt_msg/msg/detection_info.hpp>
#include <mcmt_msg/msg/raw_images.hpp>

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt 
{
class McmtDetectNode : public rclcpp::Node {
  public:
    McmtDetectNode(std::string index);
		
		// declare node parameters
    rclcpp::Node::SharedPtr node_handle_;
		std::string cam_index_, topic_name_;

		// declare video parameters
    cv::VideoCapture cap_;
		cv::Mat frame_, masked_, gray_, mask_, element_, removebg_;
		std::string video_input_;
    int frame_w_, frame_h_, fps_, frame_id_, next_id_;
		float scale_factor_, aspect_ratio_;
		bool is_realtime_, downsample_;

		// declare tracking variables
		std::vector<std::shared_ptr<mcmt::Track>> tracks_, good_tracks_;
		std::vector<int> dead_tracks_;

		// declare detection variables
		std::vector<float> sizes_;
		std::vector<cv::Point2f> centroids_;

		// declare blob detector and background subtractor
		cv::Ptr<cv::SimpleBlobDetector> detector_;
		cv::Ptr<cv::BackgroundSubtractorMOG2> fgbg_;

		// declare tracking variables
		std::vector<int> origin_, unassigned_tracks_, unassigned_detections_;
		// we store the matched track index and detection index in the assigments vector
		std::vector<std::vector<int>> assignments_;
		std::vector<int> tracks_to_be_removed_;

		// declare ROS2 video parameters
		rclcpp::Parameter IS_REALTIME_param, VIDEO_INPUT_param, FRAME_WIDTH_param, FRAME_HEIGHT_param,
											CAM_INDEX_param, VIDEO_FPS_param, MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_param;

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
		rclcpp::Publisher<mcmt_msg::msg::DetectionInfo>::SharedPtr detection_pub_;

		// declare image variables

		// declare node functions
		void declare_parameters();
		void get_parameters();
		void publish_info();

		// declare detection and tracking functions
		void detect_objects();
		cv::Mat remove_ground();
		cv::Mat apply_bg_subtractions();
		void predict_new_locations_of_tracks();
		void detection_to_track_assignment();
		void update_assigned_tracks();
		void update_unassigned_tracks();
		void create_new_tracks();
		void delete_lost_tracks();
		std::vector<std::shared_ptr<mcmt::Track>> filter_tracks();

		// declare utility functions
		double euclideanDist(cv::Point2f & p, cv::Point2f & q);
		std::vector<int> apply_hungarian_algo(std::vector<std::vector<double>> & cost_matrix);
		int average_brightness();
    std::string mat_type2encoding(int mat_type);
		int encoding2mat_type(const std::string & encoding);
};
}

#endif    // MCMT_UVC_DRIVER_HPP_