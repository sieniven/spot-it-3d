/** MCMT UVCDriver Node
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the UVCDriver node class that runs our camera, and publish the 
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

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt 
{
class McmtMultiDetectNode : public rclcpp::Node {
  public:
    McmtMultiDetectNode();
		
		// declare node parameters
    rclcpp::Node::SharedPtr node_handle_;
		std::string topic_name_;

		// declare Camera variables
		std::vector<std::shared_ptr<mcmt::Camera>> cameras_;
		bool is_realtime_;
		int frame_id_;
		std::string video_input_1_, video_input_2_;
		cv::Mat element_;

		// declare tracking variables
		std::vector<int> origin_;

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

		// declare ROS2 sun compemsation parameters
		rclcpp::Parameter BRIGHTNESS_THRES_param, SKY_THRES_param;
		
		// declare video parameters
		int FRAME_WIDTH_, FRAME_HEIGHT_, VIDEO_FPS_, MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_;

		// declare filter parameters
		float VISIBILITY_RATIO_, VISIBILITY_THRESH_, CONSECUTIVE_THRESH_, AGE_THRESH_, SEC_FILTER_DELAY_;
		int SECONDARY_FILTER_;
		
		// declare background subtractor parameters
		int FGBG_HISTORY_, NMIXTURES_, BRIGHTNESS_GAIN_, DILATION_ITER_;
		float BACKGROUND_RATIO_, FGBG_LEARNING_RATE_, REMOVE_GROUND_ITER_, BACKGROUND_CONTOUR_CIRCULARITY_;

		// declare sun compensation parameters
		int BRIGHTNESS_THRES, SKY_THRES;

		// detector functions
		void start_record();
    void stop_record();

  private:
		rclcpp::Publisher<mcmt_msg::msg::MultiDetectionInfo>::SharedPtr detection_pub_;

		// declare node functions
		void declare_parameters();
		void get_parameters();
		void publish_info();

		// declare detection and tracking functions
		void detect_objects(std::shared_ptr<mcmt::Camera> & camera);
		cv::Mat remove_ground(std::shared_ptr<mcmt::Camera> & camera);
		cv::Mat apply_bg_subtractions(std::shared_ptr<mcmt::Camera> & camera);
		void extract_sky(std::shared_ptr<mcmt::Camera> & camera);
		void predict_new_locations_of_tracks(std::shared_ptr<mcmt::Camera> & camera);
		void detection_to_track_assignment(std::shared_ptr<mcmt::Camera> & camera);
		void update_assigned_tracks(std::shared_ptr<mcmt::Camera> & camera);
		void update_unassigned_tracks(std::shared_ptr<mcmt::Camera> & camera);
		void create_new_tracks(std::shared_ptr<mcmt::Camera> & camera);
		void delete_lost_tracks(std::shared_ptr<mcmt::Camera> & camera);
		std::vector<std::shared_ptr<mcmt::Track>> filter_tracks(std::shared_ptr<mcmt::Camera> & camera);

		// declare utility functions
		double euclideanDist(cv::Point2f & p, cv::Point2f & q);
		std::vector<int> apply_hungarian_algo(std::vector<std::vector<double>> & cost_matrix);
		int average_brightness(std::shared_ptr<mcmt::Camera> & camera,
			cv::ColorConversionCodes colortype, int channel);
    std::string mat_type2encoding(int mat_type);
		int encoding2mat_type(const std::string & encoding);
};
}

#endif    // MCMT_MULTI_DETECTOR_NODE_HPP_