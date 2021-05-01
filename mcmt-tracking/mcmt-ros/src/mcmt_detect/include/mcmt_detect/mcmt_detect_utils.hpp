/** Detection package utilities
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the common functions and classes Camera and Track, used in our 
 * main detection algorithm.
 */

#ifndef MCMT_DETECT_UTILS_HPP_
#define MCMT_DETECT_UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/tracking.hpp>
#include <mcmt_detect/mcmt_params.hpp>
#include <std_msgs/msg/header.hpp>

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt
{
class Track {
	public:
		Track(int track_id, float size);

	private:
		// declare kalman filter and DCF tracker
		cv::KalmanFilter kf_;
		cv::Ptr<cv::Tracker> tracker_;

		// size of detected blob
		float size_;
		int id_, age_, totalVisibleCount_, consecutiveInvisibleCount_;
		bool is_goodtrack_, outOfSync_;
		cv::Rect2d box_;
		cv::Mat state_;

		// declare class functions
		void createConstantVelocityKF();
};


class Camera {
	public:
		Camera(McmtParams & params, int & cam_index, int frame_w, int frame_h);
		cv::Mat frame_, masked_;
		int frame_id_;

		// declare class functions
		void detect_and_track();

	private:
		// declare required variables
		int index_, frame_w_, frame_h_, fps_, next_id_; 
		float scale_factor_, aspect_ratio_;
		bool downsample_;
		std::vector<Track> tracks_, dead_tracks_;
		std::vector<int> origin_;

		// declare blob detector and background subtractor
		cv::Ptr<cv::SimpleBlobDetector> detector_;
		cv::Ptr<cv::BackgroundSubtractor> fgbg_;

		// declare class functions
		void detect_objects();
		void remove_ground(int & dilation_iteration, float & background_contour_circularity, int & index);
		
};

}

#endif			// MCMT_DETECT_UTILS_HPP_