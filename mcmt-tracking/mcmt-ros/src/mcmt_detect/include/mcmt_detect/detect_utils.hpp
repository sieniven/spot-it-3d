// Detection package utilities
// Author: Niven Sie, sieniven@gmail.com
// 
// This code contains the common functions and classes Camera and Track, used in our 
// main detection algorithm.

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

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt
{
/** 
 * This class is for tracking the detected blobs, and using state estimators 
 * (KF and DCF) to predict the location of the track in the next frame.
*/
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
}

/** 
 * This class is for keeping track of the respective camera's information. 
 * It stores information of all tracked targets and initializes our blob detector.
 */
class Camera {
	public:
		Camera(const int index, const int fps);

	private:
		// declare required variables
		int index_, frame_w_, frame_h_, fps_, scale_factor_, aspect_ratio_, next_id_;
		bool downsample_;
		std::vector<Track> tracks_, dead_tracks_;
		std::vector<int> origin_;
		cv::Mat mask;

		// declare blob detector and background subtractor
		cv::SimpleBlobDetector::Params params_;
		cv::Ptr<cv::SimpleBlobDetector> detector_;
		cv::Ptr<cv::BackgroundSubtractor> fgbg_;

		// declare class functions
		setup_system_objects();
}

/**
 * This function
 * 
 */

}