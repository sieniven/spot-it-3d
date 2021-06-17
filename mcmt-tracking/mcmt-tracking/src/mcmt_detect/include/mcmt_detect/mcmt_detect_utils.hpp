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
#include <opencv2/tracking.hpp>

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
		Track(
			int track_id, 
			float size,
			cv::Point2f centroid,
			int video_fps,
			float sec_filter_delay);
		
  	virtual ~Track() {}
		
		// video parameters
		int vid_fps_;
		float sec_filter_delay_;

		// variable to store predicted and actual locations from kf
		cv::Point2f centroid_, predicted_;

		// declare DCF bool variable
		bool is_dcf_init_, outOfSync_;

		// size of detected blob
		float size_;

		// declare tracking variables
		int id_, age_, totalVisibleCount_, consecutiveInvisibleCount_;
		bool is_goodtrack_;

		// declare class functions
		void predictKF();
		void updateKF(cv::Point2f & measurement);
		void predictDCF(cv::Mat & frame);
		void checkDCF(cv::Point2f & measurement, cv::Mat & frame);

		// declare kf variables
		std::shared_ptr<cv::KalmanFilter> kf_;

		// declare dcf variables
		cv::Ptr<cv::Tracker> tracker_;
		cv::Rect box_;
};

class Camera {
	public:
		Camera(
			int cam_index,
			bool is_realtime,
			std::string video_input,
			int fps,
			int max_frame_width,
			int max_frame_height,
			int fgbg_history,
			float background_ratio,
			int nmixtures
		);

  	virtual ~Camera() {}

		// declare video parameters
    cv::VideoCapture cap_;
		cv::Mat frame_, masked_, gray_, mask_, removebg_;
		std::string video_input_;
    int cam_index_, frame_w_, frame_h_, fps_, next_id_;
		float scale_factor_, aspect_ratio_;
		bool downsample_;

		// declare tracking variables
		std::vector<std::shared_ptr<mcmt::Track>> tracks_, good_tracks_;
		std::vector<int> dead_tracks_;

		// declare detection variables
		std::vector<float> sizes_;
		std::vector<cv::Point2f> centroids_;

		// declare tracking variables
		std::vector<int> unassigned_tracks_, unassigned_detections_;
		std::vector<int> unassigned_tracks_kf_, unassigned_detections_kf_;
		std::vector<int> unassigned_tracks_dcf_, unassigned_detections_dcf_;
		
		// we store the matched track index and detection index in the assigments vector
		std::vector<std::vector<int>> assignments_;
		std::vector<std::vector<int>> assignments_kf_;
		std::vector<std::vector<int>> assignments_dcf_;
		std::vector<int> tracks_to_be_removed_;

		// declare blob detector and background subtractor
		cv::Ptr<cv::SimpleBlobDetector> detector_;
		cv::Ptr<cv::BackgroundSubtractorMOG2> fgbg_;
};
}

#endif			// MCMT_DETECT_UTILS_HPP_