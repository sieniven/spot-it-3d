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
		Track(int track_id, float size, cv::Point2f centroid);
		
		// variable to store predicted locations from kf
		cv::Point predicted_;

		// bool variable to flag if dcf has been initialized
		bool is_dcf_init_;

	private:
		// declare kf variables
		cv::KalmanFilter kf_;
		cv::Mat state_;

		// size of detected blob
		float size_;

		// declare tracking variables
		int id_, age_, totalVisibleCount_, consecutiveInvisibleCount_;
		bool is_goodtrack_;

		// declare dcf variables
		cv::Ptr<cv::Tracker> tracker_;
		bool outOfSync_;
		cv::Rect2d box_;

		// declare class functions
		void createConstantVelocityKF();
};

class Camera {
	public:
		Camera(McmtParams & params, std::string & cam_index, int frame_w, int frame_h);
		cv::Mat frame_, gray_, masked_, mask_, element_;
		int frame_id_;
		std::string cam_index_;
		McmtParams params_;

		// declare class functions
		void detect_and_track();

	private:
		// declare camera variables
		int index_, frame_w_, frame_h_, fps_, next_id_; 
		float scale_factor_, aspect_ratio_;
		bool downsample_;

		// declare tracking variables
		std::vector<Track> tracks_;
		std::vector<int> dead_tracks_, origin_;
		std::vector<float> sizes_;
		std::vector<cv::Point2f> centroids_;

		// declare blob detector and background subtractor
		cv::Ptr<cv::SimpleBlobDetector> detector_;
		cv::Ptr<cv::BackgroundSubtractor> fgbg_;

		// declare class functions
		void detect_objects();
		void remove_ground();
		int average_brightness();
		void predict_new_locations_of_tracks();
		void detection_to_track_assignment();
		void update_assigned_tracks();
		void update_unassigned_tracks();
		void delete_lost_tracks();
		void create_new_tracks();
		void filter_tracks();
};
}

#endif			// MCMT_DETECT_UTILS_HPP_