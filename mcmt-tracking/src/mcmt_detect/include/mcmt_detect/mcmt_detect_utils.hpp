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
		
		// video parameters
		int vid_fps_;
		float sec_filter_delay_;

		// variable to store predicted and actual locations from kf
		cv::Point2f centroid_, predicted_;

		// declare DCF bool variable
		bool dcf_flag_, is_dcf_init_, outOfSync_;

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
}

#endif			// MCMT_DETECT_UTILS_HPP_