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

	private:
		// declare kf variables
		cv::KalmanFilter kf_;
		cv::Mat state_;

		// declare dcf variables
		cv::Ptr<cv::Tracker> tracker_;
		cv::Rect box_;

		// declare class functions
		void createConstantVelocityKF(cv::Point2f & cen);
		void createDCF();
};

class Camera {
	public:
		Camera(McmtParams & params, int frame_w, int frame_h);
		Camera();
		cv::Mat frame_, masked_, mask_, element_, removebg_;
		int frame_id_;
		McmtParams params_;

		// declare detection and tracking functions
		cv::Mat remove_ground();
		void detect_objects();
		void predict_new_locations_of_tracks();
		void detection_to_track_assignment();
		void update_assigned_tracks();
		void update_unassigned_tracks();
		void delete_lost_tracks();
		void create_new_tracks();
		std::vector<mcmt::Track> filter_tracks();

		// declare utility functions
		float euclideanDist(cv::Point2f & p, cv::Point2f & q);
		std::vector<int> apply_hungarian_algo(std::vector<std::vector<double>> & cost_matrix);

		// declare tracking variables
		std::vector<Track> tracks_, good_tracks_;
		std::vector<int> dead_tracks_;

		// declare blob detector and background subtractor
		cv::Ptr<cv::SimpleBlobDetector> detector_;

		// declare camera variables
		int frame_w_, frame_h_, fps_, next_id_; 
		float scale_factor_, aspect_ratio_;
		bool downsample_;

		// declare detection variables
		std::vector<float> sizes_;
		std::vector<cv::Point2f> centroids_;

		// declare tracking variables
		std::vector<int> origin_, unassigned_tracks_, unassigned_detections_;
		// we store the matched track index and detection index in the assigments vector
		std::vector<std::vector<int>> assignments_;
		std::vector<int> tracks_to_be_removed_;
};
}

#endif			// MCMT_DETECT_UTILS_HPP_