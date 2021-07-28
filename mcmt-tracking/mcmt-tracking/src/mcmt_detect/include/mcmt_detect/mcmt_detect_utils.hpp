/**
 * @file multi_cam_detect_utils.hpp
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
 * This file contains the declaration of the classes (Camera and Track) and their
 * associated methods, which is primarily used in the detection pipeline.
 * The full definition of the classes and their methods can be found in the
 * file multi_cam_detect_utils.cpp. 
 */

#ifndef MCMT_DETECT_UTILS_HPP_
#define MCMT_DETECT_UTILS_HPP_

// opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/tracking.hpp>

// standard package imports
#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt {

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
			cv::Mat frame_, frame_ec_, color_converted_, element_;
			std::array<cv::Mat, 2> masked_, removebg_;
			std::string video_input_;
			int cam_index_, frame_w_, frame_h_, fps_, next_id_;
			float scale_factor_, aspect_ratio_;
			bool downsample_;

			// declare tracking variables
			std::vector<std::shared_ptr<mcmt::Track>> tracks_, good_tracks_;
			std::vector<int> dead_tracks_;

			// declare detection variables
			std::array<std::vector<float>,2> sizes_temp_;
			std::vector<float> sizes_;
			std::array<std::vector<cv::Point2f>,2> centroids_temp_;
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
			std::array<cv::Ptr<cv::BackgroundSubtractorMOG2>, 2> fgbg_;
	};
}

#endif			// MCMT_DETECT_UTILS_HPP_