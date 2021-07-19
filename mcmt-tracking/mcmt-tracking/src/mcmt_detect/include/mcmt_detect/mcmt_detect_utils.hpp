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

using namespace std;
using namespace cv;

namespace mcmt {

	class Track {
		public:
			Track(
				int track_id, 
				float size,
				Point2f centroid,
				int video_fps,
				float sec_filter_delay);
			
		virtual ~Track() {}
			
			// video parameters
			int vid_fps_;
			float sec_filter_delay_;

			// variable to store predicted and actual locations from kf
			Point2f centroid_, predicted_;

			// declare DCF bool variable
			bool is_dcf_init_, outOfSync_;

			// size of detected blob
			float size_;

			// declare tracking variables
			int id_, age_, totalVisibleCount_, consecutiveInvisibleCount_;
			bool is_goodtrack_;

			// declare class functions
			void predictKF();
			void updateKF(Point2f & measurement);
			void predictDCF(Mat & frame);
			void checkDCF(Point2f & measurement, Mat & frame);

			// declare kf variables
			shared_ptr<KalmanFilter> kf_;

			// declare dcf variables
			Ptr<Tracker> tracker_;
			Rect box_;
	};

	class Camera {
		public:
			Camera(
				int cam_index,
				bool is_realtime,
				string video_input,
				int fps,
				int max_frame_width,
				int max_frame_height,
				int fgbg_history,
				float background_ratio,
				int nmixtures
			);

		virtual ~Camera() {}

			// declare video parameters
			VideoCapture cap_;
			Mat frame_, masked_, gray_, mask_, removebg_;
			string video_input_;
			int cam_index_, frame_w_, frame_h_, fps_, next_id_;
			float scale_factor_, aspect_ratio_;
			bool downsample_;

			// declare tracking variables
			vector<shared_ptr<Track>> tracks_, good_tracks_;
			vector<int> dead_tracks_;

			// declare detection variables
			vector<float> sizes_;
			vector<Point2f> centroids_;

			// declare tracking variables
			vector<int> unassigned_tracks_, unassigned_detections_;
			vector<int> unassigned_tracks_kf_, unassigned_detections_kf_;
			vector<int> unassigned_tracks_dcf_, unassigned_detections_dcf_;
			
			// we store the matched track index and detection index in the assigments vector
			vector<vector<int>> assignments_;
			vector<vector<int>> assignments_kf_;
			vector<vector<int>> assignments_dcf_;
			vector<int> tracks_to_be_removed_;

			// declare blob detector and background subtractor
			Ptr<SimpleBlobDetector> detector_;
			Ptr<BackgroundSubtractorMOG2> fgbg_;
	};
}

#endif			// MCMT_DETECT_UTILS_HPP_