/**
 * @file mcmt_detect_utils.cpp
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
 * This file contains the definition of the classes (Camera and Track) and their
 * associated methods, which is primarily used in the detection pipeline.
 */

// mathematical constants
#define _USE_MATH_DEFINES

// local header files
#include <mcmt_detect/mcmt_detect_utils.hpp>

// standard package imports
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <algorithm>
#include <functional>

using namespace mcmt;

/** 
 * This class is for tracking the detected blobs, and using state estimators 
 * (KF and DCF) to predict the location of the track in the next frame.
*/
Track::Track(int track_id, float size, cv::Point2f centroid, int video_fps,	float sec_filter_delay) {
	vid_fps_ = video_fps;
	sec_filter_delay_ = sec_filter_delay;
	
	// set the id and size of detected track
	id_ = track_id;
	size_ = size;
	age_ = 1;

	// initialize variables to store info on track 
	totalVisibleCount_ = 1;
	consecutiveInvisibleCount_ = 0;
	is_goodtrack_ = false;
	outOfSync_ = false;

	// initialize centroid location
	centroid_ = centroid;
	predicted_ = cv::Point2f(0.0, 0.0);
	
	// initialize kf. We define the KF's parameters for constant velocity model,
	// and inputs detected target's location into the KF tracker.
	kf_ = std::shared_ptr<cv::KalmanFilter>(
		new cv::KalmanFilter(4, 2, 0, CV_32F));
	
	// set transition matrix (F)
	// 	1   0   1   0
	// 	0   1   0   1
	// 	0   0   1   0
	// 	0   0   0   1
	cv::setIdentity(kf_->transitionMatrix);
	kf_->transitionMatrix.at<float>(0, 2) = 1;
	kf_->transitionMatrix.at<float>(1, 3) = 1;

	// set measurement matrix	(H)
	// 	1   0   0   0
	// 	0   1   0   0
	kf_->measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
	kf_->measurementMatrix.at<float>(0, 0) = 1;
	kf_->measurementMatrix.at<float>(1, 1) = 1;

	// set process noise matrix (Q)
	// 	100 0   0   0
	// 	0   100 0   0
	// 	0   0   25  0
	// 	0   0   0   25
	kf_->processNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
	kf_->processNoiseCov.at<float>(0, 0) = 100;
	kf_->processNoiseCov.at<float>(1, 1) = 100;
	kf_->processNoiseCov.at<float>(2, 2) = 25;
	kf_->processNoiseCov.at<float>(3, 3) = 25;

	// set measurement noise covariance matrix (R)
	// 	100   0  
	// 	0   	100
	kf_->measurementNoiseCov.at<float>(0, 0) = 100;
	kf_->measurementNoiseCov.at<float>(0, 1) = 100;
	kf_->measurementNoiseCov.at<float>(1, 0) = 100;
	kf_->measurementNoiseCov.at<float>(1, 1) = 100;

	// set post error covariance matrix
	// 	1   0   0   0
	// 	0   1   0   0
	// 	0   0   1   0
	// 	0   0   0   1
	cv::setIdentity(kf_->errorCovPost, cv::Scalar(1));

	// set pre error covariance matrix
	// 	1	  0   0   0
	// 	0   1   0   0
	// 	0   0   1   0
	// 	0   0   0   1
	cv::setIdentity(kf_->errorCovPost, cv::Scalar(1));

	// input detected centroid location
	// initialize states
	kf_->statePost.at<float>(0) = centroid_.x;
	kf_->statePost.at<float>(1) = centroid_.y;

	// create DCF. we set the box coordinates at the originduring the
	// object class initialization
	tracker_ = cv::TrackerCSRT::create();
	is_dcf_init_ = false;
}

/**
 * This function uses the kalman filter of the track to predict the next known location.
 */
void Track::predictKF()
{
	cv::Mat prediction = kf_->predict();
	predicted_.x = prediction.at<float>(0);
	predicted_.y = prediction.at<float>(1);
}

/**
 * This function uses the kalman filter of the track to update the filter with the measured
 * location of the detected blob in the current frame.
 */
void Track::updateKF(cv::Point2f & measurement)
{
	cv::Mat measure = cv::Mat::zeros(2, 1, CV_32F);
	measure.at<float>(0) = measurement.x;
	measure.at<float>(1) = measurement.y;

	// update
	cv::Mat prediction = kf_->correct(measure);
	centroid_.x = prediction.at<float>(0);
	centroid_.y = prediction.at<float>(1);
}

/**
 * This function uses the DCF of the track to predict the next known location.
 */
void Track::predictDCF(cv::Mat & frame)
{
	if (age_ >= vid_fps_ && is_dcf_init_ == true) {
		bool ok = tracker_->update(frame, box_);
	}
}

/**
 * This function intializes DCF and checks if the DCF measurements are far off from the
 * KF measurements. If the measurements are far off, we will take flag them as outOfSync,
 * and dump the track.
 */
void Track::checkDCF(cv::Point2f & measurement, cv::Mat & frame)
{
	// check if track age is sufficiently large. if it equals to the set prarameter, initialize the DCF tracker
	if (age_ == int(std::max((sec_filter_delay_ * vid_fps_), float(30.0)) - 1)) {
		cv::Rect box_((measurement.x - (size_ / 2)), (measurement.y - (size_ / 2)), size_, size_);
		tracker_->init(frame, box_);
		is_dcf_init_ = true;
	}

	// check if the measured track is not too far away from DCF predicted position. if it is too far away,
	// we will mark it as out of sync with the DCF tracker
	if (age_ >= int(std::max((sec_filter_delay_ * vid_fps_), float(30.0)))) {
		if (((measurement.x < (box_.x - (1 * box_.width))) || 
					(measurement.x > (box_.x + (2 * box_.width)))) &&
				((measurement.y < (box_.y - (1 * box_.height))) ||
					(measurement.y > (box_.y - (2 * box_.height))))) {
			outOfSync_ = true;
		} else {
			outOfSync_ = false;
		}
	}
}

/**
 * This class is our Camera class for storing our detections' and tracks' variables
 */
Camera::Camera(
	int index,
	bool is_realtime,
	std::string video_input,
	int fps,
	int max_frame_width,
	int max_frame_height,
	int fgbg_history,
	float background_ratio,
	int nmixtures
) {
	// get video input and camera index
	video_input_ = video_input;
	cam_index_ = index;

	// open video capturing or video file
	if (is_realtime == true) {
		cap_ = cv::VideoCapture(std::stoi(video_input_));
	} else {
		cap_ = cv::VideoCapture(video_input_);
	}

	// get video parameters
	frame_w_ = int(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
	frame_h_ = int(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
	fps_ = fps;
	scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	aspect_ratio_ = frame_w_ / frame_h_;
	next_id_ = 1000;

	// if video frame size is too big, downsize
	downsample_ = false;
	if ((frame_w_ * frame_h_) > (max_frame_width * max_frame_height)) {
		downsample_ = true;
		frame_w_ = max_frame_width;
		frame_h_ = int(max_frame_width / aspect_ratio_);
		scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	}

	if (!cap_.isOpened()) {
	std::cout << "Error: Cannot open camera! Please check!" << std::endl;
	} else {
		std::cout << "Camera opened successful!" << std::endl;
	}
	
	cap_.set(cv::CAP_PROP_FPS, 30);

	// initialize blob detector
	cv::SimpleBlobDetector::Params blob_params;
	blob_params.filterByConvexity = false;
	blob_params.filterByCircularity = false;
	detector_ = cv::SimpleBlobDetector::create(blob_params);

	// initialize background subtractor
	int hist = int(fgbg_history * fps_);
	double varThresh = double(4 / scale_factor_);
	bool detectShad = false;
	fgbg_ = cv::createBackgroundSubtractorMOG2(hist, varThresh, detectShad);
	fgbg_->setBackgroundRatio(background_ratio);
	fgbg_->setNMixtures(nmixtures);
}