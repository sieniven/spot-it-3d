/** Detection package utilities
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the common functions and classes Camera and Track, used in our 
 * main detection algorithm.
 */

#include <mcmt_detect/mcmt_detect_utils.hpp>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <functional>

using namespace mcmt;

/** 
 * This class is for tracking the detected blobs, and using state estimators 
 * (KF and DCF) to predict the location of the track in the next frame.
*/
Track::Track(int track_id, float size, cv::Point2d centroid)
{
	// set the id and size of detected track
	id_ = track_id;
	size_ = size;
	age_ = 1;

	// initialize variables to store info on track 
	totalVisibleCount_ = 1;
	consecutiveInvisibleCount_ = 0;
	is_goodtrack_ = false
	outOfSync_ = false

	// initialize kalman filter
	createConstantVelocityKF();
	
	// initialize DCF tracker
	tracker_ = cv::TrackerCSRT::create();
	box_(0, 0, 0, 0);
}

/** 
 * This function initializes openCV's Kalman Filter class. We define the 
 * KF's parameters for constant velocity model, and inputs detected target's
 * location into the KF tracker.
 */
void Track::createConstantVelocityKF(){
	kf_(4, 2, 0);
	
	float transitionMatrix[16] = { 1, 0, 1, 0,
																 0, 1, 0, 1,
																 0, 0, 1, 0,
																 0, 0, 0, 1 };

	float measurementMatrix[8] = { 1, 0, 0, 0,
																 0, 1, 0, 0 };
	
	float processNoiseCov[16] = { 100, 0, 0, 0,
																 0, 100, 0, 0,
																 0, 0, 25, 0,
																 0, 0, 0, 25 };
	
	kf_.transitionMatrix = cv::Mat(4, 4, CV_32F, transitionMatrix);
	kf_.measurementMatrix = cv::Mat(2, 4, CV_32F, measurementMatrix);
	kf_.processNoiseCov = cv::Mat(4, 4, CV_32F, processNoiseCov);
	
	cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(100)); 	// R, 2x2
	cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));

	int stateSize = 4;
	state_(stateSize, 1, CV_32F);

	// input detected centroid location
	state_.at<float>(0) = centroid.x;
	state_.at<float>(1) = centroid.y;
	state_.at<float>(2) = 0;
	state_.at<float>(3) = 0;
	kf_.statePost = state_;
}


/** 
 * This class is for keeping track of the respective camera's information. 
 * It stores information of all tracked targets and initializes our blob detector.
 */
Camera::Camera(McmtParams & params, int & cam_index, int frame_w, int frame_h)
{
	if (cam_index == 0) {
		index_ = params.VIDEO_INPUT_0_;
	} else {
		index_ = params.VIDEO_INPUT_1_;
	}
	frame_w_ = frame_w;
	frame_h_ = frame_h;
	fps_ = params.VIDEO_FPS_;
	scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	aspect_ratio_ = frame_w_ / frame_h_;

	downsample_ = false;
	if ((frame_w_ * frame_h_) > (1920 * 1080)) {
		downsample_ = true;
		frame_w_ = 1920;
		frame_h_ = int(1920 / aspect_ratio_);
		scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	}

	// initialize background subtractor
	fgbg_ = cv::createBackgroundSubtractorMOG2(history = (params.FGBG_HISTORY_ * params.VIDEO_FPS_),
																						 varThreshold= (4 / scale_factor_),
																						 detectShadows=false);
	fgbg_->setBackgroundRatio(params.BACKGROUD_RATIO_);
	fgbg_->setNMixture(params.NMIXTURES_);

	// initialize blob detector
	cv::SimpleBlobDetector::Params blob_params;
	blob_params.filterByConvexity = false;
	blob_params.filterByCircularity = false;
	detector_ = cv::SimpleBlobDetector::create(blob_params);
	
	// initialize origin (0, 0) and track id
	origin_.push_back(0);
	origin_.push_back(0);
	next_id_ = 1000;
}

/** 
 * This is our main image processing pipeline. This function adjusts contrast and brightness of the image to
 * make the foreground stand out more. It also applies background subtraction to remove background from history
 * of frames, and removes noise using morphological transformations. We subsequently binarize our image frame by
 * apply thresholding, and invert the frame to make black pixels our foreground. 
 */
void Camera::detect_objects()
{
	masked = 
}

/** 
 * This function uses the background subtractor (fgbg_) to subtract the history from the current frame.
 */
void Camera::remove_ground(int & dilation_iteration, float & background_contour_circularity, int & index)
{

}