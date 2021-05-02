/** Detection package utilities
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the classes Camera and Track, used in our main detection
 * and tracking algorithm.
 */

#define _USE_MATH_DEFINES

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
Track::Track(int track_id, float size, cv::Point2f centroid)
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
	is_dcf_init_ = false;
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
Camera::Camera(McmtParams & params, std::string & cam_index, int frame_w, int frame_h)
{
	// initialize McmtParams class
	params_ = params;
	cam_index_ = cam_index;

	// initialize camera video index port
	if (cam_index_ == "0") {
		index_ = params_.VIDEO_INPUT_0_;
	} else {
		index_ = params_.VIDEO_INPUT_1_;
	}

	// initialize camera video parameters
	frame_w_ = frame_w;
	frame_h_ = frame_h;
	fps_ = params_.VIDEO_FPS_;
	scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	aspect_ratio_ = frame_w_ / frame_h_;

	// if video frame size is too big, downsize
	downsample_ = false;
	if ((frame_w_ * frame_h_) > (1920 * 1080)) {
		downsample_ = true;
		frame_w_ = 1920;
		frame_h_ = int(1920 / aspect_ratio_);
		scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	}

	// initialize background subtractor
	fgbg_ = cv::createBackgroundSubtractorMOG2(history = (params_.FGBG_HISTORY_ * params_.VIDEO_FPS_),
																						 varThreshold= (4 / scale_factor_),
																						 detectShadows=false);
	fgbg_->setBackgroundRatio(params_.BACKGROUD_RATIO_);
	fgbg_->setNMixture(params_.NMIXTURES_);

	// initialize blob detector
	cv::SimpleBlobDetector::Params blob_params;
	blob_params.filterByConvexity = false;
	blob_params.filterByCircularity = false;
	detector_ = cv::SimpleBlobDetector::create(blob_params);
	
	// initialize origin (0, 0) and track id
	origin_.push_back(0);
	origin_.push_back(0);
	next_id_ = 1000;

	// initialize kernel used for morphological transformations
	element_ = cv::getStructuringElement(0, cv::Size(5, 5));

	// initialize openCV namedWindows
	cv::namedWindow("Masked " + cam_index_);
	cv::namedWindow("Remove Ground " + cam_index_);
}

/**
 * This is our main detection and tracking algorithm. This function runs for every image callback 
 * that our raw image subscriber receives in the McmtProcessorNode. We run our detection algorithm 
 * (detect_objects()) and tracking algorithm here in this pipelines.
 */
void Camera::detect_and_track()
{
	// initialize new mask of zeros
	mask_ = new cv::Mat::zeros(frame_h_, frame_w_, CV_8UC1);
	
	// detection and tracking pipeline
	detect_objects();
	predict_new_locations_of_tracks();
	detection_to_track_assignment();
	update_assigned_tracks();
	update_unassigned_tracks();
	delete_lost_tracks();
	create_new_tracks();

	// convert masked to BGR
	cv::cvtColor(masked_, masked_, cv::COLOR_GRAY2BGR);

	filter_tracks();

	// show masked frame
	cv::imshow(("Masked " + cam_index_), masked_);

}

/** 
 * This is our main image processing pipeline. The function's pipeline is:
 * 1. Adjust contrast and brightness of the image to make the foreground stand out more
 * 2. Apply background subtraction to remove background from the history of frames
 * 3. Remove noise using morphological transformations 
 * 4. Binarize the image frame by apply thresholding
 * 5. Invert the frame to make black pixels our foreground
 * 6. Apply blob detection to our thresholded binarized frame to detect our targets in the frame
 */
void Camera::detect_objects()
{
	cv::convertScaleAbs(frame_, masked_, alpha=1, beta=0);
	cv::convertScaleAbs(masked_, masked_, alpha=1, 
											beta=(256 - average_brightness() + params_.BRIGHTNESS_GAIN_));

	// subtract background
	fgbg_->apply(masked_, masked_, learningRate=params_.FGBG_LEARNING_RATE_);
	remove_ground();

	// apply morphological transformation
	cv::dilate(masked_, masked_, element_, iterations=params_.DILATION_ITER_);

	// invert frame such taht black pixels are foreground
	cv::bitwise_not(masked_, masked_);

	// apply blob detection
	std::vector<cv::KeyPoint> keypoints;
	detector_->detect(masked_, keypoints);

	// clear vectors to store sizes and centroids of current frame's detected targets
	sizes_.clear();
	centroids_.clear();
	for (auto & it : keypoints) {
		centroids_.push_back(it.pt);
		sizes_.push_back(it.size);
	}
}

/** 
 * This function uses the background subtractor to subtract the history from the current frame.
 * It is implemented inside the "detect_object()" function pipeline.
 */
void Camera::remove_ground()
{
	// declare variables
	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point>> background_contours;

	// number of iterations determines how close objects need to be to be considered background
	cv::dilate(masked_, masked_, element_, iterations=int(params_.REMOVE_GROUND_ITER_ * scale_factor_));
	cv::findContours(masked_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (auto & it : contours) {
		float circularity = (4 * M_PI * cv::contourArea(it)) / (pow(cv::arcLength(it, true), 2));
		if (circularity <= params_.BACKGROUND_CONTOUR_CIRCULARITY_) {
			background_contours.push_back(it);
		}
	}

	cv::Mat bg_removed = frame_.clone();
	cv::drawContours(bg_removed, background_contours, -1, cv::Scalar(255, 0, 0), 3);
	cv::imshow(("Remove Ground " + cam_index_), bg_removed);
}

/**
 * This function calculates the average brightness value of the frame
 */
int Camera::average_brightness()
{
	// declare and initialize required variables
	cv::Mat hist;
	int channels[] = {0};
	int bins[] = {16};
	float hrange = {0, 256};
	int weighted_sum = 0;

	// get grayscale frame and calculate histogram
	cv::cvtColor(frame_, gray_, cv::COLOR_BGR2GRAY);
	cv::calcHist(gray_, 1, channels, mask_, hist, 1, bins, hrange);

	// iterate through each bin
	for (int i=0; i < hist.cols; i++) {
		weighted_sum += hist.at<int>(0, i);
	}
	
	return int((weighted_sum/cv::sum(hist)) * (256/16));
}

/**
 * This function uses the kalman filter and DCF to predict new location of current tracks
 * in the next frame.
 */
void Camera::predict_new_locations_of_tracks()
{
	for (auto & it : tracks_) {
		cv::Mat prediction = it.kf_.predict();
		it.predicted_(prediction.at<float>(0), prediction.at<float>(1));

		cv::Rect2d box;
		if (it.age_ >= params_.VIDEO_FPS_ && it.is_dcf_init_ == true) {
			bool ok = it.tracker_->update(frame_, box);
			if (ok) {
				it.box_ = box;
			}
		}
	}
}

/**
 * This function assigns detections to tracks using Munkre's algorithm. The algorithm is 
 * based on cost calculated euclidean distance, with detections being located too far 
 * away from existing tracks being designated as unassigned detections. Tracks without any
 * nearby detections are also designated as unassigned tracks
 */
void Camera::detection_to_track_assignment()
{

}