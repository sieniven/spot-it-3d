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
#include <algorithm>
#include <functional>

using namespace mcmt;

/** 
 * This class is for tracking the detected blobs, and using state estimators 
 * (KF and DCF) to predict the location of the track in the next frame.
*/
Track::Track(
	int track_id,
	float size,
	cv::Point2f centroid,
	int video_fps,
	float sec_filter_delay)
{
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
	dcf_flag_ = true;
	outOfSync_ = false;

	// initialize centroid location
	centroid_ = centroid;
	
	// initialize kf and DCF
	createConstantVelocityKF(centroid_);
	createDCF();
}

/** 
 * This function initializes openCV's Kalman Filter class. We define the 
 * KF's parameters for constant velocity model, and inputs detected target's
 * location into the KF tracker.
 */
void Track::createConstantVelocityKF(cv::Point2f & cen)
{
	cv::KalmanFilter kf_(4, 2, 0);
	
	kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);
	
	kf_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0);

	kf_.processNoiseCov = (cv::Mat_<float>(4, 4) <<
		100, 0, 0, 0,
		0, 100, 0, 0,
		0, 0, 25, 0,
		0, 0, 0, 25);

	cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(100)); 	// R, 2x2
	cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));

	int stateSize = 2;
	cv::Mat state_(stateSize, 1, CV_32F);

	// input detected centroid location
	state_.at<float>(0) = cen.x;
	state_.at<float>(1) = cen.y;

	kf_.statePost = state_;
}

/**
 * This function initializes openCV's Discriminative Correlation Filter class. We set
 * box coordiantes at origin during the object class initialization
 */
void Track::createDCF()
{
	tracker_ = cv::TrackerCSRT::create();
	dcf_flag_ = true;
	is_dcf_init_ = false;
}

/**
 * This function uses the kalman filter of the track to predict the next known location.
 */
void Track::predictKF()
{
	cv::Mat prediction = kf_.predict();
	std::cout << "hihi" << prediction << std::endl;
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
	cv::Mat prediction = kf_.correct(measure);
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
	if (dcf_flag_ == true) {
		if (age_ == int(std::max((sec_filter_delay_ * vid_fps_), float(30.0)) - 1)) {
			cv::Rect box_((measurement.x - (size_ / 2)), (measurement.y - (size_ / 2)), size_, size_);
			tracker_->init(frame, box_);
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
}