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
#include "Hungarian.h"

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

/** 
 * This class is for keeping track of the respective camera's information. 
 * It stores information of all tracked targets and initializes our blob detector.
 */

// define default constructor
Camera::Camera(){}

Camera::Camera(McmtParams & params, int frame_w, int frame_h)
{
	// initialize McmtParams class
	params_ = params;

	// initialize camera video parameters
	frame_w_ = frame_w;
	frame_h_ = frame_h;
	fps_ = params_.VIDEO_FPS_;
	scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	aspect_ratio_ = frame_w_ / frame_h_;

	// if video frame size is too big, downsize
	downsample_ = false;
	if ((frame_w_ * frame_h_) > (params_.FRAME_WIDTH_ * params_.FRAME_HEIGHT_)) {
		downsample_ = true;
		frame_w_ = params_.FRAME_WIDTH_;
		frame_h_ = int(params_.FRAME_WIDTH_ / aspect_ratio_);
		scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	}

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
	removebg_ = remove_ground();

	// apply morphological transformation
	cv::dilate(masked_, masked_, element_, cv::Point(), params_.DILATION_ITER_);

	// invert frame such that black pixels are foreground
	cv::bitwise_not(masked_, masked_);

	// apply blob detection
	std::vector<cv::KeyPoint> keypoints;
	std::cout << "hi" << std::endl;
	detector_->detect(masked_, keypoints);
	std::cout << "hi" << std::endl;

	// clear vectors to store sizes and centroids of current frame's detected targets
	for (auto & it : keypoints) {
		centroids_.push_back(it.pt);
		sizes_.push_back(it.size);
	}
}

/** 
 * This function uses the background subtractor to subtract the history from the current frame.
 * It is implemented inside the "detect_object()" function pipeline.
 */
cv::Mat Camera::remove_ground()
{
	// declare variables
	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point>> background_contours;

	// number of iterations determines how close objects need to be to be considered background
	cv::dilate(masked_, masked_, element_, 
		cv::Point(), int(params_.REMOVE_GROUND_ITER_ * scale_factor_));
	cv::findContours(masked_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (auto & it : contours) {
		float circularity = 4 * M_PI * cv::contourArea(it) / (pow(cv::arcLength(it, true), 2));
		if (circularity <= params_.BACKGROUND_CONTOUR_CIRCULARITY_) {
			background_contours.push_back(it);
		}
	}

	cv::Mat bg_removed = frame_.clone();
	cv::drawContours(bg_removed, background_contours, -1, cv::Scalar(0, 255, 0), 3);
	return bg_removed;
}

/**
 * This function uses the kalman filter and DCF to predict new location of current tracks
 * in the next frame.
 */
void Camera::predict_new_locations_of_tracks()
{
	for (auto & it : tracks_) {
		// predict next location using KF and DCF
		it.predictKF();
		it.predictDCF(frame_);
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
	// declare non assignment cost
	float cost_of_non_assignment = 10 * scale_factor_;

	// num of tracks and centroids, and get min and std::max sizes
	int num_of_tracks = tracks_.size();
	int num_of_centroids = centroids_.size();
	int total_size = num_of_tracks + num_of_centroids;

	// declare 2-D cost matrix and required variables
	std::vector<double> empty_rows(total_size, 0.0);
	std::vector<std::vector<double>> cost(total_size, empty_rows);
	int row_index = 0;
	int col_index = 0;
	std::vector<int> assignments_all;

	// get euclidean distance of every detected centroid with each track's predicted location
	for (auto & track : tracks_) {
		for (auto & centroid : centroids_) {
			cost[row_index][col_index] = euclideanDist(track.predicted_, centroid);
			col_index++;
		}
		// padding cost matrix with dummy columns to account for unassigned tracks, used to fill
		// the top right corner of the cost matrix
		for (int i = 0; i < num_of_tracks; i++) {
			cost[row_index][col_index] = cost_of_non_assignment;
			col_index++;
		}
		row_index++;
		col_index = 0;
	}

	// padding for cost matrix to account for unassigned detections, used to fill the bottom
	// left corner of the cost matrix
	std::vector<double> append_row;
	for (int i = num_of_tracks; i < total_size; i++) {
		for (int j = 0; j < num_of_centroids; j++) {
			cost[i][j] = cost_of_non_assignment;
		}
	}

	// the bottom right corner of the cost matrix, corresponding to dummy detections being 
	// matched to dummy tracks, is left with 0 cost to ensure that the excess dummies are
	// always matched to each other

	// apply Hungarian algorithm to get assignment of detections to tracked targets
	assignments_all = apply_hungarian_algo(cost);

	int track_index = 0;

	// get assignments, unassigned tracks, and unassigned detections
	for (auto & assignment : assignments_all) {
		if (track_index < num_of_tracks) {
			// track index is less than no. of tracks, thus the current assignment is a valid track
			if (assignment < num_of_centroids) {
				// if in the top left of the cost matrix (no. of tracks x no. of detections), these 
				// assignments are successfully matched detections and tracks. For this, the assigned 
				// detection index must be less than number of centroids. if so, we will store the 
				// track indexes and detection indexes in the assignments vector
				std::vector<int> indexes(2, 0);
				indexes[0] = track_index;
				indexes[1] = assignment;
				assignments_.push_back(indexes);
			} else {
				// at the top right of the cost matrix. if detection index is equal to or more than 
				// the no. of detections, then the track is assigned to a dummy detection. As such, 
				// it is an unassigned track
				unassigned_tracks_.push_back(track_index);
			}
		} else {
			// at the lower half of cost matrix. Thus, if detection index is less than no. of 
			// detections, then the detection is assigned to a dummy track. As such it is an 
			// unassigned detections
			if (assignment < num_of_centroids) {
				unassigned_detections_.push_back(assignment);
			}
			// if not, then the last case is when excess dummies are matching with each other. 
			// we will ignore these cases
		}
		track_index++;
	}
}

/**
 * This function processes the valid assignments of tracks and detections using the detection
 * and track indices, and updates the tracks with the matched detections
 */
void Camera::update_assigned_tracks()
{
	for (auto & assignment : assignments_) {
		int track_index = assignment[0];
		int detection_index = assignment[1];

		cv::Point2f cen = centroids_[detection_index];
		float size = sizes_[detection_index];
		mcmt::Track track = tracks_[track_index];

		// update kalman filter
		track.updateKF(cen);

		// update DCF
		track.checkDCF(cen, frame_);
		
		// update track info
		track.size_ = size;
		track.age_++;
		track.totalVisibleCount_++;
		track.consecutiveInvisibleCount_ = 0;
	}
}

/**
 * This function updates the unassigned tracks obtained from our detection_to_track_assignments.
 * we process the tracks do not have an existing matched detection in the current frame by 
 * increasing their consecutive invisible count. It also gets any track that has been invisible
 * for too long, and stores them in the vector tracks_to_be_removed_
 */
void Camera::update_unassigned_tracks()
{
	int invisible_for_too_long = int(params_.CONSECUTIVE_THRESH_ * params_.VIDEO_FPS_);
	int age_threshold = int(params_.AGE_THRESH_ * params_.VIDEO_FPS_);

	for (auto & track_index : unassigned_tracks_) {
		mcmt::Track track = tracks_[track_index];
		track.age_++;
		track.consecutiveInvisibleCount_++;
		
		int visibility = int(track.totalVisibleCount_ / track.age_);

		// if invisible for too long, append track to be removed
		if ((track.age_ < age_threshold && visibility < params_.VISIBILITY_RATIO_) ||
				(track.consecutiveInvisibleCount_ >= invisible_for_too_long) ||
				(track.outOfSync_ == true)) {
			tracks_to_be_removed_.push_back(track_index);
		}
	}
}

/**
 * This function removes the tracks to be removed, in the vector tracks_to_be_removed_
 */
void Camera::delete_lost_tracks()
{
	for (auto & track_index : tracks_to_be_removed_) {
		dead_tracks_.push_back(track_index);
		tracks_.erase(tracks_.begin() + track_index);
	}
}

/**
 * This function creates new tracks for detections that are not assigned to any existing
 * track. We will initialize a new Track with the location of the detection
 */
void Camera::create_new_tracks()
{
	for (auto & unassigned_detection : unassigned_detections_) {
		cv::Point2f cen = centroids_[unassigned_detection];
		float size = sizes_[unassigned_detection];
		// initialize new track
		mcmt::Track new_track(next_id_, size, cen, params_.VIDEO_FPS_, params_.SEC_FILTER_DELAY_);
		tracks_.push_back(new_track);
		next_id_++;
	}
}

/**
 * This function filters out good tracks to be considered for re-identification. it also
 * filters out noise and only shows tracks that are considered "good" in our output processed
 * frames. it draws bounding boxes into these tracks into our camera frame to continuously 
 * identify and track the detected good tracks
 */
std::vector<mcmt::Track> Camera::filter_tracks()
{
	std::vector<mcmt::Track> good_tracks;
	int min_track_age = int(std::max((params_.AGE_THRESH_ * params_.VIDEO_FPS_), float(30.0)));
	int min_visible_count = int(std::max((params_.VISIBILITY_THRESH_ * params_.VIDEO_FPS_), float(30.0)));

	if (tracks_.size() != 0) {
		for (auto & track : tracks_) {
			if (track.age_ > min_track_age && track.totalVisibleCount_ > min_visible_count) {
				// requirement for track to be considered in re-identification
				// note that min no. of frames being too small may lead to loss of continuous tracking
				if (track.consecutiveInvisibleCount_ <= 5) {
					track.is_goodtrack_ = true;
					good_tracks.push_back(track);
				}
				
				cv::Point2i rect_top_left((track.centroid_.x - (track.size_ / 2)), 
																	(track.centroid_.y - (track.size_ / 2)));
				
				cv::Point2i rect_bottom_right((track.centroid_.x + (track.size_ / 2)), 
																			(track.centroid_.y + (track.size_ / 2)));

				if (track.consecutiveInvisibleCount_ == 0) {
					// green color bounding box if track is detected in the current frame
					cv::rectangle(frame_, rect_top_left, rect_bottom_right, cv::Scalar(0, 255, 0), 1);
					cv::rectangle(masked_, rect_top_left, rect_bottom_right, cv::Scalar(0, 255, 0), 1);
				} else {
					// red color bounding box if track is not detected in the current frame
					cv::rectangle(frame_, rect_top_left, rect_bottom_right, cv::Scalar(0, 0, 255), 1);
					cv::rectangle(masked_, rect_top_left, rect_bottom_right, cv::Scalar(0, 0, 255), 1);
				}
			}
		}
	}
	return good_tracks;
}

/**
 * This function calculates the euclidean distance between two points
 */
float Camera::euclideanDist(cv::Point2f & p, cv::Point2f & q) {
	cv::Point2f diff = p - q;
	return sqrt(diff.x*diff.x + diff.y*diff.y);
}

/**
 * This function applies hungarian algorithm to obtain the optimal assignment for 
 * our cost matrix of tracks and detections
 */
std::vector<int> Camera::apply_hungarian_algo(std::vector<std::vector<double>> & cost_matrix) {
	// declare function variables
	HungarianAlgorithm hungAlgo;
	vector<int> assignment;

	double cost = hungAlgo.Solve(cost_matrix, assignment);
	return assignment;
}