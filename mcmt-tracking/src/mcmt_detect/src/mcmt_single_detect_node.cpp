/** MCMT McmtSingleDetectNode Node
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the McmtSingleDetectNode node class that runs our camera, and publish the 
 * raw frames into our ROS2 DDS-RTPS ecosystem.
 */

#include <mcmt_detect/mcmt_single_detect_node.hpp>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include <memory>
#include <algorithm>
#include <functional>
#include "Hungarian.h"

using namespace mcmt;

McmtSingleDetectNode::McmtSingleDetectNode()
: Node("McmtSingleDetectNode")
{
	node_handle_ = std::shared_ptr<::rclcpp::Node>(this, [](::rclcpp::Node *) {});	
	declare_parameters();
	get_parameters();
	RCLCPP_INFO(this->get_logger(), "Initializing Mcmt Detector Node");

	if (is_realtime_ == true) {
		cap_ = cv::VideoCapture(std::stoi(video_input_));
	} else {
		cap_ = cv::VideoCapture(video_input_);
	}

	frame_w_ = int(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
	frame_h_ = int(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
	fps_ = VIDEO_FPS_;
	scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	aspect_ratio_ = frame_w_ / frame_h_;

	// if video frame size is too big, downsize
	downsample_ = false;
	if ((frame_w_ * frame_h_) > (FRAME_WIDTH_ * FRAME_HEIGHT_)) {
		downsample_ = true;
		frame_w_ = FRAME_WIDTH_;
		frame_h_ = int(FRAME_WIDTH_ / aspect_ratio_);
		scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	}

	// initialize blob detector
	cv::SimpleBlobDetector::Params blob_params;
	blob_params.filterByConvexity = false;
	blob_params.filterByCircularity = false;
	detector_ = cv::SimpleBlobDetector::create(blob_params);

	// initialize background subtractor
	int hist = int(FGBG_HISTORY_ * VIDEO_FPS_);
	double varThresh = double(4 / scale_factor_);
	bool detectShad = false;
	fgbg_ = cv::createBackgroundSubtractorMOG2(hist, varThresh, detectShad);
	fgbg_->setBackgroundRatio(BACKGROUND_RATIO_);
	fgbg_->setNMixtures(NMIXTURES_);

	// initialize origin (0, 0) and track id
	origin_.push_back(0);
	origin_.push_back(0);
	next_id_ = 1000;

	// initialize kernel used for morphological transformations
	element_ = cv::getStructuringElement(0, cv::Size(5, 5));

	if (!cap_.isOpened()) {
    std::cout << "Error: Cannot open camera! Please check!" << std::endl;
  }
	else {
		std::cout << "Camera opened successful!" << std::endl;
	}
	cap_.set(cv::CAP_PROP_FPS, 30);

	// create detection info publisher with topic name "mcmt/detection_info"
	topic_name_ = "mcmt/detection_info";
	detection_pub_ = this->create_publisher<mcmt_msg::msg::SingleDetectionInfo> (topic_name_, 1000);
}

/**
 * This is our main detection and tracking algorithm. This function takes in an image frame 
 * that our camera gets using our openCV video capture. We run our detection algorithm 
 * (detect_objects()) and tracking algorithm here in this pipelines.
 */
void McmtSingleDetectNode::start_record()
{
	frame_id_ = 1;
	while (1) {
		auto start = std::chrono::system_clock::now();
		// get camera frame
		cap_ >> frame_;
		// check if getting frame was successful
		if (frame_.empty()) {
			std::cout << "Error: Video camera is disconnected!" << std::endl;
			break;
		}

		// apply background subtraction
		masked_ = apply_bg_subtractions();

		// clear detection variable vectors
		sizes_.clear();
		centroids_.clear();
		
		// get detections
		detect_objects();
		cv::imshow("Remove Ground", removebg_);
		
		// apply state estimation filters
		predict_new_locations_of_tracks();

		// clear tracking variable vectors
		assignments_.clear();
		unassigned_tracks_.clear();
		unassigned_detections_.clear();
		tracks_to_be_removed_.clear();

		// get cost matrix and match detections and track targets
		detection_to_track_assignment();

		// updated assigned tracks
		update_assigned_tracks();

		// update unassigned tracks, and delete lost tracks
		update_unassigned_tracks();
		delete_lost_tracks();

		// create new tracks
		create_new_tracks();

		// convert masked to BGR
		cv::cvtColor(masked_, masked_, cv::COLOR_GRAY2BGR);

		// filter the tracks
		good_tracks_ = filter_tracks();

		// show masked and frame
		// cv::imshow("Frame", frame_);
		cv::imshow("Masked", masked_);
		cv::waitKey(1);

		// publish detection and tracking information
		publish_info();

		//  spin the McmtSingleDetectNode node once
		rclcpp::spin_some(node_handle_);

		frame_id_++;
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-start;
		std::cout << "Total number of tracks: " << tracks_.size() << std::endl;
		std::cout << "Detection took: " << elapsed_seconds.count() << "s\n";

	}
}

/**
 *  This function to stops the video capture
 */
void McmtSingleDetectNode::stop_record()
{
	std::cout << "Stop capturing camera completed!" << std::endl;
	cap_.release();
}

/**"
 * This function applies background subtraction to the raw image frames to obtain 
 * thresholded mask image.
 */
cv::Mat McmtSingleDetectNode::apply_bg_subtractions()
{
	cv::Mat masked, converted_mask;

	// When there is sunlight reflecting off the target, target becomes white and "blends" into sky
	// Thus, sun compensation is applied when the sunlight level is above a certain threshold
	// Increasing contrast of the frame solves the issue but generates false positives among treeline
	// Instead, the sky is extracted from the frame, and a localised contrast increase applied to it
	// The "non-sky" parts are then restored back to the frame for subsequent masking operations
	if (average_brightness(cv::COLOR_BGR2HSV, 2) > BRIGHTNESS_THRES) {
		extract_sky();
		cv::convertScaleAbs(sky_, sky_, 2);
		cv::add(sky_, non_sky_, masked);
		// Resest the sky and non-sky for future iterations
		sky_ = cv::Scalar(0,0,0);
		non_sky_ = cv::Scalar(0,0,0);
	}
	else {
		// When sun compensation is not active, a simple contrast change is applied to the frame
		cv::convertScaleAbs(frame_, masked);
	}
	cv::convertScaleAbs(masked, masked, 1, (256 - average_brightness(cv::COLOR_BGR2GRAY, 0) + BRIGHTNESS_GAIN_));
	
	// subtract background
	fgbg_->apply(masked, masked, FGBG_LEARNING_RATE_);
	masked.convertTo(converted_mask, CV_8UC1);
	return converted_mask;
}

/**
 * This function takes in a frame, and convert them into two frames
 * One contains the sky while the other contains non-sky components
 * This is part of the sun compensation algorithm.
 */
void McmtSingleDetectNode::extract_sky()
{
	cv::Mat hsv, sky_temp, non_sky_temp;
	
	// Convert image from RGB to HSV
	cv::cvtColor(frame_, hsv, cv::COLOR_BGR2HSV);

	// Threshold the HSV image to extract the sky and put it in sky_ frame
	// The lower bound of V for clear, sunlit sky is given in SKY_THRES
	auto lower = cv::Scalar(0, 0, SKY_THRES);
	auto upper = cv::Scalar(180, 255, 255);
	cv::inRange(hsv, lower, upper, sky_temp);

	// Also extract the non-sky component and put it in non_sky_ frame
	lower = cv::Scalar(0, 0, 0);
	upper = cv::Scalar(180, 255, SKY_THRES);
	cv::inRange(hsv, lower, upper, non_sky_temp);

	// Image opening to remove small patches of sky among the treeline
	// These small patches of sky may become noise if not removed
	cv::morphologyEx(sky_temp, sky_temp, cv::MORPH_OPEN, element_, cv::Point(), DILATION_ITER_);

	// Retrieve original RGB images with extracted sky/non-sky using bitwise and
	cv::bitwise_and(frame_, frame_, sky_, sky_temp);
	cv::bitwise_and(frame_, frame_, non_sky_, non_sky_temp);
}

void McmtSingleDetectNode::detect_objects()
{
	removebg_ = remove_ground();

	// apply morphological transformation
	cv::dilate(masked_, masked_, element_, cv::Point(), DILATION_ITER_);

	// invert frame such that black pixels are foreground
	cv::bitwise_not(masked_, masked_);

	// apply blob detection
	std::vector<cv::KeyPoint> keypoints;
	detector_->detect(masked_, keypoints);

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
cv::Mat McmtSingleDetectNode::remove_ground()
{
	// declare variables
	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point>> background_contours;

	// number of iterations determines how close objects need to be to be considered background
	cv::Mat dilated;
	cv::dilate(masked_, dilated, element_, cv::Point(), int(REMOVE_GROUND_ITER_ * scale_factor_));
	cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (auto & it : contours) {
		float circularity = 4 * M_PI * cv::contourArea(it) / (pow(cv::arcLength(it, true), 2));
		if (circularity <= BACKGROUND_CONTOUR_CIRCULARITY_) {
			background_contours.push_back(it);
		}
	}
	
	// show removed background on raw image frames
	cv::Mat bg_removed = frame_.clone();
	cv::drawContours(bg_removed, background_contours, -1, cv::Scalar(0, 255, 0), 3);

	// draw contours on masked frame to remove background
	cv::drawContours(masked_, background_contours, -1, cv::Scalar(0, 0, 0), -1);
	return bg_removed;
}

/**
 * This function uses the kalman filter and DCF to predict new location of current tracks
 * in the next frame.
 */
void McmtSingleDetectNode::predict_new_locations_of_tracks()
{
	for (auto & it : tracks_) {
		// predict next location using KF and DCF
		it->predictKF();
		it->predictDCF(frame_);
	}
}

/**
 * This function assigns detections to tracks using Munkre's algorithm. The algorithm is 
 * based on cost calculated euclidean distance, with detections being located too far 
 * away from existing tracks being designated as unassigned detections. Tracks without any
 * nearby detections are also designated as unassigned tracks
 */
void McmtSingleDetectNode::detection_to_track_assignment()
{
	// declare non assignment cost
	float cost_of_non_assignment = 10 * scale_factor_;

	// num of tracks and centroids, and get min and std::max sizes
	int num_of_tracks = tracks_.size();
	int num_of_centroids = centroids_.size();
	int total_size = num_of_tracks + num_of_centroids;

	// declare 2-D cost matrix and required variables
	std::vector<std::vector<double>> cost(total_size, std::vector<double>(total_size, 0.0));
	int row_index = 0;
	int col_index = 0;
	std::vector<int> assignments_all;

	// get euclidean distance of every detected centroid with each track's predicted location
	for (auto & track : tracks_) {
		for (auto & centroid : centroids_) {
			cost[row_index][col_index] = euclideanDist(track->predicted_, centroid);
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
	if (total_size > 0) {
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
}

/**
 * This function processes the valid assignments of tracks and detections using the detection
 * and track indices, and updates the tracks with the matched detections
 */
void McmtSingleDetectNode::update_assigned_tracks()
{
	for (auto & assignment : assignments_) {
		int track_index = assignment[0];
		int detection_index = assignment[1];

		cv::Point2f cen = centroids_[detection_index];
		float size = sizes_[detection_index];
		std::shared_ptr<mcmt::Track> track = tracks_[track_index];

		// update kalman filter
		track->updateKF(cen);

		// update DCF
		track->checkDCF(cen, frame_);
		
		// update track info
		track->size_ = size;
		track->age_++;
		track->totalVisibleCount_++;
		track->consecutiveInvisibleCount_ = 0;
	}
}

/**
 * This function updates the unassigned tracks obtained from our detection_to_track_assignments.
 * we process the tracks do not have an existing matched detection in the current frame by 
 * increasing their consecutive invisible count. It also gets any track that has been invisible
 * for too long, and stores them in the vector tracks_to_be_removed_
 */
void McmtSingleDetectNode::update_unassigned_tracks()
{
	int invisible_for_too_long = int(CONSECUTIVE_THRESH_ * VIDEO_FPS_);
	int age_threshold = int(AGE_THRESH_ * VIDEO_FPS_);

	for (auto & track_index : unassigned_tracks_) {
		std::shared_ptr<mcmt::Track> track = tracks_[track_index];
		track->age_++;
		track->consecutiveInvisibleCount_++;
		
		float visibility = float(track->totalVisibleCount_) / float(track->age_);

		// if invisible for too long, append track to be removed
		if ((track->age_ < age_threshold && visibility < VISIBILITY_RATIO_) ||
				(track->consecutiveInvisibleCount_ >= invisible_for_too_long) ||
				(track->outOfSync_ == true)) {
			tracks_to_be_removed_.push_back(track_index);
		}
	}
}

/**
 * This function creates new tracks for detections that are not assigned to any existing
 * track. We will initialize a new Track with the location of the detection
 */
void McmtSingleDetectNode::create_new_tracks()
{
	for (auto & unassigned_detection : unassigned_detections_) {
		cv::Point2f cen = centroids_[unassigned_detection];
		float size = sizes_[unassigned_detection];
		// initialize new track
		auto new_track = std::shared_ptr<Track>(
			new Track(next_id_, size, cen, VIDEO_FPS_, SEC_FILTER_DELAY_));
		tracks_.push_back(new_track);
		next_id_++;
	}
}

/**
 * This function removes the tracks to be removed, in the vector tracks_to_be_removed_
 */
void McmtSingleDetectNode::delete_lost_tracks()
{
	for (auto & track_index : tracks_to_be_removed_) {
		dead_tracks_.push_back(track_index);
		tracks_.erase(tracks_.begin() + track_index);
	}
}

/**
 * This function filters out good tracks to be considered for re-identification. it also
 * filters out noise and only shows tracks that are considered "good" in our output processed
 * frames. it draws bounding boxes into these tracks into our camera frame to continuously 
 * identify and track the detected good tracks
 */
std::vector<std::shared_ptr<Track>> McmtSingleDetectNode::filter_tracks()
{
	std::vector<std::shared_ptr<Track>> good_tracks;
	int min_track_age = int(std::max((AGE_THRESH_ * VIDEO_FPS_), float(30.0)));
	int min_visible_count = int(std::max((VISIBILITY_THRESH_ * VIDEO_FPS_), float(30.0)));

	if (tracks_.size() != 0) {
		for (auto & track : tracks_) {
			if (track->age_ > min_track_age && track->totalVisibleCount_ > min_visible_count) {
				// requirement for track to be considered in re-identification
				// note that min no. of frames being too small may lead to loss of continuous tracking
				if (track->consecutiveInvisibleCount_ <= 5) {
					track->is_goodtrack_ = true;
					good_tracks.push_back(track);
				}
				std::cout << track->size_ << std::endl;
				cv::Point2i rect_top_left((track->centroid_.x - (track->size_ / 2)), 
																	(track->centroid_.y - (track->size_ / 2)));
				
				cv::Point2i rect_bottom_right((track->centroid_.x + (track->size_ / 2)), 
																			(track->centroid_.y + (track->size_ / 2)));
				
				if (track->consecutiveInvisibleCount_ == 0) {
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
double McmtSingleDetectNode::euclideanDist(cv::Point2f & p, cv::Point2f & q) {
	cv::Point2f diff = p - q;
	return sqrt(diff.x*diff.x + diff.y*diff.y);
}

/**
 * This function applies hungarian algorithm to obtain the optimal assignment for 
 * our cost matrix of tracks and detections
 */
std::vector<int> McmtSingleDetectNode::apply_hungarian_algo(
	std::vector<std::vector<double>> & cost_matrix) {
	// declare function variables
	HungarianAlgorithm hungAlgo;
	vector<int> assignment;

	double cost = hungAlgo.Solve(cost_matrix, assignment);
	return assignment;
}

/**
 * This function calculates the average brightness value of the frame.
 * Takes in color conversion type (e.g. BGR2GRAY, BGR2HSV) and pointer to list of
 * color channels that represent brightness (e.g. for HSV, use Channel 2, which is V)
 * Returns the average brightness
 */
int McmtSingleDetectNode::average_brightness(cv::ColorConversionCodes colortype, int channel)
{	
	// declare and initialize required variables
	cv::Mat hist;
	int bins = 16;
	float hrange[] = {0, 256};
	const float* range = {hrange};
	float weighted_sum = 0;
	int chan[1] = {channel};

	// get color converted frame and calculate histogram
	cv::cvtColor(frame_, color_converted_, colortype);
	cv::calcHist(&color_converted_, 1, chan, cv::Mat(), hist, 1, &bins, &range, true, false);
	cv::Scalar total_sum = cv::sum(hist);

	// iterate through each bin
	for (int i=0; i < 16; i++) {
		weighted_sum += (i * (hist.at<float>(i)));
	}
	return int((weighted_sum/total_sum.val[0]) * (256/16));
}

/**
 * This function declares our mcmt software parameters as ROS2 parameters.
 */
void McmtSingleDetectNode::declare_parameters()
{
	// declare ROS2 video parameters
	this->declare_parameter("IS_REALTIME");
	this->declare_parameter("CAMERA_INDEX");
	this->declare_parameter("VIDEO_INPUT");
	this->declare_parameter("FRAME_WIDTH");
	this->declare_parameter("FRAME_HEIGHT");
	this->declare_parameter("VIDEO_FPS");
	this->declare_parameter("MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES");
	
	// declare ROS2 filter parameters
	this->declare_parameter("VISIBILITY_RATIO");
	this->declare_parameter("VISIBILITY_THRESH");
	this->declare_parameter("CONSECUTIVE_THRESH");
	this->declare_parameter("AGE_THRESH");
	this->declare_parameter("SECONDARY_FILTER");
	this->declare_parameter("SEC_FILTER_DELAY");

	// declare ROS2 background subtractor parameters
	this->declare_parameter("FGBG_HISTORY");
	this->declare_parameter("BACKGROUND_RATIO");
	this->declare_parameter("NMIXTURES");
	this->declare_parameter("BRIGHTNESS_GAIN");
	this->declare_parameter("FGBG_LEARNING_RATE");
	this->declare_parameter("DILATION_ITER");
	this->declare_parameter("REMOVE_GROUND_ITER");
	this->declare_parameter("BACKGROUND_CONTOUR_CIRCULARITY");

	// declare sun compensation parameters
	this->declare_parameter("BRIGHTNESS_THRES");
	this->declare_parameter("SKY_THRES");
}

/**
 * This function gets the mcmt parameters from the ROS2 parameters
 */
void McmtSingleDetectNode::get_parameters()
{
	// get video parameters
	IS_REALTIME_param = this->get_parameter("IS_REALTIME");
	VIDEO_INPUT_param = this->get_parameter("VIDEO_INPUT");
	FRAME_WIDTH_param = this->get_parameter("FRAME_WIDTH");
	FRAME_HEIGHT_param = this->get_parameter("FRAME_HEIGHT");
	VIDEO_FPS_param = this->get_parameter("VIDEO_FPS");
	MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_param = this->
		get_parameter("MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES");
	
	// get filter parameters
	VISIBILITY_RATIO_param = this->get_parameter("VISIBILITY_RATIO");
	VISIBILITY_THRESH_param = this->get_parameter("VISIBILITY_THRESH");
	CONSECUTIVE_THRESH_param = this->get_parameter("CONSECUTIVE_THRESH");
	AGE_THRESH_param = this->get_parameter("AGE_THRESH");
	SECONDARY_FILTER_param = this->get_parameter("SECONDARY_FILTER");
	SEC_FILTER_DELAY_param = this->get_parameter("SEC_FILTER_DELAY");

	// get background subtractor parameters
	FGBG_HISTORY_param = this->get_parameter("FGBG_HISTORY");
	BACKGROUND_RATIO_param = this->get_parameter("BACKGROUND_RATIO");
	NMIXTURES_param = this->get_parameter("NMIXTURES");
	BRIGHTNESS_GAIN_param = this->get_parameter("BRIGHTNESS_GAIN");
	FGBG_LEARNING_RATE_param = this->get_parameter("FGBG_LEARNING_RATE");
	DILATION_ITER_param = this->get_parameter("DILATION_ITER");
	REMOVE_GROUND_ITER_param = this->get_parameter("REMOVE_GROUND_ITER");
	BACKGROUND_CONTOUR_CIRCULARITY_param = this->
		get_parameter("BACKGROUND_CONTOUR_CIRCULARITY");

	// get sun compensation params
	BRIGHTNESS_THRES_param = this->get_parameter("BRIGHTNESS_THRES");
	SKY_THRES_param = this->get_parameter("SKY_THRES");

	// initialize and get the parameter values
	FRAME_WIDTH_ = FRAME_WIDTH_param.as_int(),
	FRAME_HEIGHT_ = FRAME_HEIGHT_param.as_int(), 
	VIDEO_FPS_ = VIDEO_FPS_param.as_int(),
	MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_ = 
		MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_param.as_int(),
	VISIBILITY_RATIO_ = VISIBILITY_RATIO_param.as_double(),
	VISIBILITY_THRESH_ = VISIBILITY_THRESH_param.as_double(),
	CONSECUTIVE_THRESH_ = CONSECUTIVE_THRESH_param.as_double(),
	AGE_THRESH_ = AGE_THRESH_param.as_double(),
	SECONDARY_FILTER_ = SECONDARY_FILTER_param.as_int(),
	SEC_FILTER_DELAY_ = SEC_FILTER_DELAY_param.as_double(),
	FGBG_HISTORY_ = FGBG_HISTORY_param.as_int(),
	BACKGROUND_RATIO_ = BACKGROUND_RATIO_param.as_double(),
	NMIXTURES_ = NMIXTURES_param.as_int(),
	BRIGHTNESS_GAIN_ = BRIGHTNESS_GAIN_param.as_int(),
	FGBG_LEARNING_RATE_ = FGBG_LEARNING_RATE_param.as_double(),
	DILATION_ITER_ = DILATION_ITER_param.as_int(),
	REMOVE_GROUND_ITER_ = REMOVE_GROUND_ITER_param.as_double(),
	BACKGROUND_CONTOUR_CIRCULARITY_ = BACKGROUND_CONTOUR_CIRCULARITY_param.as_double();
	BRIGHTNESS_THRES = BRIGHTNESS_THRES_param.as_int();
	SKY_THRES = SKY_THRES_param.as_int();

	// initialize video parameters
	is_realtime_ = IS_REALTIME_param.as_bool();
	video_input_ = VIDEO_INPUT_param.as_string();
}

/**
 * This function publishes detection information into our ROS2 DDS Ecosystem,
 * that is required by our tracking package for re-identification of tracks 
 * between cameras. We will publish information on deadtracks, good tracks 
 * and identified targets in image frames to the tracking package.
 */
void McmtSingleDetectNode::publish_info()
{
	rclcpp::Time timestamp = this->now();
	std_msgs::msg::Header header;
	std::string encoding;
	std::string frame_id_str;

	header.stamp = timestamp;
	frame_id_str = std::to_string(frame_id_);
	header.frame_id = frame_id_str;

	mcmt_msg::msg::SingleDetectionInfo dect_info;

	// convert cv::Mat frame to ROS2 msg type frame
	encoding = mat_type2encoding(frame_.type());
	sensor_msgs::msg::Image::SharedPtr detect_img_msg = cv_bridge::CvImage(
		header, encoding, frame_).toImageMsg();

	std::vector<int16_t> goodtrack_id_list;
	std::vector<int16_t> goodtrack_x_list;
	std::vector<int16_t> goodtrack_y_list;
	std::vector<int16_t> deadtrack_id_list;

	// get good track's information
	for (auto & track : good_tracks_) {
		goodtrack_id_list.push_back(track->id_);
		goodtrack_x_list.push_back(track->centroid_.x);
		goodtrack_y_list.push_back(track->centroid_.y);
	}

	// get gone track ids
	for (auto & deadtrack_index : dead_tracks_) {
		deadtrack_id_list.push_back(deadtrack_index);
	}

	// get SingleDetectionInfo message
	dect_info.header = header;
	dect_info.image = *detect_img_msg;
	dect_info.goodtracks_id = goodtrack_id_list;
	dect_info.goodtracks_x = goodtrack_x_list;
	dect_info.goodtracks_y = goodtrack_y_list;
	dect_info.gonetracks_id = deadtrack_id_list;

	// publish detection info
	detection_pub_->publish(dect_info);
}

std::string McmtSingleDetectNode::mat_type2encoding(int mat_type)
{
	switch (mat_type) {
		case CV_8UC1:
			return "mono8";
		case CV_8UC3:
			return "bgr8";
		case CV_16SC1:
			return "mono16";
		case CV_8UC4:
			return "rgba8";
		default:
			throw std::runtime_error("Unsupported encoding type");
	}
}

int McmtSingleDetectNode::encoding2mat_type(const std::string & encoding)
{
	if (encoding == "mono8") {
		return CV_8UC1;
	} else if (encoding == "bgr8") {
		return CV_8UC3;
	} else if (encoding == "mono16") {
		return CV_16SC1;
	} else if (encoding == "rgba8") {
		return CV_8UC4;
	} else if (encoding == "bgra8") {
		return CV_8UC4;
	} else if (encoding == "32FC1") {
		return CV_32FC1;
	} else if (encoding == "rgb8") {
		return CV_8UC3;
	} else {
		throw std::runtime_error("Unsupported encoding type");
	}
}