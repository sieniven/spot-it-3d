/** MCMT McmtSingleDetectNode Node
 * Authors:
 * Niven Sie (sieniven@gmail.com)
 * Seah Shao Xuan (shaoxuan.seah@gmail.com)
 * Lau Yan Han (sps08.lauyanhan@gmail.com)
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
			std::raise(SIGINT);
			break;
		}

		// apply background subtraction
		masked_ = apply_bg_subtractions();

		// clear detection variable vectors
		sizes_.clear();
		centroids_.clear();
		
		// get detections
		detect_objects();
		// cv::imshow("Remove Ground", removebg_);
		
		// apply state estimation filters
		predict_new_locations_of_tracks();

			// clear tracking variable vectors
			assignments_.clear();
			unassigned_tracks_.clear();
			unassigned_detections_.clear();
			tracks_to_be_removed_.clear();
			
			assignments_kf_.clear();
			unassigned_tracks_kf_.clear();
			unassigned_detections_kf_.clear();

			assignments_dcf_.clear();
			unassigned_tracks_dcf_.clear();
			unassigned_detections_dcf_.clear();

		// get KF cost matrix and match detections and track targets
		detection_to_track_assignment_KF();

		// get DCF cost matrix and match detections and track targets
		detection_to_track_assignment_DCF();

		// compare DCF and KF cost matrix
		compare_cost_matrices();

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
		// cv::imshow("Masked", masked_);
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

	// Apply sun compensation when the sunlight level is above a certain threshold
	if (average_brightness(cv::COLOR_BGR2HSV, 2) > BRIGHTNESS_THRES) {
		masked = apply_sun_compensation();
		cv::imshow("localised contrast frame", masked);
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
 * Apply sun compensation on frame and return the sun compensated frame.
 * Sun compensation is needed when there is sunlight illuminating the target and making
 * it "blend" into sky. Increasing contrast of the entire frame is a solution, but it
 * generates false positives among treeline. Instead, sun compensation works by applying
 * localised contrast increase to regions of the frame identified as sky
 */
cv::Mat McmtSingleDetectNode::apply_sun_compensation()
{
	cv::Mat hsv, sky, non_sky, mask, result;

	result = frame_.clone();
	cv::inRange(result, cv::Scalar(0, 0, 0), cv::Scalar(110, 110, 110), mask);
	result.setTo(cv::Scalar(0, 0, 0), mask);

	cv::cvtColor(result, result, cv::COLOR_BGR2HSV);
	cv::multiply(result, cv::Scalar(1, 0.3, 1), result);
	cv::cvtColor(result, result, cv::COLOR_HSV2BGR);
	
	// // Get HSV version of the frame
	// cv::cvtColor(frame_, hsv, cv::COLOR_BGR2HSV);

	// // Threshold the HSV image to extract the sky and put it in sky frame
	// // The lower bound of V for clear, sunlit sky is given in SKY_THRES
	// // The sky frame is kept in HSV for further operations
	// auto lower = cv::Scalar(0, 0, SKY_THRES);
	// auto upper = cv::Scalar(180, 255, 255);
	// cv::inRange(hsv, lower, upper, mask);
	// cv::bitwise_and(hsv, hsv, sky, mask);
	// // cv::imshow("sky", sky);

	// // Extract the treeline and put it in non_sky frame
	// // The mask for the treeline is the inversion of the sky mask
	// // The treeline is converted back to RGB by using frame_ in the bitwise_and cmd
	// cv::bitwise_not(mask, mask);
	// cv::bitwise_and(frame_, frame_, non_sky, mask);
	// // cv::imshow("non sky", non_sky);

	// // Decrease saturation value of the sky frame to avoid whiteout
	// // After this operation, the sky can be converted back to RGB
	// cv::multiply(sky, cv::Scalar(1, 0.3, 1), sky);
	// cv::cvtColor(sky, sky, cv::COLOR_HSV2BGR);

	// sky.convertTo(sky, -1, MAX_SUN_CONTRAST_GAIN, SUN_BRIGHTNESS_GAIN);
	// cv::erode(sky, sky, cv::getStructuringElement(0, cv::Size(5,5)));

	// // Recombine the sky and treeline
	// // cv::add(blue, non_blue, sky);
	// cv::add(sky, non_sky, result);

	// // Set dark components to black
	// lower = cv::Scalar(0, 0, 0);
	// upper = cv::Scalar(110, 110, 110);
	// cv::inRange(result, lower, upper, mask);
	// result.setTo(cv::Scalar(0, 0, 0), mask);

	return result;
}

/**
 * This function calculates the value of contrast gain to be used in sun compensation
 * The gain is calculated based on the percentage of pixels in the sky that are white
 * The more white areas, the lower the gain should be to reduce probability of saturation
 */
float McmtSingleDetectNode::calc_sun_contrast_gain(cv::Mat sky)
{
	cv::Mat gray_sky, white;

	// Convert the sky_ frame to grayscale to easily determine which pixels are white
	cv::cvtColor(sky, gray_sky, cv::COLOR_BGR2GRAY);
	// cv::imshow("Gray sky", gray_sky);

	// The total number of pixels in the sky (rest of the image is black)
	float total_sky_pix = cv::countNonZero(gray_sky);

	// Threshold the gray sky image to find the white pixels and put them in the "white" frame
	// Empirical observation shows white pixels in sky typically have grayscale value > 175
	auto lower = cv::Scalar(175);
    auto upper = cv::Scalar(255);
    cv::inRange(gray_sky, lower, upper, white);
	// cv::imshow("White parts", white);

	// Total number of white pixels in the sky
	float white_sky_pix = cv::countNonZero(white);

	// Percentage of sky pixels that are white, from 0 - 1
	float white_sky_percent = white_sky_pix / total_sky_pix;

	// Sun contrast gain (a) assumed to have negative linear relationship with white_sky_percent (N)
	// At N = 0, a = max sun contrast gain; At N = 1, a = 1
	return MAX_SUN_CONTRAST_GAIN - (MAX_SUN_CONTRAST_GAIN - 1) * white_sky_percent;
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
 * based on cost calculated euclidean distance between detections and tracks' prediction location
 * based on KF. Detections being located too far away from existing tracks will be designated as
 * unassigned detections. Tracks without any nearby detections are also designated as unassigned tracks
 */
void McmtSingleDetectNode::detection_to_track_assignment_KF()
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
					assignments_kf_.push_back(indexes);
				} else {
					// at the top right of the cost matrix. if detection index is equal to or more than 
					// the no. of detections, then the track is assigned to a dummy detection. As such, 
					// it is an unassigned track
					unassigned_tracks_kf_.push_back(track_index);
				}
			} else {
				// at the lower half of cost matrix. Thus, if detection index is less than no. of 
				// detections, then the detection is assigned to a dummy track. As such it is an 
				// unassigned detections
				if (assignment < num_of_centroids) {
					unassigned_detections_kf_.push_back(assignment);
				}
				// if not, then the last case is when excess dummies are matching with each other. 
				// we will ignore these cases
			}
			track_index++;
		}
	}
}

/**
 * This function assigns detections to tracks using Munkre's algorithm. The algorithm is 
 * based on cost calculated euclidean distance between detections and tracks' prediction location
 * based on DCF. Detections being located too far away from existing tracks will be designated as
 * unassigned detections. Tracks without any nearby detections are also designated as unassigned tracks
 */
void McmtSingleDetectNode::detection_to_track_assignment_DCF()
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
			cv::Point2f point;
			point.x = track->box_.x + (0.5 * track->box_.width);
			point.y = track->box_.y + (0.5 * track->box_.height);
			cost[row_index][col_index] = euclideanDist(point, centroid);
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
					assignments_dcf_.push_back(indexes);
				} else {
					// at the top right of the cost matrix. if detection index is equal to or more than 
					// the no. of detections, then the track is assigned to a dummy detection. As such, 
					// it is an unassigned track
					unassigned_tracks_dcf_.push_back(track_index);
				}
			} else {
				// at the lower half of cost matrix. Thus, if detection index is less than no. of 
				// detections, then the detection is assigned to a dummy track. As such it is an 
				// unassigned detections
				if (assignment < num_of_centroids) {
					unassigned_detections_dcf_.push_back(assignment);
				}
				// if not, then the last case is when excess dummies are matching with each other. 
				// we will ignore these cases
			}
			track_index++;
		}
	}
}

/**
 * This function processes the two cost matrices from the DCF and KF predictions for the matching
 * of detections to tracks. We obtain the final assignments, unassigned_tracks, and unassigned
 * detections vectors from this function.
 */
void McmtSingleDetectNode::compare_cost_matrices()
{
	// check to see if it is the case where there are no assignments in the current frame
	if (assignments_kf_.size() == 0 && assignments_dcf_.size() == 0) {
			assignments_ = assignments_kf_;
			unassigned_tracks_ = unassigned_tracks_kf_;
			unassigned_detections_ = unassigned_detections_kf_;
		return;
	}

	// check to see if assignments by kf and dcf are equal
	else if (assignments_kf_.size() == assignments_dcf_.size()) {
		// get bool if the kf and dcf assignments are equal
		bool is_equal = std::equal(
			assignments_kf_.begin(), assignments_kf_.end(), assignments_dcf_.begin());
		
		if (is_equal == true) {
			// assignments are the same. get the final tracking assignment vectors
			assignments_ = assignments_kf_;
			unassigned_tracks_ = unassigned_tracks_kf_;
			unassigned_detections_ = unassigned_detections_kf_;
			return;
		} 
		else {
			// we will always choose to use detection-to-track assignments using the dcf
			assignments_ = assignments_dcf_;
			unassigned_tracks_ = unassigned_tracks_dcf_;
			unassigned_detections_ = unassigned_detections_dcf_;
			return;
		}
	}

	// when kf and dcf assignments are not zero, and they are not equal as well. in this code
	// block, we know that the dcf and kf differ in terms of assigning the tracks and detections.
	// this means that either the dcf or the kf is able to assign a detection to a track, while the
	// other filter was not able to. in this case, we will loop through their assignments, and 
	// implement all successful assignments to the final tracking assignment vectors
	else {
		// declare flags
		bool different_flag;
		bool different_assignment_flag;
		bool already_assigned_flag;
		std::vector<int> assigned_tracks;
		std::vector<int> assigned_detections;

		// iterate through every dcf assignment
		for (auto & dcf_assignment : assignments_dcf_) {
			different_flag = true;
			different_assignment_flag = false;
			
			// interate through kf assignments
			for (auto & kf_assignment : assignments_kf_) {
				// check if dcf assignment is the same as the kf assignment. if it is, break immediately
				// condition: different_flag = false, different_assignment_flag = false
				if (dcf_assignment == kf_assignment) {
					different_flag = false;
					break;
				}
				// check if dcf assignment track index equal to the kf assignment track index
				// if it is, then the kf and dcf assigned the same track to a different detection
				// condition: different_flag = true, different_assignment_flag = true
				else {
					if (dcf_assignment[0] == kf_assignment[0]) {
						different_assignment_flag = true;
						break;
					}
				}
			}

			// both kf and dcf assigned the same detection to track
			// condition: different_flag = false
			if (different_flag == false) {
				assignments_.push_back(dcf_assignment);
				assigned_tracks.push_back(dcf_assignment[0]);
				assigned_detections.push_back(dcf_assignment[1]);
			}

			// kf and dcf did not assign the same detection to track
			// condition: different_flag = true
			else {
				// see if kf and dcf assigned the track to a different detections
				// condition: different_flag = false, different_assignment_flag = true
				// for this case, we will always go with dcf predictions
				if (different_assignment_flag == true) {
					assignments_.push_back(dcf_assignment);
					assigned_tracks.push_back(dcf_assignment[0]);
					assigned_detections.push_back(dcf_assignment[1]);
				}
				// dcf assigned the track to a detection, but the kf did not. 
				// condition: different_flag = false, different_assignment_flag = false
				// for this case, we will assign the track to the detection that the dcf assigned it to.
				else {
					assignments_.push_back(dcf_assignment);
					assigned_tracks.push_back(dcf_assignment[0]);
					assigned_detections.push_back(dcf_assignment[1]);
				}
			}
		}

		// iterate through every kf assignment. in this iteration, we will find for any tracks that
		// the kf assigned, but the dcf did not
		for (auto & kf_assignment : assignments_kf_) {
			already_assigned_flag = false;
			different_assignment_flag = false;

			// interate through dcf assignments
			for (auto & dcf_assignment : assignments_dcf_) {
				// check if kf assignment is the same as the dcf assignment. if it is, immediately break
				// and move on to the next dcf track
				if (dcf_assignment == kf_assignment) {
					break;
				}
				// check if kf assignment track index equal to the dcf assignment track index
				// if it is, then the kf and dcf assigned the same track to a different detection
				// condition: different_flag = true
				else {
					if (dcf_assignment[0] == kf_assignment[0]) {
						different_assignment_flag = true;
						break;
					}
				}
			}

			// code block are for cases where dcf_assignment and kf_assignment are different, and
			// that the kf assigned the track to a detection, but the dcf did not
			if (already_assigned_flag == false || different_assignment_flag == false) {
				// check first to see if the detection is already part of an assignment
				// if so, do not add it as it potentially results in multiple tracks assigned to a single detection
				if (std::find(assigned_detections.begin(), assigned_detections.end(), kf_assignment[1]) != assigned_detections.end()) {
					// existing assignment to this detection exists. since this is likely a crossover event, we prioritise KF
					// look for confliction DCF assignment
					for (auto & dcf_assignment : assignments_dcf_) {
						if (kf_assignment[1] == dcf_assignment[1]) {
							// once conflicting DCF assignment found, delete it from assignments
							assignments_.erase(std::remove(assignments_.begin(), assignments_.end(), dcf_assignment), assignments_.end());
							assigned_tracks.erase(std::remove(assigned_tracks.begin(), assigned_tracks.end(), dcf_assignment[0]), assigned_tracks.end());
							assigned_detections.erase(std::remove(assigned_detections.begin(), assigned_detections.end(), dcf_assignment[1]), assigned_detections.end());
							// attempt to assign conflicting track with prior KF assignment, if it exists
							// otherwise, don't bother
							for (auto & prior_kf_assignment : assignments_kf_) {
								// check to ensure that the new assignment detection is unassigned
								if (prior_kf_assignment[0] == dcf_assignment[0] && std::find(assigned_detections.begin(), assigned_detections.end(), prior_kf_assignment[1]) != assigned_detections.end()) {
									assignments_.push_back(prior_kf_assignment);
									assigned_tracks.push_back(prior_kf_assignment[0]);
									assigned_detections.push_back(prior_kf_assignment[1]);
									break;
								}
							}
							break;
						}
					}
				}
				// update the KF assignment
				assignments_.push_back(kf_assignment);
				assigned_tracks.push_back(kf_assignment[0]);
				assigned_detections.push_back(kf_assignment[1]);
			}
			// for the case where kf and dcf assigned the same track different detections (condition
			// when different_assignment_flag = true), we will take the dcf assignments
		}

		// get unassigned tracks
		for (auto & unassigned_track : unassigned_tracks_dcf_) {
			if (std::find(assigned_tracks.begin(), assigned_tracks.end(), unassigned_track) != assigned_tracks.end()) {
				continue;
			}
			unassigned_tracks_.push_back(unassigned_track);
		}

		// get unassigned detections
		for (auto & unassigned_detection : unassigned_detections_dcf_) {
			if (std::find(assigned_detections.begin(), assigned_detections.end(), unassigned_detection) != assigned_detections.end()) {
				continue;
			}
			unassigned_detections_.push_back(unassigned_detection);
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
	this->declare_parameter("MAX_SUN_CONTRAST_GAIN");
	this->declare_parameter("SUN_BRIGHTNESS_GAIN");
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
	MAX_SUN_CONTRAST_GAIN_param = this->get_parameter("MAX_SUN_CONTRAST_GAIN");
	SUN_BRIGHTNESS_GAIN_param = this->get_parameter("SUN_BRIGHTNESS_GAIN");

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
	MAX_SUN_CONTRAST_GAIN = MAX_SUN_CONTRAST_GAIN_param.as_double();
	SUN_BRIGHTNESS_GAIN = SUN_BRIGHTNESS_GAIN_param.as_int();

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