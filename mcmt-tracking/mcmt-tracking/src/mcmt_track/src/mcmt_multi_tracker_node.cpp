/**
 * @file mcmt_multi_tracker_node.cpp
 * @author Niven Sie, sieniven@gmail.com
 * @author Seah Shao Xuan
 * 
 * This code contains the McmtMultiTrackerNode class that runs our tracking and
 * re-identification process
 */

#include <mcmt_track/mcmt_multi_tracker_node.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <csignal>
#include <chrono>
#include <math.h>
#include <memory>
#include <map>
#include <set>
#include <algorithm>
#include <functional>
#include <eigen3/Eigen/Dense>

using namespace mcmt;

McmtMultiTrackerNode::McmtMultiTrackerNode()
:Node("MultiTrackerNode")
{
	node_handle_ = std::shared_ptr<rclcpp::Node>(this, [](::rclcpp::Node *) {});
	declare_parameters();
	get_parameters();
	RCLCPP_INFO(this->get_logger(), "Initializing Mcmt Multi Tracker Node");

	// get camera parameters
	if (is_realtime_ == true) {
		cap_ = cv::VideoCapture(std::stoi(video_input_1_));
	} else {
		cap_ = cv::VideoCapture(video_input_1_);
	}

	frame_w_ = int(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
	frame_h_ = int(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
	scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	aspect_ratio_ = frame_w_ / frame_h_;
	fps_ = int(cap_.get(cv::CAP_PROP_FPS));
	// fps_ = VIDEO_FPS_;

	// if video frame size is too big, downsize
	downsample_ = false;
	if ((frame_w_ * frame_h_) > (FRAME_WIDTH_ * FRAME_HEIGHT_)) {
		downsample_ = true;
		frame_w_ = FRAME_WIDTH_;
		frame_h_ = int(FRAME_WIDTH_ / aspect_ratio_);
		scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	}

	// intialize video writer;
	recording_.open(output_vid_path_, cv::VideoWriter::fourcc('M','J','P','G'), fps_, 
		cv::Size(int(frame_w_ * 2), frame_h_));
	cap_.release();

	// initialize origin (0, 0), frame count and track id
	origin_.push_back(0);
	origin_.push_back(0);
	next_id_ = 0;
	frame_count_ = 0;

	// initialze plotting parameters
	plot_history_ = 200;
	font_scale_ = 0.5;
	for (int i = 0; i < plot_history_; i++) {
		colors_.push_back(scalar_to_rgb(i, plot_history_));
	}

	// initialize cumulative camera tracks
	cumulative_tracks_[0] = std::shared_ptr<CameraTracks>(new CameraTracks(0));
	cumulative_tracks_[1] = std::shared_ptr<CameraTracks>(new CameraTracks(1));

	process_detection_callback();
}

/**
 * This function creates subscription to the detection info publisher from the McmtMultiDetectorNode
 * and contains the main pipeline for the tracker node callback. The pipeline includes:
 * 1. Processing of new tracks
 * 2. Re-identification between tracks
 * 3. Plotting of info on each camera's frames
 */
void McmtMultiTrackerNode::process_detection_callback()
{
	// set qos and topic name
	auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
	topic_name_ = "mcmt/detection_info";
	detection_sub_ = this->create_subscription<mcmt_msg::msg::MultiDetectionInfo> (
		topic_name_,
		qos,
		[this](const mcmt_msg::msg::MultiDetectionInfo::SharedPtr msg) -> void
		{
			// get start time
			auto start = std::chrono::system_clock::now();

			// declare tracking arrays
			std::array<std::shared_ptr<cv::Mat>, 2> frames_;
			std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> good_tracks_, filter_good_tracks_;
			std::array<std::vector<int>, 2> dead_tracks_;

			// process detection info
			process_msg_info(msg, frames_, good_tracks_, dead_tracks_);

			// create filter copy of good_tracks list
			filter_good_tracks_[0] = good_tracks_[0];
			filter_good_tracks_[1] = good_tracks_[1];

			update_cumulative_tracks(0, good_tracks_);
			update_cumulative_tracks(1, good_tracks_);

			process_new_tracks(0, 1, good_tracks_, filter_good_tracks_, dead_tracks_);
			process_new_tracks(1, 0, good_tracks_, filter_good_tracks_, dead_tracks_);

			verify_existing_tracks();

			calculate_3D();

			prune_tracks(0, good_tracks_);
			prune_tracks(1, good_tracks_);

			// draw tracks on opencv GUI to monitor the detected tracks
			// lopp through each camera frame
			std::map<int, std::shared_ptr<mcmt::TrackPlot>>::iterator track;
			for (int i = 0; i < 2; i++) {
				cv::putText(*frames_[i].get(), "CAMERA " + std::to_string(i), cv::Point(20, 30),
					cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 0.85, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
				
				cv::putText(*frames_[i].get(), "Frame Count " + std::to_string(frame_count_), cv::Point(20, 50),
					cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 0.85, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
				
				// loop through every track plot
				if (cumulative_tracks_[i]->track_plots_.empty() == false) {
					for (track = cumulative_tracks_[i]->track_plots_.begin(); 
						track != cumulative_tracks_[i]->track_plots_.end(); track++) {
						if ((frame_count_ - track->second->lastSeen_) <= fps_) {
							// get last frames up till plot history (200)
							shown_indexes_.clear();

							for(int j = track->second->frameNos_.size() - 1; j >= 0; j--) {
								if (track->second->frameNos_[j] > (frame_count_ - plot_history_)) {
									shown_indexes_.push_back(j);
								} else {
									break;
								}
							}

							// draw the track's path history on opencv GUI
							for (auto & idx : shown_indexes_) {
								int color_idx = track->second->frameNos_[idx] - frame_count_ + plot_history_ - 1;
								cv::circle(*frames_[i].get(), cv::Point(track->second->xs_[idx], track->second->ys_[idx]), 3,
									cv::Scalar(colors_[color_idx][2], colors_[color_idx][1], colors_[color_idx][0]), -1);
							}
							
							// put ID and XYZ coordinates on opencv GUI
							if (shown_indexes_.empty() == false) {
								cv::putText(*frames_[i].get(), "ID: " + std::to_string(track->second->id_), 
									cv::Point(track->second->xs_.back(), track->second->ys_.back() + 15), cv::FONT_HERSHEY_SIMPLEX,
									font_scale_, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
								
								if (track->second->xyz_.empty() == false) {
									cv::putText(*frames_[i].get(), "X: " + std::to_string(track->second->xyz_[0]),
										cv::Point(track->second->xs_.back(), track->second->ys_.back() + 30), cv::FONT_HERSHEY_SIMPLEX,
										font_scale_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

									cv::putText(*frames_[i].get(), "Y: " + std::to_string(track->second->xyz_[1]),
										cv::Point(track->second->xs_.back(), track->second->ys_.back() + 45), cv::FONT_HERSHEY_SIMPLEX,
										font_scale_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

									cv::putText(*frames_[i].get(), "Z: " + std::to_string(track->second->xyz_[2]),
										cv::Point(track->second->xs_.back(), track->second->ys_.back() + 60), cv::FONT_HERSHEY_SIMPLEX,
										font_scale_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
								}
							}
						}
					}
				}
			}

			// get trackplot process time
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - start;
			std::cout << "Trackplot process took: " << elapsed_seconds.count() << "s\n";

			// show and save video combined tracking frame
			frame_count_ += 1;
			cv::Mat combined_frame;
			cv::hconcat(*frames_[0].get(), *frames_[1].get(), combined_frame);
			std::string window_name = "Detection";
			imshow_resized_dual(window_name, combined_frame);
			cv::waitKey(1);
		}
	);
}

/**
 * This function processes the detection messsage information
 */
void McmtMultiTrackerNode::process_msg_info(mcmt_msg::msg::MultiDetectionInfo::SharedPtr msg,
	std::array<std::shared_ptr<cv::Mat>, 2> & frames,
	std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & good_tracks,
	std::array<std::vector<int>, 2> & dead_tracks)
{
	// get both camera frames
	auto frame_1 = std::shared_ptr<cv::Mat>(
		new cv::Mat(
			msg->image_one.height, msg->image_one.width, encoding2mat_type(msg->image_one.encoding),
			const_cast<unsigned char *>(msg->image_one.data.data()), msg->image_one.step));
	
	auto frame_2 = std::shared_ptr<cv::Mat>(
		new cv::Mat(
			msg->image_two.height, msg->image_two.width, encoding2mat_type(msg->image_two.encoding),
			const_cast<unsigned char *>(msg->image_two.data.data()), msg->image_two.step));
	
	frames[0] = frame_1;
	frames[1] = frame_2;

	// get dead tracks
	for (auto & it : msg->gonetracks_id_one) {
		dead_tracks[0].push_back(int(it));
	}

	for (auto & it : msg->gonetracks_id_two) {
		dead_tracks[1].push_back(int(it));
	}

	// get good tracks
	int total_num_tracks = msg->goodtracks_id_one.size();
	for (int i = 0; i < total_num_tracks; i++) {
		auto good_track = std::shared_ptr<GoodTrack>(new GoodTrack());
		good_track->id = msg->goodtracks_id_one[i];
		good_track->x = msg->goodtracks_x_one[i];
		good_track->y = msg->goodtracks_y_one[i];
		good_tracks[0].push_back(good_track);
	}

	total_num_tracks = msg->goodtracks_id_two.size();
	for (int i = 0; i < total_num_tracks; i++) {
		auto good_track = std::shared_ptr<GoodTrack>(new GoodTrack());
		good_track->id = msg->goodtracks_id_two[i];
		good_track->x = msg->goodtracks_x_two[i];
		good_track->y = msg->goodtracks_y_two[i];
		good_tracks[1].push_back(good_track);
	}
}

/**
 * this function creates new tracks and the addition to the cumulative tracks log for each frame
 */
void McmtMultiTrackerNode::update_cumulative_tracks(
	int index,
	std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & good_tracks)
{
	int track_id, centroid_x, centroid_y;
	for (auto & track : good_tracks[index]) {
		track_id = track->id;
		centroid_x = track->x;
		centroid_y = track->y;

		// occurance of a new track
		if (matching_dict_[index].find(track_id) == matching_dict_[index].end()) {
			cumulative_tracks_[index]->track_new_plots_[track_id] = std::shared_ptr<TrackPlot>(
				new TrackPlot(track_id));
			matching_dict_[index][track_id] = track_id;
		}
	}
}

/**
 * this function removes dead tracks if they have not appeared for more than 300 frames
 */
void McmtMultiTrackerNode::prune_tracks(
	int index,
	std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & good_tracks)
{
	std::vector<int> prune;
	std::map<int, int>::iterator track;

	// prune dead tracks if they have not appeared for more than 300 frames
	for (track = cumulative_tracks_[index]->track_new_plots_.begin();
		track != cumulative_tracks_[index]->track_new_plots_.end(); track++) {
		if (frame_count_ - track->second->lastSeen_) > 60 {
			prune.push_back(track-second->id_);
		}
	}

	for (auto & track_id : prune) {
		cumulative_tracks_[index]->track_new_plots_.erase(track_id);
		matching_dict_[index].erase(track_id);
		std::cout << track_id << " pruned" << std::endl;
	}
}

/**
 * checks matched tracks to see if they are still valid. also checks if multiple tracks
 * within each camera are tracking the same target
 */
void McmtMultiTrackerNode::verify_existing_tracks()
{
	std::map<int, std::shared_ptr<mcmt::TrackPlot>>::iterator track, other_track;
	std::vector<int> matched_ids;

	for (track = cumulative_tracks_[0]->track_plots_.begin(); 
		track != cumulative_track[0]->track_plots_.end(); track++) {
		// search if alternate trackplot contains the same key
		other_track = cumulative_tracks_[1]->track_plots_.find(track->first);
		
		if (other_track != cumulative_tracks_[1]->track_plots_.end()) {
			// alternate trackplot contains the same key
			matched_ids.push_back(track->first);
		}
	}

	for (auto & matched_id : matched_ids) {
		std::shared_ptr<TrackPlot> track_plot_0 = cumulative_tracks[0]->track_plots_[matched_id];
		std::shared_ptr<TrackPlot> track_plot_1 = cumulative_tracks[1]->track_plots_[matched_id];

		// normalization of cross correlation values
		track_plot_normalize_xj = normalise_track_plot(track_plot_0);
		alt_track_plot_normalize_xj = normalise_track_plot(track_plot_1);

		// track feature variable correlation strength
		auto r_value = correlationCoefficient(track_plot_normalize_xj,
			alt_track_plot_normalize_xj, track_plot_normalize_xj.size());

		// heading deviation error score
		float heading_err = heading_error(track_plot_0, track_plot_1, 30);

		if ((r_value < 0.4 && heading_err > 0.1 && track_plot_0->frameNos_.size() > 180 && track_plot_1->frameNos_.size() > 180) || 
			(track_plot_0.check_stationary() != track_plot_1.check_stationary())) {
			track_plot_0->mismatch_count_ += 1;
			track_plot_1->mismatch_count_ += 1;
		} else {
			track_plot_0->mismatch_count_ = 0;
			track_plot_1->mismatch_count_ = 0;
		}

		if (track_plot_0->mismatch_count_ >= 30 && track_plot_1->mismatch_count_ >= 30) {
			track_plot_0->mismatch_count_ = 0;
			track_plot_1->mismatch_count_ = 0;

			std::map<int, int>::iterator it;
			for (it = matching_dict_[0].begin(); it != matching_dict_[0].end(); it++) {
				if (it->second == track_plot_0->id_) {
					int original_track_id_0 = it->first;
					break;
				}
			}

			for (it = matching_dict_[1].begin(); it != matching_dict_[1].end(); it++) {
				if (it->second == track_plot_1->id_) {
					int original_track_id_1 = it->first;
					break;
				}
			}

			track_plot_0->id_ = original_track_id_0;
			track_plot_1->id_ = original_track_id_1;

			cumulative_tracks_[0]->track_new_plots_[original_track_id_0] = track_plot_0;
			cumulative_tracks_[0]->track_plots_.erase(matched_id);

			cumulative_tracks_[1]->track_new_plots_[original_track_id_1] = track_plot_1;
			cumulative_tracks_[1]->track_plots_.erase(matched_id);

			matching_dict_[0][original_track_id_0] = original_track_id_0;
			matching_dict_[1][original_track_id_1] = original_track_id_1;
	}
}

void McmtMultiTrackerNode::process_new_tracks(
	int index, int alt,
	std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & good_tracks,
	std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & filter_good_tracks,
	std::array<std::vector<int>, 2> & dead_tracks)
{
	get_total_number_of_tracks();
	std::map<int, std::map<int, float>> corrValues;
	std::set<int> removeSet;
	int track_id, centroid_x, centroid_y;
	int row = 0;

	for (auto & track : good_tracks[index]) {
		// Extract details from the track
		track_id = track->id;
		centroid_x = track->x;
		centroid_y = track->y;

		if (std::find(cumulative_tracks_[index]->track_plots_.begin(),
									cumulative_tracks_[index]->track_plots_.end(),
									matching_dict[index][track_id]) != cumulative_tracks_[index]->track_plots_.end())
		{

			// Update track_new_plots with centroid and feature variable of every new frame
			auto track_plot = cumulative_tracks_[index]->track_new_plots_[track_id];
			track_plot.update(std::vector(centroid_x, centroid_y), frame_count_);
			track_plot.calculate_track_feature_variable(frame_count_, fps_);

			// Check if track feature variable has non-zero elements
			float sum = 0 for (int i = 0; i < track_plot.track_feature_variable_.size(); i++)
			{
				sum += track_plot.track_feature_variable_[i];
			}

			// if track is not a new track, we use 90 frames as the minimum requirement before matching occurs
			if (track_plot.frameNos_.size() >= 30 && track_plot.track_feature_variable_.size() >= 30 && sum != 0)
			{
				// look into 2nd camera's new tracks (new tracks first)
				for (auto &it : alt_track_plot : cumulative_tracks_[alt]->track_new_plots_)
				{

					sum = 0;
					for (int i = 0; i < alt_track_plot.track_feature_variable_.size(); i++)
					{
						sum += alt_track_plot.track_feature_variable_[i];
					}
					// track in 2nd camera must fulfills requirements to have at least 90 frames to match
					if (alt_track_plot.frameNos_.size() >= 30 && alt_track_plot.track_feature_variable_.size() >= 30 && sum != 0)
					{

						float score = compute_matching_score(track_plot, alt_track_plot, index, alt);
						if (score != 0)
						{
							corrValues[track_id][alt_track_plot.id_] = score;
						}
					}
				}
				// look into other camera's matched tracks list (old tracks last)
				for (auto &it : alt_track_plot : cumulative_tracks_[alt]->track_plots_)
				{

					bool eligibility_flag = true;
					// do not consider dead tracks from the other camera
					for (auto &it : dead_track : dead_tracks[alt])
					{
						if (matching_dict[alt][dead_track->id] == alt_track_plot.id_)
						{
							eligibility_flag = false; // 2nd camera's track has already been lost. skip the process of matching for this track
						}
					}

					for (auto &it : alt_track : good_tracks[index])
					{
						if (alt_track_plot.id_ == matching_dict[index][alt_track->id])
						{
							eligibility_flag = false; // 2nd camera's track has already been matched. skip the process of matching for this track
						}
					}

					sum = 0;
					for (int i = 0; i < alt_track_plot.track_feature_variable_.size(); i++)
					{
						sum += alt_track_plot.track_feature_variable_[i];
					}

					if (!eligibility_flag && sum != 0)
					{
						float score = compute_matching_score(track_plot, alt_track_plot, index, alt);
						if (score != 0)
						{
							corrValues[track_id][alt_track_plot.id_] = score;
						}
					}
				}
			}

			row += 1;
		}

		else
		{
			auto track_plot = cumulative_tracks_[index]->track_plots_[matching_dict[index][track_id]];
			track_plot.update(std::vector(centroid_x, centroid_y), frame_count_);
			track_plot.calculate_track_feature_variable(frame_count_, fps_);
			filter_good_tracks[index].erase(filter_good_tracks[index].begin() + row);
		}

		for (auto &it : track : filter_good_tracks[index])
		{
			std::map<int, float> maxValues = corrValues[track->id];
			int maxID = -1;
			int global_max_flag = 0;

			// for the selected max track in the 2nd camera, we check to see if the track has a higher
			// cross correlation value with another track in current camera

			while (global_max_flag == 0 && maxValues.size() != 0)
			{
				maxID = std::max_element(maxValues.begin(), maxValues.end(),
																 [](const pair<int, float> &p1, const pair<int, float> &p2)
																 { return p1.second < p2.second; });

				float maxValue = maxValues[maxID];

				// search through current camera's tracks again, for the selected track that we wish to re-id with.
				// we can note that if there is a track in the current camera that has a higher cross correlation value
				// than the track we wish to match with, then the matching will not occur.

				for (auto &it : track_1 : filter_good_tracks[index])
				{
					if (std::find(corrValues[track_1->id].begin(), corrValues[track_1->id].end(), maxID) != corrValues[track_1->id].end())
					{
						if (corrValues[track_1->id][maxID] > maxValue)
						{
							maxValues.erase(maxID);
							global_max_flag = 1;
							break;
						}
					}
				}

				if (global_max_flag == 1)
				{
					// there existed a value larger than the current maxValue. thus, re-id cannot occur
					global_max_flag = 0;
					continue;
				}
				else
				{
					// went through the whole loop without breaking, thus it is the maximum value. re-id can occur
					global_max_flag = 2;
				}
			}

			// re-id process
			if (global_max_flag == 2)
			{

				// if track is in 2nd camera's new track list
				if (maxID != 1 && std::find(cumulative_tracks_[alt]->track_new_plots_.begin(),
																		cumulative_tracks_[alt]->track_new_plots_.end(), maxID) !=
															cumulative_tracks_[alt]->track_new_plots_.end())
				{

					// remove track plot in new tracks' list and add into matched tracks' list for alternate camera
					cumulative_tracks_[alt]->track_new_plots_[maxID].id_ = next_id_;
					cumulative_tracks_[alt]->track_plots_.insert({next_id_, cumulative_tracks_[alt]->track_new_plots_[maxID]});
					// update dictionary matching
					matching_dict[alt][maxID] = next_id_;
					removeSet.insert(maxID);

					// remove track plot in new tracks' list and add into matched tracks' list for current camera
					int track_id = track->id;
					auto track_plot = cumulative_tracks_[index]->track_new_plots_[track_id];
					track_plot.id_ = next_id_;

					cumulative_tracks_[index]->track_plots_.insert({next_id_, track_plot});
					cumulative_tracks_[index]->track_new_plots_.erase(track_id);

					// update dictionary matching list
					matching_dict[index][track_id] = next_id_;
					next_id_ += 1;
				}

				// if track is in 2nd camera's matched track list
				else
				{
					int track_id = track->id;
					auto track_plot = cumulative_tracks_[index]->track_new_plots_[track_id];
					track_plot.id_ = cumulative_tracks_[alt]->track_plots_[maxID].id_;

					// update track plot in the original track ID
					combine_track_plots_(track_plot.id_, cumulative_tracks_[index], track_plot, frame_count_);

					// update dictionary matching list
					for (auto &it : old_id : matching_dict[index])
					{
						if (matching_dict[index][old_id] == track_plot.id_)
						{
							matching_dict[index][old_id] = old_id;
							cumulative_tracks_[index]->track_new_plots_.insert({old_id, cumulative_tracks_[index].track_plots_[matching_dict[index][old_id]]});
							cumulative_tracks_[index]->track_new_plots_[old_id].id_ = old_id;
							break;
						}
					}

					// remove track plot in new tracks' list
					cumulative_tracks_[index]->track_plots_[track_plot.id_] = track_plot;
					cumulative_tracks_[index]->track_new_plots__.erase(track_id);
					matching_dict[index].insert({track_id, track_plot.id});
				}
			}
		}

		for (auto &it : remove_id : removeSet)
		{
			cumulative_tracks[alt]->track_new_plots_.erase(remove_id);
		}
	}
}

void McmtMultiTrackerNode::get_total_number_of_tracks()
{
	total_tracks_[0] = cumulative_tracks_[0]->track_new_plots_.size() + 
		cumulative_tracks_[0]->track_plots_.size();
	total_tracks_[1] = cumulative_tracks_[1]->track_new_plots_.size() + 
		cumulative_tracks_[1]->track_plots_.size();
}

/**
 * Normalises the existing track plot based on mean and sd
 */
std::vector<float> McmtMultiTrackerNode::normalise_track_plot(std::shared_ptr<mcmt::TrackPlot> track_plot)
{
	int total_track_feature = track_plot->track_feature_variable_.size();
	float mean = 0, variance = 0, std;
	std::vector<float> result;

	// Mean
	for (int i = 0; i < total_track_feature; i++){
		mean += track_plot->track_feature_variable_[i];
	}
	mean = mean / total_track_feature;

	// Variance and stdev
	for (int i = 0; i < total_track_feature; i++){
		variance += pow(track_plot->track_feature_variable_[i] - mean, 2);
	}
	variance = variance / total_track_feature;
	std = sqrt(variance);

	// Normalise
	for (int i = 0; i < total_track_feature; i++){
		float res = (track_plot->track_feature_variable_[i] - mean) / (std * sqrt(total_track_feature));
		result.push_back(res);
	}

	return result;
}

float McmtMultiTrackerNode::compute_matching_score(std::shared_ptr<mcmt::TrackPlot> track_plot,
		std::shared_ptr<mcmt::TrackPlot> alt_track_plot, int index, int alt)
{
	// Normalization of cross correlation values
	auto track_plot_normalize_xj = normalise_track_plot(track_plot);
	auto alt_track_plot_normalize_xj = normalise_track_plot(alt_track_plot);

	// Updating of tracks in the local neighbourhood
	mcmt::update_other_tracks(track_plot, cumulative_tracks_[index]);
	mcmt::update_other_tracks(alt_track_plot, cumulative_tracks_[alt]);

	// Track feature variable correlation strength
	// How to set mode = "full" like in numpy correlate?
	auto r_value = correlationCoefficient(track_plot_normalize_xj,
					alt_track_plot_normalize_xj, track_plot_normalize_xj.size());

	// Geometric track matching strength value
	float geometric_strength = geometric_similarity(track_plot->other_tracks_, alt_track_plot->other_tracks_);

	// Heading deviation error value
	int track_history = 30;
	float heading_err = heading_error(track_plot, alt_track_plot, track_history);

	float w1 = 0.3;
	float w2 = 0.4;
	float w3 = 0.3;
	float score = (w1 * r_value) + (w2 * geometric_strength) + (w3 * (1 - heading_err));

	if (r_value > 0.4 && (geometric_strength == 0 || geometric_strength >= 0.5) && heading_err < 0.1){
		return score;
	}
	else{
		return 0;
	}
	
	return 0;
}

/**
 * Find cross correlation of two 1D arrays with size n
 * Source: https://www.geeksforgeeks.org/program-find-correlation-coefficient/
*/
float McmtMultiTrackerNode::correlationCoefficient(std::vector<float> X, std::vector<float> Y, int n)
{
 
		float sum_X = 0, sum_Y = 0, sum_XY = 0;
		float squareSum_X = 0, squareSum_Y = 0;
 
		for (int i = 0; i < n; i++)
		{
				// sum of elements of array X.
				sum_X = sum_X + X[i];
 
				// sum of elements of array Y.
				sum_Y = sum_Y + Y[i];
 
				// sum of X[i] * Y[i].
				sum_XY = sum_XY + X[i] * Y[i];
 
				// sum of square of array elements.
				squareSum_X = squareSum_X + X[i] * X[i];
				squareSum_Y = squareSum_Y + Y[i] * Y[i];
		}
 
		// use formula for calculating correlation coefficient.
		float corr = (float)(n * sum_XY - sum_X * sum_Y)
									/ sqrt((n * squareSum_X - sum_X * sum_X)
											* (n * squareSum_Y - sum_Y * sum_Y));
 
		return corr;
}

float McmtMultiTrackerNode::geometric_similarity(
	std::vector<mcmt::TrackPlot::OtherTrack> & other_tracks_0, 
	std::vector<mcmt::TrackPlot::OtherTrack> & other_tracks_1)
{
	std::vector<float> relative_distances, shortest_distances;

	int total_num_other_tracks_0 = other_tracks_0.size();
	int total_num_other_tracks_1 = other_tracks_1.size();
	for (int i = 0; i < total_num_other_tracks_0; i++){
		float a_angle = other_tracks_0[i].angle;
		float a_dist =  other_tracks_0[i].dist;
		relative_distances.clear();
		for (int j = 0; j < total_num_other_tracks_1; j++){
			float b_angle = other_tracks_1[i].angle;
			float b_dist=  other_tracks_1[i].dist;

			relative_distances.push_back((std::min<float>(std::abs(a_angle - b_angle),
				(2 * M_PI) - std::abs(a_angle - b_angle))) / M_PI * 
				std::min<float>(a_dist / b_dist, b_dist / a_dist));
		}

		int total_num_relative_distances = relative_distances.size();
		if (total_num_relative_distances > 0){
			float minimum = relative_distances.front();
			for (int i = 0; i < total_num_relative_distances; i++){
				if (relative_distances[i] < minimum){
					minimum = relative_distances[i];
				}
			}
			shortest_distances.push_back(minimum);
		}
	}

	int total_num_shortest_distances = shortest_distances.size();
	if (total_num_shortest_distances > 0){
		// Get average value of shortest_distances
		int avg_shortest_distances = 0;
		for (int i = 0; i < total_num_shortest_distances; i++){
			avg_shortest_distances += shortest_distances[i];
		}
		avg_shortest_distances = avg_shortest_distances / total_num_shortest_distances;
		return std::max(0, avg_shortest_distances)*10;
	}
	else {
		return 0;
	}
}

float McmtMultiTrackerNode::heading_error(std::shared_ptr<mcmt::TrackPlot> track_plot, 
	std::shared_ptr<mcmt::TrackPlot> alt_track_plot, int & history)
{
	int deviation = 0;
	auto dx_0 = track_plot->xs_.back() - track_plot->xs_[track_plot->xs_.size() - 2];
	auto dy_0 = track_plot->ys_.back() - track_plot->xs_[track_plot->xs_.size() - 2];
	auto rotation_0 = (atan2(dy_0, dx_0) + M_PI) / (2 * M_PI);

	auto dx_1 = alt_track_plot->xs_.back() - alt_track_plot->xs_[alt_track_plot->xs_.size() - 2];
	auto dy_1 = alt_track_plot->ys_.back() - alt_track_plot->xs_[alt_track_plot->ys_.size() - 2];
	auto rotation_1 = (atan2(dy_1, dx_1) + M_PI) / (2 * M_PI);

	for (int i = -2; i > 1-history; i--) {
		dx_0 = track_plot->xs_[i] - track_plot->xs_[i-1];
		dy_0 = track_plot->ys_[i] - track_plot->ys_[i-1];
		auto angle_0 = (atan2(dy_0, dx_0) + M_PI) / (2 * M_PI);

		dx_1 = track_plot->xs_[i] - track_plot->xs_[i-1];
		dy_1 = track_plot->ys_[i] - track_plot->ys_[i-1];
		auto angle_1 = (atan2(dy_0, dx_0) + M_PI) / (2 * M_PI);

		auto relative_0 = std::fmod(angle_0 - rotation_0, 1.0);
		auto relative_1 = std::fmod(angle_1 - rotation_1, 1.0);

		deviation += std::min(std::abs(std::fmod(relative_0 - relative_1,1)),
					std::abs(std::fmod(relative_1 - relative_0, 1)));
	}

	return (deviation / 19);
}

/**
 * Computes the 3D position of a matched drone through triangulation methods.
 */
void McmtMultiTrackerNode::calculate_3D()
{
	float fx = 1454.6;
	float cx = 960.9;
	float fy = 1450.3;
	float cy = 543.7;
	float B = 1.5;
	int epsilon = 7;

	// Check for IDs that belong to both cumulative tracks 0 and 1
	std::set<int> matched_ids;
	int total_num_cumulative_tracks_0 = cumulative_tracks_[0]->track_plots_.size();
	int total_num_cumulative_tracks_1 = cumulative_tracks_[1]->track_plots_.size();
	for (int i = 0; i < total_num_cumulative_tracks_0; i++){
		for (int j = 0; j < total_num_cumulative_tracks_1; j++){
			if (cumulative_tracks_[0]->track_plots_[i]->id_ == cumulative_tracks_[1]->track_plots_[j]->id_){
				matched_ids.insert(cumulative_tracks_[0]->track_plots_[i]->id_);
			}
		}
	}

	int total_num_matched_ids = matched_ids.size();
	for (std::set<int>::iterator it = matched_ids.begin(); it != matched_ids.end(); it++){
		// Help me check if I am accessing the track_plots indexes correctly!
		auto track_plot_0 = cumulative_tracks_[0]->track_plots_[*it];
		auto track_plot_1 = cumulative_tracks_[1]->track_plots_[*it];

		if ((track_plot_0->lastSeen_ == frame_count_) && (track_plot_1->lastSeen_ == frame_count_)){
			int x_L = track_plot_0->xs_.back();
			int y_L = track_plot_0->ys_.back();
			int x_R = track_plot_1->xs_.back();
			int y_R = track_plot_1->ys_.back();

			auto alpha_L = atan2(x_L - cx, fx) / M_PI * 180;
			auto alpha_R = atan2(x_R - cx, fx) / M_PI * 180;

			auto gamma = epsilon + alpha_L - alpha_R; // unused - shld we remove?

			auto Z = B / (tan((alpha_L + epsilon / 2) * (M_PI / 180)) - tan((alpha_L - epsilon / 2) * (M_PI / 180)));
			auto X = (Z * tan((alpha_L + epsilon / 2) * (M_PI / 180)) - B / 2
						+ Z * tan((alpha_L - epsilon / 2) * (M_PI / 180)) + B / 2) / 2;
			auto Y = (Z * -(y_L - cy) / fy + Z * -(y_R - cy) / fy) / 2;

			auto tilt = 10 * M_PI / 180;
			Eigen::Matrix3d R;
			R << 1, 0, 0,
				0, cos(tilt), sin(tilt),
				0, -sin(tilt), cos(tilt);
			Eigen::Vector3d XYZ_original;
			XYZ_original << X, Y, Z;
			auto XYZ = R * XYZ_original;
			X = XYZ(0);
			Y = XYZ(1);
			Z = XYZ(2);

			Y += 1;

			X = (std::round(X*100))/100;
			Y = (std::round(Y*100))/100;
			Z = (std::round(Z*100))/100;

			track_plot_0->xyz_.clear();
			track_plot_0->xyz_.push_back(X);
			track_plot_0->xyz_.push_back(Y);
			track_plot_0->xyz_.push_back(Z);
			track_plot_1->xyz_.clear();
			track_plot_1->xyz_.push_back(X);
			track_plot_1->xyz_.push_back(Y);
			track_plot_1->xyz_.push_back(Z);

		}
		else {
			track_plot_0->xyz_.clear();
			track_plot_1->xyz_.clear();
		}
	}
}

void McmtMultiTrackerNode::imshow_resized_dual(std::string & window_name, cv::Mat & img)
{
	cv::Size img_size = img.size();

	float aspect_ratio = img_size.width / img_size.height;

	cv::Size window_size;
	window_size.width = 1920;
	window_size.height = 1920 / aspect_ratio;
	
	cv::resize(img, img, window_size, 0, 0, cv::INTER_CUBIC);
	cv::imshow(window_name, img);
}

/**
 * This function declares our mcmt software parameters as ROS2 parameters.
 */
void McmtMultiTrackerNode::declare_parameters()
{
	// declare ROS2 video parameters
	this->declare_parameter("IS_REALTIME");
	this->declare_parameter("VIDEO_INPUT_1");
	this->declare_parameter("VIDEO_INPUT_2");
	this->declare_parameter("FRAME_WIDTH");
	this->declare_parameter("FRAME_HEIGHT");
	this->declare_parameter("OUTPUT_VIDEO_PATH");
	this->declare_parameter("OUTPUT_CSV_PATH_1");
	this->declare_parameter("OUTPUT_CSV_PATH_2");
}

/**
 * This function gets the mcmt parameters from the ROS2 parameters
 */
void McmtMultiTrackerNode::get_parameters()
{
	// get video parameters
	IS_REALTIME_param = this->get_parameter("IS_REALTIME");
	VIDEO_INPUT_1_param = this->get_parameter("VIDEO_INPUT_1");
	VIDEO_INPUT_2_param = this->get_parameter("VIDEO_INPUT_2");
	FRAME_WIDTH_param = this->get_parameter("FRAME_WIDTH");
	FRAME_HEIGHT_param = this->get_parameter("FRAME_HEIGHT");
	OUTPUT_VIDEO_PATH_param = this->get_parameter("OUTPUT_VIDEO_PATH");
	OUTPUT_CSV_PATH_1_param = this->get_parameter("OUTPUT_CSV_PATH_1");
	OUTPUT_CSV_PATH_2_param = this->get_parameter("OUTPUT_CSV_PATH_2");

	// initialize video parameters
	is_realtime_ = IS_REALTIME_param.as_bool();
	video_input_1_ = VIDEO_INPUT_1_param.as_string();
	video_input_2_ = VIDEO_INPUT_2_param.as_string();
	FRAME_WIDTH_ = FRAME_WIDTH_param.as_int();
	FRAME_HEIGHT_ = FRAME_HEIGHT_param.as_int();
	output_vid_path_ = OUTPUT_VIDEO_PATH_param.as_string();
	output_csv_path_1_ = OUTPUT_CSV_PATH_1_param.as_string();
	output_csv_path_2_ = OUTPUT_CSV_PATH_2_param.as_string();
}

int McmtMultiTrackerNode::encoding2mat_type(const std::string & encoding)
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