/**
 * @file mcmt_multi_tracker_node.cpp
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
 * This file contains the declarations of the functions primarily used in the
 * tracking and re-identification pipeline. These functions interact with the 
 * key classes CameraTracks and TrackPlots which are essential to the tracking 
 * and re-identification process. These classes may be found at 
 * mcmt_track_utils.cpp. This code also contains the McmtMultiTrackerNode class
 * which handles the subscription from our ROS2 DDS-RTPS ecosystem.
 */

// local header files
#include <mcmt_track/mcmt_multi_tracker_node.hpp>

// standard package imports
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <csignal>
#include <chrono>
#include <math.h>
#include <memory>
#include <map>
#include <set>
#include <algorithm>
#include <functional>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace cv;
using namespace mcmt;

McmtMultiTrackerNode::McmtMultiTrackerNode() : Node("MultiTrackerNode") {

	node_handle_ = shared_ptr<rclcpp::Node>(this, [](::rclcpp::Node *) {});
	declare_parameters();
	get_parameters();
	RCLCPP_INFO(this->get_logger(), "Initializing Mcmt Multi Tracker Node");

	// get camera parameters
	if (is_realtime_ == true) {
		cap_ = VideoCapture(stoi(video_input_1_));
	} else {
		cap_ = VideoCapture(video_input_1_);
	}

	frame_w_ = int(cap_.get(CAP_PROP_FRAME_WIDTH));
	frame_h_ = int(cap_.get(CAP_PROP_FRAME_HEIGHT));
	scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	aspect_ratio_ = frame_w_ / frame_h_;
	fps_ = int(cap_.get(CAP_PROP_FPS));

	// if video frame size is too big, downsize
	downsample_ = false;
	if ((frame_w_ * frame_h_) > (FRAME_WIDTH_ * FRAME_HEIGHT_)) {
		downsample_ = true;
		frame_w_ = FRAME_WIDTH_;
		frame_h_ = int(FRAME_WIDTH_ / aspect_ratio_);
		scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	}

	// intialize video writer;
	recording_ = VideoWriter(output_vid_path_, VideoWriter::fourcc('M','P','4','V'), fps_, 
		Size(1920, 640));
	cap_.release();

	// initialize frame count and track id
	next_id_ = 0;
	frame_count_ = 0;

	// initialze plotting parameters
	plot_history_ = 200;
	font_scale_ = 0.5;

	// initialize cumulative camera tracks
	cumulative_tracks_[0] = shared_ptr<CameraTracks>(new CameraTracks(0));
	cumulative_tracks_[1] = shared_ptr<CameraTracks>(new CameraTracks(1));

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
			auto start = chrono::system_clock::now();	

			// declare tracking arrays
			array<shared_ptr<Mat>, 2> frames_;
			array<vector<shared_ptr<GoodTrack>>, 2> good_tracks_, filter_good_tracks_;
			array<vector<int>, 2> dead_tracks_;

			// process detection info
			process_msg_info(msg, frames_, good_tracks_, dead_tracks_);

			// create filter copy of good_tracks list
			filter_good_tracks_[0] = good_tracks_[0];
			filter_good_tracks_[1] = good_tracks_[1];

			update_cumulative_tracks(0, good_tracks_);
			update_cumulative_tracks(1, good_tracks_);

			// lines.clear();

			process_new_tracks(0, 1, good_tracks_, filter_good_tracks_, dead_tracks_);
			process_new_tracks(1, 0, good_tracks_, filter_good_tracks_, dead_tracks_);

			verify_existing_tracks();

			calculate_3D();

			prune_tracks(0);
			prune_tracks(1);

			print_frame_summary();

			annotate_frames(frames_, cumulative_tracks_);

			// show and save video combined tracking frame
			Mat combined_frame;
			hconcat(*frames_[0].get(), *frames_[1].get(), combined_frame);

			// for (auto line : lines) {
			// 	line(combined_frame, Point((int) line[0], (int)line[1]), Point((int) line[2], (int) line[3]), Scalar(0, (int) (line[4] * 255), (int) ((1 - line[4]) * 255)), 1);
			// 	string scores;
				
			// 	if (line[6] != 0) {
			// 		scores = to_string(line[5]).substr(0,4) + " x " + to_string(line[6]).substr(0,4) + " x " + to_string(line[7]).substr(0,4);
			// 	} else {
			// 		scores = to_string(line[5]).substr(0,4) + " x " + "0" + " x " + to_string(line[7]).substr(0,4);
			// 	}

			// 	putText(combined_frame, scores, Point((int) ((line[0] + line[2]) / 2), (int) ((line[1] + line[3]) / 2)),  
			// 					FONT_HERSHEY_SIMPLEX, font_scale_ * 1.5, Scalar(0, (int) (line[4] * 255), (int) ((1 - line[4]) * 255)), 3, LINE_AA);
			// }

			graphical_UI(combined_frame, cumulative_tracks_);

			// get trackplot process time
			auto end = chrono::system_clock::now();
			chrono::duration<double> elapsed_seconds = end - start;
			cout << "Trackplot process took: " << elapsed_seconds.count() << "s\n";		

			// show cv window
			string window_name = "Annotated";
			imshow_resized_dual(window_name, combined_frame);
			recording_.write(combined_frame);

			frame_count_ += 1;

			waitKey(1);
		}
	);
}

/**
 * This function processes the detection messsage information
 */
void McmtMultiTrackerNode::process_msg_info(mcmt_msg::msg::MultiDetectionInfo::SharedPtr msg,
	array<shared_ptr<Mat>, 2> & frames,
	array<vector<shared_ptr<GoodTrack>>, 2> & good_tracks,
	array<vector<int>, 2> & dead_tracks)
{
	// get both camera frames
	auto frame_1 = shared_ptr<Mat>(
		new Mat(
			msg->image_one.height, msg->image_one.width, encoding2mat_type(msg->image_one.encoding),
			const_cast<unsigned char *>(msg->image_one.data.data()), msg->image_one.step));
	
	auto frame_2 = shared_ptr<Mat>(
		new Mat(
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
		auto good_track = shared_ptr<GoodTrack>(new GoodTrack());
		good_track->id = msg->goodtracks_id_one[i];
		good_track->x = msg->goodtracks_x_one[i];
		good_track->y = msg->goodtracks_y_one[i];
		good_track->size = msg->goodtracks_size_one[i];
		good_tracks[0].push_back(good_track);
	}

	total_num_tracks = msg->goodtracks_id_two.size();

	for (int i = 0; i < total_num_tracks; i++) {
		auto good_track = shared_ptr<GoodTrack>(new GoodTrack());
		good_track->id = msg->goodtracks_id_two[i];
		good_track->x = msg->goodtracks_x_two[i];
		good_track->y = msg->goodtracks_y_two[i];
		good_track->size = msg->goodtracks_size_two[i];
		good_tracks[1].push_back(good_track);
	}
}

/**
 * this function creates new tracks and the addition to the cumulative tracks log for each frame
 */
void McmtMultiTrackerNode::update_cumulative_tracks( int index,
	array<vector<shared_ptr<GoodTrack>>, 2> & good_tracks) {

	int track_id;
	for (auto & track : good_tracks[index]) {
		track_id = track->id;

		// occurance of a new track
		if (matching_dict_[index].find(track_id) == matching_dict_[index].end()) {
			cumulative_tracks_[index]->track_new_plots_[track_id] = shared_ptr<TrackPlot>(
				new TrackPlot(track_id));
			matching_dict_[index][track_id] = track_id;
		}	
	}
}

/**
 * this function removes dead tracks if they have not appeared for more than 300 frames
 */
void McmtMultiTrackerNode::prune_tracks(int index) {

	vector<int> prune;
	map<int, shared_ptr<TrackPlot>>::iterator track;

	// prune dead tracks if they have not appeared for more than 300 frames
	for (track = cumulative_tracks_[index]->track_new_plots_.begin();
		track != cumulative_tracks_[index]->track_new_plots_.end(); track++) {
		if ((frame_count_ - track->second->lastSeen_) > 60) {
			prune.push_back(track->second->id_);
		}
	}

	for (auto & track_id : prune) {
		cumulative_tracks_[index]->track_new_plots_.erase(track_id);
		matching_dict_[index].erase(track_id);
	}
}

/**
 * checks matched tracks to see if they are still valid. also checks if multiple tracks
 * within each camera are tracking the same target
 */
void McmtMultiTrackerNode::verify_existing_tracks() {
	map<int, shared_ptr<TrackPlot>>::iterator track, other_track;
	vector<int> matched_ids;
	int original_track_id_0, original_track_id_1;

	for (track = cumulative_tracks_[0]->track_plots_.begin(); 
		track != cumulative_tracks_[0]->track_plots_.end(); track++) {
		// search if alternate trackplot contains the same key
		other_track = cumulative_tracks_[1]->track_plots_.find(track->first);
		
		if (other_track != cumulative_tracks_[1]->track_plots_.end()) {
			// alternate trackplot contains the same key
			matched_ids.push_back(track->first);
		}
	}

	for (auto & matched_id : matched_ids) {

		auto verify_start = chrono::system_clock::now();
		shared_ptr<TrackPlot> track_plot_0 = cumulative_tracks_[0]->track_plots_[matched_id];
		shared_ptr<TrackPlot> track_plot_1 = cumulative_tracks_[1]->track_plots_[matched_id];

		// normalization of cross correlation values
		vector<double> track_plot_normalize_xj = normalise_track_plot(track_plot_0);
		vector<double> alt_track_plot_normalize_xj = normalise_track_plot(track_plot_1);

		if (track_plot_normalize_xj.size() > 120) {
			vector<double> track_plot_normalize_xj_trunc(track_plot_normalize_xj.end() - 120, track_plot_normalize_xj.end());
			track_plot_normalize_xj = track_plot_normalize_xj_trunc;
		}
		if (alt_track_plot_normalize_xj.size() > 120) {
			vector<double> alt_track_plot_normalize_xj_trunc(alt_track_plot_normalize_xj.end() - 120, alt_track_plot_normalize_xj.end());
			alt_track_plot_normalize_xj = alt_track_plot_normalize_xj_trunc;
		}

		// track feature variable correlation strength
		auto r_value = crossCorrelation(track_plot_normalize_xj, alt_track_plot_normalize_xj);

		// heading deviation error score
		double heading_err = heading_error(track_plot_0, track_plot_1, 30);

		// vector<double> line{track_plot_0->xs_.back(), track_plot_0->ys_.back(), track_plot_1->xs_.back() + 1920, track_plot_1->ys_.back(), 0, r_value, 0, 1 - heading_err};
		// lines.push_back(line);
		
		if (r_value < 0.4 && heading_err > 0.2 && track_plot_0->frameNos_.size() > 180 && track_plot_1->frameNos_.size() > 180) {
				if (track_plot_0->check_stationary() != track_plot_1->check_stationary()) {
					track_plot_0->mismatch_count_ += 4;
					track_plot_1->mismatch_count_ += 4;
				} else {
					track_plot_0->mismatch_count_ += 1;
					track_plot_1->mismatch_count_ += 1;
				}
			
		} else {
			track_plot_0->mismatch_count_ = 0;
			track_plot_1->mismatch_count_ = 0;
		}

		if (track_plot_0->mismatch_count_ >= 120 && track_plot_1->mismatch_count_ >= 120) {
			track_plot_0->mismatch_count_ = 0;
			track_plot_1->mismatch_count_ = 0;

			debug_messages.push_back("Target ID " +  to_string(track_plot_0->id_)  + " is dropped due to mismatch ");

			map<int, int>::iterator it;
			for (it = matching_dict_[0].begin(); it != matching_dict_[0].end(); it++) {
				if (it->second == track_plot_0->id_) {
					original_track_id_0 = it->first;
					break;
				}
			}

			for (it = matching_dict_[1].begin(); it != matching_dict_[1].end(); it++) {
				if (it->second == track_plot_1->id_) {
					original_track_id_1 = it->first;
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
}

void McmtMultiTrackerNode::process_new_tracks(
	int index, int alt,
	array<vector<shared_ptr<GoodTrack>>, 2> & good_tracks,
	array<vector<shared_ptr<GoodTrack>>, 2> & filter_good_tracks,
	array<vector<int>, 2> & dead_tracks) {

	get_total_number_of_tracks();
	map<int, map<int, double>> corrValues;
	set<int> removeSet;
	int track_id, centroid_x, centroid_y, size;
	int row = 0;

	for (auto & track : good_tracks[index]) {
		// Extract details from the track
		track_id = track->id;
		centroid_x = track->x;
		centroid_y = track->y;
		size = track->size;

		if (cumulative_tracks_[index]->track_plots_.find(matching_dict_[index][track_id]) 
			== cumulative_tracks_[index]->track_plots_.end()) {

			vector<int> location;
			location.push_back(centroid_x);
			location.push_back(centroid_y);

			// Update track_new_plots with centroid and feature variable of every new frame
			auto track_plot = cumulative_tracks_[index]->track_new_plots_[track_id];			
			track_plot->update(location, size, frame_count_);
			track_plot->calculate_track_feature_variable(frame_count_, fps_);

			// Check if track feature variable has non-zero elements
			double sum = 0;
			for (int i = 0; i < static_cast<int>(track_plot->track_feature_variable_.size()); i++) {
				sum += track_plot->track_feature_variable_[i];
			}

			// if track is not a new track, we use 90 frames as the minimum requirement before matching occurs
			if (track_plot->frameNos_.size() >= 30 && track_plot->track_feature_variable_.size() >= 30 && sum != 0)	{

				// look into 2nd camera's new tracks (new tracks first)
				map<int, shared_ptr<TrackPlot>>::iterator alt_track_plot;
				for (alt_track_plot = cumulative_tracks_[alt]->track_new_plots_.begin();
					alt_track_plot != cumulative_tracks_[alt]->track_new_plots_.end(); alt_track_plot++) {

					sum = 0;
					for (int i = 0; i < static_cast<int>(alt_track_plot->second->track_feature_variable_.size()); i++) {
						sum += alt_track_plot->second->track_feature_variable_[i];
					}
					// track in 2nd camera must fulfills requirements to have at least 90 frames to match
					if (alt_track_plot->second->frameNos_.size() >= 30 && alt_track_plot->second->track_feature_variable_.size() >= 30 
						&& sum != 0) {
						double score = compute_matching_score(track_plot, alt_track_plot->second, index, alt);
						if (score != 0)	{
							corrValues[track_id][alt_track_plot->second->id_] = score;
						}
					}
				}

				// look into other camera's matched tracks list (old tracks last)
				for (alt_track_plot = cumulative_tracks_[alt]->track_plots_.begin();
					alt_track_plot != cumulative_tracks_[alt]->track_plots_.end(); alt_track_plot++) {

					bool eligibility_flag = true;
					
					// do not consider dead tracks from the other camera
					for (auto & dead_track : dead_tracks[alt]) {
						if (matching_dict_[alt].find(dead_track) != matching_dict_[alt].end() && matching_dict_[alt][dead_track] == alt_track_plot->second->id_) {
							eligibility_flag = false; // 2nd camera's track has already been lost. skip the process of matching for this track
						}
					}
											
					// test to see if alternate camera's track is currently being matched with current camera                        
					for (auto & alt_track : good_tracks[index])	{
						if (alt_track_plot->second->id_ == matching_dict_[index][alt_track->id]) {
							eligibility_flag = false; // 2nd camera's track has already been matched. skip the process of matching for this track
						}
					}

					sum = 0;
					for (int i = 0; i < static_cast<int>(alt_track_plot->second->track_feature_variable_.size()); i++) {
						sum += alt_track_plot->second->track_feature_variable_[i];
					}

					if (eligibility_flag && sum != 0) {
						double score = compute_matching_score(track_plot, alt_track_plot->second, index, alt);
						if (score != 0)	{
							corrValues[track_id][alt_track_plot->second->id_] = score;
						}
					}
				}
			}

			row += 1;

		} else {
			vector<int> location;
			location.push_back(centroid_x);
			location.push_back(centroid_y);

			auto track_plot = cumulative_tracks_[index]->track_plots_[matching_dict_[index][track_id]];
			track_plot->update(location, size, frame_count_);
			track_plot->calculate_track_feature_variable(frame_count_, fps_);
			filter_good_tracks[index].erase(filter_good_tracks[index].begin() + row);
		}

	}

	for (auto & track : filter_good_tracks[index]) {
		map<int, double> maxValues = corrValues[track->id];
		int maxID = -1;
		double maxValue = -1;
		int global_max_flag = 0;

		// for the selected max track in the 2nd camera, we check to see if the track has a higher
		// cross correlation value with another track in current camera

		while (global_max_flag == 0 && maxValues.size() != 0) {
			for (auto it = maxValues.begin(); it != maxValues.end(); it++) {
				if (maxValue < it->second) {
					maxID = it->first;
				}
			}
			maxValue = maxValues[maxID];

			// search through current camera's tracks again, for the selected track that we wish to re-id with.
			// we can note that if there is a track in the current camera that has a higher cross correlation value
			// than the track we wish to match with, then the matching will not occur.
			for (auto & track_1 : filter_good_tracks[index]) {
				if (corrValues[track_1->id].find(maxID) != corrValues[track_1->id].end()) {
					if (corrValues[track_1->id][maxID] > maxValue) {
						maxValues.erase(maxID);
						global_max_flag = 1;
						break;
					}
				}
			}

			if (global_max_flag == 1) {
				// there existed a value larger than the current maxValue. thus, re-id cannot occur
				global_max_flag = 0;
				continue;
			} else {
				// went through the whole loop without breaking, thus it is the maximum value. re-id can occur
				global_max_flag = 2;
			}
		}

		// re-id process
		if (global_max_flag == 2) {

			// if track is in 2nd camera's new track list
			if (maxID != 1 && 
				(cumulative_tracks_[alt]->track_new_plots_.find(maxID) != cumulative_tracks_[alt]->track_new_plots_.end()))	{
				// add notification message
				debug_messages.push_back("New target ID " +  to_string(next_id_)  + " acquired with a score of " + to_string(maxValue));

				// remove track plot in new tracks' list and add into matched tracks' list for alternate camera
				cumulative_tracks_[alt]->track_new_plots_[maxID]->id_ = next_id_;
				cumulative_tracks_[alt]->track_plots_.insert(
						pair<int, shared_ptr<TrackPlot>>(next_id_, cumulative_tracks_[alt]->track_new_plots_[maxID]));
				// update dictionary matching
				matching_dict_[alt][maxID] = next_id_;
				removeSet.insert(maxID);

				// remove track plot in new tracks' list and add into matched tracks' list for current camera
				int track_id = track->id;
				auto track_plot = cumulative_tracks_[index]->track_new_plots_[track_id];
				track_plot->id_ = next_id_;

				cumulative_tracks_[index]->track_plots_.insert({next_id_, track_plot});
				cumulative_tracks_[index]->track_new_plots_.erase(track_id);

				// update dictionary matching list
				matching_dict_[index][track_id] = next_id_;
				next_id_ += 1;

			// if track is in 2nd camera's matched track list
			} else {
				int track_id = track->id;
				auto track_plot = cumulative_tracks_[index]->track_new_plots_[track_id];
				track_plot->id_ = cumulative_tracks_[alt]->track_plots_[maxID]->id_;

				// add notification message
				debug_messages.push_back("New target ID " +  to_string(track_plot->id_)  + " acquired with a score of " + to_string(maxValue));

				// update track plot in the original track ID
				combine_track_plots(track_plot->id_, cumulative_tracks_[index], track_plot, frame_count_);

				// update dictionary matching list
				for (map<int, int>::iterator old_id = matching_dict_[index].begin(); old_id != matching_dict_[index].end(); old_id++) {
					if (old_id->second == track_plot->id_) {
						old_id->second = old_id->first;
						cumulative_tracks_[index]->track_new_plots_[old_id->first] = cumulative_tracks_[index]->track_plots_[track_plot->id_];
						cumulative_tracks_[index]->track_new_plots_[old_id->first]->id_ = old_id->first;
						break;
					}
				}

				// remove track plot in new tracks' list
				cumulative_tracks_[index]->track_plots_[track_plot->id_] = track_plot;
				cumulative_tracks_[index]->track_new_plots_.erase(track_id);
				matching_dict_[index][track_id] = track_plot->id_;

			}
		}
	}

	for (auto & remove_id : removeSet) {
		cumulative_tracks_[alt]->track_new_plots_.erase(remove_id);
	}
}

void McmtMultiTrackerNode::get_total_number_of_tracks() {
	total_tracks_[0] = cumulative_tracks_[0]->track_new_plots_.size() + 
		cumulative_tracks_[0]->track_plots_.size();
	total_tracks_[1] = cumulative_tracks_[1]->track_new_plots_.size() + 
		cumulative_tracks_[1]->track_plots_.size();
}

/**
 * Normalises the existing track plot based on mean and sd
 */
vector<double> McmtMultiTrackerNode::normalise_track_plot(shared_ptr<TrackPlot> track_plot) {
	int total_track_feature = track_plot->track_feature_variable_.size();
	double mean = 0, variance = 0, std;
	vector<double> result;

	// Mean
	for (int i = 0; i < total_track_feature; i++) {
		mean += track_plot->track_feature_variable_[i];
	}
	mean = mean / total_track_feature;

	// Variance and stdev
	for (int i = 0; i < total_track_feature; i++) {
		variance += pow(track_plot->track_feature_variable_[i] - mean, 2);
	}
	variance = variance / total_track_feature;
	std = sqrt(variance);

	// Normalise
	for (int i = 0; i < total_track_feature; i++) {
		double res = (track_plot->track_feature_variable_[i] - mean) / (std * sqrt(total_track_feature));
		result.push_back(res);
	}

	return result;
}

double McmtMultiTrackerNode::compute_matching_score(shared_ptr<TrackPlot> track_plot,
		shared_ptr<TrackPlot> alt_track_plot, int index, int alt) {

	// Normalization of cross correlation values
	auto track_plot_normalize_xj = normalise_track_plot(track_plot);
	auto alt_track_plot_normalize_xj = normalise_track_plot(alt_track_plot);

	// Updating of tracks in the local neighbourhood
	update_other_tracks(track_plot, cumulative_tracks_[index]);
	update_other_tracks(alt_track_plot, cumulative_tracks_[alt]);

	// Track feature variable correlation strength
	auto r_value = crossCorrelation(track_plot_normalize_xj, alt_track_plot_normalize_xj);

	// Geometric track matching strength value
	double geometric_strength = geometric_similarity(track_plot->other_tracks_, alt_track_plot->other_tracks_);

	// Heading deviation error value
	int track_history = 30;
	double heading_err = heading_error(track_plot, alt_track_plot, track_history);

	double w1 = 0.3;
	double w2 = 0.4;
	double w3 = 0.3;
	double score = (w1 * r_value) + (w2 * geometric_strength) + (w3 * (1 - heading_err));

	// if (index == 0) {
	// 	vector<double> line{track_plot->xs_.back(), track_plot->ys_.back(), alt_track_plot->xs_.back() + 1920, alt_track_plot->ys_.back(), score, r_value, geometric_strength, 1 - heading_err};
	// 	lines.push_back(line);
	// }

	if (r_value > 0.4 && (geometric_strength == 0 || geometric_strength >= 0.5) && heading_err < 0.1 && score >= 0.75) {
		return score;
	} else {
		return 0;
	}
}

/**
 * Find cross correlation of two 1D arrays with size n
 * Involves the convolution of array X with array Y by sliding vector Y from left to right
 */
double McmtMultiTrackerNode::crossCorrelation(vector<double> X, vector<double> Y) {
	
	double max = 0;
	vector<double> A;
	vector<double> K;

	if (X.size() >= Y.size()) {
		A = X;
		K = Y;
	} else {
		A = Y;
		K = X;
	}
	
	for (int i = 1; i <= A.size() + K.size() - 1; i++) {
		double sum = 0;

		// Kernel is outside (to the left of) the array
		if (i <= K.size() - 1) {
			for (int k = K.size() - i; k < K.size(); k++) {
				sum += K[k] * A[i + k - K.size()];
			}
		// Kernel is outside (to the left of) the array
		} else if (i >= A.size() + 1) {
			for (int k = 0; k < A.size() + K.size() - i; k++) {
				sum += K[k] * A[i + k - K.size()];
			}
		// Kernel is fully within the array
		} else {
			for (int k = 0; k < K.size(); k++) {
				sum += K[k] * A[i + k - K.size()];
			}
		}
		// Only keep the peak cross-correlation
		if (sum > max) {
			max = sum;
		}			
	}

	return max;
}

double McmtMultiTrackerNode::geometric_similarity(
	vector<shared_ptr<TrackPlot::OtherTrack>> & other_tracks_0, 
	vector<shared_ptr<TrackPlot::OtherTrack>> & other_tracks_1) {
	vector<double> relative_distances, shortest_distances;

	int total_num_other_tracks_0 = other_tracks_0.size();
	int total_num_other_tracks_1 = other_tracks_1.size();

	for (int i = 0; i < total_num_other_tracks_0; i++) {
		double a_angle = other_tracks_0[i]->angle;
		double a_dist = other_tracks_0[i]->dist;

		relative_distances.clear();
		for (int j = 0; j < total_num_other_tracks_1; j++) {
			double b_angle = other_tracks_1[j]->angle;
			double b_dist = other_tracks_1[j]->dist;

			relative_distances.push_back((min<double>(abs(a_angle - b_angle),
				(2 * M_PI) - abs(a_angle - b_angle))) / M_PI * 
				min<double>(a_dist / b_dist, b_dist / a_dist));
		}

		int total_num_relative_distances = relative_distances.size();
		if (total_num_relative_distances > 0){
			double minimum = relative_distances.front();
			for (int i = 0; i < total_num_relative_distances; i++) {
				if (relative_distances[i] < minimum) {
					minimum = relative_distances[i];
				}
			}
			shortest_distances.push_back(minimum);
		}
	}

	int total_num_shortest_distances = shortest_distances.size();
	if (total_num_shortest_distances > 0) {
		// Get average value of shortest_distances
		double avg_shortest_distances = 0;
		for (int i = 0; i < total_num_shortest_distances; i++) {
			avg_shortest_distances += shortest_distances[i];
		}
		avg_shortest_distances = avg_shortest_distances / total_num_shortest_distances;
		return max<double>(0.001, 0.2 - avg_shortest_distances) * 5;
	} else {
		return 0;
	}

}

double McmtMultiTrackerNode::heading_error(shared_ptr<TrackPlot> track_plot, 
	shared_ptr<TrackPlot> alt_track_plot, int history) {

	double deviation = 0;
	int dx_0 = track_plot->xs_.back() - track_plot->xs_[track_plot->xs_.size() - 2];
	int dy_0 = track_plot->ys_.back() - track_plot->ys_[track_plot->ys_.size() - 2];
	double rotation_0 = (atan2((double) dy_0, (double) dx_0) + M_PI) / (2 * M_PI);

	int dx_1 = alt_track_plot->xs_.back() - alt_track_plot->xs_[alt_track_plot->xs_.size() - 2];
	int dy_1 = alt_track_plot->ys_.back() - alt_track_plot->ys_[alt_track_plot->ys_.size() - 2];
	double rotation_1 = (atan2((double) dy_1, (double) dx_1) + M_PI) / (2 * M_PI);

	for (int i = -2; i > 1-history; i--) {
		dx_0 = track_plot->xs_[track_plot->xs_.size() - 1 + i] - track_plot->xs_[track_plot->xs_.size() - 2 + i];
		dy_0 = track_plot->ys_[track_plot->ys_.size() - 1 + i] - track_plot->ys_[track_plot->ys_.size() - 2 + i];
		double angle_0 = (atan2((double) dy_0, (double) dx_0) + M_PI) / (2 * M_PI);

		dx_1 = alt_track_plot->xs_[alt_track_plot->xs_.size() - 1 + i] - alt_track_plot->xs_[alt_track_plot->xs_.size() - 2 + i];
		dy_1 = alt_track_plot->ys_[alt_track_plot->ys_.size() - 1 + i] - alt_track_plot->ys_[alt_track_plot->ys_.size() - 2 + i];
		double angle_1 = (atan2((double) dy_1, (double) dx_1) + M_PI) / (2 * M_PI);

		double relative_0 = angle_0 - rotation_0;
		double relative_1 = angle_1 - rotation_1;

		if (relative_0 < 0) {
			relative_0 += 1;
		}

		if (relative_1 < 0) {
			relative_1 += 1;
		}

		deviation += min(abs(relative_0 - relative_1), 1 - abs(relative_0 - relative_1));

	}
	
	return (deviation / (history - 1));
}

/**
 * Computes the 3D position of a matched drone through triangulation methods.
 */
void McmtMultiTrackerNode::calculate_3D() {
	double fx = 1454.6;
	double cx = 960.9;
	double fy = 1450.3;
	double cy = 543.7;
	double B = 1.5;
	int epsilon = 7;

	// Check for IDs that belong to both cumulative tracks 0 and 1
	set<int> matched_ids;
	for (auto i = cumulative_tracks_[0]->track_plots_.begin(); i != cumulative_tracks_[0]->track_plots_.end(); i++) {
		for (auto j = cumulative_tracks_[1]->track_plots_.begin(); j != cumulative_tracks_[1]->track_plots_.end(); j++) {
			if (i->first == j->first) {
				matched_ids.insert(i->first);
			}
		}
	}

	for (set<int>::iterator it = matched_ids.begin(); it != matched_ids.end(); it++) {
		auto track_plot_0 = cumulative_tracks_[0]->track_plots_[*it];
		auto track_plot_1 = cumulative_tracks_[1]->track_plots_[*it];

		if ((track_plot_0->lastSeen_ == frame_count_) && (track_plot_1->lastSeen_ == frame_count_)) {
			int x_L = track_plot_0->xs_.back();
			int y_L = track_plot_0->ys_.back();
			int x_R = track_plot_1->xs_.back();
			int y_R = track_plot_1->ys_.back();

			double alpha_L = atan2(x_L - cx, fx) / M_PI * 180;
			double alpha_R = atan2(x_R - cx, fx) / M_PI * 180;

			double Z = B / (tan((alpha_L + epsilon / 2) * (M_PI / 180)) - tan((alpha_L - epsilon / 2) * (M_PI / 180)));
			double X = (Z * tan((alpha_L + epsilon / 2) * (M_PI / 180)) - B / 2
						+ Z * tan((alpha_R - epsilon / 2) * (M_PI / 180)) + B / 2) / 2;
			double Y = (Z * - (y_L - cy) / fy + Z * - (y_R - cy) / fy) / 2;

			double tilt = 10 * M_PI / 180;
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

			X = (round(X*100))/100;
			Y = (round(Y*100))/100;
			Z = (round(Z*100))/100;

			track_plot_0->xyz_.clear();
			track_plot_0->xyz_.push_back(X);
			track_plot_0->xyz_.push_back(Y);
			track_plot_0->xyz_.push_back(Z);
			track_plot_1->xyz_.clear();
			track_plot_1->xyz_.push_back(X);
			track_plot_1->xyz_.push_back(Y);
			track_plot_1->xyz_.push_back(Z);

		} else {
			track_plot_0->xyz_.clear();
			track_plot_1->xyz_.clear();
		}
	}
}

void McmtMultiTrackerNode::print_frame_summary() {

	cout << "SUMMARY OF FRAME " << frame_count_ << endl;
	cout << "Camera 0 New Tracks: ";
	for (auto it = cumulative_tracks_[0]->track_new_plots_.begin(); it != cumulative_tracks_[0]->track_new_plots_.end(); it++) {
		cout << "(" << it->first << ": " << it->second->id_ << ") | ";
	}
	cout << endl;
	cout << "Camera 0 Tracks: ";
	for (auto it = cumulative_tracks_[0]->track_plots_.begin(); it != cumulative_tracks_[0]->track_plots_.end(); it++) {
		cout << "(" << it->first << ": " << it->second->id_ << ") | ";
	}
	cout << endl;
	cout << "Camera 0 Matching: ";
	for (auto it = matching_dict_[0].begin(); it != matching_dict_[0].end(); it++) {
		cout << "(" << it->first << ": " << it->second << ") | ";
	}
	cout << endl;
	cout << "Camera 1 New Tracks: ";
	for (auto it = cumulative_tracks_[1]->track_new_plots_.begin(); it != cumulative_tracks_[1]->track_new_plots_.end(); it++) {
		cout << "(" << it->first << ": " << it->second->id_ << ") | ";
	}
	cout << endl;
	cout << "Camera 1 Tracks: ";
	for (auto it = cumulative_tracks_[1]->track_plots_.begin(); it != cumulative_tracks_[1]->track_plots_.end(); it++) {
		cout << "(" << it->first << ": " << it->second->id_ << ") | ";
	}
	cout << endl;
	cout << "Camera 1 Matching: ";
	for (auto it = matching_dict_[1].begin(); it != matching_dict_[1].end(); it++) {
		cout << "(" << it->first << ": " << it->second << ") | ";
	}
	cout << endl;
}
void McmtMultiTrackerNode::annotate_frames(array<shared_ptr<Mat>, 2> frames_, array<shared_ptr<CameraTracks>, 2> cumulative_tracks_) {

	// draw tracks on opencv GUI to monitor the detected tracks
	// lopp through each camera frame
	for (int i = 0; i < 2; i++) {

		putText(*frames_[i].get(), "CAMERA " + to_string(i), Point(20, 30),
			FONT_HERSHEY_SIMPLEX, font_scale_ * 0.85, Scalar(255, 0, 0), 2, LINE_AA);
		
		putText(*frames_[i].get(), "Frame Count " + to_string(frame_count_), Point(20, 50),
			FONT_HERSHEY_SIMPLEX, font_scale_ * 0.85, Scalar(255, 0, 0), 2, LINE_AA);
		
		// loop through every track plot
		if (cumulative_tracks_[i]->track_plots_.empty() == false) {
			for (auto track = cumulative_tracks_[i]->track_plots_.begin(); 
				track != cumulative_tracks_[i]->track_plots_.end(); track++) {
				if ((frame_count_ - track->second->lastSeen_) <= fps_) {
					
					Point2i rect_top_left((track->second->xs_.back() - (track->second->size_.back())), 
															(track->second->ys_.back() - (track->second->size_.back())));
		
					Point2i rect_bottom_right((track->second->xs_.back() + (track->second->size_.back())), 
																	(track->second->ys_.back() + (track->second->size_.back())));
		
					
					rectangle(*frames_[i].get(), rect_top_left, rect_bottom_right, colors[track->second->id_ % 10], 2);

					Scalar status_color;
					if (track->second->lastSeen_ == frame_count_) {
						status_color = Scalar(0, 255, 0);
					} else {
						status_color = Scalar(0, 0, 255);
					}
					
					// get last frames up till plot history (200)
					shown_indexes_.clear();

					for(int j = track->second->frameNos_.size() - 1; j >= 0; j--) {
						if (track->second->frameNos_[j] > (frame_count_ - plot_history_)) {
							shown_indexes_.push_back(j);
						} else {
							break;
						}
					}

					for (auto & idx : shown_indexes_) {
						int color_idx = track->second->frameNos_[idx] - frame_count_ + plot_history_ - 1;
						double alpha = 0.5 + (double) color_idx / 400;
						double beta = 1 - alpha;
						Vec3b pixelColor = (*frames_[i].get()).at<Vec3b>(track->second->ys_[idx], track->second->xs_[idx]);

						circle(*frames_[i].get(), Point(track->second->xs_[idx], track->second->ys_[idx]), 3,
							Scalar((int) (pixelColor[0] * beta + (colors[track->second->id_ % 10][0] * alpha)),
										(int) (pixelColor[1] * beta + (colors[track->second->id_ % 10][1] * alpha)),
										(int) (pixelColor[2] * beta + (colors[track->second->id_ % 10][2] * alpha))), -1);
					}

					// put ID and XYZ coordinates on opencv GUI
					if (shown_indexes_.empty() == false) {
						putText(*frames_[i].get(), "ID: " + to_string(track->second->id_).substr(0,4), 
							Point(rect_top_left.x + 20, rect_top_left.y - 5), FONT_HERSHEY_SIMPLEX,
							font_scale_, colors[track->second->id_ % 10], 1, LINE_AA);
						
						if (track->second->xyz_.empty() == false) {
							putText(*frames_[i].get(), "X: " + to_string(track->second->xyz_[0]).substr(0,4),
								Point(rect_bottom_right.x + 10, rect_top_left.y + 10), FONT_HERSHEY_SIMPLEX,
								font_scale_, colors[track->second->id_ % 10], 1, LINE_AA);

							putText(*frames_[i].get(), "Y: " + to_string(track->second->xyz_[1]).substr(0,4),
								Point(rect_bottom_right.x + 10, rect_top_left.y + 25), FONT_HERSHEY_SIMPLEX,
								font_scale_, colors[track->second->id_ % 10], 1, LINE_AA);

							putText(*frames_[i].get(), "Z: " + to_string(track->second->xyz_[2]).substr(0,4),
								Point(rect_bottom_right.x + 10, rect_top_left.y + 40), FONT_HERSHEY_SIMPLEX,
								font_scale_, colors[track->second->id_ % 10], 1, LINE_AA);

						}
					}
					
					circle(*frames_[i].get(), Point(rect_top_left.x + 5, rect_top_left.y - 10), 5, status_color, -1);	

				}

				// if (track->second->check_stationary()) {
				// 	putText(*frames_[i].get(), "S", Point(track->second->xs_.back(), track->second->ys_.back() - 40),
				// 			FONT_HERSHEY_SIMPLEX, font_scale_ * 2, Scalar(255, 0, 0), 2, LINE_AA);
				// }

			}
		}

	}
}

void McmtMultiTrackerNode::graphical_UI(Mat combined_frame, array<shared_ptr<CameraTracks>, 2> cumulative_tracks_) {
	
	// Summary box
	rectangle(combined_frame, Point(190, 860), Point(800, 900), Scalar(220,220,220), -1);
	rectangle(combined_frame, Point(190, 860), Point(800, 900), Scalar(110,110,110), 4);
	int spacing = 0;
	int drones_on_screen = 0;
	if (cumulative_tracks_[0]->track_plots_.empty() == false) {
		for (auto track = cumulative_tracks_[0]->track_plots_.begin(); 
			track != cumulative_tracks_[0]->track_plots_.end(); track++) {
			if ((frame_count_ - track->second->lastSeen_) <= fps_) {
				drones_on_screen++;
				putText(combined_frame, "ID: " + to_string(track->second->id_).substr(0,4), Point(210 + spacing, 890), FONT_HERSHEY_SIMPLEX,
						font_scale_ * 1.5, colors[track->second->id_ % 10], 2, LINE_AA);
				spacing += 100;	
			}
		}
	}

	// Notification box
	rectangle(combined_frame, Point(20, 920), Point(800, 1060), Scalar(200,200,200), -1);
	int num_of_messages = 4;
	spacing = 0;
	for (int i = 0; i < num_of_messages && i < debug_messages.size(); i++, spacing -= 30) {
		putText(combined_frame, debug_messages[debug_messages.size() - 1 - i], Point(40, 1040 + spacing), 
						FONT_HERSHEY_SIMPLEX, font_scale_ * 1.5, Scalar(0,0,0), 2, LINE_AA);
	}

	// Targets box
	rectangle(combined_frame, Point(20, 780), Point(170, 900), Scalar(220,220,220), -1);
	rectangle(combined_frame, Point(20, 780), Point(170, 900), Scalar(110,110,110), 4);
	putText(combined_frame, "TARGETS", Point(45, 805), FONT_HERSHEY_SIMPLEX,
				font_scale_ * 1.5, Scalar(0,0,0), 2, LINE_AA);
	putText(combined_frame, to_string(drones_on_screen), Point(60, 885), FONT_HERSHEY_SIMPLEX,
				font_scale_ * 6, Scalar(0,0,0), 6, LINE_AA);
}

void McmtMultiTrackerNode::imshow_resized_dual(string & window_name, Mat & img) {
	Size img_size = img.size();

	double aspect_ratio = img_size.width / img_size.height;

	Size window_size;
	window_size.width = 1920;
	window_size.height = 1920 / aspect_ratio;
	
	resize(img, img, window_size, 0, 0, INTER_CUBIC);
	imshow(window_name, img);
}

/**
 * This function declares our mcmt software parameters as ROS2 parameters.
 */
void McmtMultiTrackerNode::declare_parameters() {
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
void McmtMultiTrackerNode::get_parameters() {
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

int McmtMultiTrackerNode::encoding2mat_type(const string & encoding) {
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
			throw runtime_error("Unsupported encoding type");
	}
}