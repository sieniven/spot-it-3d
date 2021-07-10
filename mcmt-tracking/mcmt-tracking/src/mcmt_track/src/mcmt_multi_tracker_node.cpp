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

	// if video frame size is too big, downsize
	downsample_ = false;
	if ((frame_w_ * frame_h_) > (FRAME_WIDTH_ * FRAME_HEIGHT_)) {
		downsample_ = true;
		frame_w_ = FRAME_WIDTH_;
		frame_h_ = int(FRAME_WIDTH_ / aspect_ratio_);
		scale_factor_ = (sqrt(pow(frame_w_, 2) + pow(frame_h_, 2))) / (sqrt(pow(848, 2) + pow(480, 2)));
	}

	// intialize video writer;
	recording_ = cv::VideoWriter(output_vid_path_, cv::VideoWriter::fourcc('M','P','4','V'), fps_, 
		cv::Size(1920, 640));
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

			// lines.clear();

			process_new_tracks(0, 1, good_tracks_, filter_good_tracks_, dead_tracks_);
			process_new_tracks(1, 0, good_tracks_, filter_good_tracks_, dead_tracks_);

			verify_existing_tracks();

			calculate_3D();

			prune_tracks(0);
			prune_tracks(1);

			std::cout << "SUMMARY OF FRAME " << frame_count_ << std::endl;
			std::cout << "Camera 0 New Tracks: ";
			for (auto it = cumulative_tracks_[0]->track_new_plots_.begin(); it != cumulative_tracks_[0]->track_new_plots_.end(); it++) {
				std::cout << "(" << it->first << ": " << it->second->id_ << ") | ";
			}
			std::cout << std::endl;
			std::cout << "Camera 0 Tracks: ";
			for (auto it = cumulative_tracks_[0]->track_plots_.begin(); it != cumulative_tracks_[0]->track_plots_.end(); it++) {
				std::cout << "(" << it->first << ": " << it->second->id_ << ") | ";
			}
			std::cout << std::endl;
			std::cout << "Camera 0 Matching: ";
			for (auto it = matching_dict_[0].begin(); it != matching_dict_[0].end(); it++) {
				std::cout << "(" << it->first << ": " << it->second << ") | ";
			}
			std::cout << std::endl;
			std::cout << "Camera 1 New Tracks: ";
			for (auto it = cumulative_tracks_[1]->track_new_plots_.begin(); it != cumulative_tracks_[1]->track_new_plots_.end(); it++) {
				std::cout << "(" << it->first << ": " << it->second->id_ << ") | ";
			}
			std::cout << std::endl;
			std::cout << "Camera 1 Tracks: ";
			for (auto it = cumulative_tracks_[1]->track_plots_.begin(); it != cumulative_tracks_[1]->track_plots_.end(); it++) {
				std::cout << "(" << it->first << ": " << it->second->id_ << ") | ";
			}
			std::cout << std::endl;
			std::cout << "Camera 1 Matching: ";
			for (auto it = matching_dict_[1].begin(); it != matching_dict_[1].end(); it++) {
				std::cout << "(" << it->first << ": " << it->second << ") | ";
			}
			std::cout << std::endl;

			std::vector<cv::Scalar> colors = {
				cv::Scalar(124, 104, 66), // 1
				cv::Scalar(20, 60, 96), // 2
				cv::Scalar(46, 188, 243), // 3
				cv::Scalar(143, 89, 255), // 4
				cv::Scalar(6, 39, 156), // 5
				cv::Scalar(92, 215, 206), // 6
				cv::Scalar(105, 139, 246), // 7
				cv::Scalar(84, 43, 0), // 8
				cv::Scalar(137, 171, 197), // 9
				cv::Scalar(147, 226, 255) // 10
			};
			

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
							
							cv::Point2i rect_top_left((track->second->xs_.back() - (track->second->size_.back())), 
																	(track->second->ys_.back() - (track->second->size_.back())));
				
							cv::Point2i rect_bottom_right((track->second->xs_.back() + (track->second->size_.back())), 
																			(track->second->ys_.back() + (track->second->size_.back())));
				
							
							cv::rectangle(*frames_[i].get(), rect_top_left, rect_bottom_right, colors[track->second->id_ % 10], 2);

							cv::Scalar status_color;
							if (track->second->lastSeen_ == frame_count_) {
								status_color = cv::Scalar(0, 255, 0);
							} else {
								status_color = cv::Scalar(0, 0, 255);
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
								cv::Vec3b pixelColor = (*frames_[i].get()).at<cv::Vec3b>(track->second->ys_[idx], track->second->xs_[idx]);

								cv::circle(*frames_[i].get(), cv::Point(track->second->xs_[idx], track->second->ys_[idx]), 3,
									cv::Scalar((int) (pixelColor[0] * beta + (colors[track->second->id_ % 10][0] * alpha)),
												(int) (pixelColor[1] * beta + (colors[track->second->id_ % 10][1] * alpha)),
												(int) (pixelColor[2] * beta + (colors[track->second->id_ % 10][2] * alpha))), -1);
							}

							// put ID and XYZ coordinates on opencv GUI
							if (shown_indexes_.empty() == false) {
								cv::putText(*frames_[i].get(), "ID: " + std::to_string(track->second->id_).substr(0,4), 
									cv::Point(rect_top_left.x + 20, rect_top_left.y - 5), cv::FONT_HERSHEY_SIMPLEX,
									font_scale_, colors[track->second->id_ % 10], 1, cv::LINE_AA);
								
								if (track->second->xyz_.empty() == false) {
									cv::putText(*frames_[i].get(), "X: " + std::to_string(track->second->xyz_[0]).substr(0,4),
										cv::Point(rect_bottom_right.x + 10, rect_top_left.y + 10), cv::FONT_HERSHEY_SIMPLEX,
										font_scale_, colors[track->second->id_ % 10], 1, cv::LINE_AA);

									cv::putText(*frames_[i].get(), "Y: " + std::to_string(track->second->xyz_[1]).substr(0,4),
										cv::Point(rect_bottom_right.x + 10, rect_top_left.y + 25), cv::FONT_HERSHEY_SIMPLEX,
										font_scale_, colors[track->second->id_ % 10], 1, cv::LINE_AA);

									cv::putText(*frames_[i].get(), "Z: " + std::to_string(track->second->xyz_[2]).substr(0,4),
										cv::Point(rect_bottom_right.x + 10, rect_top_left.y + 40), cv::FONT_HERSHEY_SIMPLEX,
										font_scale_, colors[track->second->id_ % 10], 1, cv::LINE_AA);
								}
							}
							
							cv::circle(*frames_[i].get(), cv::Point(rect_top_left.x + 5, rect_top_left.y - 10), 5, status_color, -1);	

						}

						// if (track->second->check_stationary()) {
						// 	cv::putText(*frames_[i].get(), "S", cv::Point(track->second->xs_.back(), track->second->ys_.back() - 40),
						// 			cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 2, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
						// }
					}
				}

			}

			// get trackplot process time
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - start;
			std::cout << "Trackplot process took: " << elapsed_seconds.count() << "s\n";

			// trackplot_file.open("/home/mcmt/spot-it-3d/data/trackplot_time.csv",std::ios_base::app);
			// trackplot_file << elapsed_seconds.count() << "\n"; 
			// trackplot_file.close();
			
			// show and save video combined tracking frame
			frame_count_ += 1;
			cv::Mat combined_frame;
			cv::hconcat(*frames_[0].get(), *frames_[1].get(), combined_frame);

			// for (auto line : lines) {
			// 	cv::line(combined_frame, cv::Point((int) line[0], (int)line[1]), cv::Point((int) line[2], (int) line[3]), cv::Scalar(0, (int) (line[4] * 255), (int) ((1 - line[4]) * 255)), 1);
			// 	std::string scores;
				
			// 	if (line[6] != 0) {
			// 		scores = std::to_string(line[5]).substr(0,4) + " x " + std::to_string(line[6]).substr(0,4) + " x " + std::to_string(line[7]).substr(0,4);
			// 	} else {
			// 		scores = std::to_string(line[5]).substr(0,4) + " x " + "0" + " x " + std::to_string(line[7]).substr(0,4);
			// 	}

			// 	cv::putText(combined_frame, scores, cv::Point((int) ((line[0] + line[2]) / 2), (int) ((line[1] + line[3]) / 2)),  
			// 					cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 1.5, cv::Scalar(0, (int) (line[4] * 255), (int) ((1 - line[4]) * 255)), 3, cv::LINE_AA);
			// }

			// Summary box
			cv::rectangle(combined_frame, cv::Point(190, 860), cv::Point(800, 900), cv::Scalar(220,220,220), -1);
			cv::rectangle(combined_frame, cv::Point(190, 860), cv::Point(800, 900), cv::Scalar(110,110,110), 4);
			int spacing = 0;
			int drones_on_screen = 0;
			if (cumulative_tracks_[0]->track_plots_.empty() == false) {
				for (track = cumulative_tracks_[0]->track_plots_.begin(); 
					track != cumulative_tracks_[0]->track_plots_.end(); track++) {
					if ((frame_count_ - track->second->lastSeen_) <= fps_) {
						drones_on_screen++;
						cv::putText(combined_frame, "ID: " + std::to_string(track->second->id_).substr(0,4), cv::Point(210 + spacing, 890), cv::FONT_HERSHEY_SIMPLEX,
								font_scale_ * 1.5, colors[track->second->id_ % 10], 2, cv::LINE_AA);
						spacing += 100;	
					}
				}
			}

			// Notification box
			cv::rectangle(combined_frame, cv::Point(20, 920), cv::Point(800, 1060), cv::Scalar(200,200,200), -1);
			int num_of_messages = 4;
			spacing = 0;
			for (int i = 0; i < num_of_messages && i < debug_messages.size(); i++, spacing -= 30) {
				cv::putText(combined_frame, debug_messages[debug_messages.size() - 1 - i], cv::Point(40, 1040 + spacing), 
								cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 1.5, cv::Scalar(0,0,0), 2, cv::LINE_AA);
			}

			// Targets box
			cv::rectangle(combined_frame, cv::Point(20, 780), cv::Point(170, 900), cv::Scalar(220,220,220), -1);
			cv::rectangle(combined_frame, cv::Point(20, 780), cv::Point(170, 900), cv::Scalar(110,110,110), 4);
			cv::putText(combined_frame, "TARGETS", cv::Point(45, 805), cv::FONT_HERSHEY_SIMPLEX,
						font_scale_ * 1.5, cv::Scalar(0,0,0), 2, cv::LINE_AA);
			cv::putText(combined_frame, std::to_string(drones_on_screen), cv::Point(60, 885), cv::FONT_HERSHEY_SIMPLEX,
						font_scale_ * 6, cv::Scalar(0,0,0), 6, cv::LINE_AA);

						

			std::string window_name = "Detection";
			imshow_resized_dual(window_name, combined_frame);
			recording_.write(combined_frame);

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
		good_track->size = msg->goodtracks_size_one[i];
		good_tracks[0].push_back(good_track);
	}

	total_num_tracks = msg->goodtracks_id_two.size();

	for (int i = 0; i < total_num_tracks; i++) {
		auto good_track = std::shared_ptr<GoodTrack>(new GoodTrack());
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
void McmtMultiTrackerNode::update_cumulative_tracks(
	int index,
	std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & good_tracks)
{
	int track_id;
	for (auto & track : good_tracks[index]) {
		track_id = track->id;

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
void McmtMultiTrackerNode::prune_tracks(int index)
{
	std::vector<int> prune;
	std::map<int, std::shared_ptr<mcmt::TrackPlot>>::iterator track;

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
void McmtMultiTrackerNode::verify_existing_tracks()
{
	std::map<int, std::shared_ptr<mcmt::TrackPlot>>::iterator track, other_track;
	std::vector<int> matched_ids;
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

		auto verify_start = std::chrono::system_clock::now();
		std::shared_ptr<TrackPlot> track_plot_0 = cumulative_tracks_[0]->track_plots_[matched_id];
		std::shared_ptr<TrackPlot> track_plot_1 = cumulative_tracks_[1]->track_plots_[matched_id];

		// normalization of cross correlation values
		std::vector<double> track_plot_normalize_xj = normalise_track_plot(track_plot_0);
		std::vector<double> alt_track_plot_normalize_xj = normalise_track_plot(track_plot_1);

		if (track_plot_normalize_xj.size() > 120) {
			std::vector<double> track_plot_normalize_xj_trunc(track_plot_normalize_xj.end() - 120, track_plot_normalize_xj.end());
			track_plot_normalize_xj = track_plot_normalize_xj_trunc;
		}
		if (alt_track_plot_normalize_xj.size() > 120) {
			std::vector<double> alt_track_plot_normalize_xj_trunc(alt_track_plot_normalize_xj.end() - 120, alt_track_plot_normalize_xj.end());
			alt_track_plot_normalize_xj = alt_track_plot_normalize_xj_trunc;
		}

		// track feature variable correlation strength
		auto r_value = crossCorrelation(track_plot_normalize_xj, alt_track_plot_normalize_xj);

		// heading deviation error score
		double heading_err = heading_error(track_plot_0, track_plot_1, 30);

		// std::vector<double> line{track_plot_0->xs_.back(), track_plot_0->ys_.back(), track_plot_1->xs_.back() + 1920, track_plot_1->ys_.back(), 0, r_value, 0, 1 - heading_err};
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

			debug_messages.push_back("Target ID " +  std::to_string(track_plot_0->id_)  + " is dropped due to mismatch ");

			std::map<int, int>::iterator it;
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
	std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & good_tracks,
	std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> & filter_good_tracks,
	std::array<std::vector<int>, 2> & dead_tracks)
{
	get_total_number_of_tracks();
	std::map<int, std::map<int, double>> corrValues;
	std::set<int> removeSet;
	int track_id, centroid_x, centroid_y, size;
	int row = 0;

	for (auto & track : good_tracks[index]) {
		// Extract details from the track
		track_id = track->id;
		centroid_x = track->x;
		centroid_y = track->y;
		size = track->size;

		if (cumulative_tracks_[index]->track_plots_.find(matching_dict_[index][track_id]) 
			== cumulative_tracks_[index]->track_plots_.end())
		{
			std::vector<int> location;
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
			if (track_plot->frameNos_.size() >= 30 && track_plot->track_feature_variable_.size() >= 30 && sum != 0)
			{

				// look into 2nd camera's new tracks (new tracks first)
				std::map<int, std::shared_ptr<mcmt::TrackPlot>>::iterator alt_track_plot;
				for (alt_track_plot = cumulative_tracks_[alt]->track_new_plots_.begin();
					alt_track_plot != cumulative_tracks_[alt]->track_new_plots_.end(); alt_track_plot++)
				{

					sum = 0;
					for (int i = 0; i < static_cast<int>(alt_track_plot->second->track_feature_variable_.size()); i++)
					{
						sum += alt_track_plot->second->track_feature_variable_[i];
					}
					// track in 2nd camera must fulfills requirements to have at least 90 frames to match
					if (alt_track_plot->second->frameNos_.size() >= 30 && alt_track_plot->second->track_feature_variable_.size() >= 30 
						&& sum != 0)
					{
						double score = compute_matching_score(track_plot, alt_track_plot->second, index, alt);
						if (score != 0)
						{
							corrValues[track_id][alt_track_plot->second->id_] = score;
						}
					}
				}

				// look into other camera's matched tracks list (old tracks last)
				for (alt_track_plot = cumulative_tracks_[alt]->track_plots_.begin();
					alt_track_plot != cumulative_tracks_[alt]->track_plots_.end(); alt_track_plot++)
				{

					bool eligibility_flag = true;
					
					// do not consider dead tracks from the other camera
					for (auto & dead_track : dead_tracks[alt])
					{
						if (matching_dict_[alt].find(dead_track) != matching_dict_[alt].end() && matching_dict_[alt][dead_track] == alt_track_plot->second->id_)
						{
							eligibility_flag = false; // 2nd camera's track has already been lost. skip the process of matching for this track
						}
					}
											
					// test to see if alternate camera's track is currently being matched with current camera                        
					for (auto & alt_track : good_tracks[index])
					{
						if (alt_track_plot->second->id_ == matching_dict_[index][alt_track->id])
						{
							eligibility_flag = false; // 2nd camera's track has already been matched. skip the process of matching for this track
						}
					}

					sum = 0;
					for (int i = 0; i < static_cast<int>(alt_track_plot->second->track_feature_variable_.size()); i++)
					{
						sum += alt_track_plot->second->track_feature_variable_[i];
					}

					if (eligibility_flag && sum != 0)
					{
						double score = compute_matching_score(track_plot, alt_track_plot->second, index, alt);
						if (score != 0)
						{
							corrValues[track_id][alt_track_plot->second->id_] = score;
						}
					}
				}
			}

			row += 1;
		} 
		else
		{
			std::vector<int> location;
			location.push_back(centroid_x);
			location.push_back(centroid_y);

			auto track_plot = cumulative_tracks_[index]->track_plots_[matching_dict_[index][track_id]];
			track_plot->update(location, size, frame_count_);
			track_plot->calculate_track_feature_variable(frame_count_, fps_);
			filter_good_tracks[index].erase(filter_good_tracks[index].begin() + row);
		}

	}


	for (auto & track : filter_good_tracks[index])
	{
		std::map<int, double> maxValues = corrValues[track->id];
		int maxID = -1;
		double maxValue = -1;
		int global_max_flag = 0;

		// for the selected max track in the 2nd camera, we check to see if the track has a higher
		// cross correlation value with another track in current camera

		while (global_max_flag == 0 && maxValues.size() != 0)
		{
			for (auto it = maxValues.begin(); it != maxValues.end(); it++) {
				if (maxValue < it->second) {
					maxID = it->first;
				}
			}
			maxValue = maxValues[maxID];

			// search through current camera's tracks again, for the selected track that we wish to re-id with.
			// we can note that if there is a track in the current camera that has a higher cross correlation value
			// than the track we wish to match with, then the matching will not occur.
			for (auto & track_1 : filter_good_tracks[index])
			{
				if (corrValues[track_1->id].find(maxID) != corrValues[track_1->id].end())
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
			if (maxID != 1 && 
				(cumulative_tracks_[alt]->track_new_plots_.find(maxID) != cumulative_tracks_[alt]->track_new_plots_.end()))
			{
				// add notification message
				debug_messages.push_back("New target ID " +  std::to_string(next_id_)  + " acquired with a score of " + std::to_string(maxValue));

				// remove track plot in new tracks' list and add into matched tracks' list for alternate camera
				cumulative_tracks_[alt]->track_new_plots_[maxID]->id_ = next_id_;
				cumulative_tracks_[alt]->track_plots_.insert(
					std::pair<int, std::shared_ptr<mcmt::TrackPlot>>(next_id_, cumulative_tracks_[alt]->track_new_plots_[maxID]));
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
			}

			// if track is in 2nd camera's matched track list
			else
			{
				int track_id = track->id;
				auto track_plot = cumulative_tracks_[index]->track_new_plots_[track_id];
				track_plot->id_ = cumulative_tracks_[alt]->track_plots_[maxID]->id_;

				// add notification message
				debug_messages.push_back("New target ID " +  std::to_string(track_plot->id_)  + " acquired with a score of " + std::to_string(maxValue));

				// update track plot in the original track ID
				combine_track_plots(track_plot->id_, cumulative_tracks_[index], track_plot, frame_count_);

				// update dictionary matching list
				for (std::map<int, int>::iterator old_id = matching_dict_[index].begin(); old_id != matching_dict_[index].end(); old_id++)
				{
					if (old_id->second == track_plot->id_)
					{
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

	for (auto & remove_id : removeSet)
	{
		cumulative_tracks_[alt]->track_new_plots_.erase(remove_id);
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
std::vector<double> McmtMultiTrackerNode::normalise_track_plot(std::shared_ptr<mcmt::TrackPlot> track_plot)
{
	int total_track_feature = track_plot->track_feature_variable_.size();
	double mean = 0, variance = 0, std;
	std::vector<double> result;

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
		double res = (track_plot->track_feature_variable_[i] - mean) / (std * sqrt(total_track_feature));
		result.push_back(res);
	}

	return result;
}

double McmtMultiTrackerNode::compute_matching_score(std::shared_ptr<mcmt::TrackPlot> track_plot,
		std::shared_ptr<mcmt::TrackPlot> alt_track_plot, int index, int alt)
{
	// Normalization of cross correlation values
	auto track_plot_normalize_xj = normalise_track_plot(track_plot);
	auto alt_track_plot_normalize_xj = normalise_track_plot(alt_track_plot);

	// Updating of tracks in the local neighbourhood
	mcmt::update_other_tracks(track_plot, cumulative_tracks_[index]);
	mcmt::update_other_tracks(alt_track_plot, cumulative_tracks_[alt]);

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
	// 	std::vector<double> line{track_plot->xs_.back(), track_plot->ys_.back(), alt_track_plot->xs_.back() + 1920, alt_track_plot->ys_.back(), score, r_value, geometric_strength, 1 - heading_err};
	// 	lines.push_back(line);
	// }

	if (r_value > 0.4 && (geometric_strength == 0 || geometric_strength >= 0.5) && heading_err < 0.1 && score >= 0.75){
	//if (r_value > 0.45 && heading_err < 0.1){
		return score;
	}
	else {
		return 0;
	}
}

/**
 * Find cross correlation of two 1D arrays with size n
 * Involves the convolution of array X with array Y by sliding vector Y from left to right
 */
double McmtMultiTrackerNode::crossCorrelation(std::vector<double> X, std::vector<double> Y)
{
		double max = 0;
        std::vector<double> A;
        std::vector<double> K;

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
	std::vector<std::shared_ptr<TrackPlot::OtherTrack>> & other_tracks_0, 
	std::vector<std::shared_ptr<TrackPlot::OtherTrack>> & other_tracks_1)
{
	std::vector<double> relative_distances, shortest_distances;

	int total_num_other_tracks_0 = other_tracks_0.size();
	int total_num_other_tracks_1 = other_tracks_1.size();

	for (int i = 0; i < total_num_other_tracks_0; i++){
		double a_angle = other_tracks_0[i]->angle;
		double a_dist = other_tracks_0[i]->dist;

		relative_distances.clear();
		for (int j = 0; j < total_num_other_tracks_1; j++){
			double b_angle = other_tracks_1[j]->angle;
			double b_dist = other_tracks_1[j]->dist;

			relative_distances.push_back((std::min<double>(std::abs(a_angle - b_angle),
				(2 * M_PI) - std::abs(a_angle - b_angle))) / M_PI * 
				std::min<double>(a_dist / b_dist, b_dist / a_dist));
		}

		int total_num_relative_distances = relative_distances.size();
		if (total_num_relative_distances > 0){
			double minimum = relative_distances.front();
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
		double avg_shortest_distances = 0;
		for (int i = 0; i < total_num_shortest_distances; i++){
			avg_shortest_distances += shortest_distances[i];
		}
		avg_shortest_distances = avg_shortest_distances / total_num_shortest_distances;
		return std::max<double>(0.001, 0.2 - avg_shortest_distances) * 5;
	}
	else {
		return 0;
	}

}

double McmtMultiTrackerNode::heading_error(std::shared_ptr<mcmt::TrackPlot> track_plot, 
	std::shared_ptr<mcmt::TrackPlot> alt_track_plot, int history)
{
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

		deviation += std::min(std::abs(relative_0 - relative_1), 1 - std::abs(relative_0 - relative_1));

	}
		return (deviation / (history - 1));
}

/**
 * Computes the 3D position of a matched drone through triangulation methods.
 */
void McmtMultiTrackerNode::calculate_3D()
{
	double fx = 1454.6;
	double cx = 960.9;
	double fy = 1450.3;
	double cy = 543.7;
	double B = 1.5;
	int epsilon = 7;

	// Check for IDs that belong to both cumulative tracks 0 and 1
	std::set<int> matched_ids;
	for (auto i = cumulative_tracks_[0]->track_plots_.begin(); i != cumulative_tracks_[0]->track_plots_.end(); i++){
		for (auto j = cumulative_tracks_[1]->track_plots_.begin(); j != cumulative_tracks_[1]->track_plots_.end(); j++){
			if (i->first == j->first){
				matched_ids.insert(i->first);
			}
		}
	}

	for (std::set<int>::iterator it = matched_ids.begin(); it != matched_ids.end(); it++) {
		auto track_plot_0 = cumulative_tracks_[0]->track_plots_[*it];
		auto track_plot_1 = cumulative_tracks_[1]->track_plots_[*it];

		if ((track_plot_0->lastSeen_ == frame_count_) && (track_plot_1->lastSeen_ == frame_count_)){
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

	double aspect_ratio = img_size.width / img_size.height;

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