/**
 * @file mcmt_single_tracker_node.cpp
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
 * tracking pipeline. These functions interact with the key classes CameraTracks 
 * and TrackPlots which are essential to the tracking process. These classes may 
 * be found at mcmt_track_utils.cpp. This code also contains the McmtSingleTrackerNode 
 * class which handles the subscription from our ROS2 DDS-RTPS ecosystem.
 */

// local header files
#include <mcmt_track/mcmt_single_tracker_node.hpp>

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

using namespace mcmt;

McmtSingleTrackerNode::McmtSingleTrackerNode() : Node("SingleTrackerNode") {

	node_handle_ = std::shared_ptr<rclcpp::Node>(this, [](::rclcpp::Node *) {});
	declare_parameters();
	get_parameters();
	RCLCPP_INFO(this->get_logger(), "Initializing Mcmt Single Tracker Node");

	// get camera parameters
	if (is_realtime_ == true) {
		cap_ = cv::VideoCapture(std::stoi(video_input_));
	} else {
		cap_ = cv::VideoCapture(video_input_);
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
		cv::Size(frame_w_, frame_h_));
	cap_.release();

	// initialize frame count
	frame_count_ = 0;

	// initialze plotting parameters
	plot_history_ = 200;
	font_scale_ = 0.5;

	cumulative_tracks_ = std::shared_ptr<CameraTracks>(new CameraTracks(0));

	process_detection_callback();

}

/**
 * This function creates subscription to the detection info publisher from the McmtSingleDetectorNode
 * and contains the main pipeline for the tracker node callback. The pipeline includes:
 * 1. Processing of new tracks
 * 2. Plotting of info on each camera's frames
 */
void McmtSingleTrackerNode::process_detection_callback()
{
	// set qos and topic name
	auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
	topic_name_ = "mcmt/detection_info";
	detection_sub_ = this->create_subscription<mcmt_msg::msg::SingleDetectionInfo> (
		topic_name_,
		qos,
		[this](const mcmt_msg::msg::SingleDetectionInfo::SharedPtr msg) -> void
		{
			// get start time
			auto start = std::chrono::system_clock::now();	

			// declare tracking variables
			std::shared_ptr<cv::Mat> frame_;
			std::vector<std::shared_ptr<GoodTrack>> good_tracks_;

			// process detection info
			process_msg_info(msg, frame_, good_tracks_);

			int track_id, centroid_x, centroid_y, size;
			for (auto & track : good_tracks_) {
				track_id = track->id;
				centroid_x = track->x;
				centroid_y = track->y;
				size = track->size;

				std::vector<int> location;
				location.push_back(centroid_x);
				location.push_back(centroid_y);

				if (cumulative_tracks_->track_plots_.find(track_id) == cumulative_tracks_->track_plots_.end()) {
					cumulative_tracks_->track_plots_[track_id] = std::shared_ptr<TrackPlot>(new TrackPlot(track_id));
					debug_messages.push_back("New target ID " +  std::to_string(track_id));
				}

				cumulative_tracks_->track_plots_[track_id]->update(location, size, frame_count_);
			}

			print_frame_summary(good_tracks_);
			annotate_frames(frame_, cumulative_tracks_);
			graphical_UI(frame_, cumulative_tracks_);

			// get trackplot process time
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - start;
			std::cout << "Trackplot process took: " << elapsed_seconds.count() << "s\n";		

			// show cv window
			std::string window_name = "Annotated";
			imshow_resized_dual(window_name, *frame_.get());
			recording_.write(*frame_.get());

			frame_count_ += 1;

			cv::waitKey(1);
		}
	);
}

/**
 * This function processes the detection messsage information
 */
void McmtSingleTrackerNode::process_msg_info(mcmt_msg::msg::SingleDetectionInfo::SharedPtr msg,
	std::shared_ptr<cv::Mat> & frame,
	std::vector<std::shared_ptr<GoodTrack>> & good_tracks) {
	
	// get camera frame from message
	frame = std::shared_ptr<cv::Mat>(
		new cv::Mat(
			msg->image.height, msg->image.width, encoding2mat_type(msg->image.encoding),
			const_cast<unsigned char *>(msg->image.data.data()), msg->image.step));

	// get good tracks
	int total_num_tracks = msg->goodtracks_id.size();

	for (int i = 0; i < total_num_tracks; i++) {
		auto good_track = std::shared_ptr<GoodTrack>(new GoodTrack());
		good_track->id = msg->goodtracks_id[i];
		good_track->x = msg->goodtracks_x[i];
		good_track->y = msg->goodtracks_y[i];
		good_track->size = msg->goodtracks_size[i];
		good_tracks.push_back(good_track);
	}
}


void McmtSingleTrackerNode::print_frame_summary(std::vector<std::shared_ptr<GoodTrack>> good_tracks_) {

	std::cout << "SUMMARY OF FRAME " << frame_count_ << std::endl;
	std::cout << "Tracks: ";
	for (auto good_track : good_tracks_) {
		std::cout << "(" << good_track->id << ") | ";
	}
	std::cout << std::endl;
}

void McmtSingleTrackerNode::annotate_frames(std::shared_ptr<cv::Mat> frame_, std::shared_ptr<CameraTracks> cumulative_tracks_) {

	
	// draw tracks on opencv GUI to monitor the detected tracks
	// loop through each camera frame
	std::map<int, std::shared_ptr<mcmt::TrackPlot>>::iterator track;

	cv::putText(*frame_.get(), "Frame Count " + std::to_string(frame_count_), cv::Point(20, 25),
		cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 0.85, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
	
	// loop through every track plot
	if (cumulative_tracks_->track_plots_.empty() == false) {
		for (track = cumulative_tracks_->track_plots_.begin(); track != cumulative_tracks_->track_plots_.end(); track++) {
			if ((frame_count_ - track->second->lastSeen_) <= fps_) {
				
				cv::Point2i rect_top_left((track->second->xs_.back() - (track->second->size_.back())), 
														(track->second->ys_.back() - (track->second->size_.back())));
	
				cv::Point2i rect_bottom_right((track->second->xs_.back() + (track->second->size_.back())), 
																(track->second->ys_.back() + (track->second->size_.back())));
	
				
				cv::rectangle(*frame_.get(), rect_top_left, rect_bottom_right, colors[track->second->id_ % 10], 2);

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
					cv::Vec3b pixelColor = (*frame_.get()).at<cv::Vec3b>(track->second->ys_[idx], track->second->xs_[idx]);

					cv::circle(*frame_.get(), cv::Point(track->second->xs_[idx], track->second->ys_[idx]), 3,
						cv::Scalar((int) (pixelColor[0] * beta + (colors[track->second->id_ % 10][0] * alpha)),
									(int) (pixelColor[1] * beta + (colors[track->second->id_ % 10][1] * alpha)),
									(int) (pixelColor[2] * beta + (colors[track->second->id_ % 10][2] * alpha))), -1);
				}

				// put ID on opencv GUI
				if (shown_indexes_.empty() == false) {
					cv::putText(*frame_.get(), "ID: " + std::to_string(track->second->id_).substr(0,4), 
						cv::Point(rect_top_left.x + 20, rect_top_left.y - 5), cv::FONT_HERSHEY_SIMPLEX,
						font_scale_, colors[track->second->id_ % 10], 1, cv::LINE_AA);
				}
				
				cv::circle(*frame_.get(), cv::Point(rect_top_left.x + 5, rect_top_left.y - 10), 5, status_color, -1);	

			}

			// if (track->second->check_stationary()) {
			// 	cv::putText(*frame_.get(), "S", cv::Point(track->second->xs_.back(), track->second->ys_.back() - 40),
			// 			cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 2, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
			// }
		}
	}
}

void McmtSingleTrackerNode::graphical_UI(std::shared_ptr<cv::Mat> frame_, std::shared_ptr<CameraTracks> cumulative_tracks_) {
	
	// Summary box
	cv::rectangle(*frame_.get(), cv::Point(190, 860), cv::Point(1000, 900), cv::Scalar(220,220,220), -1);
	cv::rectangle(*frame_.get(), cv::Point(190, 860), cv::Point(1000, 900), cv::Scalar(110,110,110), 4);
	int spacing = 0;
	int drones_on_screen = 0;
	if (cumulative_tracks_->track_plots_.empty() == false) {
		for (auto track = cumulative_tracks_->track_plots_.begin(); 
			track != cumulative_tracks_->track_plots_.end(); track++) {
			if ((frame_count_ - track->second->lastSeen_) <= fps_) {
				drones_on_screen++;
				cv::putText(*frame_.get(), "ID: " + std::to_string(track->second->id_).substr(0,4), cv::Point(210 + spacing, 890), cv::FONT_HERSHEY_SIMPLEX,
						font_scale_ * 1.5, colors[track->second->id_ % 10], 2, cv::LINE_AA);
				spacing += 150;	
			}
		}
	}

	// Notification box
	cv::rectangle(*frame_.get(), cv::Point(20, 920), cv::Point(1000, 1060), cv::Scalar(200,200,200), -1);
	int num_of_messages = 4;
	spacing = 0;
	for (int i = 0; i < num_of_messages && i < debug_messages.size(); i++, spacing -= 30) {
		cv::putText(*frame_.get(), debug_messages[debug_messages.size() - 1 - i], cv::Point(40, 1040 + spacing), 
						cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 1.5, cv::Scalar(0,0,0), 2, cv::LINE_AA);
	}

	// Targets box
	cv::rectangle(*frame_.get(), cv::Point(20, 780), cv::Point(170, 900), cv::Scalar(220,220,220), -1);
	cv::rectangle(*frame_.get(), cv::Point(20, 780), cv::Point(170, 900), cv::Scalar(110,110,110), 4);
	cv::putText(*frame_.get(), "TARGETS", cv::Point(45, 805), cv::FONT_HERSHEY_SIMPLEX,
				font_scale_ * 1.5, cv::Scalar(0,0,0), 2, cv::LINE_AA);
	cv::putText(*frame_.get(), std::to_string(drones_on_screen), cv::Point(60, 885), cv::FONT_HERSHEY_SIMPLEX,
				font_scale_ * 6, cv::Scalar(0,0,0), 6, cv::LINE_AA);
}

void McmtSingleTrackerNode::imshow_resized_dual(std::string & window_name, cv::Mat & img) {
	cv::Size img_size = img.size();

	double aspect_ratio = (double) img_size.width / (double) img_size.height;

	cv::Size window_size;
	window_size.width = 1080;
	window_size.height = (int) (1080.0 / aspect_ratio);
	
	cv::Mat img_;
	cv::resize(img, img_, window_size, 0, 0, cv::INTER_CUBIC);
	cv::imshow(window_name, img_);
}

/**
 * This function declares our mcmt software parameters as ROS2 parameters.
 */
void McmtSingleTrackerNode::declare_parameters() {
	// declare ROS2 video parameters
	this->declare_parameter("IS_REALTIME");
	this->declare_parameter("VIDEO_INPUT");
	this->declare_parameter("FRAME_WIDTH");
	this->declare_parameter("FRAME_HEIGHT");
	this->declare_parameter("OUTPUT_VIDEO_PATH");
	this->declare_parameter("OUTPUT_CSV_PATH");
}

/**
 * This function gets the mcmt parameters from the ROS2 parameters
 */
void McmtSingleTrackerNode::get_parameters() {
	// get video parameters
	IS_REALTIME_param = this->get_parameter("IS_REALTIME");
	VIDEO_INPUT_param = this->get_parameter("VIDEO_INPUT");
	FRAME_WIDTH_param = this->get_parameter("FRAME_WIDTH");
	FRAME_HEIGHT_param = this->get_parameter("FRAME_HEIGHT");
	OUTPUT_VIDEO_PATH_param = this->get_parameter("OUTPUT_VIDEO_PATH");
	OUTPUT_CSV_PATH_param = this->get_parameter("OUTPUT_CSV_PATH");

	// initialize video parameters
	is_realtime_ = IS_REALTIME_param.as_bool();
	video_input_ = VIDEO_INPUT_param.as_string();
	FRAME_WIDTH_ = FRAME_WIDTH_param.as_int();
	FRAME_HEIGHT_ = FRAME_HEIGHT_param.as_int();
	output_vid_path_ = OUTPUT_VIDEO_PATH_param.as_string();
	output_csv_path_ = OUTPUT_CSV_PATH_param.as_string();
}

int McmtSingleTrackerNode::encoding2mat_type(const std::string & encoding) {
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