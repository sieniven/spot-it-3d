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
#include <algorithm>
#include <functional>

using namespace mcmt;

McmtMultiTrackerNode::McmtMultiTrackerNode()
:Node("MultiTrackerNode")
{
	node_handler_ = std::shared_ptr<rclcpp::Node>(this, [](::rclcpp::Node *) {});
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

	// intialize video writer
	int codec = cv::CV_FOURCC('M', 'J', 'P', 'G');
	recording_.open(output_vid_path_, codec, fps_, cv::Size(int(frame_w_ * 2), frame_h_));
	cap_.release();

	// initialize timer and frame count
	frame_count_ = 0;
	start_ = std::chrono::system_clock::now();
	end_ = std::chrono::system_clock::now();

	// initialize origin (0, 0) and track id
	origin_.push_back(0);
	origin_.push_back(0);
	next_id_ = 0;

	// initialze plotting parameters
	plot_history_ = 200;
	font_scale_ = 0.5;
	for (int i = 0; i < plot_history_; i++) {
		colors_.push_back(scalar_to_rgb(i, plot_history_));
	}

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
			start_ = std::chrono::system_clock::now();

			// declare tracking arrays
			std::array<std::shared_ptr<cv::Mat>, 2> frames_;
			std::array<std::vector<std::shared_ptr<GoodTrack>>, 2> good_tracks_, filter_good_tracks_;
			std::array<std::vector<int>, 2> dead_tracks_;

			// process detection info
			process_msg_info(msg, frames_, good_tracks_, dead_tracks_);

			// get time taken to get message
			std::chrono::duration<double> elapsed_seconds = start_ - end_;
			std::cout << "Time taken to get message: " << elapsed_seconds.count() << "s\n";

			update_cumulative_tracks(0);
			update_cumulative_tracks(1);

			process_new_tracks(0, 1);
			process_new_tracks(1, 0);

			verify_existing_tracks(0, 1);
			verify_existing_tracks(1, 0);

			calculate_3D();

			prune_tracks(0);
			prune_tracks(1);

			// draw tracks on opencv GUI to monitor the detected tracks
			// lopp through each camera frame
			for (int i = 0, i < 2; i++) {
				cv::putText(*frames_[i].get(), "CAMERA " + std::to_string(i), cv::Point(20, 30),
					cv::FONT_HERSHEY_SIMPLEX, front_scale_ * 0.85, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
				
				cv::putText(*frames_[i].get(), "Frame Count " + std::to_string(frame_count_), cv::Point(20, 50),
					cv::FONT_HERSHEY_SIMPLEX, front_scale_ * 0.85, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
				
				// loop through every track plot
				for (auto & track_plot : cumulative_tracks_[i]->track_plots_) {
					if ((frame_count_ - track_plot->lastSeen_) <= fps_) {
						// get last frames up till plot history (200)
						shown_indexes_.clear();

						for(int j = track_plot->frameNos_.size() - 1; j >= 0; j--) {
							if (track_plot->frameNos_[j] > (frame_count_ - plot_history_) {
								shown_indexes_.push_back(j);
							} else {
								break;
							}
						}

						// draw the track's path history on opencv GUI
						for (auto & idx : shown_indexes_) {
							int color_idx = track_plot->frameNos_[idx] - frame_count_ + plot_history_ - 1;
							cv::circle(*frames_[i].get(), cv::Point(track_plot->xs_[idx], track_plot->ys_[idx]), 3,
								cv::Scalar(colors_[color_idx][2], colors_[color_idx][1], colors_[color_idx][0]), -1);
						}
						
						// put ID and XYZ coordinates on opencv GUI
						if (shown_indexes_.size() != 0) {
							cv::putText(*frames_[i].get(), "ID: " + std::to_string(track_plot->id_), 
								cv::Point(track_plot->xs_[-1], track_plot->ys_[-1] + 15), cv::FONT_HERSHEY_SIMPLEX,
								front_scale_, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
							
							if (track_plot->xyz_.size() != 0) {
								cv::putText(*frames_[i].get(), "X: " + std::to_string(track_plot->xyz_[0]),
									cv::Point(track_plot->xs_[-1], track_plot->ys_[-1] + 30), cv::FONT_HERSHEY_SIMPLEX,
									front_scale_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

								cv::putText(*frames_[i].get(), "Y: " + std::to_string(track_plot->xyz_[1]),
									cv::Point(track_plot->xs_[-1], track_plot->ys_[-1] + 45), cv::FONT_HERSHEY_SIMPLEX,
									front_scale_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

								cv::putText(*frames_[i].get(), "Z: " + std::to_string(track_plot->xyz_[2]),
									cv::Point(track_plot->xs_[-1], track_plot->ys_[-1] + 60), cv::FONT_HERSHEY_SIMPLEX,
									front_scale_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
							}
						}
					}
				}
			}

			// get trackplot process time
			end_ = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end_ - start_;
			std::cout << "Trackplot process took: " << elapsed_seconds.count() << "s\n";

			// show and save video combined tracking frame
			frame_count_ += 1;
			cv::Mat combined_frame;
			cv::hconcat(*frames_[0].get(), *frames_[1].get(), combined_frame);
			imshow_resized_dual("Detection", combined_frame);
			end_ = std::chrono::system_clock::now();
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
	
	frames.push_back(frame_1);
	frames.push_back(frame_2);

	// get dead tracks
	std::vector<int> gonetracks_1(msg->gonetracks_id_one);
	std::vector<int> gonetracks_2(msg->gonetracks_id_two);
	dead_tracks.push_back(gonetracks_1);
	dead_tracks.push_back(gonetracks_2);

	// get good tracks
	int total_num_tracks = msg->goodtracks_id_one.size();
	for (i = 0; i < total_num_tracks; i++) {
		auto good_track = std::shared_ptr<GoodTrack>(new GoodTrack());
		good_track->id = msg->goodtracks_id_one[i];
		good_track->x = msg->goodtracks_x_one[i];
		good_track->y = msg->goodtracks_y_one[i];
		good_tracks[0].push_back(good_track);
	}

	int total_num_tracks = msg->goodtracks_id_two.size();
	for (i = 0; i < total_num_tracks; i++) {
		auto good_track = std::shared_ptr<GoodTrack>(new GoodTrack());
		good_track->id = msg->goodtracks_id_two[i];
		good_track->x = msg->goodtracks_x_two[i];
		good_track->y = msg->goodtracks_y_two[i];
		good_tracks[1].push_back(good_track);
	}
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

std::string McmtMultiTrackerNode::mat_type2encoding(int mat_type)
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