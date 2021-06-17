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

			// update_cumulative_tracks(0);
			// update_cumulative_tracks(1);

			// process_new_tracks(0, 1);
			// process_new_tracks(1, 0);

			// verify_existing_tracks(0, 1);
			// verify_existing_tracks(1, 0);

			// calculate_3D();

			// prune_tracks(0);
			// prune_tracks(1);

			// draw tracks on opencv GUI to monitor the detected tracks
			// lopp through each camera frame
			for (int i = 0; i < 2; i++) {
				cv::putText(*frames_[i].get(), "CAMERA " + std::to_string(i), cv::Point(20, 30),
					cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 0.85, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
				
				cv::putText(*frames_[i].get(), "Frame Count " + std::to_string(frame_count_), cv::Point(20, 50),
					cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 0.85, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
				
				// loop through every track plot
				std::cout << "hi" << std::endl;
				std::cout << cumulative_tracks_[i]->track_plots_.empty() << std::endl;
				if (cumulative_tracks_[i]->track_plots_.empty() == false) {
					for (auto & track_plot : cumulative_tracks_[i]->track_plots_) {
						if ((frame_count_ - track_plot->lastSeen_) <= fps_) {
							// get last frames up till plot history (200)
							shown_indexes_.clear();

							for(int j = track_plot->frameNos_.size() - 1; j >= 0; j--) {
								if (track_plot->frameNos_[j] > (frame_count_ - plot_history_)) {
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
							if (shown_indexes_.empty() == false) {
								cv::putText(*frames_[i].get(), "ID: " + std::to_string(track_plot->id_), 
									cv::Point(track_plot->xs_[-1], track_plot->ys_[-1] + 15), cv::FONT_HERSHEY_SIMPLEX,
									font_scale_, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
								
								if (track_plot->xyz_.empty() == false) {
									cv::putText(*frames_[i].get(), "X: " + std::to_string(track_plot->xyz_[0]),
										cv::Point(track_plot->xs_[-1], track_plot->ys_[-1] + 30), cv::FONT_HERSHEY_SIMPLEX,
										font_scale_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

									cv::putText(*frames_[i].get(), "Y: " + std::to_string(track_plot->xyz_[1]),
										cv::Point(track_plot->xs_[-1], track_plot->ys_[-1] + 45), cv::FONT_HERSHEY_SIMPLEX,
										font_scale_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

									cv::putText(*frames_[i].get(), "Z: " + std::to_string(track_plot->xyz_[2]),
										cv::Point(track_plot->xs_[-1], track_plot->ys_[-1] + 60), cv::FONT_HERSHEY_SIMPLEX,
										font_scale_, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
								}
							}
						}
					}
				}
			std::cout << "hi" << std::endl;

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

void McmtMultiTrackerNode::heading_error(TrackPlot track_plot, TrackPlot alt_track_plot, int history)
{
	int deviation = 0;
	auto dx_0 = track_plot.xs.back() - track_plot.xs[track_plot.xs.size() - 2];
	auto dy_0 = track_ploy.ys.back() - track_plot.xs[track_plot.ys.size() - 2];
	auto angle_0 = (atan2(dy_0, dx_0) + M_PI) / (2 * M_PI);

	auto dx_1 = alt_track_plot.xs.back() - alt_track_plot.xs[alt_track_plot.xs.size() - 2];
	auto dy_1 = alt_track_ploy.ys.back() - alt_track_plot.xs[alt_track_plot.ys.size() - 2];
	auto angle_1 = (atan2(dy_1, dx_1) + M_PI) / (2 * M_PI);

	for (int i = -2; i > 1-history; i--){ // Can help me double check if should be i > 1-history or i >= 1-history?
		dx_0 = track_plot.xs[i] - track_plot.xs[track_plot.xs[i-1];
		dy_0 = track_plot.ys[i] - track_plot.ys[track_plot.ys[i-1];
		angle_0 = (atan2(dy_0, dx_0) + M_PI) / (2 * M_PI);

		dx_1 = track_plot.xs[i] - track_plot.xs[track_plot.xs[i-1];
		dy_1 = track_plot.ys[i] - track_plot.ys[track_plot.ys[i-1];
		angle_1 = (atan2(dy_0, dx_0) + M_PI) / (2 * M_PI);

		auto relative_0 = (angle_0 - rotation_0) % 1;
		auto relative_1 = (angle_1 - rotation_1) % 1;

		deviation += std::min(std::abs((relative_0 - relative_1) % 1), std::abs((relative_1 - relative_0) % 1);
	}

	return deviation / 19;
}

/**
 * Computes the 3D position of a matched drone through triangulation methods.
 */
void McmtMultiTrackerNode::calculate3D()
{
	float fx = 1454.6;
	float cx = 960.9;
	float fy = 1450.3;
	float cy = 543.7;
	float B = 1.5;
	int epsilon = 7;

	std::array<std::shared_ptr<mcmt::CameraTracks>, 2> matched_ids;
	// Placeholder. Need to find intersect of cumulative_tracks

	int total_num_matched_ids = matched_ids.size();
	for (int matched_id = 0; matched_id < matched_ids.size(); matched_id++){
		auto track_plot_0 = cumulative_tracks_[0]->track_plots_[matched_id];
		auto track_plot_1 = cumulative_tracks_[1]->track_plots_[matched_id];

		if ((track_plot_0.lastSeen_ == frame_count) && (track_plot_1.lastSeen_ == frame_count)){
			int x_L = track_plot_0.xs.back();
			int y_L = track_plot_0.ys.back();
			int x_R = track_plot_1.xs.back();
			int y_R = track_plot_1.ys.back();

			auto alpha_L = atan2(x_L - cx, fx) / M_PI * 180;
			auto alpha_R = atan2(x_R - cx, fx) / M_PI * 180;

			auto gamma = epsilon + alpha_L - alpha_R; // unused - shld we remove?

			auto Z = B / (tan((alpha_L + epsilon / 2) * (M_PI / 180)) - tan((alpha_L - epsilon / 2) * (M_PI / 180)));
			auto X = (Z * tan((alpha_L + epsilon / 2) * (M_PI / 180)) - B / 2
						Z * tan((alpha_L - epsilon / 2) * (M_PI / 180)) + B / 2) / 2;
			auto Y = (Z * -(y_L - cy) / fy + Z * -(y_R - cy) / fy) / 2;

			auto tilt = 10 * M_PI / 180;
			Eigen::Matrix3d R;
			R << 1, 0, 0,
				0, cos(tilt), sin(tilt),
				0, -sin(tilt); cos(tilt);
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

			track_plot_0.xyz_.clear();
			track_plot_0.xyz_.push_back(X);
			track_plot_0.xyz_.push_back(Y);
			track_plot_0.xyz_.push_back(Z);
			track_plot_1.xyz_.clear();
			track_plot_1.xyz_.push_back(X);
			track_plot_1.xyz_.push_back(Y);
			track_plot_1.xyz_.push_back(Z);

		}
		else{
			track_plot_0.xyz_.clear();
			track_plot_1.xyz_.clear();
		}
	}
}