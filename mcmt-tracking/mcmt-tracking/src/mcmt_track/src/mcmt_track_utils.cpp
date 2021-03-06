/**
 * @file mcmt_track_utils.cpp
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
 * This file contains the definition of the classes (CameraTracks and TrackPlots)
 * and their associated methods, which is primarily used in the re-identification
 * pipeline.
 */

// mathematical constants
#define _USE_MATH_DEFINES

// local header files
#include <mcmt_track/mcmt_track_utils.hpp>

// standard package imports
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <algorithm>
#include <functional>

using namespace mcmt;

namespace mcmt {

	/**
	 * This class is for tracking the camera's track plots
	 */
	CameraTracks::CameraTracks(int index) {
		// set camera index
		index_ = index;
	}

	/**
	 * This class is for storing our tracks' information, calculating track feature
	 * variable, and caluclating geometrical angles and distances of the tracks.
	 */
	TrackPlot::TrackPlot(int track_id) {
		// set the trackid and the mismatch count to initial value of zero
		id_ = track_id;
		mismatch_count_ = 0;
		lastSeen_ = 0;
	}

	/**
	 * This function updates the track with the latest track information
	 */
	void TrackPlot::update(std::vector<int> & location, int & size, int & frame_no) {
		xs_.push_back(location[0]);
		ys_.push_back(location[1]);
		frameNos_.push_back(frame_no);
		size_.push_back(size);
		lastSeen_ = frame_no;
	}

	/**
	 * This function calculates the track feature variable of the track in the 
	 * current frame, and stores the value of the track feature variable inside
	 * the vector track_feature_variable_
	 */
	void TrackPlot::calculate_track_feature_variable(int & frame_no, int & fps) {
		// checks if there is enough data to calculate the turning angle (at least
		// 3 points) and that the data is from the current frame
		if (frameNos_.size() >= 3 && frameNos_.back() == frame_no) {
			// check if the last 3 frames are consecutive
			if ((frameNos_.end()[-2] == (frameNos_.end()[-1] - 1)) &&
					(frameNos_.end()[-3] == (frameNos_.end()[-2] - 1))) {
				// retrieve the x and y values of the last 3 points
				int t = 2;
				std::vector<int> x(xs_.end() - 3, xs_.end());
				std::vector<int> y(ys_.end() - 3, ys_.end());

				// Turning angle and curvature
				
				/** 
				 * This code block has not been ported yet as the old method of calculation
				 * is not in the current pipeline. to see the old methodology, refer to the
				 * python version of mcmt_track_utils.py.
				 */

				// Pace
				float d = hypot((x[t] - x[t - 1]), (y[t] - y[t - 1]));
				float pace = d * fps;
				pace_.push_back(pace);

				// append track feature variable. the current pipeline uses pace as our
				// track feature variable value
				track_feature_variable_.push_back(pace);
			}
		}
	}

	/**
	 * This function checks if the current TrackPlot has been stationary for too long,
	 * by calculating the distance the track moved for the last 9 frames. it returns
	 * a bool True if its average distance moved per frame is less than 3.0, and returns
	 * False if otherwise
	 */
	bool TrackPlot::check_stationary() {
		// declare function variable to store total Euclidean distance traveled
		float distance = 0.0;

		for (int i = -1; i >= -9; i--) {
			int dx = xs_[xs_.size() + i] - xs_[xs_.size() + i - 1];
			int dy = ys_[ys_.size() + i] - ys_[ys_.size() + i - 1];

			// get euclidean distance, and add to total distance
			distance += hypot(dx, dy);
		}

		distance /= 9;

		if (distance < 3.0) {
			return true;
		} else {
			return false;
		}
	}

	/**
	 * this function updates the current frame's other_tracks list
	 */
	void update_other_tracks(std::shared_ptr<TrackPlot> trackplot,
		std::shared_ptr<CameraTracks> & cumulative_track) {
		
		// clear other_tracks vector
		trackplot->other_tracks_.clear();

		// iterate through the camera's tracks
		std::map<int, std::shared_ptr<mcmt::TrackPlot>>::iterator other_track;
		for (other_track = cumulative_track->track_plots_.begin(); 
			other_track != cumulative_track->track_plots_.end(); other_track++) {

			if (other_track->second->xs_.size() != 0 && other_track->second->ys_.size() != 0) {
				int dx = other_track->second->xs_.end()[-1] - trackplot->xs_.end()[-1];
				int dy = other_track->second->ys_.end()[-1] - trackplot->ys_.end()[-1];

				if (dx != 0 && dy != 0) {
					auto new_other_track = std::shared_ptr<TrackPlot::OtherTrack>(new TrackPlot::OtherTrack());
					new_other_track->angle = atan2(dy, dx);
					new_other_track->dist = hypot(dx, dy);
					trackplot->other_tracks_.push_back(new_other_track);
				}
			}
		}

		for (other_track = cumulative_track->track_new_plots_.begin(); 
			other_track != cumulative_track->track_new_plots_.end(); other_track++) {
			if (other_track->second->xs_.size() != 0 && other_track->second->ys_.size() != 0) {
				int dx = other_track->second->xs_.end()[-1] - trackplot->xs_.end()[-1];
				int dy = other_track->second->ys_.end()[-1] - trackplot->ys_.end()[-1];

				if (dx != 0 && dy != 0) {
					auto new_other_track = std::shared_ptr<TrackPlot::OtherTrack>(new TrackPlot::OtherTrack());
					new_other_track->angle = atan2(dy, dx);
					new_other_track->dist = hypot(dx, dy);
					trackplot->other_tracks_.push_back(new_other_track);
				}
			}
		}
	}

	void combine_track_plots(
		int & index,
		std::shared_ptr<CameraTracks> camera_tracks,
		std::shared_ptr<TrackPlot> track_plot,
		int & frame_count) {
		
		// append the various variable value vectors
		camera_tracks->track_plots_[index]->xs_.insert(
			camera_tracks->track_plots_[index]->xs_.end(),
			track_plot->xs_.begin(),
			track_plot->xs_.end());
		
		camera_tracks->track_plots_[index]->ys_.insert(
			camera_tracks->track_plots_[index]->ys_.end(),
			track_plot->ys_.begin(),
			track_plot->ys_.end());
		
		camera_tracks->track_plots_[index]->frameNos_.insert(
			camera_tracks->track_plots_[index]->frameNos_.end(),
			track_plot->frameNos_.begin(),
			track_plot->frameNos_.end());

		camera_tracks->track_plots_[index]->turning_angle_.insert(
			camera_tracks->track_plots_[index]->turning_angle_.end(),
			track_plot->turning_angle_.begin(),
			track_plot->turning_angle_.end());

		camera_tracks->track_plots_[index]->curvature_.insert(
			camera_tracks->track_plots_[index]->curvature_.end(),
			track_plot->curvature_.begin(),
			track_plot->curvature_.end());

		camera_tracks->track_plots_[index]->pace_.insert(
			camera_tracks->track_plots_[index]->pace_.end(),
			track_plot->pace_.begin(),
			track_plot->pace_.end());

		camera_tracks->track_plots_[index]->track_feature_variable_.insert(
			camera_tracks->track_plots_[index]->track_feature_variable_.end(),
			track_plot->track_feature_variable_.begin(),
			track_plot->track_feature_variable_.end());

		camera_tracks->track_plots_[index]->lastSeen_ = frame_count;
	}

}