/**
 * @file mcmt_track_utils.cpp
 * @author your name (you@domain.com)
 * @author Niven Sie, sieniven@gmail.com
 * @author Seah Shao Xuan
 * 
 * This code contains the common functions and classes CameraTracks and TrackPlot
 * that are used in our main tracking and re-identifcation algorithm.
 */

#define _USE_MATH_DEFINES

#include <mcmt_track/mcmt_track_utils.hpp>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <algorithm>
#include <functional>

using namespace mcmt;

/**
 * This class is for tracking the camera's track plots
 */
CameraTracks::CameraTracks(int index)
{
	// set camera index
	index_ = index;
}

/**
 * This class is for storing our tracks' information, calculating track feature
 * variable, and caluclating geometrical angles and distances of the tracks.
 */
TrackPlot::TrackPlot(int track_id)
{
	// set the trackid and the mismatch count to initial value of zero
	id_ = track_id;
	mismatch_count_ = 0;
	lastSeen_ = 0;
}

/**
 * This function updates the track with the latest track information
 */
void TrackPlot::update(std::vector<int> & location, int & frame_no)
{
	xs_.push_back(location[0]);
	ys_.push_back(location[1]);
	frameNos_.push_back(frame_no);
	lastSeen_ = frame_no;
}

/**
 * This function calculates the track feature variable of the track in the 
 * current frame, and stores the value of the track feature variable inside
 * the vector track_feature_variable_
 */
void TrackPlot::calculate_track_feature_variable(int & frame_no, int & fps)
{
	// checks if there is enough data to calculate the turning angle (at least
	// 3 points) and that the data is from the current frame
	if (frameNos_.size() >= 3 && frameNos.back() == frame_no) {
		// check if the last 3 frames are consecutive
		if ((frameNos_.end()[-2] == (frameNos_.end()[-1] - 1)) &&
				(frameNos_.end()[-3] == (frameNos_.end()[-2] - 1))) {
			// retrieve the x and y values of the last 3 points
			int t = 2
			std::vector<int> x(xs_.end() - 3, xs_.end());
			std::vector<int> y(ys_.end() - 3, ys_.end());

			// Turning angle
			// ~this code block has not been ported yet as the old method of calculation
			// is not in the current pipeline. to see the old methodology, refer to the
			// python version of mcmt_track_utils.py

			// Curvature
			// ~this code block has not been ported yet as the old method of calculation
			// is not in the current pipeline. to see the old methodology, refer to the
			// python version of mcmt_track_utils.py

			// Pace
			// using dt = 1/fps
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
 * this function updates the current frame's other_tracks list
 */
void TrackPlot::update_other_tracks(std::shared_ptr<CameraTracks> & cumulative_tracks)
{
	// clear other_tracks vector
	other_tracks_.clear();

	// iterate through the camera's tracks
	for (auto & other_track : cumulative_tracks->track_plots_) {
		if (other_track->xs_.size() != 0 && other_track->ys_.size() != 0) {
			int dx = other_track->xs_.end()[-1] - xs_.end()[-1];
			int dy = other_track->ys_.end()[-1] - ys_.end()[-1];

			if (dx != 0 && dy != 0) {
				auto new_other_track = std::shared_ptr<OtherTrack>(new OtherTrack());
				new_other_track->angle = atan2(dy, dx);
				new_other_track->dist = hypot(dx, dy);
				other_tracks_.push_back(new_other_track);
			}
		}
	}

	for (auto & other_track : cumulative_tracks->track_new_plots_) {
		if (other_track->xs_.size() != 0 && other_track->ys_.size() != 0) {
			int dx = other_track->xs_.end()[-1] - xs_.end()[-1];
			int dy = other_track->ys_.end()[-1] - ys_.end()[-1];

			if (dx != 0 && dy != 0) {
				auto new_other_track = std::shared_ptr<OtherTrack>(new OtherTrack());
				new_other_track->angle = atan2(dy, dx);
				new_other_track->dist = hypot(dx, dy);
				other_tracks_.push_back(new_other_track);
			}
		}
	}
}

/**
 * This function checks if the current TrackPlot has been stationary for too long,
 * by calculating the distance the track moved for the last 9 frames. it returns
 * a bool True if its average distance moved per frame is less than 3.0, and returns
 * False if otherwise
 */
bool TrackPlot::check_stationary()
{
	// declare function variable to store total Euclidean distance traveled
	float distance = 0.0;

	for (int i = -1; i >= -9; i--) {
		int dx = xs.end()[i] - xs.end()[i-1];
		int dy = ys.end()[i] - ys.end()[i-1];

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

void combine_track_plots(
	int & index,
	std::shared_ptr<CameraTracks> & camera_tracks,
	std::shared_ptr<mcmt::TrackPlot> & track_plot,
	int & frame_count)
{
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

/**
 * this function converts scalar value into rbg values
 */
std::vector<int> scalar_to_rgb(int & scalar_value, int & max_value)
{
	// create empty vector
	std::vector<int> output;

	float a = (1 - (scalar_value / max_value)) * 5;
	int x = int(floor(a));
	int y = int(floor(255 * (a - x)));

	if (x == 0) {
		output.push_back(255);
		output.push_back(y);
		output.push_back(0);
		return output;
	} else if (x == 1) {
		output.push_back(255);
		output.push_back(255);
		output.push_back(0);
		return output;
	} else if (x == 2) {
		output.push_back(0);
		output.push_back(255);
		output.push_back(y);
		return output;
	} else if (x == 3) {
		output.push_back(0);
		output.push_back(255);
		output.push_back(255);
		return output;
	} else if (x == 4) {
		output.push_back(y);
		output.push_back(0);
		output.push_back(255);
		return output;
	} else {	// x = 5
		output.push_back(255);
		output.push_back(0);
		output.push_back(255);
		return output;
	}
}