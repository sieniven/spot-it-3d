/**
 * @file mcmt_track_utils.hpp
 * @author your name (you@domain.com)
 * @author Niven Sie, sieniven@gmail.com
 * @author Seah Shao Xuan
 * 
 * This code contains the common functions and classes CameraTracks and TrackPlot
 * that are used in our main tracking and re-identifcation algorithm.
 */

#ifndef MCMT_TRACK_UTILS_HPP_
#define MCMT_TRACK_UTILS_HPP_

#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt
{
class TrackPlot {
	public:
		TrackPlot(int track_id);
		virtual ~TrackPlot() {}
		
		// define other_tracks
		typedef struct OtherTrack {
			float angle;
			float dist;
		} OtherTrack;

		// declare track information
		int id_, lastSeen_, mismatch_count_;
		std::vector<int> xs_, ys_, frameNos_, xyz_;
		std::vector<float> turning_angle_, curvature_, pace_, track_feature_variable_;
		std::vector<std::shared_ptr<OtherTrack>> other_tracks_;

		// declare methods
		void update(std::vector<int> & location, int & frame_no);
		void calculate_track_feature_variable(int & frame_no, int & fps);
		bool check_stationary();
};

class CameraTracks {
	public:
		CameraTracks(int index);
		virtual ~CameraTracks() {}

		// declare camera parameters
		int index_;

		// declare track plot variables
		std::map<int, std::shared_ptr<mcmt::TrackPlot>> track_plots_, track_new_plots_;
};

void update_other_tracks(TrackPlot & trackplot, std::shared_ptr<mcmt::CameraTracks> & cumulative_tracks);

void combine_track_plots(
	int & index,
	std::shared_ptr<mcmt::CameraTracks> & camera_tracks,
	std::shared_ptr<mcmt::TrackPlot> & track_plot,
	int & frame_count);

std::vector<int> scalar_to_rgb(int & scalar_value, int & max_value);
}

#endif	// MCMT_TRACK_UTILS_HPP_