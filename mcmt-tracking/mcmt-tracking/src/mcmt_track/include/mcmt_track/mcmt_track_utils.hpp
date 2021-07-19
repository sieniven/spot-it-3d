/**
 * @file mcmt_track_utils.hpp
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
 * This file contains the declaration of the classes (CameraTracks and TrackPlots)
 * and their associated methods, which is primarily used in the re-identification
 * pipeline. The full definition of the classes and their methods can be found 
 * in the file mcmt_track_utils.cpp. 
 */

#ifndef MCMT_TRACK_UTILS_HPP_
#define MCMT_TRACK_UTILS_HPP_

// standard package imports
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

using namespace std;

namespace mcmt {

	/**
	 * This class is for storing our tracks' information, calculating track feature
	 * variable, and caluclating geometrical angles and distances of the tracks.
	 */	
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
			vector<int> xs_, ys_, size_, frameNos_;
			vector<float> xyz_, turning_angle_, curvature_, pace_, track_feature_variable_;
			vector<shared_ptr<OtherTrack>> other_tracks_;

			// declare methods
			void update(vector<int> & location, int & size, int & frame_no);
			void calculate_track_feature_variable(int & frame_no, int & fps);
			bool check_stationary();
	};

	/**
	 * This class is for tracking the camera's track plots.
	 */
	class CameraTracks {
		public:
			CameraTracks(int index);
			virtual ~CameraTracks() {}

			// declare camera parameters
			int index_;

			// declare track plot variables
			map<int, shared_ptr<TrackPlot>> track_plots_, track_new_plots_;
	};

	void update_other_tracks(shared_ptr<TrackPlot> trackplot, 
		shared_ptr<CameraTracks> & cumulative_track);

	void combine_track_plots(
		int & index,
		shared_ptr<CameraTracks> camera_tracks,
		shared_ptr<TrackPlot> track_plot,
		int & frame_count);

}

#endif			// MCMT_TRACK_UTILS_HPP_