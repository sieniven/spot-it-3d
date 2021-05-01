/** Params class
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the class for storing the MCMT parameters configured in mcmt_config.yaml/
 */

#ifndef MCMT_PARAMS_HPP
#define MCMT_PARAMS_HPP

#include <string>

namespace mcmt
{
class McmtParams {
	public:
		McmtParams(
			bool is_realtime,
			const int VIDEO_INPUT_0,
			const int VIDEO_INPUT_1,
			std::string FILENAME_0,
			std::string FILENAME_1,
			const int FRAME_WIDTH,
			const int FRAME_HEIGHT,
			const int VIDEO_FPS,
			std::string OUTPUT,
			std::string OUTPUT_FILE,
			std::string TRACK_CSV,
			const int MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES,
			const float VISIBILITY_RATIO,
			const float VISIBILITY_THRESH,
			const float CONSECUTIVE_THRESH,
			const float AGE_THRESH,
			const int SECONDARY_FILTER,
			const float SEC_FILTER_DELAY,
			const int FGBG_HISTORY,
			const float BACKGROUND_RATIO,
			const int NMIXTURES,
			const int BRIGHTNESS_GAIN,
			const float FGBG_LEARNING_RATE,
			const int DILATION_ITER
			);
		
		// declare video parameters
		bool is_realtime_;
		int VIDEO_INPUT_0_, VIDEO_INPUT_1_, FRAME_WIDTH_, FRAME_HEIGHT_, VIDEO_FPS_, MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_;
		std::string FILENAME_0_, FILENAME_1_, OUTPUT_, OUTPUT_FILE_, TRACK_CSV_;

		// declare filter parameters
		float VISIBILITY_RATIO_, VISIBILITY_THRESH_, CONSECUTIVE_THRESH_, AGE_THRESH_, SEC_FILTER_DELAY_;
		int SECONDARY_FILTER_;
		
		// declare background subtractor parameters
		int FGBG_HISTORY_, NMIXTURES_, BRIGHTNESS_GAIN_, DILATION_ITER_;
		float BACKGROUND_RATIO_, FGBG_LEARNING_RATE_;
};
}