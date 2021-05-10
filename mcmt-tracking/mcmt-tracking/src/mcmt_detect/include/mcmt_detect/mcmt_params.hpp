/** Params class
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the class for storing the MCMT parameters configured in mcmt_config.yaml/
 */

#ifndef MCMT_PARAMS_HPP_
#define MCMT_PARAMS_HPP_

#include <string>

namespace mcmt
{
class McmtParams {
	public:
		McmtParams(
			const int FRAME_WIDTH,
			const int FRAME_HEIGHT,
			const int VIDEO_FPS,
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
			const int DILATION_ITER,
			const float REMOVE_GROUND_ITER,
			const float BACKGROUND_CONTOUR_CIRCULARITY);
		
		// declare video parameters
		int FRAME_WIDTH_, FRAME_HEIGHT_, VIDEO_FPS_, MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES_;

		// declare filter parameters
		float VISIBILITY_RATIO_, VISIBILITY_THRESH_, CONSECUTIVE_THRESH_, AGE_THRESH_, SEC_FILTER_DELAY_;
		int SECONDARY_FILTER_;
		
		// declare background subtractor parameters
		int FGBG_HISTORY_, NMIXTURES_, BRIGHTNESS_GAIN_, DILATION_ITER_;
		float BACKGROUND_RATIO_, FGBG_LEARNING_RATE_, REMOVE_GROUND_ITER_, BACKGROUND_CONTOUR_CIRCULARITY_;
};
}

#endif			// MCMT_PARAMS_HPP_
