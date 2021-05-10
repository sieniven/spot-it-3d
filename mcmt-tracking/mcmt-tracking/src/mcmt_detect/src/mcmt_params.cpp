/** Params class
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the class for storing the MCMT parameters configured in mcmt_config.yaml/
 */

#include <mcmt_detect/mcmt_params.hpp>

using mcmt::McmtParams;

McmtParams::McmtParams(
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
	const float BACKGROUND_CONTOUR_CIRCULARITY)
{
	// initialize video parameters
	FRAME_WIDTH_ = FRAME_WIDTH;
	FRAME_HEIGHT_ = FRAME_HEIGHT;
	VIDEO_FPS_ = VIDEO_FPS;

	// initialize filter parameters
	VISIBILITY_RATIO_ = VISIBILITY_RATIO;
	VISIBILITY_THRESH_ = VISIBILITY_THRESH;
	CONSECUTIVE_THRESH_ = CONSECUTIVE_THRESH;
	AGE_THRESH_ = AGE_THRESH;
	SECONDARY_FILTER_ = SECONDARY_FILTER;
	SEC_FILTER_DELAY_ = SEC_FILTER_DELAY;

	// initialize background subtractor parameters
	FGBG_HISTORY_ = FGBG_HISTORY;
	BACKGROUND_RATIO_ = BACKGROUND_RATIO;
	NMIXTURES_ = NMIXTURES;
	BRIGHTNESS_GAIN_ = BRIGHTNESS_GAIN;
	FGBG_LEARNING_RATE_ = FGBG_LEARNING_RATE;
	DILATION_ITER_ = DILATION_ITER;
	REMOVE_GROUND_ITER_ = REMOVE_GROUND_ITER;
	BACKGROUND_CONTOUR_CIRCULARITY_ = BACKGROUND_CONTOUR_CIRCULARITY;
}