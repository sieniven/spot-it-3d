/** Params class
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the class for storing the MCMT parameters configured in mcmt_config.yaml/
 */

#include <mcmt_detect/mcmt_params.hpp>

using mcmt::McmtParams;

McmtParams::McmtParams(
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
	const int DILATION_ITER) 
{
	// initialize video parameters
	is_realtime_ = is_realtime;
	VIDEO_INPUT_0_ = VIDEO_INPUT_0;
	VIDEO_INPUT_1_ = VIDEO_INPUT_1;
	FILENAME_0_ = FILENAME_0;
	FILENAME_1_ = FILENAME_1;
	FRAME_WIDTH_ = FRAME_WIDTH;
	FRAME_HEIGHT_ = FRAME_HEIGHT;
	VIDEO_FPS_ = VIDEO_FPS;
	OUTPUT_ = OUTPUT;
	OUTPUT_FILE_ = OUTPUT_FILE;
	TRACK_CSV_ = TRACK_CSV;

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
}