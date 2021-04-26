'''
VIDEO PARAMETERS
'''

# STRING
# First video input. Specify filename or integer value for livefeed.
VIDEO_INPUT_1 = 0

# STRING
# Second video input. Specify filename or integer value for livefeed.
VIDEO_INPUT_2 = 2

# INT
# Target frame width of video input. Will default to a preset value if not possible.
FRAME_WIDTH = 1920

# INT
# Target frame width of video input. Will default to a preset value if not possible.
FRAME_HEIGHT = 1080

# INT
# Target frame width of video input. Will default to a preset value if not possible.
VIDEO_FPS = 30

# STRING
# File path to output feed.
OUTPUT = '../data/output.mp4'

# STRING
# File path to annotated output feed.
OUTPUT_FILE = '../data/output_file.mp4'

# STRING
# File path to export CSV track data.
TRACK_CSV = '../data/plot.csv'

# INT
# Maximum number of failed consecutive read frames.
MAX_TOLERATED_CONSECUTIVE_DROPPED_FRAMES = 5


'''
FILTER PARAMETERS
'''

# DOUBLE
# Visibility ratio. Number of frames track is present / total track age.
VISIBILITY_RATIO = 0.8

# DOUBLE
# Minimum visibility before considered a good track in seconds.
VISIBILITY_THRESH = 1.0

# DOUBLE
# Maximum consecutive invisible count in seconds. Track is deleted after that.
CONSECUTIVE_THRESH = 1.0

# DOUBLE
# Minimum age to be exempt from visibility threshold in seconds.
AGE_THRESH = 1.0

# INT
# Secondary filter type. 0: No Secondary Filter. 1: Kernalised Correlation Filter. 2: Discriminative Correlation Filter.
SECONDARY_FILTER = 2

# DOUBLE
# Delay before secondary filter is initiated in seconds.
SEC_FILTER_DELAY = 1.0


'''
BACKGROUND SUBTRACTOR PARAMETERS
'''

# INT
# History to be subtracted by background subtractor in seconds.
FGBG_HISTORY = 5

# DOUBLE
# Fraction of history a frame must be present
BACKGROUND_RATIO = 0.05

# INT
# Sets the number of gaussian components in the background model.
NMIXTURES = 5

# INT
# Increases brightness by a certain gain.
BRIGHTNESS_GAIN = 15

# DOUBLE
# Learning rate affects how often the model is updated.
FGBG_LEARNING_RATE = 0.2

# INT
# Number of iterations of pixel dilation.
DILATION_ITER = 2