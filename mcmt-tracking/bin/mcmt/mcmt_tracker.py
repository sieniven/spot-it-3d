import multiprocessing
from detection_process import SingleCameraDetector, MultiCameraDetector
from trackplot_process import SingleCameraTrackPlot, MultiCameraTrackPlot


# define camera parameters
FPS = 30.0

# for single camera multi-target tracking
def single_camera_tracking(filename):
    """
    executes multi processes for single cam tracking
    """
    global FPS, FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR
    queue = multiprocessing.Queue()

    # initialize trackplot_process
    trackplot_process = SingleCameraTrackPlot(filename, queue, FPS)

    # initialize detection_process
    detection_process = SingleCameraDetector(filename, queue, FPS)

    detection_process.start()
    trackplot_process.start()
    detection_process.join()
    trackplot_process.join()


def multi_camera_tracking(filenames):
    """
    executes multi processes for multi cam tracking
    """
    global FPS, FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR
    queue = multiprocessing.Queue()

    # initialize trackplot_process
    trackplot_process = MultiCameraTrackPlot(filenames, queue, FPS)

    # initialize detection_process
    detection_process = MultiCameraDetector(filenames, queue, FPS)

    detection_process.start()
    trackplot_process.start()
    detection_process.join()
    trackplot_process.join()


if __name__ == "__main__":
    """
    execute mcmt software
    """
    # single camera test:
    # cameras = [0]
    # cameras = ['../../data/vidtest3.mp4']

    # multi cameras test:
    # cameras = [2, 4]
    # cameras = ['../../data/vidtest3.mp4', '../../../../data/vidtest4.mp4']
    # cameras = ['../../../../data/00012_Trim_1.mp4', '../../../../data/MVI_6690_Trim_1.mp4']
    # cameras = ['../../data/00012_Trim_2.mp4', '../../data/MVI_6690_Trim_2.mp4']
    # cameras = ['../../data/00014_Trim.mp4', '../../data/IMG_2059_HEVC.mp4']
    # cameras = ['../../data/VID_20201110_120105_Trim.mp4', '../../data/00209_Trim.mp4']

    if len(cameras) == 1:
        single_camera_tracking(cameras[0])

    else:
        multi_camera_tracking(cameras)
