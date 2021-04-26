import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

from filterpy.kalman import KalmanFilter
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from automatic_brightness import average_brightness
from object_tracking_rt import scalar_to_rgb
from reidentification_niven import re_identification


class Camera:
    def __init__(self, index):
        self.index = index
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(3, FRAME_WIDTH)
        self.cap.set(4, FRAME_HEIGHT)
        self.cap.set(5, FPS)
        self.tracks = []
        self.origin = np.array([0, 0])
        self.next_id = 1000

        self.fgbg = None

        self.output_log = []

        self.track_plots_ids = []
        self.track_new_plots_ids = []
        self.track_plots = []
        self.track_new_plots = []

    def init_fgbg(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=int(5 * FPS), varThreshold=256 / SCALE_FACTOR,
                                                       detectShadows=False)
        self.fgbg.setBackgroundRatio(0.05)
        self.fgbg.setNMixtures(5)


class Track:
    def __init__(self, track_id, size):
        self.id = track_id
        self.size = size
        # Constant Velocity Model
        self.kalmanFilter = KalmanFilter(dim_x=4, dim_z=2)
        # # Constant Acceleration Model
        # self.kalmanFilter = KalmanFilter(dim_x=6, dim_z=2)
        self.age = 1
        self.totalVisibleCount = 1
        self.consecutiveInvisibleCount = 0


def setup_system_objects():
    fgbg = cv2.createBackgroundSubtractorMOG2(history=int(5 * FPS), varThreshold=16 / SCALE_FACTOR,
                                              detectShadows=False)

    fgbg.setBackgroundRatio(0.05)
    fgbg.setNMixtures(5)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByConvexity = False
    params.filterByCircularity = False
    detector = cv2.SimpleBlobDetector_create(params)

    return fgbg, detector


def remove_ground(im_in, dilation_iterations, background_contour_circularity, frame):
    kernel_dilation = np.ones((5, 5), np.uint8)
    # Number of iterations determines how close objects need to be to be considered background
    dilated = cv2.dilate(im_in, kernel_dilation, iterations=dilation_iterations)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    background_contours = []
    for contour in contours:
        # Identify background from foreground by the circularity of their dilated contours
        circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
        if circularity <= background_contour_circularity:
            background_contours.append(contour)

    # This bit is used to find a suitable level of dilation to remove background objects
    # while keeping objects to be detected
    # im_debug = cv2.cvtColor(im_in.copy(), cv2.COLOR_GRAY2BGR)
    im_debug = frame.copy()
    cv2.drawContours(im_debug, background_contours, -1, (0, 255, 0), 3)

    im_out = im_in.copy()
    cv2.drawContours(im_out, background_contours, -1, 0, -1)

    return im_out


def detect_objects(frame, mask, fgbg, detector, origin):
    masked = cv2.convertScaleAbs(frame, alpha=1, beta=0)
    gain = 15
    masked = cv2.convertScaleAbs(masked, alpha=1, beta=256 - average_brightness(16, frame, mask) + gain)

    masked = fgbg.apply(masked, learningRate=-1)

    masked = remove_ground(masked, int(13 / (2.26 / SCALE_FACTOR)), 0.6, frame)

    kernel_dilation = np.ones((5, 5), np.uint8)
    masked = cv2.dilate(masked, kernel_dilation, iterations=2)

    masked = cv2.bitwise_not(masked)

    # keypoints = []
    # Blob detection
    keypoints = detector.detect(masked)

    n_keypoints = len(keypoints)
    centroids = np.zeros((n_keypoints, 2))
    sizes = np.zeros((n_keypoints, 2))
    for i in range(n_keypoints):
        centroids[i] = keypoints[i].pt
        centroids[i] += origin
        sizes[i] = keypoints[i].size

    return centroids, sizes, masked


def predict_new_locations_of_tracks(tracks):
    for track in tracks:
        track.kalmanFilter.predict()


def detection_to_track_assignment(tracks, centroids, cost_of_non_assignment):
    m, n = len(tracks), len(centroids)
    k, l = min(m, n), max(m, n)

    cost = np.zeros((k + l, k + l))

    for i in range(len(tracks)):
        track = tracks[i]
        track_location = track.kalmanFilter.x[:2]
        cost[i, :n] = np.array([distance.euclidean(track_location, centroid) for centroid in centroids])

    unassigned_track_cost = cost_of_non_assignment
    unassigned_detection_cost = cost_of_non_assignment

    extra_tracks = 0
    extra_detections = 0
    if m > n:  # More tracks than detections
        extra_tracks = m - n
    elif n > m:  # More detections than tracks
        extra_detections = n - m
    elif n == m:
        pass

    detection_padding = np.ones((m, m)) * unassigned_track_cost
    cost[:m, n:] = detection_padding

    track_padding = np.ones((n, n)) * unassigned_detection_cost
    cost[m:, :n] = track_padding

    row_ind, col_ind = linear_sum_assignment(cost)
    assignments_all = np.column_stack((row_ind, col_ind))

    assignments = assignments_all[(assignments_all < [m, n]).all(axis=1)]
    unassigned_tracks = assignments_all[
        (assignments_all >= [0, n]).all(axis=1) & (assignments_all < [m, k + l]).all(axis=1)]
    unassigned_detections = assignments_all[
        (assignments_all >= [m, 0]).all(axis=1) & (assignments_all < [k + l, n]).all(axis=1)]

    return assignments, unassigned_tracks, unassigned_detections


def update_assigned_tracks(assignments, tracks, centroids, sizes):
    for assignment in assignments:
        track_idx = assignment[0]
        detection_idx = assignment[1]
        centroid = centroids[detection_idx]
        size = sizes[detection_idx]

        track = tracks[track_idx]

        kf = track.kalmanFilter
        kf.update(centroid)

        track.size = size
        track.age += 1

        track.totalVisibleCount += 1
        track.consecutiveInvisibleCount = 0


def update_unassigned_tracks(unassigned_tracks, tracks):
    for unassignedTrack in unassigned_tracks:
        track_idx = unassignedTrack[0]

        track = tracks[track_idx]

        track.age += 1
        track.consecutiveInvisibleCount += 1


def delete_lost_tracks(tracks):
    if len(tracks) == 0:
        return tracks

    invisible_for_too_long = 2 * FPS
    age_threshold = 1 * FPS

    tracks_to_be_removed = []

    for track in tracks:
        visibility = track.totalVisibleCount / track.age

        if (track.age < age_threshold and visibility < 0.8) \
                or track.consecutiveInvisibleCount >= invisible_for_too_long:
            tracks_to_be_removed.append(track)

    tracks = [track for track in tracks if track not in tracks_to_be_removed]

    return tracks


def create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes):
    for unassignedDetection in unassigned_detections:
        detection_idx = unassignedDetection[1]
        centroid = centroids[detection_idx]
        size = sizes[detection_idx]

        dt = 1 / FPS  # Time step between measurements in seconds

        track = Track(next_id, size)

        # Constant velocity model
        # Initial Location
        track.kalmanFilter.x = [centroid[0], centroid[1], 0, 0]
        # State Transition Matrix
        track.kalmanFilter.F = np.array([[1., 0, 1, 0],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]])
        # Measurement Function
        track.kalmanFilter.H = np.array([[1., 0, 0, 0],
                                         [0, 1, 0, 0]])
        # Ah I really don't know what I'm doing here
        # Covariance Matrix
        track.kalmanFilter.P = np.diag([200., 200, 50, 50])
        # Motion Noise
        track.kalmanFilter.Q = np.diag([100., 100, 25, 25])
        # Measurement Noise
        track.kalmanFilter.R = 100
        # # Constant acceleration model

        tracks.append(track)

        next_id += 1

    return next_id


def filter_tracks(frame, masked, tracks, origin):
    min_track_age = max(1.0 * FPS, 30)  # seconds * FPS to give number of frames in seconds
    min_visible_count = max(1.0 * FPS, 30)

    good_tracks = []

    if len(tracks) != 0:
        for track in tracks:
            if track.age > min_track_age and track.totalVisibleCount > min_visible_count:
                centroid = track.kalmanFilter.x[:2]
                size = track.size

                # requirement for track to be considered in re-identification
                # note that no. of frames being too small may lead to loss of continuous tracking,
                # due to reidentification.py -> line 250
                if track.consecutiveInvisibleCount <= 15:
                    good_tracks.append([track.id, track.age, size, (centroid[0], centroid[1])])

                centroid = track.kalmanFilter.x[:2] - origin

                # Display filtered tracks
                rect_top_left = (int(centroid[0] - size[0] / 2), int(centroid[1] - size[1] / 2))
                rect_bottom_right = (int(centroid[0] + size[0] / 2), int(centroid[1] + size[1] / 2))

                colour = (0, 255, 0) if track.consecutiveInvisibleCount == 0 else (0, 0, 255)

                thickness = 1
                cv2.rectangle(frame, rect_top_left, rect_bottom_right, colour, thickness)
                cv2.rectangle(masked, rect_top_left, rect_bottom_right, colour, thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(frame, str(track.id), (rect_bottom_right[0], rect_top_left[1]),
                            font, font_scale, colour, thickness, cv2.LINE_AA)
                cv2.putText(masked, str(track.id), (rect_bottom_right[0], rect_top_left[1]),
                            font, font_scale, colour, thickness, cv2.LINE_AA)

    return good_tracks


if __name__ == '__main__':

    # f_x0 = 540
    # f_y0 = 540
    # c_x0 = 640/2
    # c_y0 = 480/2
    # K_0 = np.array([[f_x0, 0, c_x0],
    #                 [0, f_y0, c_y0],
    #                 [0, 0, 1]])
    #
    # R_c0 = np.array([[1, 0, 0],
    #                 [0, 1, 0],
    #                 [0, 0, 1]])
    # C_0 = np.array([[0, 0, 0]])
    # E_0 = np.concatenate((R_c0.T, -np.matmul(R_c0.T, C_0)), axis=1)
    # P_0 = np.matmul(K_0, E_0)
    #
    # f_x1 = 540
    # f_y1 = 540
    # c_x1 = 640/2
    # c_y1 = 480/2
    # K_1 = np.array([[f_x1, 0, c_x1],
    #                 [0, f_y1, c_y1],
    #                 [0, 0, 1]])
    # R_c1 = np.array([[1, 0, 0],
    #                 [0, 1, 0],
    #                 [0, 0, 1]])
    # C_1 = np.array([[1, 0, 0]])
    # E_1 = np.concatenate((R_c1.T, -np.matmul(R_c1.T, C_1)), axis=1)
    # P_1 = np.matmul(K_1, E_1)

    # Assume all webcams are 30 FPS and 640x480
    # Can be changed later for a more accurate set-up
    global FPS, FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR
    FPS = 30.0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    SCALE_FACTOR = 0.8209970330862828

    # add video_file name
    # Use camera_indices = [0, 1] to real time caption from 2 webcams
    # camera_indices = ['vidtest3.mp4', 'vidtest4.mp4']
    # camera_indices = ['00012_Trim_1.mp4', 'MVI_6690_Trim_1.mp4']
    # camera_indices = ['00012_Trim_2.mp4', 'MVI_6690_Trim_2.mp4']
    # camera_indices = ['00014_Trim.mp4', '']
    camera_indices = ['VID_20201110_120105_Trim.mp4', '00209_Trim.mp4']
    # camera_indices = [0, 1]

    # Sort through which indices are valid streams
    cameras = []

    for index in camera_indices:
        camera = Camera(index)

        ret, frame = camera.cap.read()
        if ret:
            cameras.append(camera)
            print(f"Video Capture {index}: PASS")
            camera.init_fgbg()
        else:
            print(f"Video Capture {index}: FAIL")
            camera.cap.release()

    _, detector = setup_system_objects()

    plot_history = 200
    colours = [''] * plot_history
    for i in range(plot_history):
        colours[i] = scalar_to_rgb(i, plot_history)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    master_tracks = []

    frame_count = 0
    nextID = 0
    recording = cv2.VideoWriter(f"recording.avi", cv2.VideoWriter_fourcc(*'MJPG'), FPS, (2 * FRAME_WIDTH, FRAME_HEIGHT))

    while True:
        for index, camera in enumerate(cameras):
            ret, frame = camera.cap.read()

            if ret:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            mask = np.ones((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8) * 255
            centroids_f, sizes_f, masked = detect_objects(frame, mask, camera.fgbg, detector, camera.origin)
            centroids = centroids_f
            sizes = sizes_f

            predict_new_locations_of_tracks(camera.tracks)

            assignments, unassigned_tracks, unassigned_detections \
                = detection_to_track_assignment(camera.tracks, centroids, 10 * SCALE_FACTOR)

            update_assigned_tracks(assignments, camera.tracks, centroids, sizes)

            update_unassigned_tracks(unassigned_tracks, camera.tracks)
            camera.tracks = delete_lost_tracks(camera.tracks)
            camera.next_id = create_new_tracks(unassigned_detections, camera.next_id, camera.tracks, centroids, sizes)

            masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
            good_tracks = filter_tracks(frame, masked, camera.tracks, camera.origin)

            cv2.imshow(f"Masked {index}", masked)

            camera.output_log.append([])

            # re-identification of camera targets
            removeList = []
            re_identification(removeList, index, nextID, frame, colours, font, font_scale, good_tracks, camera, cameras,
                              frame_count, plot_history, FPS)

            if index == 0:
                frame0 = np.copy(frame)
            else:
                if ret:
                    combine_frame = np.hstack((frame0, frame))
                    cv2.imshow(f"Original", combine_frame)
                    recording.write(combine_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fig, axs = plt.subplots(2, 1, figsize=(6, 9))

    for index, camera in enumerate(cameras):
        camera.cap.release()
        with open(f"data_out_{index}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in camera.output_log:
                writer.writerow(row)

        axs[0].set_yscale('log')
        axs[0].set_xlabel('Frame Number')
        axs[0].set_ylabel('Track Feature Variable Xj')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Frame Number')
        axs[0].set_ylabel('Track Feature Variable Xj')

        axs[1].set_yscale('log')
        axs[1].set_xlabel('Frame Number')
        axs[1].set_ylabel('Track Feature Variable Xj')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Frame Number')
        axs[1].set_ylabel('Track Feature Variable Xj')

        for track_plot in camera.track_plots:
            # Check that there's track feature variable data at all
            if track_plot.track_feature_variable.size != 0:
                feature_size = track_plot.track_feature_variable.size + 1
                axs[index].plot(track_plot.frameNos[:-feature_size:-1], track_plot.track_feature_variable)
                axs[index].text(track_plot.frameNos[-1], track_plot.track_feature_variable[-1], f"{track_plot.id}")

    fig.tight_layout()
    plt.show()