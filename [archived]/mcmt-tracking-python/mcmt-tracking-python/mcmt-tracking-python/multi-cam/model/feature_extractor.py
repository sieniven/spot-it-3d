import cv2
import numpy as np
import csv
import sys

sys.path.append('../utility/')
from object_tracking_util import scalar_to_rgb, Track
from trackplots_util import TrackPlot
from automatic_brightness import average_brightness
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


FPS = 30.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SCALE_FACTOR = 0.8209970330862828
FIRSTPASS = False # Set to True if extracting IDs of a video for first time. False if using IDs to generate the CSV plots


def setup_system_objects():

    fgbg = cv2.createBackgroundSubtractorMOG2(history=int(5*FPS), varThreshold=4/SCALE_FACTOR,
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
        circularity = 4*np.pi*cv2.contourArea(contour)/(cv2.arcLength(contour, True)**2)
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
    masked = cv2.convertScaleAbs(masked, alpha=1, beta=256-average_brightness(16, frame, mask)+gain)

    masked = fgbg.apply(masked, learningRate=-1)

    masked = remove_ground(masked, int(13/(2.26/SCALE_FACTOR)), 0.6, frame)

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


def get_lost_tracks(tracks):
    invisible_for_too_long = FPS
    age_threshold = 0.5 * FPS

    tracks_to_be_removed = []

    for track in tracks:
        visibility = track.totalVisibleCount / track.age
        # A new created track with a low visibility is likely to have been generated by noise and is to be removed
        # Tracks that have not been seen for too long (The threshold determined by the reliability of the filter)
        # cannot be accurately located and are also be removed
        if (track.age < age_threshold and visibility < 0.8) \
                or track.consecutiveInvisibleCount >= invisible_for_too_long:
            tracks_to_be_removed.append(track)

    return tracks_to_be_removed


def delete_lost_tracks(tracks, tracks_to_be_removed):
    if len(tracks) == 0 or len(tracks_to_be_removed) == 0:
        return tracks

    tracks = [track for track in tracks if track not in tracks_to_be_removed]
    return tracks



def create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes):
    for unassignedDetection in unassigned_detections:
        detection_idx = unassignedDetection[1]
        centroid = centroids[detection_idx]
        size = sizes[detection_idx]

        dt = 1/FPS  # Time step between measurements in seconds

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
    min_track_age = max(2.0 * FPS, 30)    # seconds * FPS to give number of frames in seconds
    min_visible_count = max(2.0 * FPS, 30)

    good_tracks = []

    if len(tracks) != 0:
        for track in tracks:
            if track.age > min_track_age and track.totalVisibleCount > min_visible_count:
                centroid = track.kalmanFilter.x[:2]
                size = track.size

                good_tracks.append([track.id, track.age, size, (centroid[0], centroid[1])])

                centroid = track.kalmanFilter.x[:2] - origin

                # Display filtered tracks
                rect_top_left = (int(centroid[0] - size[0]/2), int(centroid[1] - size[1]/2))
                rect_bottom_right = (int(centroid[0] + size[0]/2), int(centroid[1] + size[1]/2))

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

    filename = '../../../data/vidtest3.mp4'

    cap = cv2.VideoCapture(filename)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    cap.set(5, FPS)

    if FIRSTPASS:
        recording = cv2.VideoWriter('home/niven/mcmt-tracking/data/feature_extractor_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    fgbg, detector = setup_system_objects()

    frame_count = 0

    nextID = 0
    tracks = []
    track_plots_id = []
    track_plots = []
    origin = np.array([0, 0])

    ret, frame = cap.read()
    if ret:
        print(f"Video Capture: PASS")
    else:
        print(f"Video Capture: FAIL")

    plot_history = 200
    colours = [''] * plot_history
    for i in range(plot_history):
        colours[i] = scalar_to_rgb(i, plot_history)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            mask = np.ones((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8) * 255
            centroids, sizes, masked = detect_objects(frame, mask, fgbg, detector, origin)

            predict_new_locations_of_tracks(tracks)

            assignments, unassigned_tracks, unassigned_detections \
                = detection_to_track_assignment(tracks, centroids, 10 * SCALE_FACTOR)

            update_assigned_tracks(assignments, tracks, centroids, sizes)

            update_unassigned_tracks(unassigned_tracks, tracks)
            tracks_to_be_removed = get_lost_tracks(tracks)
            tracks = delete_lost_tracks(tracks, tracks_to_be_removed)
            nextID = create_new_tracks(unassigned_detections, nextID, tracks, centroids, sizes)

            masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
            good_tracks = filter_tracks(frame, masked, tracks, origin)

            cv2.imshow(f"Masked", masked)
            time = frame_count / 30.0

            for track in good_tracks:
                track_id = track[0]
                centroid_x, centroid_y = track[3]

                if track_id not in track_plots_id:
                    track_plots_id.append(track_id)
                    track_plots.append(TrackPlot(track_id))

                track_plot = track_plots[track_plots_id.index(track_id)]
                track_plot.update((centroid_x, centroid_y), frame_count, time)
                track_plot.calculate_track_feature_variable(frame_count)

            for track_plot in track_plots:
                idxs = np.where(np.logical_and(track_plot.frameNos > frame_count - plot_history,
                                               track_plot.frameNos <= frame_count))[0]
                for idx in idxs:
                    cv2.circle(frame, (track_plot.xs[idx] - origin[0], track_plot.ys[idx] - origin[1]),
                               3, colours[track_plot.frameNos[idx] - frame_count + plot_history - 1][::-1], -1)
                if len(idxs) != 0:
                    cv2.putText(frame, f"ID: {track_plot.id}",
                                (track_plot.xs[idx] - origin[0], track_plot.ys[idx] - origin[1] + 15),
                                font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow(f"Plot", frame)
            if FIRSTPASS:
                recording.write(frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    if FIRSTPASS:
        recording.release()
        sys.exit() # Don't generate CSV files since we don't know the ID list yet

    ID_list = [340, 785, 1230]    # hardcoded to extract out only IDs that are tagged to identified drones in the video
    targets_output_log = []     # to grab data of all targets in ID_list
    junk_output_log = []    # to grab data of non targets
    trackList = []
    junkList = []

    for index, track_plot in enumerate(track_plots):
        if track_plot.id in ID_list:
            # get no. of rounds. minus 30 because last 30 frames are inaccurate. divide by checkpoints of 50.
            # no. of checkpoints = ((total_frameNos - 30) / 50) + 1 (from starting point of 0) - 3 (minus off size of 150 frames)
            checkpoints = int((track_plot.frameNos.size - 30) / 50) - 2
            for checkpoint in range(checkpoints):
                # dataList = batches of trackplot_ID, time-in, time-out, frame-in, frame-out,and 150 x track feature variables
                dataList = [track_plot.id, track_plot.times[(checkpoint * 50)],
                            track_plot.times[(checkpoint * 50) + 150], track_plot.frameNos[(checkpoint * 50)],
                            track_plot.frameNos[(checkpoint * 50) + 150],
                            track_plot.track_feature_variable[(checkpoint * 50):((checkpoint * 50) + 150)]]
                print(dataList)
                targets_output_log.append(dataList)

            trackList.append([track_plot.id, track_plot.frameNos[0], track_plot.frameNos[-1], track_plot.track_feature_variable])

        else:
            checkpoints = int((track_plot.frameNos.size - 30) / 50) - 2
            for checkpoint in range(checkpoints):
                # dataList = batches of trackplot_ID, time-in, time-out, frame-in, frame-out,and 150 x track feature variables
                dataList = [track_plot.id, track_plot.times[(checkpoint * 50)],
                            track_plot.times[(checkpoint * 50) + 150], track_plot.frameNos[(checkpoint * 50)],
                            track_plot.frameNos[(checkpoint * 50) + 150],
                            track_plot.track_feature_variable[(checkpoint * 50):((checkpoint * 50) + 150)]]
                junk_output_log.append(dataList)

            junkList.append([track_plot.id, track_plot.frameNos[0], track_plot.frameNos[-1], track_plot.track_feature_variable])

    with open(f"../data/track_data.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in trackList:
            writer.writerow(row)


    with open(f"../data/target_data.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in targets_output_log:
            writer.writerow(row)

    with open(f"../data/junk_data.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in junk_output_log:
            writer.writerow(row)

    cv2.destroyAllWindows()