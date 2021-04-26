import cv2
import math
import numpy as np
import time
import queue

from camera_stabilizer import stabilize_frame
from camera_stabilizer import Camera
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from automatic_brightness import average_brightness

from object_tracker import imopen


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


# Dilates the image multiple times to get of noise in order to get a single large contour for each background object
# Identify background objects by their shape (non-circular)
# Creates a copy of the input image which has the background contour filled in
# Returns the filled image which has the background elements filled in
def remove_ground(im_in, dilation_iterations, background_contour_circularity, frame):
    kernel_dilation = np.ones((5, 5), np.uint8)
    # Number of iterations determines how close objects need to be to be considered background
    dilated = cv2.dilate(im_in, kernel_dilation, iterations=dilation_iterations)

    imshow_resized('dilated', dilated)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    background_contours = []
    for contour in contours:
        # Identify background from foreground by the circularity of their dilated contours
        circularity = 4 * math.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
        if circularity <= background_contour_circularity:
            background_contours.append(contour)

    # This bit is used to find a suitable level of dilation to remove background objects
    # while keeping objects to be detected
    # im_debug = cv2.cvtColor(im_in.copy(), cv2.COLOR_GRAY2BGR)
    im_debug = frame.copy()
    cv2.drawContours(im_debug, background_contours, -1, (0, 255, 0), 3)
    imshow_resized('to_be_removed', im_debug)

    im_out = im_in.copy()
    cv2.drawContours(im_out, background_contours, -1, 0, -1)

    return im_out


def imshow_resized(window_name, img):
    aspect_ratio = img.shape[1] / img.shape[0]

    window_size = (int(600), int(600 / aspect_ratio))
    img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, img)


def downsample_image(img):
    aspect_ratio = img.shape[1] / img.shape[0]
    img_size = (int(1920), int(1920 / aspect_ratio))
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)

    return img


def track_objects_realtime(filename):
    if filename == 0:
        realtime = True
        print('Start Video Capture')
    else:
        realtime = False

    cap = cv2.VideoCapture(filename)

    global FPS, FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    SCALE_FACTOR = math.sqrt(FRAME_WIDTH ** 2 + FRAME_HEIGHT ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
    aspect_ratio = FRAME_WIDTH / FRAME_HEIGHT

    downsample = False
    if FRAME_WIDTH * FRAME_HEIGHT > 1920 * 1080:
        downsample = True
        FRAME_WIDTH = 1920
        FRAME_HEIGHT = int(1920 / aspect_ratio)
        SCALE_FACTOR = math.sqrt(FRAME_WIDTH ** 2 + FRAME_HEIGHT ** 2) / math.sqrt(848 ** 2 + 480 ** 2)

    # recording = cv2.VideoWriter('recording.mp4', cv2.VideoWriter_fourcc(*'h264'),
    #                             FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    out_combined = cv2.VideoWriter('out_real-time.mp4', cv2.VideoWriter_fourcc(*'h264'),
                                   FPS, (FRAME_WIDTH, FRAME_HEIGHT * 2))

    camera = Camera((FRAME_WIDTH, FRAME_HEIGHT))
    fgbg, detector = setup_system_objects()

    next_id = 0
    tracks = []

    frame = None
    frame_before = None

    frame_count = 0

    fps_log = []
    frame_start = time.time()

    origin = np.array([0, 0])

    consecutive_dropped_frames = 0
    max_tolerated_consecutive_dropped_frames = 5
    while cap.isOpened():
        if realtime:
            frame_end = time.time()
            frame_time = frame_end - frame_start
            if frame_time > 0.001:
                fps_log.append(frame_time)
                if len(fps_log) > 5:
                    FPS = 1 / (sum(fps_log) / len(fps_log))
                    fps_log.pop(0)

        ret, frame = cap.read()

        frame_start = time.time()

        if ret:
            print(frame_count)

            if downsample:
                frame = downsample_image(frame)

            # frame, mask = camera.undistort(frame)
            mask = np.ones((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8) * 255

            if frame_count == 0:
                frame_before = frame
            elif frame_count >= 1:
                # Frame stabilization
                stabilized_frame, dx, dy = stabilize_frame(frame_before, frame)
                origin[0] -= int(dx)
                origin[1] -= int(dy)

                frame_before = frame
                frame = stabilized_frame
            calibration_time = time.time()

            # centroids_f, sizes_f, masked = detect_objects(frame, mask, fgbg, detector, origin) # Detect the far & small objects
            # centroids = centroids_f
            # sizes = sizes_f

            centroids_n, sizes_n, masked = detect_objects_large(frame, mask, fgbg, detector,
                                                                origin)  # Detect the near & big objects
            # centroids = np.append(centroids, centroids_n)
            # sizes = np.append(sizes, sizes_n)
            centroids = centroids_n
            sizes = sizes_n

            detection_time = time.time()

        else:  # Failed to read file
            if consecutive_dropped_frames >= max_tolerated_consecutive_dropped_frames:
                break
            else:
                consecutive_dropped_frames += 1

                # There is no frame, so we make do with the previous frame for visualization
                # We still want it as a 3 channel image but in gray
                frame = cv2.cvtColor(cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                # Empty array as the masked image
                masked = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
                # Empty data for detections
                centroids = np.zeros((0, 2))
                sizes = np.zeros(0)

        predict_new_locations_of_tracks(tracks)
        prediction_time = time.time()

        assignments, unassigned_tracks, unassigned_detections \
            = detection_to_track_assignment(tracks, centroids, 10 * SCALE_FACTOR)
        assignment_time = time.time()

        update_assigned_tracks(assignments, tracks, centroids, sizes)

        update_unassigned_tracks(unassigned_tracks, tracks)
        tracks = delete_lost_tracks(tracks)
        next_id = create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes)

        return_frame = frame.copy()
        masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        good_tracks = filter_tracks(frame, masked, tracks, frame_count, origin)

        other_track_stuff = time.time()

        # recording.write(return_frame)

        frame_out = np.zeros((FRAME_HEIGHT * 2, FRAME_WIDTH, 3), dtype=np.uint8)
        frame_out[0:FRAME_HEIGHT, 0:FRAME_WIDTH] = frame
        frame_out[FRAME_HEIGHT:FRAME_HEIGHT * 2, 0:FRAME_WIDTH] = masked
        out_combined.write(frame_out)

        imshow_resized('frame', frame)
        imshow_resized('masked', masked)

        display_time = time.time()

        print(f"The frame took {(display_time - frame_start) * 1000}ms in total.\n"
              f"Camera stabilization took {(calibration_time - frame_start) * 1000}ms.\n"
              f"Object detection took {(detection_time - calibration_time) * 1000}ms.\n"
              f"Prediction took {(prediction_time - detection_time) * 1000}ms.\n"
              f"Assignment took {(assignment_time - prediction_time) * 1000}ms.\n"
              f"Other track stuff took {(other_track_stuff - assignment_time) * 1000}ms.\n"
              f"Writing to file took {(display_time - other_track_stuff) * 1000}ms.\n\n")

        frame_count += 1

        if not realtime:
            frame_start = False

        yield good_tracks, origin, frame_count, return_frame, frame_start

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # recording.release()
    out_combined.release()
    cv2.destroyAllWindows()


# Create VideoCapture object to extract frames from,
# background subtractor object and blob detector objects for object detection
# and VideoWriters for output videos
def setup_system_objects():
    # Background subtractor works by subtracting the history from the current frame.
    # Further more this model already incldues guassian blur and morphological transformations
    # varThreshold affects the spottiness of the image. The lower it is, the more smaller spots.
    # The larger it is, these spots will combine into large foreground areas
    # fgbg = cv2.createBackgroundSubtractorMOG2(history=int(10*FPS), varThreshold=64*SCALE_FACTOR,
    #                                           detectShadows=False)
    # A lower varThreshold results in more noise which is beneficial to ground subtraction (but detrimental if you want
    # detections closer to the ground as there is more noise
    fgbg = cv2.createBackgroundSubtractorMOG2(history=int(5 * FPS), varThreshold=16 / SCALE_FACTOR,
                                              detectShadows=False)
    # Background ratio represents the fraction of the history a frame must be present
    # to be considered part of the background
    # eg. history is 5s, background ratio is 0.1, frames present for 0.5s will be considered background
    fgbg.setBackgroundRatio(0.05)
    fgbg.setNMixtures(5)

    # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.minArea = 1
    # params.maxArea = 1000
    params.filterByConvexity = False
    params.filterByCircularity = False
    detector = cv2.SimpleBlobDetector_create(params)

    return fgbg, detector


# Apply image masks to prepare frame for blob detection
# Masks: 1) Increased contrast and brightness to fade out the sky and make objects stand out
#        2) Background subtractor to remove the stationary background (Converts frame to a binary image)
#        3) Further background subtraction by means of contouring around non-circular objects
#        4) Dilation to fill holes in detected drones
#        5) Inversion to make the foreground black for the blob detector to identify foreground objects
# Perform the blob detection on the masked image
# Return detected blob centroids as well as size
def detect_objects(frame, mask, fgbg, detector, origin):
    # Adjust contrast and brightness of image to make foreground stand out more
    # alpha used to adjust contrast, where alpha < 1 reduces contrast and alpha > 1 increases it
    # beta used to increase brightness, scale of (-255 to 255) ? Needs confirmation
    # formula is im_out = alpha * im_in + beta
    # Therefore to change brightness before contrast, we need to do alpha = 1 first
    masked = cv2.convertScaleAbs(frame, alpha=1, beta=0)
    gain = 15
    masked = cv2.convertScaleAbs(masked, alpha=1, beta=256 - average_brightness(16, frame, mask) + gain)
    # masked = cv2.convertScaleAbs(masked, alpha=2, beta=128)
    # masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # masked = threshold_rgb(frame)

    imshow_resized("pre-background subtraction", masked)

    # Subtract Background
    # Learning rate affects how often the model is updated
    # High values > 0.5 tend to lead to patchy output
    # Found that 0.1 - 0.3 is a good range
    masked = fgbg.apply(masked, learningRate=-1)

    imshow_resized("background subtracted", masked)

    masked = remove_ground(masked, int(13 / (2.26 / SCALE_FACTOR)), 0.6, frame)

    # Morphological Transforms
    # Close to remove black spots
    # masked = imclose(masked, 3, 1)
    # Open to remove white holes
    # masked = imopen(masked, 3, 2)
    # masked = imfill(masked)
    kernel_dilation = np.ones((5, 5), np.uint8)
    masked = cv2.dilate(masked, kernel_dilation, iterations=2)

    # Apply foreground mask (dilated) to the image and perform detection on that
    # masked = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), masked)

    # Invert frame such that black pixels are foreground
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


def detect_objects_large(frame, mask, fgbg, detector, origin):
    masked = cv2.convertScaleAbs(frame, alpha=1, beta=0)
    gain = 15
    masked = cv2.convertScaleAbs(masked, alpha=1, beta=256 - average_brightness(16, frame, mask) + gain)

    masked = fgbg.apply(masked, learningRate=-1)

    kernel = np.ones((5, 5), np.uint8)
    # Remove Noise
    masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=int(1))

    masked = cv2.dilate(masked, kernel, iterations=int(4 * SCALE_FACTOR))

    contours, hierarchy = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_keypoints = len(contours)
    centroids = np.zeros((n_keypoints, 2))
    sizes = np.zeros((n_keypoints, 2))
    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        centroids[i] = [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]
        centroids[i] += origin
        x, y, w, h = cv2.boundingRect(contour)
        sizes[i] = (w, h)

    return centroids, sizes, masked


def predict_new_locations_of_tracks(tracks):
    for track in tracks:
        track.kalmanFilter.predict()


# Assigns detections to tracks using Munkre's Algorithm with cost based on euclidean distance,
# with detections being located too far from existing tracks being designated as unassigned detections
# and tracks without any nearby detections being designated as unassigned tracks
def detection_to_track_assignment(tracks, centroids, cost_of_non_assignment):
    # start_time = time.time()
    m, n = len(tracks), len(centroids)
    k, l = min(m, n), max(m, n)

    # Create a square 2-D cost matrix with dimensions twice the size of the larger list (detections or tracks)
    cost = np.zeros((k + l, k + l))
    # initialization_time = time.time()

    # Calculate the distance of every detection from each track,
    # filling up the rows of the cost matrix (up to column n, the number of detections) corresponding to existing tracks
    # This creates a m x n matrix
    for i in range(len(tracks)):
        start_time_distance_loop = time.time()
        track = tracks[i]
        track_location = track.kalmanFilter.x[:2]
        cost[i, :n] = np.array([distance.euclidean(track_location, centroid) for centroid in centroids])
    # distance_time = time.time()

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

    # Padding cost matrix with dummy columns to account for unassigned tracks
    # This is used to fill the top right corner of the cost matrix
    detection_padding = np.ones((m, m)) * unassigned_track_cost
    cost[:m, n:] = detection_padding

    # Padding cost matrix with dummy rows to account for unassigned detections
    # This is used to fill the bottom left corner of the cost matrix
    track_padding = np.ones((n, n)) * unassigned_detection_cost
    cost[m:, :n] = track_padding
    # padding_time = time.time()

    # The bottom right corner of the cost matrix, corresponding to dummy detections being matched to dummy tracks
    # is left with 0 cost to ensure that excess dummies are always matched to each other

    # Perform the assignment, returning the indices of assignments,
    # which are combined into a coordinate within the cost matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    assignments_all = np.column_stack((row_ind, col_ind))
    # assignment_time = time.time()

    # Assignments within the top left corner corresponding to existing tracks and detections
    # are designated as (valid) assignments
    assignments = assignments_all[(assignments_all < [m, n]).all(axis=1)]
    # Assignments within the top right corner corresponding to existing tracks matched with dummy detections
    # are designated as unassigned tracks and will later be regarded as invisible
    unassigned_tracks = assignments_all[
        (assignments_all >= [0, n]).all(axis=1) & (assignments_all < [m, k + l]).all(axis=1)]
    # Assignments within the bottom left corner corresponding to detections matched to dummy tracks
    # are designated as unassigned detections and will generate a new track
    unassigned_detections = assignments_all[
        (assignments_all >= [m, 0]).all(axis=1) & (assignments_all < [k + l, n]).all(axis=1)]
    # sorting_time = time.time()

    # print(f"Initialization took {initialization_time - start_time}ms.\n"
    #       f"Distance measuring took {distance_time - initialization_time}ms.\n"
    #       f"Padding took {padding_time - distance_time}ms.\n"
    #       f"Assignment took {assignment_time - padding_time}ms.\n"
    #       f"Sorting took {sorting_time - assignment_time}\n\n")

    return assignments, unassigned_tracks, unassigned_detections


# Using the coordinates of valid assignments which correspond to the detection and track indices,
# update the track with the matched detection
def update_assigned_tracks(assignments, tracks, centroids, sizes):
    for assignment in assignments:
        track_idx = assignment[0]
        detection_idx = assignment[1]
        centroid = centroids[detection_idx]
        size = sizes[detection_idx]

        track = tracks[track_idx]

        kf = track.kalmanFilter
        kf.update(centroid)

        # # Adaptive filtering
        # # If the residual is too large, increase the process noise
        # Q_scale_factor = 100.
        # y, S = kf.y, kf.S  # Residual and Measurement covariance
        # # Square and normalize the residual
        # eps = np.dot(y.T, np.linalg.inv(S)).dot(y)
        # kf.Q *= eps * 10.

        track.size = size
        track.age += 1

        track.totalVisibleCount += 1
        track.consecutiveInvisibleCount = 0


# Existing tracks without a matching detection are aged and considered invisible for the frame
def update_unassigned_tracks(unassigned_tracks, tracks):
    for unassignedTrack in unassigned_tracks:
        track_idx = unassignedTrack[0]

        track = tracks[track_idx]

        track.age += 1
        track.consecutiveInvisibleCount += 1


# If any track has been invisible for too long, or generated by a flash, it will be removed from the list of tracks
def delete_lost_tracks(tracks):
    if len(tracks) == 0:
        return tracks

    invisible_for_too_long = 3 * FPS
    age_threshold = 1 * FPS

    tracks_to_be_removed = []

    for track in tracks:
        visibility = track.totalVisibleCount / track.age
        # A new created track with a low visibility is likely to have been generated by noise and is to be removed
        # Tracks that have not been seen for too long (The threshold determined by the reliability of the filter)
        # cannot be accurately located and are also be removed
        if (track.age < age_threshold and visibility < 0.8) \
                or track.consecutiveInvisibleCount >= invisible_for_too_long:
            tracks_to_be_removed.append(track)

    tracks = [track for track in tracks if track not in tracks_to_be_removed]

    return tracks


# Detections not assigned an existing track are given their own track, initialized with the location of the detection
def create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes):
    for unassignedDetection in unassigned_detections:
        detection_idx = unassignedDetection[1]
        centroid = centroids[detection_idx]
        size = sizes[detection_idx]

        dt = 1 / FPS  # Time step between measurements in seconds

        track = Track(next_id, size)

        # Attempted tuning
        # # Constant velocity model
        # # Initial Location
        # track.kalmanFilter.x = [centroid[0], centroid[1], 0, 0]
        # # State Transition Matrix
        # track.kalmanFilter.F = np.array([[1., 0, dt, 0],
        #                                  [0, 1, 0, dt],
        #                                  [0, 0, 1, 0],
        #                                  [0, 0, 0, 1]])
        # # Measurement Function
        # track.kalmanFilter.H = np.array([[1., 0, 0, 0],
        #                                  [0, 1, 0, 0]])
        # # Covariance Matrix
        # track.kalmanFilter.P = np.diag([(10.*SCALE_FACTOR)**2, (10.*SCALE_FACTOR)**2,  # Positional variance
        #                                 (7*SCALE_FACTOR)**2, (7*SCALE_FACTOR)**2])  # Velocity variance
        # # Process Noise
        # # Assumes that the process noise is white
        # track.kalmanFilter.Q = Q_discrete_white_noise(dim=4, dt=dt, var=1000)
        # # Measurement Noise
        # track.kalmanFilter.R = np.diag([10.**2, 10**2])

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


def filter_tracks(frame, masked, tracks, counter, origin):
    # Minimum number of frames to remove noise seems to be somewhere in the range of 30
    # Actually, I feel having both might be redundant together with the deletion criteria
    min_track_age = max(1.0 * FPS, 30)  # seconds * FPS to give number of frames in seconds
    # This has to be less than or equal to the minimum age or it make the minimum age redundant
    min_visible_count = max(1.0 * FPS, 30)

    good_tracks = []

    if len(tracks) != 0:
        for track in tracks:
            if track.age > min_track_age and track.totalVisibleCount > min_visible_count:
                centroid = track.kalmanFilter.x[:2]
                size = track.size

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
