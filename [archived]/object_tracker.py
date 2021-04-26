import cv2
import math
import numpy as np
from camera_stabilizer import stabilize_frame
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from automatic_brightness import threshold_rgb

import time


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


# Implementation of imopen from Matlab:
def imopen(im_in, kernel_size, iterations=1):
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)/(kernel_size**2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    im_out = cv2.morphologyEx(im_in, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return im_out


# Implementation of imclose from Matlab:
def imclose(im_in, kernel_size, iterations=1):
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)/(kernel_size**2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    im_out = cv2.morphologyEx(im_in, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return im_out


# Implementation of imfill() from Matlab
# Based off https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
# The idea is to add the inverse of the holes to fill the holes
def imfill(im_in):
    # Step 1: Threshold to obtain a binary image
    # Values above 220 to 0, below 220 to 255. (Inverse threshold)
    th, im_th = cv2.threshold(im_in, 220, 225, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image
    # im_floodfill = im_th.copy

    # Step 2: Floodfill the thresholded image
    # Mask used to flood fill
    # Note mask has to be 2 pixels larger than the input image
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    _, _, im_floodfill, _ = cv2.floodFill(im_th, mask, (0, 0), 125)
    cv2.imshow('floodfill', im_floodfill)

    # Step 3: Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Step 4: Combine the two images to get the foreground image with holes filled in
    # Floodfilled image needs to be trimmed to perform the bitwise or operation.
    # Trimming is done from the outside. I.e. the "Border" is removed
    # im_out = cv2.bitwise_or(im_th, im_floodfill_inv[1:-1, 1:-1])

    im_out = im_in

    return im_out


# Dilates the image multiple times to get of noise in order to get a single large contour for each background object
# Identify background objects by their shape (non-circular)
# Creates a copy of the input image which has the background contour filled in
# Returns the filled image which has the background elements filled in
def remove_ground(im_in, dilation_iterations, background_contour_circularity):
    kernel_dilation = np.ones((5, 5), np.uint8)
    # Number of iterations determines how close objects need to be to be considered background
    dilated = cv2.dilate(im_in, kernel_dilation, iterations=dilation_iterations)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    background_contours = []
    for contour in contours:
        # Identify background from foreground by the circularity of their dilated contours
        circularity = 4 * math.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
        if circularity <= background_contour_circularity:
            background_contours.append(contour)

    # This bit is used to find a suitable level of dilation to remove background objects
    # while keeping objects to be detected
    im_debug = cv2.cvtColor(dilated.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(im_debug, background_contours, -1, (0, 255, 0), 3)
    imshow_resized('original', im_in)
    imshow_resized('to_be_removed', im_debug)

    im_out = im_in.copy()
    cv2.drawContours(im_out, background_contours, -1, 0, -1)

    return im_out


def imshow_resized(window_name, img):
    window_size = (int(848), int(480))
    img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, img)


def motion_based_multi_object_tracking(filename):
    cap = cv2.VideoCapture(filename)

    global FPS, FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Resolution: {FRAME_WIDTH} by {FRAME_HEIGHT}")
    # The scaling factor is the ratio of the diagonal of the input frame
    # to the video used to test the parameters, which in this case is 848x480
    SCALE_FACTOR = math.sqrt(FRAME_WIDTH ** 2 + FRAME_HEIGHT ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
    print(f"Scaling Factor: {SCALE_FACTOR}")

    out_original = cv2.VideoWriter('out_original.mp4', cv2.VideoWriter_fourcc(*'h264'),
                                   FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    out_masked = cv2.VideoWriter('out_masked.mp4', cv2.VideoWriter_fourcc(*'h264'),
                                 FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    fgbg, detector = setup_system_objects()

    scene_transitioning = False

    tracks = []
    next_id = 0
    frame_count = 0

    centroids_log = []
    tracks_log = []

    scene_log = [[FPS, FRAME_WIDTH, FRAME_HEIGHT], []]

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            start_time = time.time()

            absolute_translation = 0
            # if frame_count == 0:
            #     frame_before = frame
            # elif frame_count >= 1:
            #     # Frame stabilization
            #     stabilized_frame, dx, dy = stabilize_frame(frame_before, frame)
            #
            #     absolute_translation = math.sqrt(dx**2 + dy**2)
            #     movement_threshold = 3
            #     if absolute_translation > movement_threshold:
            #         translating = True
            #     else:
            #         translating = False
            #
            #     if not scene_transitioning and translating:  # Start moving
            #         scene_transitioning = True
            #     elif scene_transitioning and not translating:  # Stop moving
            #         scene_transitioning = False
            #         scene_log.append([])
            #     else:  # scene_transitioning and translating implies no change in state to be made
            #         pass
            #
            #     frame_before = frame
            #     frame = stabilized_frame

            calibration_time = time.time()

            centroids, sizes, masked = detect_objects(frame, fgbg, detector)
            detection_time = time.time()
            centroids_log.append(centroids)

            predict_new_locations_of_tracks(tracks)
            prediction_time = time.time()

            assignments, unassigned_tracks, unassigned_detections \
                = detection_to_track_assignment(tracks, centroids, 20 * (1 + 0.1 * (absolute_translation)))
            assignment_time = time.time()

            update_assigned_tracks(assignments, tracks, centroids, sizes)
            update_time = time.time()
            tracks_log.append(tracks)

            update_unassigned_tracks(unassigned_tracks, tracks)
            update_unassigned_time = time.time()
            tracks = delete_lost_tracks(tracks)
            deletion_time = time.time()
            next_id = create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes)
            creation_time = time.time()

            masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
            good_tracks = filter_tracks(frame, masked, tracks, frame_count)
            display_time = time.time()

            if good_tracks:
                scene_log[-1].append(good_tracks)

            print(f"The frame took {(display_time - start_time) * 1000}ms in total.\n"
                  f"Camera stabilization took {(calibration_time - start_time) * 1000}ms.\n"
                  f"Object detection took {(detection_time - calibration_time) * 1000}ms.\n"
                  f"Prediction took {(prediction_time - detection_time) * 1000}ms.\n"
                  f"Assignment took {(assignment_time - prediction_time) * 1000}ms.\n"
                  f"Updating took {(update_time - assignment_time) * 1000}ms.\n"
                  f"Updating unassigned tracks took {(update_unassigned_time - update_time) * 1000}.\n"
                  f"Deletion took {(deletion_time - update_unassigned_time) * 1000}ms.\n"
                  f"Track creation took {(creation_time - deletion_time) * 1000}ms.\n"
                  f"Display took {(display_time - creation_time) * 1000}ms.\n\n")

            out_original.write(frame)
            out_masked.write(masked)

            imshow_resized('frame', frame)
            imshow_resized('masked', masked)

            print(f"FPS:{1 / (time.time() - start_time)}")

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    out_original.release()
    out_masked.release()
    cv2.destroyAllWindows()

    return scene_log


# Create VideoCapture object to extract frames from,
# background subtractor object and blob detector objects for object detection
# and VideoWriters for output videos
def setup_system_objects():
    # varThreshold affects the spottiness of the image. The lower it is, the more smaller spots.
    # The larger it is, these spots will combine into large foreground areas
    fgbg = cv2.createBackgroundSubtractorMOG2(history=int(15 * FPS), varThreshold=64 * SCALE_FACTOR,
                                              detectShadows=False)
    # Background ratio represents the fraction of the history a frame must be present
    # to be considered part of the background
    # eg. history is 5s, background ratio is 0.1, frames present for 0.5s will be considered background
    fgbg.setBackgroundRatio(0.05)
    fgbg.setNMixtures(5)

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
def detect_objects(frame, fgbg, detector):
    # Adjust contrast and brightness of image to make foreground stand out more
    # alpha used to adjust contrast, where alpha < 1 reduces contrast and alpha > 1 increases it
    # beta used to increase brightness, scale of (-255 to 255) ? Needs confirmation
    # formula is im_out = alpha * im_in + beta
    # Therefore to change brightness before contrast, we need to do alpha = 1 first
    masked = cv2.convertScaleAbs(frame, alpha=1, beta=0)
    masked = cv2.convertScaleAbs(masked, alpha=1, beta=190)
    # masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # masked = threshold_rgb(frame)

    # masked = cv2.GaussianBlur(masked, (5, 5), 0)

    # Subtract Background
    # Learning rate affects how often the model is updated
    # High values > 0.5 tend to lead to patchy output
    # Found that 0.1 - 0.3 is a good range
    masked = fgbg.apply(masked, learningRate=-1)
    imshow_resized("background subtracted", masked)

    masked = remove_ground(masked, int(13 / (2.26 / SCALE_FACTOR)), 0.7)

    # Morphological Transforms
    # Close to remove black spots
    # masked = imclose(masked, 3, 1)
    # Open to remove white holes
    # masked = imopen(masked, 3, 2)
    # masked = imfill(masked)
    kernel_dilation = np.ones((5, 5), np.uint8)
    masked = cv2.dilate(masked, kernel_dilation, iterations=2)

    # Invert frame such that black pixels are foreground
    masked = cv2.bitwise_not(masked)

    # keypoints = []
    # Blob detection
    keypoints = detector.detect(masked)

    n_keypoints = len(keypoints)
    centroids = np.zeros((n_keypoints, 2))
    sizes = np.zeros(n_keypoints)
    for i in range(n_keypoints):
        centroids[i] = keypoints[i].pt
        sizes[i] = keypoints[i].size

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


def filter_tracks(frame, masked, tracks, counter):
    # Actually, I feel having both might be redundant together with the deletion criteria
    min_track_age = 1.0 * FPS  # seconds * FPS to give number of frames in seconds
    # This has to be less than or equal to the minimum age or it make the minimum age redundant
    min_visible_count = 1.0 * FPS

    good_tracks = []

    if len(tracks) != 0:
        for track in tracks:
            if track.age > min_track_age and track.totalVisibleCount > min_visible_count:
                centroid = track.kalmanFilter.x[:2]
                size = track.size

                good_tracks.append([track.id, track.age, size, (centroid[0], centroid[1])])

                # Display filtered tracks
                rect_top_left = (int(centroid[0] - size / 2), int(centroid[1] - size / 2))
                rect_bottom_right = (int(centroid[0] + size / 2), int(centroid[1] + size / 2))
                colour = (0, 0, 255)
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
