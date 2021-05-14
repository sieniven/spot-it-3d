import cv2
import math
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# local imported codes
from automatic_brightness import average_brightness
import parameters as parm


class Camera:
    def __init__(self, index, fps):
        self.index = index
        self.cap = cv2.VideoCapture(self.index)
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
        self.aspect_ratio = self.frame_w / self.frame_h

        downsample = False
        if self.frame_w * self.frame_h > 1920 * 1080:
            downsample = True
            self.frame_w = 1920
            self.frame_h = int(1920 / aspect_ratio)
            self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
        
        self.fgbg, self.detector = setup_system_objects(self.scale_factor)
        self.tracks = []
        self.origin = np.array([0, 0])
        self.next_id = 1000
        self.dead_tracks = []


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
        self.goodtrack = False

        if parm.SECONDARY_FILTER == 1:
            self.tracker = cv2.TrackerKCF_create()
        elif parm.SECONDARY_FILTER == 2:
            self.tracker = cv2.TrackerCSRT_create()
        else:
            self.tracker = None

        self.box = np.zeros(4)
        self.outOfSync = False

# Dilates the image multiple times to get of noise in order to get a single large contour for each background object
# Identify background objects by their shape (non-circular)
# Creates a copy of the input image which has the background contour filled in
# Returns the filled image which has the background elements filled in
def imopen(im_in, kernel_size, iterations=1):
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)/(kernel_size**2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    im_out = cv2.morphologyEx(im_in, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return im_out


def scalar_to_rgb(scalar_value, max_value):
    f = scalar_value / max_value
    a = (1 - f) * 5
    x = math.floor(a)
    y = math.floor(255 * (a - x))
    if x == 0:
        return 255, y, 0
    elif x == 1:
        return 255, 255, 0
    elif x == 2:
        return 0, 255, y
    elif x == 3:
        return 0, 255, 255
    elif x == 4:
        return y, 0, 255
    else:  # x == 5:
        return 255, 0, 255

# Extract sky from the image. This is used when there is bright sunlight reflecting off the drone
# Hence a localised contrast increase must be applied to the sky to make the drone stand out
# Increasing contrast of the whole image will cause false positives in the background
# The sky is extracted by converting the image from RGB to HSV and applying thresholding operations
def filter_sky(frame):
    
    # Convert image from RGB to HSV
    masked = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image. A clear, sunlit sky has high V value (200 - 255)
    lower = np.array([0, 0, 200])
    upper = np.array([180, 255, 255])
    masked = cv2.inRange(masked, lower, upper)
    
    # Retrieve original RGB image with filtered sky using bitwise and
    masked = cv2.bitwise_and(frame, frame, mask=masked)

    return masked

def remove_ground(im_in, dilation_iterations, background_contour_circularity, frame, index):
    kernel_dilation = np.ones((5, 5), np.uint8)
    # Number of iterations determines how close objects need to be to be considered background
    dilated = cv2.dilate(im_in, kernel_dilation, iterations=dilation_iterations)

    # imshow_resized('dilated_' + str(index), dilated)

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
    # imshow_resized('Remove Ground' + str(index), im_debug)

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


# Create VideoCapture object to extract frames from,
# background subtractor object and blob detector objects for object detection
# and VideoWriters for output videos
def setup_system_objects(scale_factor):
    # Background subtractor works by subtracting the history from the current frame.
    # Further more this model already incldues guassian blur and morphological transformations
    # varThreshold affects the spottiness of the image. The lower it is, the more smaller spots.
    # The larger it is, these spots will combine into large foreground areas
    # fgbg = cv2.createBackgroundSubtractorMOG2(history=int(10*FPS), varThreshold=64*SCALE_FACTOR,
    #                                           detectShadows=False)
    # A lower varThreshold results in more noise which is beneficial to ground subtraction (but detrimental if you want
    # detections closer to the ground as there is more noise
    fgbg = cv2.createBackgroundSubtractorMOG2(history=int(parm.FGBG_HISTORY * parm.VIDEO_FPS), varThreshold= 4 / scale_factor,
                                              detectShadows=False)
    # Background ratio represents the fraction of the history a frame must be present
    # to be considered part of the background
    # eg. history is 5s, background ratio is 0.1, frames present for 0.5s will be considered background
    fgbg.setBackgroundRatio(parm.BACKGROUND_RATIO)
    fgbg.setNMixtures(parm.NMIXTURES)
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

# Adjust contrast and brightness of image to make foreground stand out more
# alpha used to adjust contrast, where alpha < 1 reduces contrast and alpha > 1 increases it
# beta used to increase brightness, scale of (-255 to 255) ? Needs confirmation
# formula is im_out = alpha * im_in + beta
# Therefore to change brightness before contrast, we need to do alpha = 1 first
def detect_objects(frame, mask, fgbg, detector, origin, index, scale_factor):
    if parm.SKY_FILTER:
        masked = filter_sky(frame)
        masked = cv2.convertScaleAbs(masked, alpha=1, beta=0)
    else:
        masked = cv2.convertScaleAbs(frame, alpha=1, beta=0)
    imshow_resized("pre-backhground subtraction", masked)
    masked = cv2.convertScaleAbs(masked, alpha=1, beta=256 - average_brightness(16, frame, mask) + parm.BRIGHTNESS_GAIN)
    # masked = cv2.convertScaleAbs(masked, alpha=2, beta=128)
    # masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # masked = threshold_rgb(frame)

    # Subtract Background
    # Learning rate affects how often the model is updated
    # High values > 0.5 tend to lead to patchy output
    # Found that 0.1 - 0.3 is a good range
    masked = fgbg.apply(masked, learningRate=parm.FGBG_LEARNING_RATE)
    masked = remove_ground(masked, int(13 / (2.26 / scale_factor)), 0.5, frame, index)
    imshow_resized("post-backhground subtraction", masked)

    # Morphological Transforms
    # Close to remove black spots
    # masked = imclose(masked, 3, 1)
    # Open to remove white holes
    # masked = imopen(masked, 3, 2)
    # masked = imfill(masked)
    kernel_dilation = np.ones((5, 5), np.uint8)
    masked = cv2.dilate(masked, kernel_dilation, iterations=parm.DILATION_ITER)

    # Apply foreground mask (dilated) to the image and perform detection on that
    # masked = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), masked)

    # Invert frame such that black pixels are foreground
    masked = cv2.bitwise_not(masked)

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


def detect_objects_large(frame, mask, fgbg, detector, origin, scale_factor):
    masked = cv2.convertScaleAbs(frame, alpha=1, beta=0)
    gain = 15
    masked = cv2.convertScaleAbs(masked, alpha=1, beta=256 - average_brightness(16, frame, mask) + gain)
    masked = fgbg.apply(masked, learningRate=-1)
    kernel = np.ones((5, 5), np.uint8)

    # Remove Noise
    masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=int(1))
    masked = cv2.dilate(masked, kernel, iterations=int(4 * scale_factor))
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


def predict_new_locations_of_tracks(tracks, frame, fps):
    for track in tracks:
        track.kalmanFilter.predict()

        if track.age >= max(1.0 * fps, 30) and track.tracker is not None:
            ok, box = track.tracker.update(frame)
            if ok:
                track.box = box
            #     # Tracking success
            #     p1 = (int(box[0]), int(box[1]))
            #     p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            #     cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            #     x = int(box[0] + box[2]*0.5)
            #     y = int(box[1] + box[3]*0.5)
            #     center = (x,y)
            #     # trajectory_x.append(x)
            #     # trajectory_y.append(y)
            #     cv2.circle(frame, center, 2, (255,0,0), -1)
            # else:
            #     cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
            # multiprocessing.Process(target=secondary_tracking, args=(track, frame))

            # if ok:
            #     # Tracking success
            #     p1 = (int(box[0]), int(box[1]))
            #     p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            #     cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            #     x = int(box[0] + box[2]*0.5)
            #     y = int(box[1] + box[3]*0.5)
            #     center = (x,y)
            #     # trajectory_x.append(x)
            #     # trajectory_y.append(y)
            #     cv2.circle(frame, center, 2, (255,0,0), -1)

            # if not ok:
            #     # Tracking failure
            #     cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)


# Assigns detections to tracks using Munkre's Algorithm with cost based on euclidean distance,
# with detections being located too far from existing tracks being designated as unassigned detections
# and tracks without any nearby detections being designated as unassigned tracks
def detection_to_track_assignment(tracks, centroids, cost_of_non_assignment):
    # start_time = time.time()
    m, n = len(tracks), len(centroids)
    k, l = min(m, n), max(m, n)

    # Create a square 2-D cost matrix with dimensions twice the size of the larger list (detections or tracks)
    cost = np.zeros((k + l, k + l))

    # Calculate the distance of every detection from each track,
    # filling up the rows of the cost matrix (up to column n, the number of detections) corresponding to existing tracks
    # This creates a m x n matrix
    for i in range(len(tracks)):
        # start_time_distance_loop = time.time()
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

    # Padding cost matrix with dummy columns to account for unassigned tracks
    # This is used to fill the top right corner of the cost matrix
    detection_padding = np.ones((m, m)) * unassigned_track_cost
    cost[:m, n:] = detection_padding

    # Padding cost matrix with dummy rows to account for unassigned detections
    # This is used to fill the bottom left corner of the cost matrix
    track_padding = np.ones((n, n)) * unassigned_detection_cost
    cost[m:, :n] = track_padding

    # The bottom right corner of the cost matrix, corresponding to dummy detections being matched to dummy tracks
    # is left with 0 cost to ensure that excess dummies are always matched to each other

    # Perform the assignment, returning the indices of assignments,
    # which are combined into a coordinate within the cost matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    assignments_all = np.column_stack((row_ind, col_ind))

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

    return assignments, unassigned_tracks, unassigned_detections


# Using the coordinates of valid assignments which correspond to the detection and track indices,
# update the track with the matched detection
def update_assigned_tracks(assignments, tracks, centroids, sizes, frame):
    for assignment in assignments:
        track_idx = assignment[0]
        detection_idx = assignment[1]
        centroid = centroids[detection_idx]
        size = sizes[detection_idx]
        track = tracks[track_idx]

        track.kalmanFilter.update(centroid)

        if track.tracker is not None:
            if track.age == max(parm.SEC_FILTER_DELAY * parm.VIDEO_FPS, 30) - 1:
                track.box = (centroid[0] - (size[0] / 2), centroid[1] - (size[1] / 2), size[0], size[1])
                track.tracker.init(frame, track.box)

            if track.age >= max(parm.SEC_FILTER_DELAY * parm.VIDEO_FPS, 30):
                track.outOfSync = (centroid[0] < track.box[0] - (1 * track.box[2]) or centroid[0] > track.box[0] + (2 * track.box[2])) \
                and (centroid[1] < track.box[1] - (1 * track.box[3]) or centroid[1] > track.box[1] + (2 * track.box[3]))
                # cv2.putText(frame, "Separation detected", (100,160), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

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
def get_lost_tracks(tracks):
    invisible_for_too_long = parm.CONSECUTIVE_THRESH * parm.VIDEO_FPS
    age_threshold = parm.AGE_THRESH * parm.VIDEO_FPS

    tracks_to_be_removed = []

    for track in tracks:
        visibility = track.totalVisibleCount / track.age
        # A new created track with a low visibility is likely to have been generated by noise and is to be removed
        # Tracks that have not been seen for too long (The threshold determined by the reliability of the filter)
        # cannot be accurately located and are also be removed
        if (track.age < age_threshold and visibility < parm.VISIBILITY_RATIO) \
                or track.consecutiveInvisibleCount >= invisible_for_too_long or track.outOfSync:
            tracks_to_be_removed.append(track)

    return tracks_to_be_removed


def delete_lost_tracks(tracks, tracks_to_be_removed):
    if len(tracks) == 0 or len(tracks_to_be_removed) == 0:
        return tracks

    tracks = [track for track in tracks if track not in tracks_to_be_removed]
    return tracks


# Detections not assigned an existing track are given their own track, initialized with the location of the detection
def create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes):
    for unassignedDetection in unassigned_detections:
        detection_idx = unassignedDetection[1]
        centroid = centroids[detection_idx]
        size = sizes[detection_idx]
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
        tracks.append(track)
        next_id += 1

    return next_id


def filter_tracks(frame, masked, tracks, origin):
    # Minimum number of frames to remove noise seems to be somewhere in the range of 30
    # Actually, I feel having both might be redundant together with the deletion criteria
    min_track_age = max(parm.AGE_THRESH * parm.VIDEO_FPS, 30)  # seconds * FPS to give number of frames in seconds
    # This has to be less than or equal to the minimum age or it make the minimum age redundant
    min_visible_count = max(parm.VISIBILITY_THRESH * parm.VIDEO_FPS, 30)

    good_tracks = []

    if len(tracks) != 0:
        for track in tracks:
            if track.age > min_track_age and track.totalVisibleCount > min_visible_count:
                centroid = track.kalmanFilter.x[:2]
                size = track.size

                # requirement for track to be considered in re-identification
                # note that no. of frames being too small may lead to loss of continuous tracking,
                # due to reidentification.py -> line 250
                if track.consecutiveInvisibleCount <= 5:
                    track.goodtrack = True
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
                # cv2.putText(frame, str(track.id), (rect_bottom_right[0], rect_top_left[1]),
                #             font, font_scale, colour, thickness, cv2.LINE_AA)
                # cv2.putText(masked, str(track.id), (rect_bottom_right[0], rect_top_left[1]),
                #             font, font_scale, colour, thickness, cv2.LINE_AA)

    return good_tracks, frame


# for single camera detection
def single_cam_detector(tracks, next_id, index, fgbg, detector, fps, frame_width, frame_height, scale_factor, origin, frame):
    mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
    centroids, sizes, masked = detect_objects(frame, mask, fgbg, detector, origin, index, scale_factor)

    predict_new_locations_of_tracks(tracks, frame, fps)

    assignments, unassigned_tracks, unassigned_detections \
        = detection_to_track_assignment(tracks, centroids, 10 * scale_factor)

    update_assigned_tracks(assignments, tracks, centroids, sizes, frame)

    update_unassigned_tracks(unassigned_tracks, tracks)
    tracks_to_be_removed = get_lost_tracks(tracks)
    tracks = delete_lost_tracks(tracks, tracks_to_be_removed)

    next_id = create_new_tracks(unassigned_detections, next_id, tracks, centroids, sizes)

    masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    good_tracks, frame = filter_tracks(frame, masked, tracks, origin)
    # cv2.imshow(f"Masked {index}", masked)

    return good_tracks, tracks, next_id, frame


# for multi camera detection
def multi_cam_detector(camera, frame):
    mask = np.ones((camera.frame_h, camera.frame_w), dtype=np.uint8) * 255
    centroids, sizes, masked = detect_objects(frame, mask, camera.fgbg, camera.detector, camera.origin, camera.index, camera.scale_factor)

    predict_new_locations_of_tracks(camera.tracks, frame, camera.fps)

    assignments, unassigned_tracks, unassigned_detections \
        = detection_to_track_assignment(camera.tracks, centroids, 10 * camera.scale_factor)

    update_assigned_tracks(assignments, camera.tracks, centroids, sizes, frame)

    update_unassigned_tracks(unassigned_tracks, camera.tracks)
    tracks_to_be_removed = get_lost_tracks(camera.tracks)
    camera.tracks = delete_lost_tracks(camera.tracks, tracks_to_be_removed)

    # list to keep track of dead tracks
    for gone_track in tracks_to_be_removed:
        if gone_track.goodtrack:
            camera.dead_tracks.append(gone_track.id)

    camera.next_id = create_new_tracks(unassigned_detections, camera.next_id, camera.tracks, centroids, sizes)

    masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    good_tracks, frame = filter_tracks(frame, masked, camera.tracks, camera.origin)
    # cv2.imshow(f"Masked {camera.index}", masked)

    return good_tracks, frame