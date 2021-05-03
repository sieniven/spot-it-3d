import cv2
import math
import numpy as np
import json
import time


class Camera:
    def __init__(self, frame_shape):
        # Generic matrices, perform calibration if possible
        self.cameraMatrix = np.array([[600, 0., frame_shape[0] / 2],
                                      [0., 600, frame_shape[1] / 2],
                                      [0., 0., 1.]])
        self.distortionCoefficients = np.array([-0.05,
                                                0.05,
                                                0.,
                                                0.,
                                                -0.002])

    # This function assumes a regular 7 x 6 chessboard pattern is used for the calibration
    def calibrate_camera(self, calibration_images):
        # Termination criteria for cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points
        # This creates a 2-D set of regularly spaced points positioned in 3-D such that z=0
        # i.e. it assumes that the calibration pattern is regular and flat
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points of object in real world space (Its going to all be the same since we assume the same
        # object is used for the calibration)
        imgpoints = []  # 2d points in image plane

        for image in calibration_images:
            frame = cv2.imread(image)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find chessboard
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            if ret:
                objpoints.append(objp)

                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners_refined)

        ret, self.cameraMatrix, self.distortionCoefficients, _, _ = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        self.export_calibration_results()

    def import_calibration_results(self, filename):
        with open(filename) as file:
            data = json.loads(file.read())
            self.cameraMatrix = np.array(data['camera_matrix'])
            self.distortionCoefficients = np.array(data['distortion_coefficients'])
        print(f"Camera matrix: \n {np.array(data['camera_matrix'])}\n"
              f"Distortion Coefficients: \n {np.array(data['distortion_coefficients'])}")

    def export_calibration_results(self):
        with open('camera_parameters.txt', 'w+') as file:
            file.write(json.dumps({"camera_matrix": self.cameraMatrix.tolist(),
                                   "distortion_coefficients": self.distortionCoefficients.tolist()}))

    def undistort(self, frame):
        # getOptimalNewCameraMatrix()

        frame_undistorted = cv2.undistort(frame, self.cameraMatrix, self.distortionCoefficients)

        # Create a mask that covers the remaining bit of the frame with the image
        mask = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, -1)
        # # Erode the mask a bit to make sure we get rid of the dark border
        # mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
        # # Turn the imageless background to white
        # frame_undistorted_white_borders = cv2.bitwise_or(frame_undistorted,
        #                                                   cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR))

        return frame_undistorted, mask


def imshow_resized(window_name, img):
    window_size = (int(848), int(480))
    img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, img)


def stabilize_frame(img1_in, img2_in):
    img1_gray = cv2.cvtColor(img1_in, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_in, cv2.COLOR_BGR2GRAY)

    image_points1 = cv2.goodFeaturesToTrack(img1_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    image_points2, status, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, image_points1, None)

    # Filtering bad points
    idx = np.where(status == 1)[0]
    image_points1 = image_points1[idx]
    image_points2 = image_points2[idx]

    # Estimate affine transformation
    # Produces a transformation matrix :
    # [[cos(theta).s, -sin(theta).s, tx],
    # [sin(theta).s, cos(theta).s, ty]]
    # where theta is rotation, s is scaling and tx,ty are translation
    m = cv2.estimateAffinePartial2D(image_points1, image_points2)[0]

    # im = cv2.invertAffineTransform(m)

    # Extract translation
    dx = m[0, 2]
    dy = m[1, 2]
    # print(f"dx={dx}, dy={dy}")
    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])

    rows, columns = img2_gray.shape
    frame_stabilized = cv2.warpAffine(img2_in, m, (columns, rows))

    return frame_stabilized, dx, dy


def find_global_size(filename):
    cap = cv2.VideoCapture(filename)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    global_height, global_width = frame_height, frame_width
    origin = np.array([0, 0])
    a = 0
    top_left, bottom_right = np.array([0, 0]), np.array([frame_width, frame_height])

    camera = Camera([frame_width, frame_height])

    frame_count = 0

    transformations = []

    while cap.isOpened():
        ret, frame = cap.read()

        transformations.append([])

        if ret:
            # imshow_resized('original', frame)

            # frame, mask = camera.undistort(frame)
            mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255

            # imshow_resized('corrected', frame)
            if frame_count == 0:
                frame_before = frame
            elif frame_count >= 1:
                image1 = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
                image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                image_points1 = cv2.goodFeaturesToTrack(image1, maxCorners=300, qualityLevel=0.01, minDistance=10,
                                                        mask=mask)

                image_points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, image_points1, None)

                idx = np.where(status == 1)[0]
                image_points1 = image_points1[idx]
                image_points2 = image_points2[idx]

                # Estimate affine transformation
                # Produces a transformation matrix :
                # [[cos(theta).s, -sin(theta).s, tx],
                # [sin(theta).s, cos(theta).s, ty]]
                # where theta is rotation, s is scaling and tx,ty are translation
                m = cv2.estimateAffinePartial2D(image_points1, image_points2)[0]

                # Extract translation
                # Camera frame moving left and up is positive
                dx = int(m[0, 2])
                dy = int(m[1, 2])
                # Extract rotation angle
                da = np.arctan2(m[1, 0], m[0, 0])
                a += math.degrees(da)
                a %= 360
                # print(f"rotation: {a}")

                rows, columns = image1.shape

                rot_mat = cv2.getRotationMatrix2D((columns / 2, rows / 2), a, 1)

                transformations[-1] = (m, rot_mat)

                # print(f"dx:{dx}, dy: {dy}")

                top_left[0] -= dx
                bottom_right[0] -= dx
                top_left[1] -= dy
                bottom_right[1] -= dy

                # rounding must come before fabs to prevent rounding up vs rounding down if performed on a negative
                # float vs a positive float
                # I think
                dx_abs = abs(dx)
                dy_abs = abs(dy)
                if top_left[0] < 0:
                    global_width += dx_abs
                    origin[0] += dx_abs
                    bottom_right[0] -= top_left[0]
                    top_left[0] = 0
                elif bottom_right[0] > global_width:
                    global_width += dx_abs
                else:
                    pass
                if top_left[1] < 0:
                    global_height += dy_abs
                    origin[1] += dy_abs
                    bottom_right[1] -= top_left[1]
                    top_left[1] = 0
                elif bottom_right[1] > global_height:
                    global_height += dy_abs
                else:
                    pass

                if not ((bottom_right - top_left) == frame.shape[:2][::-1]).all():
                    print(f"Top left: {top_left}, Bottom right: {bottom_right}")
                    print(f"Error: frame shape is {(bottom_right - top_left)}")

                # Note: warp affine rotates about top left
                # cv2.imshow('original', frame)
                frame_stabilized = cv2.warpAffine(frame, m, (columns, rows))
                # cv2.imshow('frame', frame)

                frame_before = frame
                frame = frame_stabilized

            frame_count += 1
            print(f"Frames completed:{frame_count}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return filename, global_height, global_width, origin, transformations


def produce_stabilized_video(filename, global_height, global_width, origin, transformations):
    cap = cv2.VideoCapture(filename)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stabilized = cv2.VideoWriter('../data/stabilized.mp4', cv2.VideoWriter_fourcc(*'h264'),
                                 fps, (global_width, global_height))

    global_frame = np.zeros((global_height, global_width, 3), dtype=np.uint8)

    camera = Camera([frame_width, frame_height])

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # frame, mask = camera.undistort(frame)
            mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255

            if frame_count == 0:
                pass
            elif frame_count >= 1:
                m, rot_mat = transformations[frame_count]

                # Extract translation
                dx = int(m[0, 2])
                dy = int(m[1, 2])

                origin[0] -= dx
                origin[1] -= dy

                rows, columns, _ = frame.shape
                frame = cv2.warpAffine(frame, m, (columns, rows))
                frame = cv2.warpAffine(frame, rot_mat, (columns, rows))

            ROI = global_frame[origin[1]:origin[1] + frame_height, origin[0]:origin[0] + frame_width]

            mask_inv = cv2.bitwise_not(mask)
            ROI_with_hole = cv2.bitwise_and(ROI, ROI, mask=mask_inv)
            desired_part_of_frame = cv2.bitwise_and(frame, frame, mask=mask)

            dst = cv2.add(ROI_with_hole, desired_part_of_frame)

            global_frame[origin[1]:origin[1] + frame_height, origin[0]:origin[0] + frame_width] = dst

            stabilized.write(global_frame)

            imshow_resized('global frame', global_frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


# def stabilize_frame_standalone(filename):
#     cap = cv2.VideoCapture(filename)
#
#     global FRAME_WIDTH, FRAME_HEIGHT
#     FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"Video Resolution: {FRAME_WIDTH} by {FRAME_HEIGHT}")
#
#     global_height, global_width = FRAME_HEIGHT, FRAME_WIDTH
#     global_frame = np.zeros((global_height, global_width, 3), dtype=np.uint8)
#     top_left, bottom_right = [0, 0], [FRAME_WIDTH, FRAME_HEIGHT]
#
#     frame_count = 0
#     scene_transition = False
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         if ret:
#
#             imshow_resized('original', frame)
#
#             if frame_count == 0:
#                 frame_before = frame
#             elif frame_count >= 1:
#                 image1 = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
#                 image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#                 image_points1 = cv2.goodFeaturesToTrack(image1, maxCorners=100, qualityLevel=0.01, minDistance=10)
#
#                 image_points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, image_points1, None)
#
#                 idx = np.where(status == 1)[0]
#                 image_points1 = image_points1[idx]
#                 image_points2 = image_points2[idx]
#
#                 # Estimate affine transformation
#                 # Produces a transformation matrix :
#                 # [[cos(theta).s, -sin(theta).s, tx],
#                 # [sin(theta).s, cos(theta).s, ty]]
#                 # where theta is rotation, s is scaling and tx,ty are translation
#                 m = cv2.estimateAffinePartial2D(image_points1, image_points2)[0]
#
#                 # im = cv2.invertAffineTransform(m)
#
#                 # Extract translation
#                 dx = m[0, 2]
#                 dy = m[1, 2]
#                 # print(f"dx={dx}, dy={dy}")
#                 # Extract rotation angle
#                 da = np.arctan2(m[1, 0], m[0, 0])
#
#                 movement_threshold = 2
#                 if scene_transition == False:
#                     if math.fabs(dx) > movement_threshold or math.fabs(dy) > movement_threshold:
#                         scene_transition = True
#                         print('Moving')
#                 else:   # scene_transition is True
#                     if math.fabs(dx) <= movement_threshold and math.fabs(dy) <= movement_threshold:
#                         scene_transition = False
#                         print('Stopped')
#
#                 top_left[0] -= int(dx)
#                 bottom_right[0] -= int(dx)
#                 top_left[1] -= int(dy)
#                 bottom_right[1] -= int(dy)
#
#                 if top_left[0] < 0:
#                     pad_columns = np.zeros((global_height, int(math.fabs(dx)), 3), dtype=np.uint8)
#                     global_frame = np.hstack((pad_columns, global_frame))
#                     global_width += int(math.fabs(dx))
#                     top_left[0] = 0
#                     bottom_right[0] -= int(math.fabs(dx))
#                 elif bottom_right[0] > global_width:
#                     pad_columns = np.zeros((global_height, int(math.fabs(dx)), 3), dtype=np.uint8)
#                     global_frame = np.hstack((global_frame, pad_columns))
#                     global_width += int(math.fabs(dx))
#                 else:
#                     pass
#                 if top_left[1] < 0:
#                     pad_rows = np.zeros((int(math.fabs(dy)), global_width, 3), dtype=np.uint8)
#                     global_frame = np.vstack((pad_rows, global_frame))
#                     global_height += int(math.fabs(dy))
#                     top_left[1] = 0
#                     bottom_right[1] -= int(math.fabs(dy))
#                 elif bottom_right[1] > global_height:
#                     pad_rows = np.zeros((int(math.fabs(dy)), global_width, 3), dtype=np.uint8)
#                     global_frame = np.vstack((global_frame, pad_rows))
#                     global_height += int(math.fabs(dy))
#                 else:
#                     pass
#
#                 rows, columns = image1.shape
#                 frame_stabilized = cv2.warpAffine(frame, m, (columns, rows))
#
#                 frame_before = frame
#                 frame = frame_stabilized
#
#             global_frame[top_left[1]:top_left[1]+FRAME_HEIGHT, top_left[0]:top_left[0]+FRAME_WIDTH, :] = frame
#             print(f"Top left: {top_left}, Bottom Right: {bottom_right}")
#
#             display_frame = global_frame.copy()
#             cv2.rectangle(display_frame, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]),
#                           (0, 255, 255), 3)
#
#             imshow_resized('stabilized', frame)
#             imshow_resized('big picture', display_frame)
#
#             frame_count += 1
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         else:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


def create_mask_from_undistort_test(filename):
    cap = cv2.VideoCapture(filename)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    camera = Camera([frame_width, frame_height])

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame, mask = camera.undistort(frame)

            imshow_resized('mask', mask)

            out = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)

            imshow_resized('masked', out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


if __name__ == '__main__':
    # stabilize_frame_standalone('panning_video.mp4')

    filename, global_height, global_width, ORIGIN, transformations = find_global_size(
        'calibration_images/Sony RX100III/00010.MTS')
    print(f"Height:{global_height}, width: {global_width}")
    produce_stabilized_video(filename, global_height, global_width, ORIGIN, transformations)

    # create_mask_from_undistort_test('shaky_borders.mp4')
