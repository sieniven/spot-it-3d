import sys
import csv
import cv2
import math
import time
import numbers
import numpy as np
from multiprocessing import Process

# local imported codes
sys.path.append('../mcmt-tracking/multi-cam/utility/')
import parameters as parm
from object_tracking_util import Camera, scalar_to_rgb, setup_system_objects, \
                                    single_cam_detector, multi_cam_detector


class SingleCameraDetector(Process):
    """
    Process for single camera detection
    """
    def __init__(self, index, queue, FPS):
        super().__init__()
        self.queue = queue
        self.index = index
        self.realtime = isinstance(self.index, numbers.Number)
        self.fps = FPS
        self.frame_h = None
        self.frame_w = None
        self.scale_factor = None
        self.aspect_ratio = None
        self.cap = None
        self.fgbg = None
        self.detector = None
        self.video_ends_indicator = 0
        self.frame_count = 0
        self.frame = None
        self.good_tracks = None
        self.origin = np.array([0, 0])
        self.tracks = []
        self.next_id = 0

    def run(self):
        self.cap = cv2.VideoCapture(self.index)
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
        self.aspect_ratio = self.frame_w / self.frame_h
        
        downsample = False
        if self.frame_w * self.frame_h > 1920 * 1080:
            downsample = True
            self.frame_w = 1920
            self.frame_h = int(1920 / aspect_ratio)
            self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)

        self.fgbg, self.detector = setup_system_objects(self.scale_factor)

        # check if video capturing is successful
        ret, self.frame = self.cap.read()
        if ret:
            if self.realtime:
                print(f"Video Capture {self.index}: PASS")
            else:
                print(f"File Read \"{self.index}\": PASS")
        else:
            if self.realtime:
                print(f"Video Capture {self.index}: FAIL")
            else:
                print(f"File Read \"{self.index}\": FAIL")
            self.cap.release()

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret:
                self.frame = cv2.resize(self.frame, (self.frame_w, self.frame_h))

                self.good_tracks, self.tracks, self.next_id, self.frame = single_cam_detector(
                    self.tracks, self.next_id, self.index, self.fgbg, self.detector, self.fps,
                    self.frame_w, self.frame_h, self.scale_factor, self.origin, self.frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                self.video_ends_indicator = 1
                break

            self.queue.put((self.good_tracks, self.frame_count, self.frame))
            self.frame_count += 1

            if self.video_ends_indicator == 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()


class MultiCameraDetector(Process):
    """
    Process for multi camera detection
    """
    def __init__(self, filenames, queue, FPS):
        super().__init__()
        self.filenames = filenames
        self.queue = queue
        self.realtime = isinstance(self.filenames[0], numbers.Number)
        self.cameras = []
        self.fps = FPS
        self.video_ends_indicator = 0
        self.frame_count = 0
        self.good_tracks = None
        self.start_timer = None
        self.end_timer = None

    def run(self):
        for filename in self.filenames:
            camera = Camera(filename, self.fps)
            ret, self.frame = camera.cap.read()
            if ret:
                self.cameras.append(camera)
                if self.realtime:
                    print(f"Video Capture {filename}: PASS")
                else:
                    print(f"File Read \"{filename}\": PASS")
            else:
                if self.realtime:
                    print(f"Video Capture {filename}: FAIL")
                else:
                    print(f"File Read \"{filename}\": FAIL")
                camera.cap.release()

        while True:
            self.start_timer = time.time()
            sendList = []
            for index, camera in enumerate(self.cameras):
                ret, frame = camera.cap.read()

                if ret:
                    frame = cv2.resize(frame, (camera.frame_w, camera.frame_h))
                    self.good_tracks, frame = multi_cam_detector(camera, frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.video_ends_indicator = 1
                        break

                else:
                    self.video_ends_indicator = 1
                    break

                sendList.append((self.good_tracks, frame, camera.dead_tracks))

            # sendList: [(good_tracks_0, frame_0, dead_tracks_0), (good_tracks_1, frame_1, dead_tracks_1), frame_count]
            sendList.append((self.frame_count))
            self.queue.put(sendList)
            self.frame_count += 1

            if self.video_ends_indicator == 1:
                break

            self.end_timer = time.time()
            print(f"Detection process took: {self.end_timer - self.start_timer}")

        cv2.destroyAllWindows()

        for index, camera in enumerate(self.cameras):
            camera.cap.release()
            with open(f"data_out_{index}.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in camera.output_log:
                    writer.writerow(row)
