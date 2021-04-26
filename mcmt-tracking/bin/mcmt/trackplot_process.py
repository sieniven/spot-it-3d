import sys
import cv2
import math
import time
import numpy as np
from multiprocessing import Process

# local imported code
sys.path.append('../mcmt-tracking/multi-cam/utility/')
from plot_tracking_util import plot_track_feature_variable, export_data_as_csv, mcmt_plot
from object_tracking_util import scalar_to_rgb
from trackplots_util import CameraTracks, TrackPlot, combine_track_plots
import parameters as parm


class SingleCameraTrackPlot(Process):
    """
    Process for processing single camera tracks' features
    """
    def __init__(self, index, queue, FPS):
        super().__init__()
        self.index = index
        self.queue = queue
        self.fps = FPS
        self.frame_h = None
        self.frame_w = None
        self.scale_factor = None
        self.aspect_ratio = None
        self.recording = None
        self.FLAG = False

        # get camera parameters
        cap = cv2.VideoCapture(self.index)
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
        self.aspect_ratio = self.frame_w / self.frame_h
        
        downsample = False
        if self.frame_w * self.frame_h > 1920 * 1080:
            downsample = True
            self.frame_w = 1920
            self.frame_h = int(1920 / aspect_ratio)
            self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
        
        self.recording = cv2.VideoWriter("/home/niven/mcmt-tracking/data/video_plot.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                        self.fps, (self.frame_w, self.frame_h))
        
        cap.release()

        # data from detector
        self.good_tracks = None
        self.frame_count = None
        self.frame = None
        self.next_id = None

        self.last_update = time.time()
        self.new_data = False
        self.origin = np.array([0, 0])
        self.track_plots_ids = []
        self.track_plots = []

        # initialize plotting parameters
        self.plot_history = 200
        self.colours = [''] * self.plot_history
        for i in range(self.plot_history):
            self.colours[i] = scalar_to_rgb(i, self.plot_history)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5

    def run(self):
        self.last_update = time.time()

        while not self.FLAG:
            while not self.queue.empty():
                self.new_data = True
                item = self.queue.get()
                self.good_tracks, self.frame_count, self.frame = item

                for track in self.good_tracks:
                    track_id = track[0]
                    centroid_x, centroid_y = track[3]

                    if track_id not in self.track_plots_ids:  # First occurrence of the track
                        self.track_plots_ids.append(track_id)
                        self.track_plots.append(TrackPlot(track_id))

                    track_plot = self.track_plots[self.track_plots_ids.index(track_id)]
                    track_plot.update((centroid_x, centroid_y), self.frame_count)
                    track_plot.calculate_track_feature_variable(self.frame_count, self.fps)

                self.last_update = time.time()

            if self.new_data:
                for track_plot in self.track_plots:
                    idxs = np.where(np.logical_and(track_plot.frameNos > self.frame_count - self.plot_history,
                                                   track_plot.frameNos <= self.frame_count))[0]
                    for idx in idxs:
                        cv2.circle(self.frame, (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx] - self.origin[1]),
                                   3, self.colours[track_plot.frameNos[idx] - self.frame_count + self.plot_history - 1][::-1], -1)
                    if len(idxs) != 0:
                        cv2.putText(self.frame, f"ID: {track_plot.id}",
                                    (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx] - self.origin[1] + 15),
                                    self.font, self.font_scale, (0, 0, 255), 1, cv2.LINE_AA)
                        if track_plot.track_feature_variable.size != 0:
                            cv2.putText(self.frame, f"Xj: {np.mean(track_plot.track_feature_variable):.3f}",
                                        (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx] - self.origin[1] + 30),
                                        self.font, self.font_scale, (0, 255, 0), 1, cv2.LINE_AA)

                self.recording.write(self.frame)
                cv2.imshow(f"Original {self.index}", self.frame)
                self.new_data = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.FLAG = True

            if (time.time() - self.last_update) > 5 and not self.new_data:
                print("Timeout: Terminating track plot")
                self.FLAG = True

        self.recording.release()
        cv2.destroyAllWindows()

        plot_track_feature_variable(self.track_plots)
        export_data_as_csv(self.track_plots)


class MultiCameraTrackPlot(Process):
    """
    Process for processing multi camera tracks' features
    """
    def __init__(self, filenames, queue, FPS):
        super().__init__()
        self.filenames = filenames
        self.queue = queue
        self.fps = FPS
        self.FLAG = False
        
        # get camera parameters
        cap = cv2.VideoCapture(filenames[0])
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
        self.aspect_ratio = self.frame_w / self.frame_h

        downsample = False
        if self.frame_w * self.frame_h > 1920 * 1080:
            downsample = True
            self.frame_w = 1920
            self.frame_h = int(1920 / aspect_ratio)
            self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
        
        self.recording = cv2.VideoWriter("../data/video_plot.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                        self.fps, (self.frame_w * 2, self.frame_h))
        cap.release()

        # sendList data from detector
        self.good_tracks_0 = None
        self.filter_good_tracks_0 = None
        self.good_tracks_1 = None
        self.filter_good_tracks_1 = None
        self.frame_0 = None
        self.frame_1 = None
        self.dead_tracks_0 = None
        self.dead_tracks_1 = None
        self.frame_count = None

        self.last_update = time.time()
        self.next_id = 0
        self.count = 0
        self.counter = 0
        self.tester = 0
        self.combine_frame = None
        self.corrValues = None
        self.new_data = False
        self.origin = np.array([0, 0])
        self.camera_0_tracks = CameraTracks(0)
        self.camera_1_tracks = CameraTracks(1)
        self.total_tracks_0 = None
        self.total_tracks_1 = None
        self.matching_dict_0 = {}
        self.matching_dict_1 = {}
        self.maxValues = []
        self.maxIndexes = []
        self.indicator = 0
        self.maxIndex = -100
        self.maxValue = 0
        self.removeList = []
        self.start_timer = None
        self.end_timer = None

        # initialize plotting parameters
        self.plot_history = 200
        self.colours = [''] * self.plot_history
        for i in range(self.plot_history):
            self.colours[i] = scalar_to_rgb(i, self.plot_history)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5

    def run(self):
        self.last_update = time.time()

        while not self.FLAG:
            while not self.queue.empty():
                self.start_timer = time.time()
                self.new_data = True
                self.camera_0_tracks.output_log.append([])
                self.camera_1_tracks.output_log.append([])

                item = self.queue.get()
                self.good_tracks_0, self.frame_0, self.dead_tracks_0 = item[0]
                self.good_tracks_1, self.frame_1, self.dead_tracks_1 = item[1]
                self.frame_count = item[2]

            if self.new_data:
                # create filter copy of good_tracks list
                self.filter_good_tracks_0 = list(self.good_tracks_0)
                self.filter_good_tracks_1 = list(self.good_tracks_1)

                for track in self.good_tracks_0:
                    track_id = track[0]
                    centroid_x, centroid_y = track[3]
                    self.camera_0_tracks.output_log[self.frame_count].extend([track_id, centroid_x, centroid_y])

                    # occurance of a new track
                    if track_id not in self.camera_0_tracks.track_plots_ids:
                        if track_id not in self.matching_dict_0:
                            self.camera_0_tracks.track_new_plots_ids.append(track_id)
                            self.camera_0_tracks.track_new_plots.append(TrackPlot(track_id))
                            self.matching_dict_0[track_id] = track_id

                for track in self.good_tracks_1:
                    track_id = track[0]
                    centroid_x, centroid_y = track[3]
                    self.camera_1_tracks.output_log[self.frame_count].extend([track_id, centroid_x, centroid_y])

                    # occurance of a new track
                    if track_id not in self.camera_1_tracks.track_plots_ids:
                        if track_id not in self.matching_dict_1:
                            self.camera_1_tracks.track_new_plots_ids.append(track_id)
                            self.camera_1_tracks.track_new_plots.append(TrackPlot(track_id))
                            self.matching_dict_1[track_id] = track_id

                # analyze camera 0
                self.get_total_number_of_tracks()
                self.count = 0
                self.corrValues = np.zeros((1, self.total_tracks_1))
                self.removeList = []

                for track in self.good_tracks_0:
                    track_id = track[0]
                    centroid_x, centroid_y = track[3]

                    if self.matching_dict_0[track_id] not in self.camera_0_tracks.track_plots_ids:
                        if self.count != 0:
                            self.corrValues = np.append(self.corrValues, np.zeros((1, self.total_tracks_1)), axis=0)

                        track_plot = self.camera_0_tracks.track_new_plots[
                            self.camera_0_tracks.track_new_plots_ids.index(track_id)]
                        track_plot.update([centroid_x, centroid_y], self.frame_count)
                        track_plot.calculate_track_feature_variable(self.frame_count, self.fps)

                        # if track is not a new track, we use 90 frames as the minimum requirement before matching occurs
                        if track_plot.frameNos.size >= 90 and track_plot.track_feature_variable.size >= 90 and (
                                math.sqrt(np.sum(np.square(track_plot.track_feature_variable)))) != 0:
                            # normalization of cross correlation value
                            # method 1:
                            # track_plot_normalize_xj = (track_plot.track_feature_variable) / (
                            # math.sqrt(np.sum(np.square(track_plot.track_feature_variable))))

                            # method 2:
                            track_plot_normalize_xj = (track_plot.track_feature_variable - np.mean(
                                track_plot.track_feature_variable)) / (np.std(track_plot.track_feature_variable) * len(
                                track_plot.track_feature_variable))

                            self.counter = 0

                            # look into 2nd camera's new tracks
                            for alt_track_plot in self.camera_1_tracks.track_new_plots:
                                # track in 2nd camera must fulfills requirements to have at least 90 frames to match
                                if alt_track_plot.frameNos.size >= 90 and alt_track_plot.track_feature_variable.size >= 90 and (
                                        math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable)))) != 0:
                                    # normalization of cross correlation value
                                    # method 1:
                                    # alt_track_plot_normalize_xj = (alt_track_plot.track_feature_variable) / (
                                    # math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable))))

                                    # method 2
                                    alt_track_plot_normalize_xj = (alt_track_plot.track_feature_variable - np.mean(
                                        alt_track_plot.track_feature_variable)) / (np.std(alt_track_plot.track_feature_variable))

                                    r_value = max(np.correlate(track_plot_normalize_xj,
                                                               alt_track_plot_normalize_xj, mode='full'))

                                    if r_value > 0.7:
                                        self.corrValues[self.count, self.counter] = r_value
                                self.counter += 1

                            # look into 2nd camera's matched tracks list
                            for alt_track_plot in self.camera_1_tracks.track_plots:
                                tester = 0
                                for dead_track_id in self.dead_tracks_1:
                                    if self.matching_dict_1[dead_track_id] == alt_track_plot.id:
                                        tester = 1  # 2nd camera's track has already been lost. skip the process of matching for this track
                                        break

                                # test to see if alternate camera's track is currently being matched with current camera
                                for alt_tracker in self.good_tracks_0:
                                    if alt_track_plot.id == self.matching_dict_0[alt_tracker[0]]:
                                        tester = 1  # 2nd camera's track has already been matched. skip the process of matching for this track
                                        break

                                if tester == 0 and (math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable)))) != 0:
                                    # alt_track_plot_normalize_xj = (alt_track_plot.track_feature_variable) / (
                                    # math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable))))
                                    alt_track_plot_normalize_xj = (alt_track_plot.track_feature_variable - np.mean(
                                        alt_track_plot.track_feature_variable)) / (np.std(alt_track_plot.track_feature_variable))
                                    r_value = max(np.correlate(track_plot_normalize_xj,
                                                               alt_track_plot_normalize_xj, mode='full'))

                                    if r_value > 0.7:
                                        self.corrValues[self.count, self.counter] = r_value

                                self.counter += 1

                        self.count += 1

                    # if track is already a matched current track
                    else:
                        track_plot = self.camera_0_tracks.track_plots[self.camera_0_tracks.track_plots_ids.index(self.matching_dict_0[track_id])]
                        track_plot.update((centroid_x, centroid_y), self.frame_count)
                        track_plot.calculate_track_feature_variable(self.frame_count, self.fps)
                        self.filter_good_tracks_0.pop(self.count)

                # matching process, finding local maximas in the cross correlation results
                # find local maximas in corrValues. for every iteration, we search every valid track to be re-id from current camera
                for x in range(len(self.filter_good_tracks_0)):
                    self.maxValues = []
                    self.maxIndexes = []

                    # append every cross correlation value into list of maxValues, as long as they do not equal to 0
                    for y in range(self.total_tracks_1):
                        if self.corrValues[x][y] != 0:
                            self.maxValues.append(self.corrValues[x][y])
                            self.maxIndexes.append(y)

                    self.indicator = 0
                    self.maxIndex = -100

                    # for the selected max track in the 2nd camera, we check to see if the track has a higher
                    # cross correlation value with another track in current camera

                    while self.indicator == 0 and len(self.maxValues) != 0:
                        self.maxValue = max(self.maxValues)
                        self.maxIndex = self.maxIndexes[self.maxValues.index(self.maxValue)]

                        # search through current camera's tracks again, for the selected track that we wish to re-id with.
                        # we can note that if there is a track in the current camera that has a higher cross correlation value
                        # than the track we wish to match with, then the matching will not occur.
                        for x_1 in range(len(self.filter_good_tracks_0)):
                            if self.corrValues[x_1][self.maxIndex] > self.maxValue:
                                self.maxValues.remove(self.maxValue)
                                self.maxIndexes.remove(self.maxIndex)
                                self.indicator = 1
                                break

                        if self.indicator == 1:
                            # there existed a value larger than the current maxValue. thus, re-id cannot occur
                            self.indicator = 0
                            continue
                        else:
                            # went through the whole loop without breaking, thus it is the maximum value. re-id can occur
                            self.indicator = 2

                    if self.indicator == 2:  # re-id process
                        # if track is in 2nd camera's new track list
                        if self.maxIndex != -100 and self.maxIndex < len(self.camera_1_tracks.track_new_plots):
                            # remove track plot in new tracks' list and add into matched tracks' list for alternate camera
                            alt_track_id = self.camera_1_tracks.track_new_plots_ids[self.maxIndex]
                            self.camera_1_tracks.track_new_plots[self.maxIndex].id = self.next_id
                            self.camera_1_tracks.track_plots_ids.append(self.next_id)
                            self.camera_1_tracks.track_plots.append(self.camera_1_tracks.track_new_plots[self.maxIndex])
                            # update dictionary matching
                            self.matching_dict_1[alt_track_id] = self.next_id

                            self.removeList.append(self.maxIndex)

                            # remove track plot in new tracks' list and add into matched tracks' list for current camera
                            track_id = self.filter_good_tracks_0[x][0]
                            trackIndex = self.camera_0_tracks.track_new_plots_ids.index(track_id)
                            track_plot = self.camera_0_tracks.track_new_plots[trackIndex]
                            track_plot.id = self.next_id
                            self.camera_0_tracks.track_plots_ids.append(self.next_id)
                            self.camera_0_tracks.track_plots.append(track_plot)
                            self.camera_0_tracks.track_new_plots.pop(trackIndex)
                            self.camera_0_tracks.track_new_plots_ids.pop(trackIndex)
                            # update dictionary matching list
                            self.matching_dict_0[track_id] = self.next_id

                            self.next_id += 1

                        # if track is in 2nd camera's matched track list
                        else:
                            track_id = self.filter_good_tracks_0[x][0]
                            trackIndex = self.camera_0_tracks.track_new_plots_ids.index(track_id)
                            track_plot = self.camera_0_tracks.track_new_plots[trackIndex]
                            track_plot.id = self.camera_1_tracks.track_plots[
                                self.maxIndex - len(self.camera_1_tracks.track_new_plots)].id
                            track_plot_index = self.camera_0_tracks.track_plots_ids.index(track_plot.id)
                            # update track plot in the original track ID
                            # check this
                            combine_track_plots(track_plot_index, self.camera_0_tracks, track_plot, self.frame_count)
                            # remove track plot in new tracks' list
                            self.camera_0_tracks.track_new_plots.pop(trackIndex)
                            self.camera_0_tracks.track_new_plots_ids.pop(trackIndex)
                            # update dictionary matching list
                            self.matching_dict_0[track_id] = track_plot.id

                for remove_index in self.removeList:
                    self.camera_1_tracks.track_new_plots.pop(remove_index)
                    self.camera_1_tracks.track_new_plots_ids.pop(remove_index)

                # analyze camera 1
                self.get_total_number_of_tracks()
                self.count = 0
                self.corrValues = np.zeros((1, self.total_tracks_0))
                self.removeList = []

                for track in self.good_tracks_1:
                    track_id = track[0]
                    centroid_x, centroid_y = track[3]

                    if self.matching_dict_1[track_id] not in self.camera_1_tracks.track_plots_ids:
                        if self.count != 0:
                            self.corrValues = np.append(self.corrValues, np.zeros((1, self.total_tracks_0)), axis=0)

                        track_plot = self.camera_1_tracks.track_new_plots[
                            self.camera_1_tracks.track_new_plots_ids.index(track_id)]
                        track_plot.update([centroid_x, centroid_y], self.frame_count)
                        track_plot.calculate_track_feature_variable(self.frame_count, self.fps)

                        # if track is not a new track, we use 90 frames as the minimum requirement before matching occurs
                        if track_plot.frameNos.size >= 90 and track_plot.track_feature_variable.size >= 90 and (
                                math.sqrt(np.sum(np.square(track_plot.track_feature_variable)))) != 0:
                            # normalization of cross correlation value
                            # method 1:
                            # track_plot_normalize_xj = (track_plot.track_feature_variable) / (
                            # math.sqrt(np.sum(np.square(track_plot.track_feature_variable))))

                            # method 2:
                            track_plot_normalize_xj = (track_plot.track_feature_variable - np.mean(
                                track_plot.track_feature_variable)) / (np.std(track_plot.track_feature_variable) * len(
                                track_plot.track_feature_variable))

                            self.counter = 0

                            # look into 2nd camera's new tracks
                            for alt_track_plot in self.camera_0_tracks.track_new_plots:
                                # track in 2nd camera must fulfills requirements to have at least 90 frames to match
                                if alt_track_plot.frameNos.size >= 90 and alt_track_plot.track_feature_variable.size >= 90 and (
                                        math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable)))) != 0:
                                    # normalization of cross correlation value
                                    # method 1:
                                    # alt_track_plot_normalize_xj = (alt_track_plot.track_feature_variable) / (
                                    # math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable))))

                                    # method 2
                                    alt_track_plot_normalize_xj = (alt_track_plot.track_feature_variable - np.mean(
                                        alt_track_plot.track_feature_variable)) / (
                                                                      np.std(alt_track_plot.track_feature_variable))

                                    r_value = max(np.correlate(track_plot_normalize_xj,
                                                               alt_track_plot_normalize_xj, mode='full'))

                                    if r_value > 0.7:
                                        self.corrValues[self.count, self.counter] = r_value
                                self.counter += 1

                            # look into 2nd camera's matched tracks list
                            for alt_track_plot in self.camera_0_tracks.track_plots:
                                tester = 0
                                for dead_track_id in self.dead_tracks_0:
                                    if self.matching_dict_0[dead_track_id] == alt_track_plot.id:
                                        tester = 1  # 2nd camera's track has already been lost. skip the process of matching for this track
                                        break

                                # test to see if alternate camera's track is currently being matched with current camera
                                for alt_tracker in self.good_tracks_1:
                                    if alt_track_plot.id == self.matching_dict_1[alt_tracker[0]]:
                                        tester = 1  # 2nd camera's track has already been matched. skip the process of matching for this track
                                        break

                                if tester == 0 and (math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable)))) != 0:
                                    # alt_track_plot_normalize_xj = (alt_track_plot.track_feature_variable) / (
                                    # math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable))))
                                    alt_track_plot_normalize_xj = (alt_track_plot.track_feature_variable - np.mean(
                                        alt_track_plot.track_feature_variable)) / (np.std(alt_track_plot.track_feature_variable))
                                    r_value = max(np.correlate(track_plot_normalize_xj,
                                                               alt_track_plot_normalize_xj, mode='full'))

                                    if r_value > 0.7:
                                        self.corrValues[self.count, self.counter] = r_value

                                self.counter += 1

                        self.count += 1

                    # if track is already a matched current track
                    else:
                        track_plot = self.camera_1_tracks.track_plots[self.camera_1_tracks.track_plots_ids.index(self.matching_dict_1[track_id])]
                        track_plot.update((centroid_x, centroid_y), self.frame_count)
                        track_plot.calculate_track_feature_variable(self.frame_count, self.fps)
                        self.filter_good_tracks_1.pop(self.count)

                # matching process, finding local maximas in the cross correlation results
                # find local maximas in corrValues. for every iteration, we search every valid track to be re-id from current camera
                for x in range(len(self.filter_good_tracks_1)):
                    self.maxValues = []
                    self.maxIndexes = []
                    # append every cross correlation value into list of maxValues, as long as they do not equal to 0
                    for y in range(self.total_tracks_0):
                        if self.corrValues[x][y] != 0:
                            self.maxValues.append(self.corrValues[x][y])
                            self.maxIndexes.append(y)

                    self.indicator = 0
                    self.maxIndex = -100

                    # for the selected max track in the 2nd camera, we check to see if the track has a higher
                    # cross correlation value with another track in current camera

                    while self.indicator == 0 and len(self.maxValues) != 0:
                        self.maxValue = max(self.maxValues)
                        self.maxIndex = self.maxIndexes[self.maxValues.index(self.maxValue)]

                        # search through current camera's tracks again, for the selected track that we wish to re-id with.
                        # we can note that if there is a track in the current camera that has a higher cross correlation value
                        # than the track we wish to match with, then the matching will not occur.
                        for x_1 in range(len(self.filter_good_tracks_1)):
                            if self.corrValues[x_1][self.maxIndex] > self.maxValue:
                                self.maxValues.remove(self.maxValue)
                                self.maxIndexes.remove(self.maxIndex)
                                self.indicator = 1
                                break

                        if self.indicator == 1:
                            # there existed a value larger than the current maxValue. thus, re-id cannot occur
                            self.indicator = 0
                            continue
                        else:
                            # went through the whole loop without breaking, thus it is the maximum value. re-id can occur
                            self.indicator = 2

                    if self.indicator == 2:  # re-id process
                        # if track is in 2nd camera's new track list
                        if self.maxIndex != -100 and self.maxIndex < len(self.camera_0_tracks.track_new_plots):
                            # remove track plot in new tracks' list and add into matched tracks' list for alternate camera
                            alt_track_id = self.camera_0_tracks.track_new_plots_ids[self.maxIndex]
                            self.camera_0_tracks.track_new_plots[self.maxIndex].id = self.next_id
                            self.camera_0_tracks.track_plots_ids.append(self.next_id)
                            self.camera_0_tracks.track_plots.append(self.camera_0_tracks.track_new_plots[self.maxIndex])
                            # update dictionary matching
                            self.matching_dict_0[alt_track_id] = self.next_id

                            self.removeList.append(self.maxIndex)

                            # remove track plot in new tracks' list and add into matched tracks' list for current camera
                            track_id = self.filter_good_tracks_1[x][0]
                            trackIndex = self.camera_1_tracks.track_new_plots_ids.index(track_id)
                            track_plot = self.camera_1_tracks.track_new_plots[trackIndex]
                            track_plot.id = self.next_id
                            self.camera_1_tracks.track_plots_ids.append(self.next_id)
                            self.camera_1_tracks.track_plots.append(track_plot)
                            self.camera_1_tracks.track_new_plots.pop(trackIndex)
                            self.camera_1_tracks.track_new_plots_ids.pop(trackIndex)
                            # update dictionary matching list
                            self.matching_dict_1[track_id] = self.next_id

                            self.next_id += 1

                        # if track is in 2nd camera's matched track list
                        else:
                            track_id = self.filter_good_tracks_1[x][0]
                            trackIndex = self.camera_1_tracks.track_new_plots_ids.index(track_id)
                            track_plot = self.camera_1_tracks.track_new_plots[trackIndex]
                            track_plot.id = self.camera_0_tracks.track_plots[
                                self.maxIndex - len(self.camera_0_tracks.track_new_plots)].id
                            track_plot_index = self.camera_1_tracks.track_plots_ids.index(track_plot.id)
                            # update track plot in the original track ID
                            # check this
                            combine_track_plots(track_plot_index, self.camera_1_tracks, track_plot, self.frame_count)
                            # remove track plot in new tracks' list
                            self.camera_1_tracks.track_new_plots.pop(trackIndex)
                            self.camera_1_tracks.track_new_plots_ids.pop(trackIndex)
                            # update dictionary matching list
                            self.matching_dict_1[track_id] = track_plot.id

                for remove_index in self.removeList:
                    self.camera_0_tracks.track_new_plots.pop(remove_index)
                    self.camera_0_tracks.track_new_plots_ids.pop(remove_index)

                # plotting track plot trajectories into frame
                for track_plot in self.camera_0_tracks.track_plots:
                    if (self.frame_count - track_plot.lastSeen) <= self.fps:
                        shown_indexes = np.where(np.logical_and(track_plot.frameNos > self.frame_count - self.plot_history,
                                                                track_plot.frameNos <= self.frame_count))[0]
                        for idx in shown_indexes:
                            cv2.circle(self.frame_0, (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx] - self.origin[1]), 3,
                                       self.colours[track_plot.frameNos[idx] - self.frame_count + self.plot_history - 1][::-1], -1)

                        if len(shown_indexes) != 0:
                            cv2.putText(self.frame_0, f"ID: {track_plot.id}",
                                        (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx]
                                         - self.origin[1] + 15), self.font, self.font_scale, (0, 0, 255), 1, cv2.LINE_AA)

                for track_plot in self.camera_0_tracks.track_new_plots:
                    if (self.frame_count - track_plot.lastSeen) <= self.fps:
                        shown_indexes = np.where(np.logical_and(track_plot.frameNos > self.frame_count - self.plot_history,
                                                                track_plot.frameNos <= self.frame_count))[0]
                        for idx in shown_indexes:
                            cv2.circle(self.frame_0, (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx] - self.origin[1]), 3,
                                       self.colours[track_plot.frameNos[idx] - self.frame_count + self.plot_history - 1][::-1], -1)
                        if len(shown_indexes) != 0:
                            cv2.putText(self.frame_0, f"ID: {track_plot.id}",
                                        (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx]
                                         - self.origin[1] + 15), self.font, self.font_scale, (0, 0, 255), 1, cv2.LINE_AA)

                for track_plot in self.camera_1_tracks.track_plots:
                    if (self.frame_count - track_plot.lastSeen) <= self.fps:
                        shown_indexes = np.where(np.logical_and(track_plot.frameNos > self.frame_count - self.plot_history,
                                                                track_plot.frameNos <= self.frame_count))[0]
                        for idx in shown_indexes:
                            cv2.circle(self.frame_1, (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx] - self.origin[1]), 3,
                                       self.colours[track_plot.frameNos[idx] - self.frame_count + self.plot_history - 1][::-1], -1)

                        if len(shown_indexes) != 0:
                            cv2.putText(self.frame_1, f"ID: {track_plot.id}", (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx]
                                        - self.origin[1] + 15), self.font, self.font_scale, (0, 0, 255), 1, cv2.LINE_AA)

                for track_plot in self.camera_1_tracks.track_new_plots:
                    if (self.frame_count - track_plot.lastSeen) <= self.fps:
                        shown_indexes = np.where(np.logical_and(track_plot.frameNos > self.frame_count - self.plot_history,
                                                                track_plot.frameNos <= self.frame_count))[0]
                        for idx in shown_indexes:
                            cv2.circle(self.frame_1, (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx] - self.origin[1]), 3,
                                       self.colours[track_plot.frameNos[idx] - self.frame_count + self.plot_history - 1][::-1], -1)

                        if len(shown_indexes) != 0:
                            cv2.putText(self.frame_1, f"ID: {track_plot.id}", (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx]
                                        - self.origin[1] + 15), self.font, self.font_scale, (0, 0, 255), 1, cv2.LINE_AA)

                self.combine_frame = np.hstack((self.frame_0, self.frame_1))
                self.end_timer = time.time()
                print(f"Trackplot process took: {self.end_timer - self.start_timer}")

            self.last_update = time.time()
            self.new_data = False

            if self.combine_frame is not None:
                cv2.imshow(f"Original", self.combine_frame)
                self.recording.write(self.combine_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.FLAG = True

            if (time.time() - self.last_update) > 5 and not self.new_data:
                print("Timeout: Terminating track plot")
                self.FLAG = True

        print("Saving video....")
        self.recording.release()
        cv2.destroyAllWindows()

        mcmt_plot(self.camera_0_tracks, self.camera_1_tracks)


    def get_total_number_of_tracks(self):
        self.total_tracks_0 = len(self.camera_0_tracks.track_new_plots) + len(self.camera_0_tracks.track_plots)
        self.total_tracks_1 = len(self.camera_1_tracks.track_new_plots) + len(self.camera_1_tracks.track_plots)
