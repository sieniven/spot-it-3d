# basic Python frameworks
import math
import numpy as np


class CameraTracks:
    def __init__(self, index):
        self.index = index
        self.track_plots_ids = []
        self.track_new_plots_ids = []
        
        # self.track_plots = []
        # self.track_new_plots = []
        self.track_plots = {}
        self.track_new_plots = {}

        self.output_log = []


class TrackPlot:
    def __init__(self, track_id):
        self.id = track_id
        self.xs = np.array([], dtype=int)
        self.ys = np.array([], dtype=int)
        self.frameNos = np.array([], dtype=int)
        self.times = np.array([])
        self.colourized_times = []
        self.lastSeen = 0

        self.Q = 0
        self.turning_angle = np.array([])  # Degrees
        self.curvature = np.array([])
        self.pace = np.array([])
        self.track_feature_variable = np.array([])
        self.other_tracks = []

        self.xyz = None


    def update(self, location, frame_no, time=None):
        self.xs = np.append(self.xs, [int(location[0])])
        self.ys = np.append(self.ys, [int(location[1])])
        self.frameNos = np.append(self.frameNos, [frame_no])

        if time is not None:
            self.times = np.append(self.times, [time])

        self.lastSeen = frame_no


    def calculate_track_feature_variable(self, frameNo, fps):
        # Check if there is enough data to calculate turning angle (at least 3 points)
        # And that the data is still current
        if len(self.frameNos) >= 3 and self.frameNos[-1] == frameNo:
            # Check if the last 3 frames are consecutive
            if self.frameNos[-2] == self.frameNos[-1] - 1 and self.frameNos[-3] == self.frameNos[-2] - 1:
                # Retrieve the x and y values of the last 3 points and introduce t for readability
                t = 2
                x, y = self.xs[-3:], self.ys[-3:]

                # Turning angle
            #    xs = np.array([x[t] - x[t - 1], x[t - 1] - x[t - 2]])
            #    ys = np.array([y[t] - y[t - 1], y[t - 1] - y[t - 2]])

                # arctan2 returns the element-wise arc tangent, choosing the element correctly
                # Special angles excluding infinities:
                # y = +/- 0, x = +0, theta = +/- 0
                # y = +/- 0, x = -0, theta = +/- pi
                # whats a positive or negative 0?
            #    heading = np.arctan2(ys, xs) * 180 / np.pi

            #    turning_angle = heading[1] - heading[0]

            #    self.turning_angle = np.append(self.turning_angle, turning_angle)

                # Curvature
            #    a = np.sqrt((x[t] - x[t - 2]) ** 2 + (y[t] - y[t - 2]) ** 2)
            #    b = np.sqrt((x[t - 1] - x[t - 2]) ** 2 + (y[t - 1] - y[t - 2]) ** 2)
                c = np.sqrt((x[t] - x[t - 1]) ** 2 + (y[t] - y[t - 1]) ** 2)

            #    if b == 0 or c == 0:
            #        curvature = 0
            #    else:
            #       if abs((a ** 2 - b ** 2 - c ** 2) / (2 * b * c)) <= 1:
            #            curvature = np.arccos((a ** 2 - b ** 2 - c ** 2) / (2 * b * c))
                        # For whatever reason, the arccos of 1.0000000000000002 is nan
            #        else:
            #            curvature = 0
            #        if np.isnan(curvature):
            #            curvature = 0

            #    self.curvature = np.append(self.curvature, curvature)

                # Pace
                # Check if the data was returned in real time
                if self.times.size != 0:  # If so, dt is the difference in the time each consecutive frame was read
                    dt = self.times[-1] - self.times[-2]
                else:
                    # assume 30 FPS
                    dt = 1 / fps

                pace = c / dt
                self.pace = np.append(self.pace, pace)

                # track_feature_variable = np.mean(self.turning_angle) * np.mean(self.curvature) * np.mean(self.pace)
                # self.track_feature_variable = np.append(self.track_feature_variable, track_feature_variable)
                self.track_feature_variable = np.append(self.track_feature_variable, pace)


    def update_other_tracks(self, cumulative_tracks):
        
        self.other_tracks = []

        for other_track in list(cumulative_tracks.track_new_plots.values()) + list(cumulative_tracks.track_plots.values()):

            if len(other_track.xs) != 0 and len(other_track.ys) != 0: 
                dx = other_track.xs[-1] - self.xs[-1]
                dy = other_track.ys[-1] - self.ys[-1]

                if dx != 0 and dy!= 0:
                    self.other_tracks.append((np.arctan2(dy, dx), math.hypot(dx, dy)))


def combine_track_plots(index, camera_tracks, track_plot, frame_count):
    # appending of various variable value lists
    camera_tracks.track_plots[index].xs = np.append(camera_tracks.track_plots[index].xs, track_plot.xs)
    camera_tracks.track_plots[index].ys = np.append(camera_tracks.track_plots[index].ys, track_plot.ys)
    camera_tracks.track_plots[index].frameNos = np.append(camera_tracks.track_plots[index].frameNos, track_plot.frameNos)
    camera_tracks.track_plots[index].times = np.append(camera_tracks.track_plots[index].times, track_plot.times)
    camera_tracks.track_plots[index].colourized_times = np.append(camera_tracks.track_plots[index].colourized_times,
                                                                  track_plot.colourized_times)
    camera_tracks.track_plots[index].lastSeen = frame_count
    camera_tracks.track_plots[index].turning_angle = np.append(camera_tracks.track_plots[index].turning_angle,
                                                               track_plot.turning_angle)
    camera_tracks.track_plots[index].curvature = np.append(camera_tracks.track_plots[index].xs, track_plot.xs)
    camera_tracks.track_plots[index].pace = np.append(camera_tracks.track_plots[index].pace, track_plot.pace)
    camera_tracks.track_plots[index].track_feature_variable = np.append(camera_tracks.track_plots[index].track_feature_variable,
                                                                        track_plot.track_feature_variable)


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