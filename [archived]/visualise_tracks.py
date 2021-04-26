from object_tracker import motion_based_multi_object_tracking
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import multiprocessing


class TrackPlot():
    def __init__(self, track_id):
        self.id = track_id
        self.xs = []
        self.ys = []
        self.frameNos = []
        self.colourized_times = []
        self.lastSeen = 0

    def plot_track(self):
        print(f"Track {self.id} being plotted...")
        plt.scatter(self.xs, self.ys, c=self.colourized_times, marker='+')
        plt.show()

    def update(self, offset, location, frame_no):
        self.xs.append(location[0] + offset[0])
        self.ys.append(location[1] + offset[1])
        self.frameNos.append(frame_no)
        self.lastSeen = frame_no


def plot_tracks(scenes):
    fig, ax = plt.subplots()

    fps = scenes[0][0]
    frame_width, frame_height = scenes[0][1], scenes[0][2]
    scenes = scenes[1:]

    for scene in scenes:
        track_ids = []
        track_plots = []

        frame_count = len(scene)
        for frame in scene:
            idx = scene.index(frame)
            for track in frame:
                track_id = track[0]

                if track_id not in track_ids:  # First occurrence of the track
                    track_ids.append(track_id)
                    track_plots.append(TrackPlot(track_id))

                track_plot = track_plots[track_ids.index(track_id)]
                track_plot.xs.append(track[3][0])
                track_plot.ys.append(-track[3][1])
                track_plot.colourized_times.append(scalar_to_hex(idx, frame_count))

        for track_plot in track_plots:
            print(f"Track {track_plot.id} is {len(track_plot.colourized_times) / fps} seconds long.")
            if len(track_plot.colourized_times) > 5 * fps:
                # track_plot.plot_track()
                print(f"Track {track_plot.id} being plotted...")
                ax.scatter(track_plot.xs, track_plot.ys, c=track_plot.colourized_times, marker='+')
                ax.annotate(track_plot.id, (track_plot.xs[-1], track_plot.ys[-1]),
                            (track_plot.xs[-1] + 1, track_plot.ys[-1] + 1))

        ax.set_xlim(0, frame_width)
        ax.set_ylim(-frame_height, 0)
        plt.show()


def plot_tracks_realtime():
    print('b4 loop')

    q = multiprocessing.Queue()

    get_results_p = multiprocessing.Process(target=get_results, args=(q,))
    plot_results_p = multiprocessing.Process(target=plot_results, args=(q,))

    get_results_p.start()
    plot_results_p.start()

    # ax.scatter(track_plot.xs, track_plot.ys)
    #
    # fig.canvas.draw()
    # fig.canvas.flush_events()


def plot_results(q):
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()

    origin = [0, 0]

    track_ids = []
    track_plots = []

    while True:
        while not q.empty():
            item = q.get()
            tracks, (dx, dy), frame_no = item[0], item[1], item[2]
            origin[0] += dx
            origin[1] += dy
            for track in tracks:
                track_id = track[0]
                if track_id not in track_ids:  # First occurrence of the track
                    track_ids.append(track_id)
                    track_plots.append(TrackPlot(track_id))

                    track_plot = track_plots[track_ids.index(track_id)]
                    track_plot.update(origin, track[3], frame_no)

        for track_plot in track_plots:
            ax.scatter(track_plot.xs[-20:], track_plot.ys[-20:], marker='+')
            ax.annotate(track_plot.id, (track_plot.xs[-1], track_plot.ys[-1]),
                        (track_plot.xs[-1] + 1, track_plot.ys[-1] + 1))

        fig.canvas.draw()
        fig.canvas.flush_events()


def scalar_to_hex(scalar_value, max_value):
    f = scalar_value / max_value
    a = (1 - f) * 5
    x = math.floor(a)
    y = math.floor(255 * (a - x))
    if x == 0:
        return '#%02x%02x%02x' % (255, y, 0)
    elif x == 1:
        return '#%02x%02x%02x' % (255, 255, 0)
    elif x == 2:
        return '#%02x%02x%02x' % (0, 255, y)
    elif x == 3:
        return '#%02x%02x%02x' % (0, 255, 255)
    elif x == 4:
        return '#%02x%02x%02x' % (y, 0, 255)
    else:  # x == 5:
        return '#%02x%02x%02x' % (255, 0, 255)


if __name__ == "__main__":
    plot_tracks(motion_based_multi_object_tracking('drone_on_background/Samsung_v1.mp4'))
    # plot_tracks_realtime()
