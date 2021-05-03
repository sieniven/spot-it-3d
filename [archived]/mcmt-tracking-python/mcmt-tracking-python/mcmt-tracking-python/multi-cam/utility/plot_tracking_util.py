import os
import matplotlib.pyplot as plt
import csv


def delete_track_plots():
	pass


def plot_track_feature_variable(track_plots):
	fig, ax = plt.subplots()

	ax.set_yscale('log')

	ax.set_xlabel('Frame Number')
	ax.set_ylabel('Track Feature Variable Xj')

	for track_plot in track_plots:
		# Check that there's track feature variable data at all
		if track_plot.track_feature_variable.size != 0:
			ax.plot(track_plot.frameNos[2:], track_plot.track_feature_variable)
			ax.text(track_plot.frameNos[-1], track_plot.track_feature_variable[-1],
					f"{track_plot.id}")

	plt.show()


def export_data_as_csv(track_plots):
	data = []

	max_frame = 0

	for track_plot in track_plots:
		# Check that there's track feature variable data at all
		if track_plot.track_feature_variable.size != 0:

			# Check if the data has enough rows to accommodate the data
			if track_plot.frameNos[-1] > max_frame:
				# Add the required number of extra rows
				data.extend([[i] for i in range(max_frame, track_plot.frameNos[-1] + 1)])
				max_frame = track_plot.frameNos[-1]

			for idx, frame in enumerate(track_plot.frameNos):
				# Track feature variable is only calculated on the 3rd frame
				if idx >= 2:
					data[frame - 1].extend([track_plot.id,
											track_plot.xs[idx],
											track_plot.ys[idx],
											track_plot.track_feature_variable[idx - 2]])

	with open(os.path.join(os.path.expanduser('~'),
		"spot-it-3d/data", "data_out.csv"), 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for line in data:
			writer.writerow(line)


def rgb_to_hex(r, g, b):
	return '#%02x%02x%02x' % (r, g, b)


def mcmt_plot(camera_0_tracks, camera_1_tracks):
	fig, axs = plt.subplots(2, 1, figsize=(6, 9))

	with open(os.path.join(os.path.expanduser('~'),
		"spot-it-3d/data", "data_out_0.csv"), 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for row in camera_0_tracks.output_log:
			writer.writerow(row)

	with open(os.path.join(os.path.expanduser('~'),
	"spot-it-3d/data", "data_out_1.csv"), 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for row in camera_1_tracks.output_log:
			writer.writerow(row)

	axs[0].set_yscale('log')
	axs[0].set_xlabel('Frame Number')
	axs[0].set_ylabel('Track Feature Variable Xj')

	axs[1].set_yscale('log')
	axs[1].set_xlabel('Frame Number')
	axs[1].set_ylabel('Track Feature Variable Xj')

	for track_plot in camera_0_tracks.track_plots:
		# Check that there's track feature variable data at all
		if track_plot.track_feature_variable.size != 0:
			feature_size = track_plot.track_feature_variable.size + 1
			axs[0].plot(track_plot.frameNos[:-feature_size:-1], track_plot.track_feature_variable)
			axs[0].text(track_plot.frameNos[-1], track_plot.track_feature_variable[-1], f"{track_plot.id}")

	for track_plot in camera_0_tracks.track_plots:
		# Check that there's track feature variable data at all
		if track_plot.track_feature_variable.size != 0:
			feature_size = track_plot.track_feature_variable.size + 1
			axs[1].plot(track_plot.frameNos[:-feature_size:-1], track_plot.track_feature_variable)
			axs[1].text(track_plot.frameNos[-1], track_plot.track_feature_variable[-1], f"{track_plot.id}")

	fig.tight_layout()
	plt.show()