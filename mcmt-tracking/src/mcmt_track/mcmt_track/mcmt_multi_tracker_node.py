# basic Python frameworks
import cv2
import math
import time
import numpy as np

# ROS2 libraries
from rclpy.node import Node

# ROS2 message interfaces
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
from cv_bridge import CvBridge, CvBridgeError
from mcmt_msg.msg import DetectionInfo

# local imported code
from mcmt_track.mcmt_track_utils import CameraTracks, TrackPlot, combine_track_plots, scalar_to_rgb


class MultiTrackerNode(Node):
	"""
	Multi camera tracking node, for processing 2 cameras' tracks' features, re-identification and
	matching of tracks between the two cameras.
	"""
	def __init__(self):
		super().__init__("MultiTrackerNode")

		# declare video parameters
		self.filenames = None
		self.bridge = CvBridge()

		self.declare_mcmt_parameters()
		self.get_mcmt_parameters()

		# get camera parameters
		cap = cv2.VideoCapture(self.filenames[0])
		self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
		self.aspect_ratio = self.frame_w / self.frame_h
		self.fps = int(cap.get(cv2.CAP_PROP_FPS))
		if self.frame_w * self.frame_h > (self.max_frame_w * self.max_frame_h):
			self.frame_w = 1920
			self.frame_h = int(1920 / self.aspect_ratio)
			self.scale_factor = math.sqrt(self.frame_w ** 2 + self.frame_h ** 2) / math.sqrt(848 ** 2 + 480 ** 2)
		
		self.recording = cv2.VideoWriter(self.output_vid_path,
										cv2.VideoWriter_fourcc(*'mp4v'),
										self.fps, (self.frame_w, self.frame_h))
		cap.release()

		# data from detector
		self.good_tracks = [None, None]
		self.filter_good_tracks = [None, None]
		self.frame = [None, None]
		self.dead_tracks = [None, None]
		self.frame_count = 0
		self.timer = time.time()

		# tracking variables
		self.next_id = 0
		self.combine_frame = None
		self.origin = np.array([0, 0])
		self.cumulative_tracks = [CameraTracks(0), CameraTracks(1)]
		self.total_tracks = [0, 0]
		self.matching_dict = [{}, {}]

		# initialize plotting parameters
		self.plot_history = 200
		self.colours = [''] * self.plot_history
		for i in range(self.plot_history):
			self.colours[i] = scalar_to_rgb(i, self.plot_history)
		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.font_scale = 0.5

		# create publisher to the topic name "mcmt/track_image"
		topic_name = "mcmt/track_image"
		self.track_pub = self.create_publisher(Image, topic_name, 10)
		# prevent unused variable warning
		self.track_pub

		# create subscribers to two camera detection info topics
		topic_name = "mcmt/detection_info_" + self.cam_indexes[0]
		self.detection_sub_1 = Subscriber(self, DetectionInfo, topic_name)
		topic_name = "mcmt/detection_info_" + self.cam_indexes[1]
		self.detection_sub_2 = Subscriber(self, DetectionInfo, topic_name)
		
		# create approximate synchronized subscriber to synchronize message callbacks
		self.sync_sub = ApproximateTimeSynchronizer([self.detection_sub_1, self.detection_sub_2], 1000, 0.1)
		self.sync_sub.registerCallback(self.track_callback)


	def declare_mcmt_parameters(self):
		"""
		This function declares our mcmt software parameters as ROS2 parameters
		"""
		self.declare_parameter('IS_REALTIME')
		self.declare_parameter('VIDEO_INPUT_1')
		self.declare_parameter('VIDEO_INPUT_2')
		self.declare_parameter('CAMERA_INDEX_1')
		self.declare_parameter('CAMERA_INDEX_2')
		self.declare_parameter('FRAME_WIDTH')
		self.declare_parameter('FRAME_HEIGHT')
		self.declare_parameter('OUTPUT_VIDEO_PATH')
		self.declare_parameter('OUTPUT_CSV_PATH_1')
		self.declare_parameter('OUTPUT_CSV_PATH_2')


	def get_mcmt_parameters(self):
		"""
		This function gets the mcmt parameters from the ROS2 parameters
		"""
		# get camera parameters
		self.is_realtime = self.get_parameter('IS_REALTIME').value
		self.cam_indexes = [self.get_parameter('CAMERA_INDEX_1').value, self.get_parameter('CAMERA_INDEX_2').value]

		if self.is_realtime:
			self.filenames = [int(self.get_parameter('VIDEO_INPUT_1').value), int(self.get_parameter('VIDEO_INPUT_2').value)]
		else:
			self.filenames = [self.get_parameter('VIDEO_INPUT_1').value, self.get_parameter('VIDEO_INPUT_2').value]
		
		self.max_frame_w = self.get_parameter('FRAME_WIDTH').value
		self.max_frame_h = self.get_parameter('FRAME_HEIGHT').value
		self.output_vid_path = self.get_parameter('OUTPUT_VIDEO_PATH').value
		self.output_csv_path_1 = self.get_parameter('OUTPUT_CSV_PATH_1').value
		self.output_csv_path_2 = self.get_parameter('OUTPUT_CSV_PATH_2').value


	def track_callback(self, msg_1, msg_2):
		"""
		Main pipeline for tracker node callback. Pipeline includes:
		1.
		2.
		3.
		"""
		start_timer = time.time()
		self.process_msg_info(msg_1, msg_2)
		print(f"Time take to get message: {time.time() - self.timer}")

		# new entry for cumulative track lists
		self.cumulative_tracks[0].output_log.append([])
		self.cumulative_tracks[1].output_log.append([])

		# create filter copy of good_tracks list
		self.filter_good_tracks[0] = list(self.good_tracks[0])
		self.filter_good_tracks[1] = list(self.good_tracks[1])
		
		for i in range(2):
			self.update_cumulative_tracks(i)

		self.process_new_tracks(0, 1)
		self.process_new_tracks(1, 0)

		self.calculate_3D()

		# plotting track plot trajectories into frame
		for frame in range(2):
			for track_plot in self.cumulative_tracks[frame].track_plots.values():
				if (self.frame_count - track_plot.lastSeen) <= self.fps:
					shown_indexes = np.where(np.logical_and(track_plot.frameNos > self.frame_count - self.plot_history,
															track_plot.frameNos <= self.frame_count))[0]
					for idx in shown_indexes:
						cv2.circle(self.frame[frame], (track_plot.xs[idx] - self.origin[0], track_plot.ys[idx] - self.origin[1]), 3,
								self.colours[track_plot.frameNos[idx] - self.frame_count + self.plot_history - 1][::-1], -1)

					if len(shown_indexes) != 0:
						cv2.putText(self.frame[frame], f"ID: {track_plot.id}",
									(track_plot.xs[idx] - self.origin[0], track_plot.ys[idx]
									- self.origin[1] + 15), self.font, self.font_scale, (0, 0, 255), 1, cv2.LINE_AA)

						if track_plot.xyz != None:

							cv2.putText(self.frame[frame], f"X: {track_plot.xyz[0]}",
										(track_plot.xs[idx] - self.origin[0], track_plot.ys[idx]
										- self.origin[1] + 30), self.font, self.font_scale, (255, 0, 0), 1, cv2.LINE_AA)

							cv2.putText(self.frame[frame], f"Y: {track_plot.xyz[1]}",
										(track_plot.xs[idx] - self.origin[0], track_plot.ys[idx]
										- self.origin[1] + 45), self.font, self.font_scale, (255, 0, 0), 1, cv2.LINE_AA)

							cv2.putText(self.frame[frame], f"Z: {track_plot.xyz[2]}",
										(track_plot.xs[idx] - self.origin[0], track_plot.ys[idx]
										- self.origin[1] + 60), self.font, self.font_scale, (255, 0, 0), 1, cv2.LINE_AA)

		# get trackplot process timer
		end_timer = time.time()
		# print(f"Trackplot process took: {end_timer - start_timer}")

		# reset sub node flags and get next frame_count
		self.frame_count += 1

		# show and save video combined tracking frame
		self.combine_frame = np.hstack((self.frame[0], self.frame[1]))
		self.imshow_resized_dual("Detection", self.combine_frame)
		self.recording.write(self.combine_frame)
		self.timer = time.time()
		cv2.waitKey(1)

	
	def process_msg_info(self, msg_1, msg_2):
		try:
			frame_1 = self.bridge.imgmsg_to_cv2(msg_1.image, desired_encoding="passthrough")
		except CvBridgeError as e:
			print(e)

		# get gone tracks id
		gone_tracks_1 = msg_1.gonetracks_id

		# get goodtracks list
		total_num_tracks = len(msg_1.goodtracks_id)
		good_tracks_1 = []
		for i in range(total_num_tracks):
			good_tracks_1.append([msg_1.goodtracks_id[i], msg_1.goodtracks_x[i], msg_1.goodtracks_y[i]])

		try:
			frame_2 = self.bridge.imgmsg_to_cv2(msg_2.image, desired_encoding="passthrough")
		except CvBridgeError as e:
			print(e)

		# get gone tracks id
		gone_tracks_2 = msg_2.gonetracks_id

		# get goodtracks list
		total_num_tracks = len(msg_2.goodtracks_id)
		good_tracks_2 = []
		for i in range(total_num_tracks):
			good_tracks_2.append([msg_2.goodtracks_id[i], msg_2.goodtracks_x[i], msg_2.goodtracks_y[i]])

		# get frames
		self.frame = [frame_1, frame_2]

		# get good tracks
		self.good_tracks = [good_tracks_1, good_tracks_2]

		# get gone tracks
		self.dead_tracks = [gone_tracks_1, gone_tracks_2]


	def update_cumulative_tracks(self, index):
		"""
		Add func description
		"""
		for track in self.good_tracks[index]:
			track_id = track[0]
			centroid_x = track[1]
			centroid_y = track[2]
			self.cumulative_tracks[index].output_log[self.frame_count].extend([track_id, centroid_x, centroid_y])

			# occurance of a new track
			if track_id not in self.cumulative_tracks[index].track_plots_ids and track_id not in self.matching_dict[index]:
				self.cumulative_tracks[index].track_new_plots[track_id] = TrackPlot(track_id)
				self.matching_dict[index][track_id] = track_id


	def process_new_tracks(self, index, alt):
		"""
		Add func description
		"""
		self.get_total_number_of_tracks()
		corrValues = {}
		removeSet = set()
		
		"""
		METHOD 1: Track Feature Variable (best for low #)
		"""
		row = 0

		for track in self.good_tracks[index]:
			# Extract details from the track
			track_id = track[0]
			centroid_x = track[1]
			centroid_y = track[2]

			if self.matching_dict[index][track_id] not in self.cumulative_tracks[index].track_plots:
				corrValues[track_id] = {}

				# Update track_new_plots with centroid and feature variable of every new frame
				track_plot = self.cumulative_tracks[index].track_new_plots[track_id]
				track_plot.update([centroid_x, centroid_y], self.frame_count)
				track_plot.calculate_track_feature_variable(self.frame_count, self.fps)

				# if track is not a new track, we use 90 frames as the minimum requirement before matching occurs
				if track_plot.frameNos.size >= 90 and track_plot.track_feature_variable.size >= 90 and (
						math.sqrt(np.sum(np.square(track_plot.track_feature_variable)))) != 0:
					
					# normalization of cross correlation value
					track_plot_normalize_xj = self.normalise_track_plot(track_plot)
					# track_plot.update_other_tracks(self.good_tracks[index])

					# look into 2nd camera's new tracks (new tracks first)
					for alt_track_plot in self.cumulative_tracks[alt].track_new_plots.values():
						# track in 2nd camera must fulfills requirements to have at least 90 frames to match
						if alt_track_plot.frameNos.size >= 90 and alt_track_plot.track_feature_variable.size >= 90 and (
								math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable)))) != 0:
							
							# normalization of cross correlation value
							alt_track_plot_normalize_xj = self.normalise_track_plot(alt_track_plot)

							r_value = max(np.correlate(track_plot_normalize_xj,
														alt_track_plot_normalize_xj, mode='full'))

							if r_value > 0.7:
								corrValues[track_id][alt_track_plot.id] = r_value

					# look into other camera's matched tracks list (old tracks last)
					for alt_track_plot in self.cumulative_tracks[alt].track_plots.values():
						
						eligibility_flag = True
						
						# do not consider dead tracks from the other camera
						for dead_track_id in self.dead_tracks[alt]:
							if dead_track_id in self.matching_dict[alt] and self.matching_dict[alt][dead_track_id] == alt_track_plot.id:
								eligibility_flag = False  # 2nd camera's track has already been lost. skip the process of matching for this track
								break

						# test to see if alternate camera's track is currently being matched with current camera                        
						for alt_tracker in self.good_tracks[index]:
							if alt_track_plot.id == self.matching_dict[index][alt_tracker[0]]:
								eligibility_flag = False  # 2nd camera's track has already been matched. skip the process of matching for this track
								break

						if eligibility_flag is True and (math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable)))) != 0:

							
							alt_track_plot_normalize_xj = self.normalise_track_plot(alt_track_plot)
							
							r_value = max(np.correlate(track_plot_normalize_xj,
														alt_track_plot_normalize_xj, mode='full'))

							if r_value > 0.7:
								corrValues[track_id][alt_track_plot.id] = r_value

				row += 1

			# if track is already a matched current track
			else:
				track_plot = self.cumulative_tracks[index].track_plots[self.matching_dict[index][track_id]]
				track_plot.update((centroid_x, centroid_y), self.frame_count)
				track_plot.calculate_track_feature_variable(self.frame_count, self.fps)
				self.filter_good_tracks[index].pop(row)


		"""
		METHOD 2: Relative Polar Coordinates (best for high #)
		"""

		# row = 0

		# for track in self.good_tracks[index]:
		# 	# Extract details from the track
		# 	track_id = track[0]
		# 	centroid_x = track[1]
		# 	centroid_y = track[2]

		# 	if self.matching_dict[index][track_id] not in self.cumulative_tracks[index].track_plots:
		# 		corrValues[track_id] = {}

		# 		# Update track_new_plots with centroid and feature variable of every new frame
		# 		track_plot = self.cumulative_tracks[index].track_new_plots[track_id]
		# 		track_plot.update([centroid_x, centroid_y], self.frame_count)
		# 		track_plot.calculate_track_feature_variable(self.frame_count, self.fps)

		# 		# if track is not a new track, we use 90 frames as the minimum requirement before matching occurs
		# 		# if track_plot.frameNos.size >= 90 and track_plot.track_feature_variable.size >= 90 and (
		# 		#         math.sqrt(np.sum(np.square(track_plot.track_feature_variable)))) != 0:
					
		# 		# update location of other tracks
		# 		track_plot.update_other_tracks(self.cumulative_tracks[index])

		# 		# look into 2nd camera's new tracks (new tracks first)
		# 		for alt_track_plot in self.cumulative_tracks[alt].track_new_plots.values():
		# 			# track in 2nd camera must fulfills requirements to have at least 90 frames to match
		# 			# if alt_track_plot.frameNos.size >= 90 and alt_track_plot.track_feature_variable.size >= 90 and (
		# 			#         math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable)))) != 0:
						
		# 			# update location of other tracks
		# 			alt_track_plot.update_other_tracks(self.cumulative_tracks[alt])

		# 			relative_strength = self.calculate_strength(track_plot.other_tracks, alt_track_plot.other_tracks)

		# 			corrValues[track_id][alt_track_plot.id] = relative_strength

		# 		# look into other camera's matched tracks list (old tracks last)
		# 		for alt_track_plot in self.cumulative_tracks[alt].track_plots.values():
					
		# 			eligibility_flag = True
					
		# 			# do not consider dead tracks from the other camera
		# 			for dead_track_id in self.dead_tracks[alt]:
		# 				if dead_track_id in self.matching_dict[alt] and self.matching_dict[alt][dead_track_id] == alt_track_plot.id:
		# 					eligibility_flag = False  # 2nd camera's track has already been lost. skip the process of matching for this track
		# 					break

		# 			# test to see if alternate camera's track is currently being matched with current camera
		# 			for alt_tracker in self.good_tracks[index]:
		# 				if alt_track_plot.id == self.matching_dict[index][alt_tracker[0]]:
		# 					eligibility_flag = False  # 2nd camera's track has already been matched. skip the process of matching for this track
		# 					break

		# 			if eligibility_flag is True: # and (math.sqrt(np.sum(np.square(alt_track_plot.track_feature_variable)))) != 0:
						
		# 				# update location of other tracks
		# 				alt_track_plot.update_other_tracks(self.cumulative_tracks[alt])

		# 				relative_strength = self.calculate_strength(track_plot.other_tracks, alt_track_plot.other_tracks)

		# 				corrValues[track_id][alt_track_plot.id] = relative_strength

		# 		row += 1

		# 	# if track is already a matched current track
		# 	else:
		# 		track_plot = self.cumulative_tracks[index].track_plots[self.matching_dict[index][track_id]]
		# 		track_plot.update((centroid_x, centroid_y), self.frame_count)
		# 		track_plot.calculate_track_feature_variable(self.frame_count, self.fps)
		# 		self.filter_good_tracks[index].pop(row)

		# matching process, finding local maximas in the cross correlation results
		# find local maximas in corrValues. for every iteration, we search every valid track to be re-id from current camera

		for x in self.filter_good_tracks[index]:
			
			maxValues = {}
			maxID = -1
			global_max_flag = 0

			# for the selected max track in the 2nd camera, we check to see if the track has a higher
			# cross correlation value with another track in current camera

			while global_max_flag == 0 and len(corrValues[x[0]]) != 0:
				maxID = max(corrValues[x[0]])
				maxValue = corrValues[x[0]][maxID]

				# search through current camera's tracks again, for the selected track that we wish to re-id with.
				# we can note that if there is a track in the current camera that has a higher cross correlation value
				# than the track we wish to match with, then the matching will not occur.
				for x_1 in self.filter_good_tracks[index]:
					if maxID in corrValues[x_1[0]] and corrValues[x_1[0]][maxID] > maxValue:
						maxValues.pop(maxID)
						global_max_flag = 1
						break

				if global_max_flag == 1:
					# there existed a value larger than the current maxValue. thus, re-id cannot occur
					global_max_flag = 0
					continue
				else:
					# went through the whole loop without breaking, thus it is the maximum value. re-id can occur
					global_max_flag = 2

			if global_max_flag == 2:  # re-id process

				print("matching")

				# if track is in 2nd camera's new track list
				if maxID != -1 and maxID in self.cumulative_tracks[alt].track_new_plots.keys():
					# remove track plot in new tracks' list and add into matched tracks' list for alternate camera

					self.cumulative_tracks[alt].track_new_plots[maxID].id = self.next_id
					self.cumulative_tracks[alt].track_plots[self.next_id] = self.cumulative_tracks[alt].track_new_plots[maxID]

					# update dictionary matching
					self.matching_dict[alt][maxID] = self.next_id

					# self.removeList.append(alt_track_id)
					removeSet.add(maxID)

					# remove track plot in new tracks' list and add into matched tracks' list for current camera
					track_id = x[0]
					track_plot = self.cumulative_tracks[index].track_new_plots[track_id]
					track_plot.id = self.next_id

					self.cumulative_tracks[index].track_plots[self.next_id] = track_plot
					self.cumulative_tracks[index].track_new_plots.pop(track_id)
					
					# update dictionary matching list
					self.matching_dict[index][track_id] = self.next_id

					self.next_id += 1

				# if track is in 2nd camera's matched track list
				else:
					track_id = x[0]
					track_plot = self.cumulative_tracks[index].track_new_plots[track_id]
					track_plot.id = self.cumulative_tracks[alt].track_plots[maxID].id

					# update track plot in the original track ID
					# check this
					combine_track_plots(track_plot.id, self.cumulative_tracks[index], track_plot, self.frame_count)

					# remove track plot in new tracks' list
					self.cumulative_tracks[index].track_new_plots.pop(track_id)

					# update dictionary matching list
					self.matching_dict[index][track_id] = track_plot.id

		for remove_id in removeSet:
			self.cumulative_tracks[alt].track_new_plots.pop(remove_id)


	def get_total_number_of_tracks(self):
		"""
		Add func description
		"""
		self.total_tracks[0] = len(self.cumulative_tracks[0].track_new_plots) + len(self.cumulative_tracks[0].track_plots)
		self.total_tracks[1] = len(self.cumulative_tracks[1].track_new_plots) + len(self.cumulative_tracks[1].track_plots)


	def normalise_track_plot(self, track_plot):
		"""
		Add func description
		"""
		# method 1:
		# track_plot_normalize_xj = (track_plot.track_feature_variable) / (
		# math.sqrt(np.sum(np.square(track_plot.track_feature_variable))))

		# method 2:
		return (track_plot.track_feature_variable - np.mean(
			track_plot.track_feature_variable)) / (np.std(track_plot.track_feature_variable) * math.sqrt(len(
			track_plot.track_feature_variable)))


	def calculate_3D(self):
		"""
		Add func description
		"""
		# fx = parm.LENS_FX
		# cx = parm.LENS_CX
		# fy = parm.LENS_FY
		# cy = parm.LENS_CY

		# B = parm.MULTICAM_B

		fx = 1454.6
		cx = 960.9
		fy = 1450.3
		cy = 534.7

		B = 1.5

		epsilon = 7

		for matched_id in self.cumulative_tracks[0].track_plots.keys():
			track_plot_0 = self.cumulative_tracks[0].track_plots[matched_id]
			track_plot_1 = self.cumulative_tracks[1].track_plots[matched_id]

			if track_plot_0.lastSeen == self.frame_count and track_plot_1.lastSeen == self.frame_count:
				x_L = track_plot_0.xs[-1]
				y_L = track_plot_0.ys[-1]
				x_R = track_plot_1.xs[-1]
				y_R = track_plot_1.ys[-1]

				alpha_L = np.arctan2(x_L - cx, fx) / np.pi * 180
				alpha_R = np.arctan2(x_R - cx, fx) / np.pi * 180

				gamma = epsilon + alpha_L - alpha_R

				Z = B / (np.tan((alpha_L + epsilon / 2) * (np.pi / 180)) - np.tan((alpha_L + -epsilon / 2) * (np.pi / 180)))
				X = (Z * np.tan((alpha_L + epsilon / 2) * (np.pi / 180)) - B / 2
						+ Z * np.tan((alpha_R + -epsilon / 2) * (np.pi / 180)) + B / 2) / 2
				Y = (Z * -(y_L - cy) / fy + Z * -(y_R - cy) / fy) / 2

				tilt = 10 * np.pi / 180
				R = np.array([[1, 0, 0],
								[0, np.cos(tilt), np.sin(tilt)],
								[0, -np.sin(tilt), np.cos(tilt)]])

				[X, Y, Z] = np.matmul(R, np.array([X, Y, Z]))

				Y += 1

				X = np.round(X, 2)
				Y = np.round(Y, 2)
				Z = np.round(Z, 2)

				track_plot_0.xyz = (X, Y, Z)
				track_plot_1.xyz = (X, Y, Z)
			
			else:
				track_plot_0.xyz = None
				track_plot_1.xyz = None


	def calculate_strength(self, other_tracks_0, other_tracks_1):
		"""
		Add func description
		"""
		relative_strength = 0
		for a_polar in other_tracks_0:
			(a_angle, a_dist) = a_polar
			for b_polar in other_tracks_1:
				(b_angle, b_dist) = b_polar
				
				delta_angle = (a_angle - b_angle) / (2 * math.pi)
				angle_factor = 1 / (delta_angle * delta_angle) 
				dist_factor = min(a_dist, b_dist) / max(a_dist, b_dist)

				relative_strength += angle_factor * pow(dist_factor, 4)
		
		return relative_strength


	def imshow_resized_dual(self, window_name, img):
		"""
		Add func description
		"""
		aspect_ratio = img.shape[1] / img.shape[0]

		window_size = (int(1920), int(1920 / aspect_ratio))
		img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
		cv2.imshow(window_name, img)