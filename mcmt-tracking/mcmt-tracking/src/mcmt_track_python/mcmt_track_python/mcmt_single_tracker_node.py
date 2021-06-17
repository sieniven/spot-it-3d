# basic Python frameworks
import cv2
import math
import time
import numpy as np

# ROS2 libraries
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.clock import Clock

# import message interface types
from mcmt_msg.msg import SingleDetectionInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# local imported code
from mcmt_track_python.mcmt_track_utils import TrackPlot, scalar_to_rgb


class SingleTrackerNode(Node):
	"""
	Single camera tracking node, for processing the single camera tracks' features
	"""
	def __init__(self):
		super().__init__("SingleTrackerNode")
		
		# declare video parameters
		self.bridge = CvBridge()
		self.index = None
		self.timer = time.time()
		
		# declare and get mcmt ROS2 parameters
		self.declare_mcmt_parameters()
		self.get_mcmt_parameters()
		
		# get camera parameters
		cap = cv2.VideoCapture(self.index)
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
		self.good_tracks = []
		self.frame_count = 0
		self.total_num_tracks = None
		self.frame = None
		self.next_id = None

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

		# create subscriber to the topic name "mcmt/detection_info"
		topic_name = "mcmt/detection_info"
		self.detect_sub = self.create_subscription(
			SingleDetectionInfo, topic_name, self.track_callback, 10)
		# prevent unused variable warning
		self.detect_sub

		# create publisher to the topic name "mcmt/track_image"
		topic_name = "mcmt/track_image"
		self.track_pub = self.create_publisher(Image, topic_name, 1000)
		# prevent unused variable warning
		self.track_pub

	
	def declare_mcmt_parameters(self):
		"""
		This function declares our mcmt software parameters as ROS2 parameters
		"""
		self.declare_parameter('IS_REALTIME')
		self.declare_parameter('VIDEO_INPUT')
		self.declare_parameter('FRAME_WIDTH')
		self.declare_parameter('FRAME_HEIGHT')
		self.declare_parameter('OUTPUT_VIDEO_PATH')
		self.declare_parameter('OUTPUT_CSV_PATH')


	def get_mcmt_parameters(self):
		"""
		This function gets the mcmt parameters from the ROS2 paramters
		"""
		# get camera parameters
		self.is_realtime = self.get_parameter('IS_REALTIME').value
		
		if self.is_realtime:
			self.index = int(self.get_parameter('VIDEO_INPUT').value)
		else:
			self.index = self.get_parameter('VIDEO_INPUT').value
		
		self.max_frame_w = self.get_parameter('FRAME_WIDTH').value
		self.max_frame_h = self.get_parameter('FRAME_HEIGHT').value
		self.output_vid_path = self.get_parameter('OUTPUT_VIDEO_PATH').value
		self.output_csv_path = self.get_parameter('OUTPUT_CSV_PATH').value

	
	def track_callback(self, msg):
		"""
		callback function to process the camera's tracks' features
		"""
		start_timer = time.time()

		# get DetectionInfo message
		try:
			self.frame = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="passthrough")
		except CvBridgeError as e:
			print(e)
		
		# get goodtracks list
		self.total_num_tracks = len(msg.goodtracks_id)
		self.good_tracks = []
		for i in range(self.total_num_tracks):
			self.good_tracks.append([msg.goodtracks_id[i], msg.goodtracks_x[i], msg.goodtracks_y[i]])
		print(f"Time take to get message: {time.time() - self.timer}")

		# get track feature variable for each track
		for track in self.good_tracks:
			track_id = track[0]
			centroid_x = track[1]
			centroid_y = track[2]

			if track_id not in self.track_plots_ids:  # First occurrence of the track
				self.track_plots_ids.append(track_id)
				self.track_plots.append(TrackPlot(track_id))

			track_plot = self.track_plots[self.track_plots_ids.index(track_id)]
			track_plot.update((centroid_x, centroid_y), self.frame_count)
			track_plot.calculate_track_feature_variable(self.frame_count, self.fps)

		# plot trackplots
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
		
		# get timer
		end_timer = time.time()
		print(f"Trackplot process took: {end_timer - start_timer}")

		# show and save video tracking frame
		self.frame_count += 1
		self.imshow_resized_dual("Detection", self.frame)
		self.recording.write(self.frame)
		self.timer = time.time()
		cv2.waitKey(1)

	
	def imshow_resized_dual(self, window_name, img):
		"""
		Function to resize and enlarge tracking frame
		"""
		aspect_ratio = img.shape[1] / img.shape[0]

		window_size = (int(1280), int(1280 / aspect_ratio))
		img = cv2.resize(img, window_size, interpolation=cv2.INTER_CUBIC)
		cv2.imshow(window_name, img)