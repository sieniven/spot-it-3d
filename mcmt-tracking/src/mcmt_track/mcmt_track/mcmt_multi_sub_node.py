# basic Python frameworks
import cv2
import math
import time
import numpy as np

# ROS2 libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

# import message interface types
from mcmt_msg.msg import DetectionInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class MultiSubNode(Node):
	"""
	Subscriber node for subscribing to two topic channels from two cameras. This node
	subscribes and gets detection info from the two topics, and synchronizes message
	subscription for the MultiTrackerNode.
	"""
	def __init__(self, camera_idxs):
		super().__init__("MultiSubNode")

		# declare video parameters
		self.bridge = CvBridge()
		self.frame_1 = None
		self.frame_2 = None
		self.good_tracks_1 = None
		self.good_tracks_2 = None
		self.gone_tracks_1 = None
		self.gone_tracks_2 = None
		self.FLAG_1 = False
		self.FLAG_2 = False

		# create subscriber to the topic name "mcmt/detection_info_{cam_index_1}"
		topic_name = "mcmt/detection_info_" + camera_idxs[0]
		self.detect_sub_1 = self.create_subscription(
			DetectionInfo, topic_name, self.sub_callback_1, 10)
		# prevent unused variable warning
		self.detect_sub_1

		# create subscriber to the topic name "mcmt/detection_info_{cam_index_2}"
		topic_name = "mcmt/detection_info_" + camera_idxs[1]
		self.detect_sub_2 = self.create_subscription(
			DetectionInfo, topic_name, self.sub_callback_2, 10)
		# prevent unused variable warning
		self.detect_sub_2


	def sub_callback_1(self, msg):
		try:
			self.frame_1 = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="passthrough")
		except CvBridgeError as e:
			print(e)

		# get gone tracks id
		self.gone_tracks_1 = msg.gonetracks_id

		# get goodtracks list
		total_num_tracks = len(msg.goodtracks_id)
		self.good_tracks_1 = []
		for i in range(total_num_tracks):
			self.good_tracks_1.append([msg.goodtracks_id[i], msg.goodtracks_x[i], msg.goodtracks_y[i]])

		self.FLAG_1 = True

	
	def sub_callback_2(self, msg):
		try:
			self.frame_2 = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="passthrough")
		except CvBridgeError as e:
			print(e)

		# get gone tracks id
		self.gone_tracks_2 = msg.gonetracks_id

		# get goodtracks list
		total_num_tracks = len(msg.goodtracks_id)
		self.good_tracks_2 = []
		for i in range(total_num_tracks):
			self.good_tracks_2.append([msg.goodtracks_id[i], msg.goodtracks_x[i], msg.goodtracks_y[i]])

		self.FLAG_2 = True