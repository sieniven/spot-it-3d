# ROS2 libraries
import rclpy
import cv2

# local imported code
from mcmt_track.mcmt_multi_tracker_node import MultiTrackerNode
from mcmt_track.mcmt_plot_utils import mcmt_plot



def main():
	"""
	Pipeline to execute MultiTrackerNode. Handles SIGINT signal to gracefully shutdown
	the MultiTrackerNode process and save the live tracking video frames.
	"""
	try:
		rclpy.init()
		tracker = MultiTrackerNode()

		while True:
			tracker.track_callback()
			

	except KeyboardInterrupt:
		print("Saving video....")
		tracker.recording.release()
		cv2.destroyAllWindows()

		# plot track feature variable values and export the data into csv file
		# mcmt_plot(tracker.cumulative_tracks[0], tracker.cumulative_tracks[1])

	finally:
		tracker.destroy_node()
		rclpy.shutdown()