# ROS2 libraries
import rclpy
import cv2

# local imported code
from mcmt_track.mcmt_single_tracker_node import SingleTrackerNode
from mcmt_track.mcmt_plot_utils import plot_track_feature_variable, export_data_as_csv



def main():
	"""
	Pipeline to execute SingleTrackerNode. Handles SIGINT signal to gracefully shutdown
	the SingleTrackerNode process and save the live tracking video frames.
	"""
	try:
		rclpy.init()
		tracker = SingleTrackerNode()
		rclpy.spin(tracker)

	except KeyboardInterrupt:
		print("Saving video....")
		tracker.recording.release()
		cv2.destroyAllWindows()

		# plot track feature variable values and export the data into csv file
		plot_track_feature_variable(tracker.track_plots)
		export_data_as_csv(tracker.track_plots, tracker.output_csv_path)

	finally:
		tracker.destroy_node()
		rclpy.shutdown()