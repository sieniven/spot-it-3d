/**
 * @file mcmt_single_tracker_node.hpp
 * @author Niven Sie, sieniven@gmail.com
 * @author Seah Shao Xuan
 * 
 * This code contains the McmtSingleTrackerNode class that runs our tracking process
 */

#ifndef MCMT_SINGLE_TRACKER_NODE_HPP_
#define MCMT_SINGLE_TRACKER_NODE_HPP_

// opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

// ros2 header files
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

// local header files
#include <mcmt_track/mcmt_track_utils.hpp>
#include <mcmt_msg/msg/single_detection_info.hpp>

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt
{
class McmtSingleTrackerNode : public rclcpp::Node {
	public:
		McmtSingleTrackerNode();
		virtual ~McmtSingleTrackerNode() {}
	
};
}

#endif	// MCMT_SINGLE_TRACKER_NODE_HPP_