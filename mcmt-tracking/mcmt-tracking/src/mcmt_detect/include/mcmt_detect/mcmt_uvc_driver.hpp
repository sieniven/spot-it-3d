/** MCMT UVCDriver Node
 * Author: Niven Sie, sieniven@gmail.com
 * 
 * This code contains the UVCDriver node class that runs our camera, and publish the 
 * raw frames into our ROS2 DDS-RTPS ecosystem.
 */

#ifndef MCMT_UVC_DRIVER_HPP_
#define MCMT_UVC_DRIVER_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

namespace mcmt 
{
class UVCDriver : public rclcpp::Node {
  public:
    UVCDriver(const int & video_device_id, const std::string cam_index);
    void start_record();
    void stop_record();

    std::string topic_name_;
    rclcpp::Node::SharedPtr node_handle_;

  private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    int frame_id_;
    std::string frame_id_str_;

    cv::VideoCapture cap_;
    cv::Mat frame_;

    void publish_image();
    std::string mat_type2encoding(int mat_type);
};
}

#endif    // MCMT_UVC_DRIVER_HPP_