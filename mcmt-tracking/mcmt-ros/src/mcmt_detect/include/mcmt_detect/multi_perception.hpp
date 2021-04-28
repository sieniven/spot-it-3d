// Pipeline to launch UVCDrivers
// Author: Niven Sie, sieniven@gmail.com
// 
// This code contains the main pipeline to launch our multi camera detector,
// and it will launch 2 UVCDriver nodes.

#ifndef MCMT_MULTI_PERCEPTION_HPP_
#define MCMT_MULTI_PERCEPTION_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <list>
#include <array>

int frame_w, frame_h, fps;
double scale_factor, aspect_ratio;
bool downsample;

rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_one_sub;
rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_two_sub;


