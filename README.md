# Simultaneous Positioning, Observing, Tracking, Identifying Targets in 3D (SPOT-IT 3D)

## 1. Introduction

**SPOT-IT 3D** (Simultaneous Positioning, Observing, Tracking, Identifying Targets in 3D) is a project by Mechanical Engineering @ National University of Singapore. This project aims to develop a software that utilizes a multi-camera surveillance system for real-time multiple target tracking capabilities. This software capability is highly applicable for monitoring specific areas, and some use cases include monitoring airspaces, traffic junctions, etc.


## 2. Table of Contents

- [3. Publications](#3-publications)
- [4. Aim](#4-aim)
- [5. Capabilities](#5-capabilities)
- [6. Installation Guide](#6-installation-guide)
  * [6.1 Install locally](#61-install-locally)
  * [6.2 Install using Docker](#62-install-using-docker)
- [7. Configurations](#7-configurations)
- [8. Run](#8-run)
  * [8.1 Run software locally](#81-run-software-locally)
  * [8.2 Run software on Docker](#82-run-software-on-docker)
  * [1. Ensure that camera ports are configured correctly](#1-ensure-that-camera-ports-are-configured-correctly)
- [9. Acknowledgements](#9-acknowledgements)


## 3. Publications

1. Paper on trajectory-based target matching and re-identification between cameras:
	* Niven Sie Jun Liang and Sutthiphong Srigrarom. "Multi-camera multi-target tracking systems with trajectory-based target matching and re-identification." In *2021 IEEE International Conference on Unmanned Aerial Systems (ICUAS)*, IEEE, Athens, Greece, 2021.
	* Link to paper: (To be added into IEEE Xplore soon)

2. Paper on field test validations for using trajectory-based tracking with a multiple camera system for target tracking and 3-dimensional localization:
	* Niven Sie Jun Liang, Sutthiphong Srigrarom and Sunan Huang. "Field test validations of vision-based multi-camera multi-drone tracking and 3D localizing, using concurrent camera pose estimation." In *2021 IEEE 6th International Conference on Control and Robotics Engineering (ICCRE)*, IEEE, Beijing, China, 2021.
	* Link to paper: https://ieeexplore.ieee.org/abstract/document/9358454

3. Paper on state estimation filters and proposed use of implementing multiple state estimation filters in parrallel (Integrated Multiple Model):
	* Sutthiphong Srigrarom, Niven Sie Jun Liang, Jiahe Yi, Kim Hoe Chew, Floorian Holzapfel, Henrik Hesse, Teng Hooi Chan and Jalvin Jiaxiang Chen. "Vision-based drones tracking using correlation filters and Linear Integrated Multiple Model." In *2021 IEEE International Conference on Electrical Engineering/Electronics, Computer, Telecommunications and Information Technology (ECTI-CON)*, IEEE, Chiang Mai, Thailand, 2021.
	* Link to paper: (To be added into IEEE Xplore soon)

4. Paper on integrated kinematic-based detection tracking estimation system for dynamic localization of small aerial vehicles:
	* Sutthiphong Srigrarom, Shawndy Michael Lee, Mengda Lee, Foong Shaohui and Photchara Ratsamee. "An integrated vision-based detection-tracking-estimation system for dynamic localization of small aerial vehicles." In *2020 IEEE 5th International Conference on Control and Robotics Engineering (ICCRE)*, IEEE, Osaka, Japan, 2020.
	* Link to paper: https://ieeexplore.ieee.org/abstract/document/9096259

5. Paper on binocular and stereo cameras for multiple drone detection and 3-dimensional localization:
	* Yi, Jiahe, and Sutthiphong Srigrarom. "Near-Parallel Binocular-Like Camera Pair for Multi-Drone Detection and 3D Localization." In *2020 16th International Conference on Control, Automation, Robotics and Vision (ICARCV)*, pp. 204-210. IEEE, Shenzhen, China, 2020.
	* Link to paper: https://ieeexplore.ieee.org/abstract/document/9305485


## 4. Aim

This project aims to develop a methodology of identification, localization, and tracking of **small and fast moving targets**, such as flying drones, using an integrated multiple camera monitoring system. Our study focuses on using **motion-based** features to track targets, instead of traditional tracking algorithms that use appearance-based features by incorporating deep convolutional neural networks for target tracking. As we focus on small and fast moving targets, such as drones, using appearance-based features in this specific case may be difficult, as these targets often appear as small black "dots/blobs" in video frames. This would specifically mean that we use targets’ **trajectory features**, such as the derivatives of displacement, heading and turning angle, to identify and track targets instead.

Thus, our software incorporates the use of a multiple camera surveillance system to track and localize these moving targets. We aim to be able to continuously track drones in monitored airspaces, re-identify these targets between the cameras, and montor their 3-dimensional coordinates when they are inside the monitored space. 


## 5. Capabilities

The current software capabilities are:

1. Track multiple targets in each camera frame. The software implements computer vision techniques such as blob detection, background subtractions, etc. 

2. Implements the use of DeepSort algorithm to continuously track and localize these targets. We use Kalman filtering (KF) and Discriminative Correlation Filter as our state estimation techniques in our target tracking algorithm. Both filters are implemented in parrallel, as these targets move at fast and erratic speeds, and using these filters allow us to better predict their positions for continuous tracking.

3. The software has re-identification capabilities between targets. This would mean that every camera will be able to know that they are tracking the same target, purely based on their kinematic features. We implement a binary classification model that matches and re-identifies these targets between cameras.

4. Estimation of targets' location in 3-dimensional space using triangulation methodology


## 6. Installation Guide

The following step-by-step processing will guide you on the installation process. Our software requires Linux environment to run it. To run it on other OS, please install using Docker.

### 6.1 Install locally

Build our environment using Anaconda. Please install the latest version of Anaconda before continuing.

``` bash
cd mcmt-tracking/mcmt-dockerfiles/
conda env create -f mcmt-track.yml
```

### 6.2 Install using Docker

Build our Docker image using Docker. Please install the latest version of Docker before continuing.

``` bash
cd mcmt-tracking/mcmt-dockerfiles/
sudo docker-compose -f mcmt-tracker-launch.yaml build
```

## 7. Configurations

* To configure the software to run in real-time, go to *"mcmt-tracking/mcmt_tracking/multi-cam/bin"* 
* change to cameras = [{camera_index_1}, {camera_index_2}] in the main body. The camera indexes are OpenCV's camera port device indexes. 

* To configure the tracking to run on video files, change cameras list to input the respective video filenames. For example, cameras = [{filename_1}, {filename_2}].


## 8. Run

### 8.1 Run software locally

``` bash
cd mcmt-tracking/mcmt-tracking/bin
./mcmt_tracker.sh
```

### 8.2 Run software on Docker

### 1. Ensure that camera ports are configured correctly

* We will link our server ports with our container ports using the docker-compose files. Go into **"mcmt-dockerfiles/mcmt-tracker-launch.yaml"**.
* Link server video device ports with container video device ports under **"services"**.
* For example, if cameras are connected to computer video ports 0 and 1, the docker-compose **"devices"** column should look like this:
	
	``` bash 
		devices:
		- "/dev/video0:/dev/video0"
		- "/dev/video2:/dev/video2"
	```
2. Launch the software inside the Docker container using Docker-compose:

	``` bash
	cd mcmt-tracking/mcmt-dockerfiles/
	sudo docker-compose -f mcmt-tracker-launch.yaml up
	```


## 9. Acknowledgements

We would like to thank the lead researcher in this project, Dr. Sutthiphong Srigrarom, for his continuous guidance and supervision with the development of this project. We would also like to acknowledge the hard work that everyone who have played a part in developing this software. The main developers for the software are Niven Sie Jun Liang, Seah Shao Xuan and Lau Yan Han. Our research team for this project comprises of:

1. Dr. Sutthiphong Srigrarom (email: spot.srigrarom@nus.edu.sg, GitHub profile: spotkrub)
2. Niven Sie Jun Liang (email: sieniven@gmail.com, GitHub profile: sieniven)
3. Seah Shao Xuan (email: shaoxuan.seah@gmail.com, GitHub profile: seahhorse)
4. Lau Yan Han (email: sps08.lauyanhan@gmail.com, GitHub profile: disgruntled-patzer)