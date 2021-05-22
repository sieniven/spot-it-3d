# Simultaneous Positioning, Observing, Tracking, Identifying Targets in 3D (SPOT-IT 3D)

## 1. Introduction

**SPOT-IT 3D** (Simultaneous Positioning, Observing, Tracking, Identifying Targets in 3D) is a project by Mechanical Engineering @ National University of Singapore. This project aims to develop a software that utilizes a multi-camera surveillance system for real-time multiple target tracking capabilities. This software capability is highly applicable for monitoring specific areas, and some use cases include monitoring airspaces, traffic junctions, etc.


## 2. Table of Contents

- [3. Publications](#3-publications)
- [4. Aim](#4-aim)
- [5. Capabilities](#5-capabilities)
- [6. Installation Guide](#6-installation-guide)
  * [6.1 Install and build locally](#61-install-and-build-locally)
  * [6.2 Install and build using Docker](#62-install-and-build-using-docker)
- [7. Configurations](#7-configurations)
- [8. Run](#8-run)
  * [8.1 Run software locally](#81-run-software-locally)
  * [8.2 Run software on Docker](#82-run-software-on-docker)
  * [1. Ensure that camera ports are configured correctly](#1-ensure-that-camera-ports-are-configured-correctly)
- [9. Acknowledgements](#9-acknowledgements)


## 3. Publications

![Software Demo](./docs/software_demo.gif)

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

The current software runs in a Dockerized environment and it is built on Robot Operating System (ROS2) framework as our message passing ecosystem. We use ROS2's Data Distribution Service (DDS) as our centralized system of communications for message passing. DDS uses RTPS (Real-Time Publish-Subscribe) protocol and acts as a central broker for data distribution. The software is built in C++ for high runtime performance capabilities, and we use Python for data management and matching of tracks' feature values.

Our software mainly runs on two process and are handled in the respective ROS2 nodes. They are the **detection and tracking process**, and the **re-identification and trackplot process**. They are built on the two main ROS2 packages in our software stack, *"MCMT Detect"* and *"MCMT Track"*. 

Our software capabilities include:

1. Open and read from single / double camera sensors, and obtain live camera frames.

2. Apply image processing techniques to remove background noise, by distinguisihing between foreground and background. We apply background subtraction in our image processing pipeline, and subsequently identify contours that do not conform to a minimum level of circularity. We deem these contours as background, and subtract them out of our image frames.

3. Apply morphological operations such as dilation and erosion to remove noise and enhance our detection.

4. Apply thresholding and binarization of our frames to obtained masked frames to detect targets using blob detection.

5. Use of Extemded Kalman filtering (KF) and Discriminative Correlation Filter (DCF) as our state estimation techniques to predict our targets' next known location in the following frames.Both filters are implemented in our tracking pipeline, as these targets move at fast and erratic speeds, and using these filters allow us to better predict their positions for continuous tracking.

6. Implements the use of Hungarian / Munkre's algorithm to match our detections to tracks. The matching algorithm is based on a 2 dimensional cost matrix of tracks and detections, where the cost is computed by comparing every detection's euclidean distance away from each tracked target predicted location from our DCF and EKF filters. 

7. The software has re-identification capabilities of targets between cameras. This would mean that every camera will be able to know that they are tracking the same target, purely based on their kinematic features. We implement cross-correlation matching of tracks' trajectory features, and obtain a 2-dimensional correlation score between the cameras' tracks.

8. Apply graph matching algorithm for geomentry-based identification using relative coordinates. The software initializes a complete bipartite graph, that calculates the maximum sum of the weights of the edges that span across the two disjoint groups of the complete bipartite graph. We obtain a geometry-based identification using relative coordinates, and serves as a secondary verification in our re-identification process.

9. Estimation of targets' location in 3-dimensional space using triangulation and stereo camera methodology. We use disparities between the two camera frames to obtain depth estimations of the tracked targets.


## 6. Installation Guide

The following step-by-step processing will guide you on the installation process. Our software runs on Linux environment with ROS2 Eloquent. To run it on other OS, please install and run using Docker. The following step-by-step instructions will guide you on the installation process.

### 6.1 Install and build locally

1. Pull spot-it-3d repository

	``` bash
	get clone https://github.com/sieniven/spot-it-3d.git
	```

2. Install ROS2 Eloquent (Bare Bones version). Refer to: https://index.ros.org/doc/ros2/Installation/Eloquent/Linux-Install-Debians/

3. Install required dependencies. 

	``` bash
	cd ~/spot-it-3d/mcmt-tracking/bin/
	python3 -m pip install -r requirements.txt
	```

4. Install Boost libraries
	
	``` bash
	sudo apt-get update && sudo apt-get install libboost-all-dev 
	```

5. Install OpenCV dependencies
	
	``` bash
	sudo sudo apt-get purge *libopencv*
	sudo apt-get install build-essential cmake git unzip pkg-config
	sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
	sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
	sudo apt-get install libgtk2.0-dev libcanberra-gtk*
	sudo apt-get install python3-dev python3-numpy python3-pip
	sudo apt-get install libxvidcore-dev libx264-dev libgtk-3-dev
	sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev
	sudo apt-get install libv4l-dev v4l-utils
	sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
	sudo apt-get install libavresample-dev libvorbis-dev libxine2-dev
	sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev
	sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev
	sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev
	sudo apt-get install liblapack-dev libeigen3-dev gfortran
	sudo apt-get install libhdf5-dev protobuf-compiler
	sudo apt-get install libprotobuf-dev libgoogle-glog-dev libgflags-dev
	```

6. Install OpenCV and OpenCV_Contrib libraries

	* Install latest version, refer to installation guide here: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html.
	* Install OpenCV and OpenCV_Contrib

	``` bash
	wget https://github.com/opencv/opencv/archive/4.5.2.zip -O opencv-4.5.2.zip
	wget https://github.com/opencv/opencv_contrib/archive/4.5.2.zip -O opencv-contrib-4.5.2.zip
	unzip opencv-4.5.2.zip
	unzip opencv-contrib-4.5.2.zip
	mv opencv-4.5.2 opencv
	mv opencv_contrib-4.5.2 opencv_contrib
	mkdir -p build && cd build && cmake -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv
	cd build && make -j5
	cd build && sudo make install
	```

7. After all dependencies above have been configured and installed, build MCMT software:

	``` bash
	cd ~/spot-it-3d/mcmt-tracking/mcmt-tracking/
	./mcmt_build.sh
	```

### 6.2 Install and build using Docker

1. Pull spot-it-3d repository

``` bash
get clone https://github.com/sieniven/spot-it-3d.git
```

2. Install Docker Engine. Refer to https://docs.docker.com/engine/install/ to install Docker Engine.

3. Build our Docker image using Docker.

``` bash
cd spot-it-3d/mcmt-dockerfiles/
sudo docker-compose -f mcmt-multi-launch.yaml build
```

## 7. Configurations

* To change the configurations of the software, please use the configurations file ***"config.yaml"***. The file can be found in *"~/spot-it-3d/mcmt-tracking/mcmt-tracking/src/mcmt_bringup/config/config.yaml"*.

* Change IS_REALTIME to True if running the software live. Change IS_REALTIME to False if running the software on video files.

* Use VIDEO_INPUT to set video camera index for single camera detector and tracker nodes. Use VIDEO_INPUT_1 and VIDEO_INPUT_2 to configure the camera sensors for the multi camera detector and tracker nodes. The camera indexes are OpenCV's camera port device indexes.

* When running with Docker, ensure that VIDEO_INPUT_1 and VIDEO_INPUT_2 are set to "0" and "2" respectively.

* To change the image processing parameters, we can also use the **config.yaml** file tune them. Parameters such as video parameters, visibility parameters, background subtraction parameters, and dilation parameters for the software can be tuned here. 

## 8. Run

### 8.1 Run software locally

* To launch the MCMT software, we will launch the MCMT Tracker and the MCMT Detector. Note that the MCMT Tracker has to be launched first, before launching the MCMT Detector.

* To launch the multi-camera sensor system software:
	
	* In the first bash terminal:
		
		``` bash
		cd ~/spot-it-3d/mcmt-tracking/bin
		./mcmt_multi_tracker.sh
		```

	* In the second bash terminal:
		
		``` bash
		cd ~/spot-it-3d/mcmt-tracking/bin
		./mcmt_multi_detector.sh
		```

* To launch the single-camera sensor system software:

	* In the first bash terminal:
		
		``` bash
		cd ~/spot-it-3d/mcmt-tracking/bin
		./mcmt_single_tracker.sh
		```

	* In the second bash terminal:
		
		``` bash
		cd ~/spot-it-3d/mcmt-tracking/bin
		./mcmt_single_detector.sh
		```

### 8.2 Run software on Docker

### 1. Ensure that camera ports are configured correctly

* We will link our server camera sensor ports with our container ports using the docker-compose files. Go into **"mcmt-dockerfiles/mcmt-tracker-launch.yaml"**.
* Link server video device ports with container video device ports under each **"services"**.
* For example, if cameras are connected to computer video ports 0 and 1, the docker-compose **"devices"** column should look like this:
	
	``` bash 
		devices:
		- "/dev/video0:/dev/video0"
		- "/dev/video1:/dev/video2"
	```
2. Launch the software inside the Docker container using Docker-compose. To launch the multi-camera sensor system software:

	``` bash
	cd mcmt-tracking/mcmt-dockerfiles/
	sudo docker-compose -f mcmt-multi-launch.yaml up
	```

3. Similarly, to launch the single-camera sensor system software:

	``` bash
	cd mcmt-tracking/mcmt-dockerfiles/
	sudo docker-compose -f mcmt-single-launch.yaml up
	```


## 9. Acknowledgements

We would like to thank the lead researcher in this project, Dr. Sutthiphong Srigrarom, for his continuous guidance and supervision with the development of this project. We would also like to acknowledge the hard work that everyone who have played a part in developing this software. The main developers for the software are Niven Sie Jun Liang and Seah Shao Xuan. Our research team for this project comprises of:

1. Dr. Sutthiphong Srigrarom (email: spot.srigrarom@nus.edu.sg, GitHub profile: spotkrub)
2. Niven Sie Jun Liang (email: sieniven@gmail.com, GitHub profile: sieniven)
3. Seah Shao Xuan (email: shaoxuan.seah@gmail.com, GitHub profile: seahhorse)