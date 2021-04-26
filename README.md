# Simultaneous Positioning, Observing, Tracking, Identifying Targets in 3D (SPOT-IT 3D)

## 1. Introduction

**SPOT-IT 3D** (Simultaneous Positioning, Observing, Tracking, Identifying Targets in 3D) is a project by Temasek Laboratories @ National University of Singapore. This project aims to utilize a multi-camera surveillance system for real-time multiple target tracking capabilities. This software capability is highly applicable for monitoring specific areas, and some use cases include monitoring airspaces, traffic junctions, etc. 

## 2. Aim

This project aims to develop a methodology of identification, localization, and tracking of **small and fast moving targets**, such as flying drones, using an integrated multiple camera monitoring system. Our study focuses on using **motion-based** features to track targets, instead of traditional tracking algorithms that use appearance-based features by incorporating deep convolutional neural networks for target tracking. As we focus on small and fast moving targets, such as drones, using appearance-based features in this specific case may be difficult, as these targets often appear as small black "dots/blobs" in video frames. This would specifically mean that we use targets’ **trajectory features**, such as the derivatives of displacement, heading and turning angle, to identify and track targets instead.

Thus, our software incorporates the use of a multiple camera surveillance system to track and localize these moving targets. We aim to be able to continuously track drones in monitored airspaces, re-identify these targets between the cameras, and montor their 3-dimensional coordinates when they are inside the monitored space. 

## 3. Capabilities

The current software capabilities are:

1. Track multiple targets in each camera frame. The software implements computer vision techniques such as blob detection, background subtractions, etc. 

2. Implements the use of DeepSort algorithm to continuously track and localize these targets. We use Kalman filtering (KF) and Discriminative Correlation Filter as our state estimation techniques in our target tracking algorithm. Both filters are implemented in parrallel, as these targets move at fast and erratic speeds, and using these filters allow us to better predict their positions for continuous tracking.

3. The software has re-identification capabilities between targets. This would mean that every camera will be able to know that they are tracking the same target, purely based on their kinematic features. We implement a binary classification model that matches and re-identifies these targets between cameras.

4. Estimation of targets' location in 3-dimensional space using triangulation methodology

## 4. Installation Guide

The following step-by-step processing will guide you on the installation process. Our software requires Linux environment to run it. To run it on other OS, please install using Docker.

### 4.1 Install locally

Build our environment using Anaconda. Please install the latest version of Anaconda before continuing.

``` bash
cd mcmt-tracking/mcmt-dockerfiles/
conda env create -f mcmt-track.yml
```

### 4.2 Install using Docker

Build our Docker image using Docker. Please install the latest version of Docker before continuing.

``` bash
cd mcmt-tracking/mcmt-dockerfiles/
sudo docker-compose -f mcmt-tracker-launch.yaml build
```

## 5. Configurations

* To configure the software to run in real-time, go to ***"mcmt-tracking/mcmt_tracking/multi-cam/bin"*** 
* change to cameras = [{camera_index_1}, {camera_index_2}] in the main body. The camera indexes are OpenCV's camera port device indexes. 

* To configure the tracking to run on video files, change cameras list to input the respective video filenames. For example, cameras = [{filename_1}, {filename_2}].


## 6. Run

### 6.1 Run locally

``` bash
cd mcmt-tracking/mcmt-tracking/bin
./mcmt_tracker.sh
```

### 6.2 Run on Docker

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
