FROM --platform=linux/arm64 ros:foxy-ros-base-focal
LABEL maintainer="Niven"

# setup timezone
RUN echo 'Asia/Singapore' > /etc/timezone \
    && rm /etc/localtime \ 
    && ln -s /usr/share/zoneinfo/Asia/Singapore /etc/localtime

# install core linux tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils lsb-release sudo unzip wget ssh vim curl git pkg-config \
	libcanberra-gtk-module libcanberra-gtk3-module build-essential gcc \
	libfreetype6-dev libpng-dev libhdf5-serial-dev libcurl3-dev rsync \
	software-properties-common unzip zip zlib1g-dev apt-utils lsb-release \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev python3-numpy libavcodec-dev \
    libavformat-dev libswscale-dev libgtk2.0-dev libpng-dev libjpeg-dev libopenexr-dev libtiff-dev \
	libwebp-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev  libfreetype6-dev libpng-dev\
    && rm -rf /var/lib/apt/lists/

RUN apt-get update && apt-get install -y ros-foxy-launch* --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

USER root
RUN set -xe \
    && apt-get update && apt-get install -y python3-pip libboost-all-dev python-dev systemd \
    && rm -rf /var/lib/apt/lists/*

# user and permissions
ARG user=mcmt
ARG group=${user}
ARG uid=1000
ARG gid=1000
ARG home=/home/${user}
RUN mkdir -p /etc/sudoers.d \
    && groupadd -g ${gid} ${group} \
    && useradd -d ${home} -u ${uid} -g ${gid} -m -s /bin/bash ${user} \
    && echo "${user} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sudoers_${user}
USER ${user}
RUN sudo usermod -a -G video ${user}

WORKDIR ${home}

# install openCV
RUN wget https://github.com/opencv/opencv/archive/4.5.2.zip -O opencv-4.5.2.zip
RUN wget https://github.com/opencv/opencv_contrib/archive/4.5.2.zip -O opencv-contrib-4.5.2.zip
RUN unzip opencv-4.5.2.zip
RUN unzip opencv-contrib-4.5.2.zip
RUN mv opencv-4.5.2 opencv
RUN mv opencv_contrib-4.5.2 opencv_contrib
RUN mkdir -p build && cd build && cmake -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv
RUN cd build && make -j8
RUN cd build && sudo make install

# install all required dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools

# copy mcmt-tracking repository into Docker container, and install dependencies
RUN mkdir -p spot-it-3d/mcmt-tracking/
COPY --chown=${user} ./mcmt-tracking/ spot-it-3d/mcmt-tracking/
RUN python3 -m pip install -r ./spot-it-3d/mcmt-tracking/bin/requirements.txt

# update .bashrc for bash interactive mode
RUN echo "PATH=${home}/.local/bin:$PATH" >> ${home}/.bashrc

# build mcmt packages
RUN /bin/bash -c "cd spot-it-3d/mcmt-tracking/mcmt-tracking; ./mcmt_build.sh"