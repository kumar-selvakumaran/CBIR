FROM debian:buster

# Install Dependencies
RUN apt-get update && apt-get install build-essential -y
RUN apt-get install \
    cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev -y
# Create OPENCV on a layer
RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git && \
    cd /opencv && git checkout 4.2.0 && \
    cd /opencv_contrib && git checkout 4.2.0 && \
    cd /opencv && \
    mkdir build && \
    cd /opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local .. \
        -D INSTALL_C_EXAMPLES=ON \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON .. && \
    make -j5 && make install && \
    cd / && \
    rm opencv -rf && \
    rm opencv_contrib -rf

############### FURTHER SETUP INSTRUCTIONS #####################################################
 #  // TO COMPLETE SETUP RUN THE BELOW COMMAND AS WELL. FOR SOME REASON, IT CONTAINER THOWS <OPENCV_CORE cannot open shared object> error
# apt-get install -y scons libboost-all-dev 

# // DOCKER RUN COMMAND WITH VOLUMES, AND GUI FUNCTIONALITY.
# fOR WSL
# docker run -it -e DISPLAY=$DISPLAY    -v /tmp/.X11-unix:/tmp/.X11-unix    --network=host    --volume=/mnt/d/NEU/semester_2/prcv_5330/projectwsl/app:/app  --workdir=/app opencv_working:v0.0.1 /bin/bash

# FOR POWERSHELL
# docker run -it -e DISPLAY=$DISPLAY    -v /tmp/.X11-unix:/tmp/.X11-unix    --network=host    --volume=D:\NEU\semester_2\prcv_5330\opencv_setup\app:/app  --workdir=/app

# IF IT IS NOT WORKING, DOCKER DESKTOP IS PROBABLY NOT SWITCHED ON
##################################

# // CONFIGURE CPP_PROPERTIES.JSON TO ALLOW VSCODE TO DETECT HEADER FILES.
# edit c_cpp_properties.json.  i.e. : ctrl + shift + p -> edit c/c++ configuration.json -> add package paths in configurations["includePath"] 

############### compile and execute
# scons  //compilation
# ./test // execution (depending on the source code)
##################################################################################################