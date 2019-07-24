FROM tensorflow/tensorflow:1.3.0-gpu
LABEL maintainer bennettd@janelia.hhmi.org

# basic ubuntu packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
	libmlpack-dev \
	python-numpy \
	python-setuptools \
        python-pip \
	libboost-all-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*