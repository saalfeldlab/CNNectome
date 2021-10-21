FROM tensorflow/tensorflow:1.14.0-gpu-py3
LABEL maintainer heinrichl@janelia.hhmi.org

# basic ubuntu packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libboost-all-dev \
        libmlpack-dev \
        python3.6 \
        python3-numpy \
        python3-setuptools \
        python3-pip \
        strace \
        wget && \
    rm -rf /var/lib/apt/lists/*

ENV PIP=pip
ENV PYTHON=python3.6

RUN ${PIP} -V && \
    ${PIP} install pytest dask toolz memory_profiler

ENV PREFIX=$HOME


#
# malis
#
#ENV MALIS_ROOT=${PREFIX}/src/malis
#ENV PYTHONPATH ${MALIS_ROOT}:$PYTHONPATH
#ENV MALIS_REPOSITORY=https://github.com/TuragaLab/malis.git
#ENV MALIS_REVISION=beb4ee965acee89ab00a20a70205f51177003c69

RUN ${PYTHON} --version && ${PIP} install cython
#WORKDIR ${MALIS_ROOT}
#RUN git clone ${MALIS_REPOSITORY} . && \
#    git checkout ${MALIS_REVISION} && \
#    ${PYTHON} setup.py build_ext --inplace

#ENV CNNECTOME_BRANCH=v2.0
#ENV CNNECTOME_REPOSITORY=https://github.com/saalfeldlab/CNNectome.git
#RUN ${PIP} install cnnectome[synapse_postprocessing]@git+${CNNECTOME_REPOSITORY}@${CNNECTOME_BRANCH}
ENV CNNECTOME_ROOT=${PREFIX}/src/cnnectome
ADD /groups/saalfeld/home/heinrichl/dev/CNNectome ${CNNECTOME_ROOT}
WORKDIR ${CNNECTOME_ROOT}
RUN ${PIP} install .

ENV PYTHONPATH="/groups/saalfeld/home/heinrichl/dev/CNNectome:$PYTHONPATH"
ENV PYTHONPATH="/groups/saalfeld/home/heinrichl/dev/simpleference:$PYTHONPATH"
ENV PYTHONPATH="/groups/saalfeld/home/heinrichl/dev/gunpowder:$PYTHONPATH"
ENV PYTHONPATH="/groups/saalfeld/home/heinrichl/dev/corditea:$PYTHONPATH"
WORKDIR /run
