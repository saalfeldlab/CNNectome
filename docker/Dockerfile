FROM tensorflow/tensorflow:1.3.0-gpu
LABEL maintainer heinrichl@janelia.hhmi.org

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
RUN pip -V
ENV PIP=pip
ENV PYTHON=python2.7

ENV PREFIX=$HOME


ARG NUM_MAKE_CORES=10


ENV BLOSC_ROOT=${PREFIX}/src/c-blosc
ENV BLOSC_REPOSITORY=https://github.com/Blosc/c-blosc
ENV BLOSC_REVISION=1.15.1

WORKDIR ${BLOSC_ROOT}
RUN git clone ${BLOSC_REPOSITORY} . && \
    git checkout ${BLOSC_REVISION}
RUN cmake -DWITH_BLOSC=OFF -DWITH_ZLIB=ON -DWITH_BZIP2=ON -DWITH_XZ=ON -DCMAKE_CXX_FLAGS="-std=c++17" && \
    make -j $NUM_MAKE_CORES && \
    make install
ENV PYTHONPATH ${BLOSC_ROOT}:$PYTHONPATH


ENV BOOST_VERSION=1.69.0
ENV BOOST_VERSION_UNDERSCORE=1_69_0
ENV BOOST_ROOT=${PREFIX}/src/boost_${BOOST_VERSION_UNDERSCORE}
ENV BOOST_TAR=$HOME/boost_${BOOST_VERSION_UNDERSCORE}.tar.gz
ENV BOOST_SRC=https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERSCORE}.tar.gz

WORKDIR ${BOOST_ROOT}
RUN wget ${BOOST_SRC} -O ${BOOST_TAR}
RUN tar xf ${BOOST_TAR} -C $PREFIX/src
RUN ./bootstrap.sh
RUN ./b2 --with-filesystem --with-system install
# RUN ./bjam install


RUN ${PIP} install pytest


ENV PYBIND11_ROOT=${PREFIX}/src/pybind11
ENV PYBIND11_REPOSITORY=https://github.com/pybind/pybind11
ENV PYBIND11_REVISION=v2.2.4

WORKDIR ${PYBIND11_ROOT}
RUN git clone ${PYBIND11_REPOSITORY} . && \
    git checkout ${PYBIND11_REVISION}
RUN cmake -DPYTHON_EXECUTABLE=$(which ${PYTHON}) .
RUN make -j $NUM_MAKE_CORES
RUN make install

ENV XTL_ROOT=${PREFIX}/src/xtl
ENV XTL_REPOSITORY=https://github.com/QuantStack/xtl
ENV XTL_REVISION=0.5.2


WORKDIR ${XTL_ROOT}
RUN git clone ${XTL_REPOSITORY} . && \
    git checkout ${XTL_REVISION}
RUN cmake .
RUN make -j $NUM_MAKE_CORES
RUN make install


ENV XTENSOR_ROOT=${PREFIX}/src/xtensor
ENV XTENSOR_REPOSITORY=https://github.com/QuantStack/xtensor
ENV XTENSOR_REVISION=0.19.1

WORKDIR ${XTENSOR_ROOT}
RUN git clone ${XTENSOR_REPOSITORY} . && \
    git checkout ${XTENSOR_REVISION}
RUN cmake .
RUN make -j $NUM_MAKE_CORES
RUN make install


ENV XTENSOR_PYTHON_ROOT=${PREFIX}/src/xtensor-python
ENV XTENSOR_PYTHON_REPOSITORY=https://github.com/QuantStack/xtensor-python
ENV XTENSOR_PYTHON_REVISION=0.22.0

WORKDIR ${XTENSOR_PYTHON_ROOT}
RUN git clone ${XTENSOR_PYTHON_REPOSITORY} . && \
    git checkout ${XTENSOR_PYTHON_REVISION}
RUN cmake -DPYTHON_EXECUTABLE=$(which ${PYTHON}) .
RUN make -j $NUM_MAKE_CORES
RUN make install


ENV CMAKE_ROOT=${PREFIX}/src/cmake
ENV CMAKE_REPOSITORY=https://github.com/Kitware/CMake
ENV CMAKE_REVISION=v3.13.2

WORKDIR ${CMAKE_ROOT}
RUN git clone ${CMAKE_REPOSITORY} . && \
    git checkout ${CMAKE_REVISION}
RUN cmake .
RUN make -j $NUM_MAKE_CORES
RUN make install


ENV NLOHMANN_JSON_ROOT=${PREFIX}/src/json
ENV NLOHMANN_JSON_REPOSITORY=https://github.com/nlohmann/json
ENV NLOHMANN_JSON_REVISION=v3.5.0

WORKDIR ${NLOHMANN_JSON_ROOT}
RUN git clone ${NLOHMANN_JSON_REPOSITORY} . && \
    git checkout ${NLOHMANN_JSON_REVISION}
RUN cmake .
RUN make -j $NUM_MAKE_CORES
RUN make install


ENV Z5_ROOT=${PREFIX}/src/z5
ENV Z5_REPOSITORY=https://github.com/constantinpape/z5.git
ENV Z5_REVISION=e551d920d477c31bc25cdcddb06e265218228ef8

WORKDIR ${Z5_ROOT}
RUN git clone ${Z5_REPOSITORY} . && \
    git checkout ${Z5_REVISION}
RUN cmake \
    -DPYTHON_EXECUTABLE=$(which ${PYTHON}) \
    -DWITH_BLOSC=ON -DWITH_ZLIB=ON -DWITH_BZIP2=OFF \
    -DWITH_XZ=OFF \
    -DCMAKE_CXX_FLAGS="-std=c++17" \
    -DBoost_LIBRARY_DIR_RELEASE=/usr/local/lib \
    -DBoost_LIBRARY_DIR_DEBUG=/usr/local/lib \
    .
RUN make -j $NUM_MAKE_CORES
RUN make install

#
# malis
#
ENV MALIS_ROOT=${PREFIX}/src/malis
ENV MALIS_REPOSITORY=https://github.com/TuragaLab/malis.git
ENV MALIS_REVISION=beb4ee965acee89ab00a20a70205f51177003c69

RUN ${PYTHON} --version
RUN ${PIP} install cython
WORKDIR ${MALIS_ROOT}
RUN git clone ${MALIS_REPOSITORY} . && \
    git checkout ${MALIS_REVISION}
RUN ${PYTHON} setup.py build_ext --inplace
ENV PYTHONPATH ${MALIS_ROOT}:$PYTHONPATH

#
# augment
#
ENV AUGMENT_ROOT=${PREFIX}/src/augment
ENV AUGMENT_REPOSITORY=https://github.com/funkey/augment.git
ENV AUGMENT_REVISION=4a42b01ccad7607b47a1096e904220729dbcb80a

WORKDIR ${AUGMENT_ROOT}
RUN git clone ${AUGMENT_REPOSITORY} . && \
    git checkout ${AUGMENT_REVISION}
RUN ${PIP} install -r requirements.txt
ENV PYTHONPATH ${AUGMENT_ROOT}:$PYTHONPATH

#
# gunpowder
#
ENV GUNPOWDER_ROOT=${PREFIX}/src/gunpowder
ENV GUNPOWDER_REPOSITORY=https://github.com/funkey/gunpowder.git
ENV GUNPOWDER_REVISION=d49573f53e8f23d12461ed8de831d0103acb2715

WORKDIR ${GUNPOWDER_ROOT}
RUN git clone ${GUNPOWDER_REPOSITORY} . && \
    git checkout ${GUNPOWDER_REVISION}
RUN ${PIP} install -r requirements.txt
RUN ${PYTHON} setup.py build_ext --inplace
ENV PYTHONPATH ${GUNPOWDER_ROOT}:$PYTHONPATH

RUN ${PIP} install dask
RUN ${PIP} install toolz
WORKDIR /run
