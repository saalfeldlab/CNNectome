# make the environment
apt-get update && apt-get install -y --no-install-recommends \
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

    export PIP=pip
    export PYTHON=python2.7
    echo 'Installing blosc'
    export PREFIX=$HOME
    export NUM_MAKE_CORES=10
    export BLOSC_ROOT=${PREFIX}/src/c-blosc
        export BLOSC_REPO=https://github.com/Blosc/c-blosc
    export BLOSC_REV=1.15.1
    
    mkdir -p $BLOSC_ROOT
    cd $BLOSC_ROOT
    git clone ${BLOSC_REPO} . && git checkout ${BLOSC_REV}
    cmake -DWITH_BLOSC=OFF -DWITH_ZLIB=ON -DWITH_BZIP2=ON -DWITH_XZ=ON -DCMAKE_CXX_FLAGS="-std=c++17" && \
    make -j $NUM_MAKE_CORES && \
    make install
    
    export PYTHONPATH=${BLOSC_ROOT}:$PYTHONPATH

    export BOOST_VER=1.69.0
    export BOOST_VER_UNDERSCORE=1_69_0
    export BOOST_ROOT=${PREFIX}/src/boost_${BOOST_VER_UNDERSCORE}
    export BOOST_TAR=$HOME/boost_${BOOST_VER_UNDERSCORE}.tar.gz
    export BOOST_SRC=https://dl.bintray.com/boostorg/release/${BOOST_VER}/source/boost_${BOOST_VER_UNDERSCORE}.tar.gz

    mkdir -p ${BOOST_ROOT}
    cd ${BOOST_ROOT}
    wget ${BOOST_SRC} -O ${BOOST_TAR}
    tar xf ${BOOST_TAR} -C $PREFIX/src
    ./bootstrap.sh
    ./b2 --with-filesystem --with-system install
    # RUN ./bjam install

    pip install pytest

    export PYBIND11_ROOT=${PREFIX}/src/pybind11
    export PYBIND11_REPO=https://github.com/pybind/pybind11
    export PYBIND11_REV=v2.2.4

    mkdir -p $PYBIND11_ROOT
    WORKDIR ${PYBIND11_ROOT}
    git clone ${PYBIND11_REPO} . && \
    git checkout ${PYBIND11_REV}
    cmake -DPYTHON_EXECUTABLE=$(which ${PYTHON}) .
    make -j $NUM_MAKE_CORES
    make install

    XTL_ROOT=${PREFIX}/src/xtl
    XTL_REPO=https://github.com/QuantStack/xtl
    XTL_REV=0.5.2

    mkdir -p ${XTL_ROOT}
    cd ${XTL_ROOT}
    git clone ${XTL_REPO} . && git checkout ${XTL_REV}
    cmake .
    make -j $NUM_MAKE_CORES
    make install

    export XTENSOR_ROOT=${PREFIX}/src/xtensor
    export XTENSOR_REPO=https://github.com/QuantStack/xtensor
    export XTENSOR_REV=0.19.1

    mkdir -p  ${XTENSOR_ROOT} && cd ${XTENSOR_ROOT}
    git clone ${XTENSOR_REPO} . && \
    git checkout ${XTENSOR_REV}
    cmake .
    make -j $NUM_MAKE_CORES
    make install


    export XTENSOR_PYTHON_ROOT=${PREFIX}/src/xtensor-python
    export  XTENSOR_PYTHON_REPO=https://github.com/QuantStack/xtensor-python
    export  XTENSOR_PYTHON_REV=0.22.0

    mkdir -p  ${XTENSOR_PYTHON_ROOT} && cd ${XTENSOR_PYTHON_ROOT}
    git clone ${XTENSOR_PYTHON_REPO} . && \
    git checkout ${XTENSOR_PYTHON_REV}
    cmake -DPYTHON_EXECUTABLE=$(which ${PYTHON}) .
    make -j $NUM_MAKE_CORES
    make install

    export CMAKE_ROOT=${PREFIX}/src/cmake
    export CMAKE_REPO=https://github.com/Kitware/CMake
    export CMAKE_REV=v3.13.2

    mkdir -p ${CMAKE_ROOT} && cd ${CMAKE_ROOT}
    git clone ${CMAKE_REPO} . && \
    git checkout ${CMAKE_REV}
    cmake .
    make -j $NUM_MAKE_CORES
    make install

    export NLOHMANN_JSON_ROOT=${PREFIX}/src/json
    export NLOHMANN_JSON_REPO=https://github.com/nlohmann/json
    export NLOHMANN_JSON_REV=v3.5.0

    mkdir -p  ${NLOHMANN_JSON_ROOT} && cd ${NLOHMANN_JSON_ROOT}
    git clone ${NLOHMANN_JSON_REPO} . && \
    git checkout ${NLOHMANN_JSON_REV}
    cmake .
    make -j $NUM_MAKE_CORES
    make install

    export Z5_ROOT=${PREFIX}/src/z5
    export Z5_REPO=https://github.com/constantinpape/z5.git
    export Z5_REV=e551d920d477c31bc25cdcddb06e265218228ef8

    mkdir -p ${Z5_ROOT} && cd ${Z5_ROOT}
    git clone ${Z5_REPO} . && \
    git checkout ${Z5_REV}
    cmake \
    -DPYTHON_EXECUTABLE=$(which ${PYTHON}) \
    -DWITH_BLOSC=ON -DWITH_ZLIB=ON -DWITH_BZIP2=OFF \
    -DWITH_XZ=OFF \
    -DCMAKE_CXX_FLAGS="-std=c++17" \
    -DBoost_LIBRARY_DIR_RELEASE=/usr/local/lib \
    -DBoost_LIBRARY_DIR_DEBUG=/usr/local/lib \
    .
    make -j $NUM_MAKE_CORES
    make install

    #
    # malis
    #
    export MALIS_ROOT=${PREFIX}/src/malis
    export MALIS_REPO=https://github.com/TuragaLab/malis.git
    export MALIS_REV=beb4ee965acee89ab00a20a70205f51177003c69

    ${PYTHON} --version
    ${PIP} install cython
    cd -p ${MALIS_ROOT} && cd ${MALIS_ROOT}
    git clone ${MALIS_REPO} . && \
    git checkout ${MALIS_REV}
    ${PYTHON} setup.py build_ext --inplace
    export PYTHONPATH=${MALIS_ROOT}:$PYTHONPATH

    #
    # augment
    #
    export AUGMENT_ROOT=${PREFIX}/src/augment
    export AUGMENT_REPO=https://github.com/funkey/augment.git
    export AUGMENT_REV=4a42b01ccad7607b47a1096e904220729dbcb80a

    mkdir -p  ${AUGMENT_ROOT} && cd ${AUGMENT_ROOT}
    git clone ${AUGMENT_REPO} . && \
    git checkout ${AUGMENT_REV}
    ${PIP} install -r requirements.txt
    export PYTHONPATH=${AUGMENT_ROOT}:$PYTHONPATH

    #
    # gunpowder
    #
    export GUNPOWDER_ROOT=${PREFIX}/src/gunpowder
    export GUNPOWDER_REPO=https://github.com/funkey/gunpowder.git
    export GUNPOWDER_REV=d49573f53e8f23d12461ed8de831d0103acb2715

    mkdir -p ${GUNPOWDER_ROOT} && cd ${GUNPOWDER_ROOT}
    git clone ${GUNPOWDER_REPO} . && git checkout ${GUNPOWDER_REV}
    ${PIP} install -r requirements.txt
    ${PYTHON} setup.py build_ext --inplace
    export PYTHONPATH=${GUNPOWDER_ROOT}:$PYTHONPATH

    ${PIP} install dask
    ${PIP} install toolz
    mkdir -p /run cd /run

    




