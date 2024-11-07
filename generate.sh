#!/bin/bash
#MEMO : while true; do  sh generate.sh hash_sift512 /home/tobeta/FBoW/dataset/ /home/tobeta/FBoW/dst/; done
FEAT_TYPE=${1:-'bad256'} # bad256 bad512 hash_sift256 hash_sift512
SRC_DIR=${2:-'/triorb/data/log'}
DST_DIR=${3:-"/triorb/data/"}

docker run -it --rm --name fbow --privileged --net=host --runtime=nvidia --gpus all \
               --add-host=localhost:127.0.1.1 \
               -m 8g\
               -e ROS_LOCALHOST_ONLY=1 \
               -e FEAT_TYPE="${FEAT_TYPE}"\
               -e SRC_DIR="${SRC_DIR}"\
               -e DST_DIR="${DST_DIR}"\
               -v ${SRC_DIR}:${SRC_DIR} \
               -v ${DST_DIR}:${DST_DIR} \
               -v "$(pwd)":/ws \
               -w /ws \
               triorb/ros2/cuda:humble /bin/bash -c "\
               mkdir /ws/cuda-efficient-features/build;
               cd /ws/cuda-efficient-features/build &&
               cmake .. &&
               make -j4 &&
               make install &&
               export CPATH=/usr/local/cuda/targets/x86_64-linux/include:$CPATH &&
               export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH &&
               export PATH=/usr/local/cuda/bin:$PATH &&
               export cuda_efficient_features_DIR='/usr/local/lib/cmake/cuda_efficient_features/' &&
               export PATH=/usr/local/include/:${PATH} &&
               export CPATH=/usr/local/bin/:${CPATH} &&
               export LD_LIBRARY_PATH=/usr/local/lib/:${LD_LIBRARY_PATH} &&
               mkdir /ws/build;
               cd /ws/build &&
               cmake  .. -DBUILD_TESTS=ON -DBUILD_UTILS=ON &&
               make -j4 && cd ../ && sh ./.generate.sh ${FEAT_TYPE} ${SRC_DIR} ${DST_DIR}"
