#!/bin/bash
FEAT_TYPE=${1:-'bad256'}
SRC_DIR=${2:-'/triorb/data/log'}
DST_DIR=${3:-"/triorb/data"}
DUMP_FILE=${DST_DIR}/${FEAT_TYPE}.dump
FBOW_DICT_FILE=${DST_DIR}/${FEAT_TYPE}_vocab.fbow

cd ./build/utils
./fbow_dump_features ${FEAT_TYPE} ${DUMP_FILE} ${SRC_DIR} &&\
./fbow_create_vocabulary ${DUMP_FILE} ${FBOW_DICT_FILE} &&\
ls -lh ${DST_DIR} && rm ${DUMP_FILE}