#! /bin/sh
#
# run_local.sh
# Copyright (C) 2016 qingze <qingze@node39.com>
#
# Distributed under terms of the MIT license.
#

export PS=192.168.184.35:9000
export WORKER=192.168.184.35:9501

export JOB_NAME=worker
export TASK_INDEX=0

CUDA_VISIBLE_DEVICES='3' python start.py

