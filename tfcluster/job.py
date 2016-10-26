#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 qingze <qingze@node39.com>
#
# Distributed under terms of the MIT license.

"""

"""

import sys
import uuid
from collections import OrderedDict

from task import TensorflowTask

import logging
from tfcluster.utils import setup_logger
logger = logging.getLogger('tfcluster.tfcluster')
setup_logger(logger)

# one job contains multi tasks
class Job(object):
    def __init__(self, job):
        self.__uid = str(uuid.uuid1())
        self.__tasks = OrderedDict()
        self.backend = job[-1]['backend']
        del job[-1]

        if self.backend == 'tensorflow':
            ps_cnt = 0
            worker_cnt = 0
            for task in job:
                mesos_task_id = str(uuid.uuid1())
                if task['name'] == 'ps':
                    task = TensorflowTask(
                        mesos_task_id,
                        self.__uid,
                        task['name'],
                        ps_cnt,
                        task['zk_master'],
                        task['hdfs_namenode'],
                        task['model_on_hdfs'],
                        cpus = task['cpus'],
                        gpus = task['gpus'],
                        mem = task['mems'],
                        cmd = task['cmd'])

                    ps_cnt += 1
                elif task['name'] == 'worker':
                    task = TensorflowTask(
                        mesos_task_id,
                        self.__uid,
                        task['name'],
                        worker_cnt,
                        task['zk_master'],
                        task['hdfs_namenode'],
                        task['model_on_hdfs'],
                        cpus = task['cpus'],
                        gpus = task['gpus'],
                        mem = task['mems'],
                        cmd = task['cmd'])

                    worker_cnt += 1
                self.__tasks[mesos_task_id] = task
        elif self.backend == 'caffe':
            pass
        else:
            logger.error('unknow backend: {0}'.format(self.backend))
            sys.exit(0)

    @property
    def tasks(self):
        return self.__tasks

    @property
    def uid(self):
        return self.__uid

    def __call__(self):
        return self.tasks


