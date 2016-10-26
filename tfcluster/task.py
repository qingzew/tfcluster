#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 qingze <qingze@node39.com>
#
# Distributed under terms of the MIT license.

"""

"""
# import sys
import textwrap
from mesos.interface import mesos_pb2


class TensorflowTask(object):
    def __init__(self, mesos_task_id, uid, job_name, task_index, zk_master,
                 hdfs_namenode, model_on_hdfs,
                 cpus = 1.0, mem = 1024.0, gpus = 0, cmd = None):
        self.mesos_task_id = mesos_task_id
        self.job_name = job_name
        self.task_index = task_index

        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem
        self.cmd = cmd

        self.uid = uid
        self.offered = False

        self.zk_master = zk_master
        self.hdfs_namenode = hdfs_namenode
        self.model_on_hdfs = model_on_hdfs

    def __str__(self):
        return textwrap.dedent('''
                               <Task
                               job_name=%s
                               task_index=%s
                               >''' % (self.job_name, self.task_index))

    def to_task_info(self, slave_id, port, docker = False):
        ti = mesos_pb2.TaskInfo()
        ti.task_id.value = str(self.mesos_task_id)
        ti.slave_id.value = slave_id
        ti.name = '/job:%s/task:%s' % (self.job_name, self.task_index)

        cpus = ti.resources.add()
        cpus.name = 'cpus'
        cpus.type = mesos_pb2.Value.SCALAR
        cpus.scalar.value = self.cpus

        mem = ti.resources.add()
        mem.name = 'mem'
        mem.type = mesos_pb2.Value.SCALAR
        mem.scalar.value = self.mem

        if self.gpus:
            gpus = ti.resources.add()
            gpus.name = 'gpus'
            gpus.type = mesos_pb2.Value.SCALAR
            gpus.scalar.value = self.gpus

        ports = ti.resources.add()
        ports.name = 'ports'
        ports.type = mesos_pb2.Value.RANGES
        ports_range = ports.ranges.range.add()
        ports_range.begin = port
        ports_range.end = port

        ti.command.shell = True

        # ti.command.value = self.cmd
        ti.command.value = 'python -m tfcluster.startup'

        zk_master = ti.command.environment.variables.add()
        zk_master.name = 'ZK_MASTER'
        zk_master.value = self.zk_master

        hdfs = ti.command.environment.variables.add()
        hdfs.name = 'HDFS_NAMENODE'
        hdfs.value = self.hdfs_namenode

        model = ti.command.environment.variables.add()
        model.name = 'MODEL_ON_HDFS'
        model.value = self.model_on_hdfs

        uid = ti.command.environment.variables.add()
        uid.name = 'UID'
        uid.value = self.uid

        job_name = ti.command.environment.variables.add()
        job_name.name = 'JOB_NAME'
        job_name.value = self.job_name

        task_index = ti.command.environment.variables.add()
        task_index.name  = 'TASK_INDEX'
        task_index.value = str(self.task_index)

        cmd = ti.command.environment.variables.add()
        cmd.name = 'CMD'
        cmd.value = self.cmd

        # python_path = ti.command.environment.variables.add()
        # python_path.name = 'PYTHONPATH'
        # python_path.value = ':'.join(sys.path)

        # can be set in container
        ld_library_path = ti.command.environment.variables.add()
        ld_library_path.name = 'LD_LIBRARY_PATH'
        # ld_library_path.value = os.environ['LD_LIBRARY_PATH']
        ld_library_path.value = 'LD_LIBRARY_PATH:/usr/local/cuda/lib64/'

        if docker == True:
            ci = ti.container
            ci.type = mesos_pb2.ContainerInfo.MESOS

            # use docker image as rootfs
            mesos = ci.mesos
            image = mesos.image
            image.type = mesos_pb2.Image.DOCKER

            dc = image.docker
            dc.name = 'qingzew/centos'

            # mount docker image as a rootfs's dir
            # image_dir = ci.volumes.add()
            # image_dir.mode = mesos_pb2.Volume.RO

            # image = image_dir.image
            # image.type = mesos_pb2.Image.DOCKER
            # dc = image.docker
            # dc.name = 'qingzew/centos'
            # image_dir.container_path = '/tmp'

            # mount volume
            data_dir = ci.volumes.add()
            data_dir.mode = mesos_pb2.Volume.RO
            data_dir.host_path = '/tmp/cifar10/cifar10_data/'
            data_dir.container_path = '/tmp/data'

        return ti


class CaffeTask:
    pass
