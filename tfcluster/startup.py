#!/usr/bin/env python
# coding: utf-8

import os
import sys
# import time
# import threading
import logging
import subprocess
from tfcluster.utils import setup_logger
from kazoo.client import KazooClient

from snakebite.client import Client

logger = logging.getLogger('tfcluster.startup')
setup_logger(logger)

def main(argv):
    hdfs_namenode = os.environ['HDFS_NAMENODE']
    model_on_hdfs = os.environ['MODEL_ON_HDFS']
    ip, port = hdfs_namenode.rsplit(':', 1)
    client = Client(ip, int(port), use_trash = False)
    dst_dir = os.path.join('/')
    for x in client.copyToLocal([model_on_hdfs], dst_dir):
        print x

    zk_master = os.environ['ZK_MASTER']

    logger.info('job_name: {0}, task_index: {1}'.format(os.environ['JOB_NAME'], os.environ['TASK_INDEX']))
    logger.info('command: {0}'.format(os.environ['CMD']))

    zk = KazooClient(hosts = zk_master)
    zk.start()

    logger.info('job uid: {0}'.format(os.environ['UID']))
    job_zk_dir = '/' + os.environ['UID'];

    members = zk.get_children(job_zk_dir + '/member/')
    members.sort()

    cluster_def = {}
    for member in members:
        host = zk.get(job_zk_dir + '/member/' + member)[0]
        if host != '':
            logger.info('{0} running on {1}'.format(member, host))
            job_type = member.split('_')[2]

            if job_type == 'ps':
                cluster_def.setdefault('ps', []).append(host)
            elif job_type == 'worker':
                cluster_def.setdefault('worker', []).append(host)
            else:
                logger.error('unkown type: {0}'.format(job_type))


    ps = ','.join(cluster_def['ps'])
    worker = ','.join(cluster_def['worker'])

    my_env = os.environ.copy()
    logger.info(my_env)
    my_env['PS'] = ps
    my_env['WORKER'] = worker

    cmd = [os.environ['CMD']]
    child = subprocess.Popen(cmd, shell = True, env = my_env)

    child.wait()
    zk.stop()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
