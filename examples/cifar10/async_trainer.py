#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 qingze <qingze@node39.com>
#
# Distributed under terms of the MIT license.

"""

"""

import os
import time
import tensorflow as tf
import logging
from datetime import datetime

def setup_logger(logger):
    logger.setLevel(logging.DEBUG)

    FORMAT = '%(levelname).1s%(asctime)-11s  %(process)d %(filename)-9s:%(lineno)d] %(message)s'
    formatter = logging.Formatter(FORMAT, datefmt = '%m%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = logging.getLogger(__name__)
setup_logger(logger)




class Trainer(object):
    def __init__(self, logdir, steps):
        self.func_map = {}
        self.logdir = logdir
        self.steps = steps

    def register(self, name):
        def func_wrapper(func):
            self.func_map[name] = func
            return func
        return func_wrapper

    def call_method(self, name = None):
        func = self.func_map.get(name, None)
        if func is None:
            raise Exception("No function registered against - " + str(name))
        return func()

    def train(self):
        ps_hosts = os.environ['PS'].split(',')
        worker_hosts = os.environ['WORKER'].split(',')

        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                            job_name = os.environ['JOB_NAME'],
                            task_index = int(os.environ['TASK_INDEX']))

        if os.environ['JOB_NAME'] == 'ps':
            server.join()
        elif os.environ['JOB_NAME'] == 'worker':
            with tf.Graph().as_default():
                # Assigns ops to the local worker by default.

                with tf.device(tf.train.replica_device_setter(
                    worker_device = '/job:worker/task:%d' % int(os.environ['TASK_INDEX']), cluster = cluster)):

                    train_op, loss, global_step = self.call_method('build_model')
                    init_op = tf.initialize_all_variables()

                # saver = tf.train.Saver(tf.all_variables(), max_to_keep = 25, keep_checkpoint_every_n_hours = 1)
                saver = tf.train.Saver(tf.all_variables(), max_to_keep = 25)
                # Create a "supervisor", which oversees the training process.
                sv = tf.train.Supervisor(is_chief = (int(os.environ['TASK_INDEX']) == 0),
                                    logdir = self.logdir,
                                    init_op = init_op,
                                    saver = saver,
                                    global_step = global_step,
                                    save_model_secs = 600)

                config = tf.ConfigProto(allow_soft_placement = True,
                                                      log_device_placement = False)
                config.gpu_options.allow_growth = True

                # The supervisor takes care of session initialization, restoring from
                # a checkpoint, and closing when done or an error occurs.
                with sv.managed_session(server.target, config = config) as sess:
                    sess.run(init_op)
                    # Loop until the supervisor shuts down or 1000000 steps have completed.
                    step = 0
                    while not sv.should_stop() and step < self.steps:
                        # Run a training step asynchronously.
                        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                        # perform *synchronous* training.

                        start_time = time.time()

                        # step
                        _, loss_value = sess.run([train_op, loss])

                        duration = time.time() - start_time

                        if step % 1 == 0:
                            format_str = ('%s: step %d, loss = %.2f (%.3f sec)')
                            print(format_str % (datetime.now(), step, loss_value, duration))

                            # Print status to stdout.
                            # print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                        step += 1


                # Ask for all the services to stop.
                sv.stop()


