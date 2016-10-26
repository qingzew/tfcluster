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
from datetime import datetime
import numpy as np

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.
MOVING_AVERAGE_DECAY = 0.9999

class Trainer(object):
    def __init__(self, batch_size,
                 logdir,
                 num_examples_per_epoch,
                 max_steps = 1000000,
                 log_device_placement = False,
                 num_replicas_to_aggregate = -1,
                 save_model_secs = 10 * 60,
                 save_summaries_secs = 180,
                 initial_learning_rate = 0.045,
                 num_epochs_per_decay = 2.,
                 learning_rate_decay_factor = 0.94):
        self.batch_size = batch_size
        self.logdir = logdir
        self.max_steps = max_steps
        self.log_device_placement = log_device_placement
        self.num_replicas_to_aggregate = num_replicas_to_aggregate
        self.save_model_secs = save_model_secs
        self.save_summaries_secs = save_summaries_secs
        self.initial_learning_rate = initial_learning_rate
        self.num_examples_per_epoch = num_examples_per_epoch
        self.num_epochs_per_decay = num_epochs_per_decay
        self.learning_rate_decay_factor = learning_rate_decay_factor

        self.func_map = {}

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
        tf.logging.set_verbosity(tf.logging.INFO)

        ps_hosts = os.environ['PS'].split(',')
        worker_hosts = os.environ['WORKER'].split(',')

        # Number of workers and parameter servers are infered from the workers and ps
        # hosts string.
        num_parameter_servers = len(ps_hosts)
        num_workers = len(worker_hosts)

        # If no value is given, num_replicas_to_aggregate defaults to be the number of
        # workers.
        if self.num_replicas_to_aggregate == -1:
            self.num_replicas_to_aggregate = num_workers

        # Both should be greater than 0 in a distributed training.
        assert num_workers > 0 and num_parameter_servers > 0, \
                (' num_workers and ' 'num_parameter_servers' ' must be > 0.')

        # Choose worker 0 as the chief. Note that any worker could be the chief
        # but there should be only one chief.
        is_chief = (int(os.environ['TASK_INDEX']) == 0)

        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                            job_name = os.environ['JOB_NAME'],
                            task_index = int(os.environ['TASK_INDEX']))

        if os.environ['JOB_NAME'] == 'ps':
            server.join()
        elif os.environ['JOB_NAME'] == 'worker':
            # with tf.device('/job:worker/task:%d' % int(os.environ['TASK_INDEX'])):
            with tf.device(tf.train.replica_device_setter(
                worker_device = '/job:worker/task:%d' % int(os.environ['TASK_INDEX']), cluster = cluster)):

                # Create a variable to count the number of train() calls. This equals the
                # number of updates applied to the variables.
                global_step = tf.Variable(0, trainable = False)

                # Calculate the learning rate schedule.
                num_batches_per_epoch = (self.num_examples_per_epoch / self.batch_size)

                # Decay steps need to be divided by the number of replicas to aggregate.
                decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay /
                            self.num_replicas_to_aggregate)

                # Decay the learning rate exponentially based on the number of steps.
                lr = tf.train.exponential_decay(self.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        self.learning_rate_decay_factor,
                                        staircase = True)
                # Add a summary to track the learning rate.
                tf.scalar_summary('learning_rate', lr)

                # Create an optimizer that performs gradient descent.
                opt = tf.train.RMSPropOptimizer(lr,
                                        RMSPROP_DECAY,
                                        momentum = RMSPROP_MOMENTUM,
                                        epsilon = RMSPROP_EPSILON)

                self.call_method('build_model')
                losses = tf.get_collection('losses')

                total_loss = tf.add_n(losses, name = 'total_loss')

                if is_chief:
                    # Compute the moving average of all individual losses and the
                    # total loss.
                    loss_averages = tf.train.ExponentialMovingAverage(0.9, name = 'avg')
                    loss_averages_op = loss_averages.apply(losses + [total_loss])

                    # Attach a scalar summmary to all individual losses and the total loss;
                    # do the same for the averaged version of the losses.
                    for l in losses + [total_loss]:
                        loss_name = l.op.name
                        # Name each loss as '(raw)' and name the moving average version of the
                        # loss as the original loss name.
                        tf.scalar_summary(loss_name + ' (raw)', l)
                        tf.scalar_summary(loss_name, loss_averages.average(l))

                    # Add dependency to compute loss_averages.
                    with tf.control_dependencies([loss_averages_op]):
                        total_loss = tf.identity(total_loss)

                # Track the moving averages of all trainable variables.
                # Note that we maintain a 'double-average' of the BatchNormalization
                # global statistics.
                # This is not needed when the number of replicas are small but important
                # for synchronous distributed training with tens of workers/replicas.
                exp_moving_averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                                    global_step)

                variables_to_average = (
                        tf.trainable_variables() + tf.moving_average_variables())

                # Add histograms for model variables.
                for var in variables_to_average:
                    tf.histogram_summary(var.op.name, var)

                # Create synchronous replica optimizer.
                opt = tf.train.SyncReplicasOptimizer(
                    opt,
                    replicas_to_aggregate = self.num_replicas_to_aggregate,
                    replica_id = int(os.environ['TASK_INDEX']),
                    total_num_replicas = num_workers,
                    variable_averages = exp_moving_averager,
                    variables_to_average =  variables_to_average)


                # Compute gradients with respect to the loss.
                grads = opt.compute_gradients(total_loss)

                # Add histograms for gradients.
                for grad, var in grads:
                    if grad is not None:
                        tf.histogram_summary(var.op.name + '/gradients', grad)

                apply_gradients_op = opt.apply_gradients(grads, global_step = global_step)

                with tf.control_dependencies([apply_gradients_op]):
                    train_op = tf.identity(total_loss, name = 'train_op')

                # Get chief queue_runners, init_tokens and clean_up_op, which is used to
                # synchronize replicas.
                # More details can be found in sync_replicas_optimizer.
                chief_queue_runners = [opt.get_chief_queue_runner()]
                init_tokens_op = opt.get_init_tokens_op()
                clean_up_op = opt.get_clean_up_op()

                # Create a saver.
                saver = tf.train.Saver()

                # Build the summary operation based on the TF collection of Summaries.
                summary_op = tf.merge_all_summaries()

                # Build an initialization operation to run below.
                init_op = tf.initialize_all_variables()

                # We run the summaries in the same thread as the training operations by
                # passing in None for summary_op to avoid a summary_thread being started.
                # Running summaries and training operations in parallel could run out of
                # GPU memory.
                sv = tf.train.Supervisor(is_chief = is_chief,
                                    logdir = self.logdir,
                                    init_op = init_op,
                                    summary_op = None,
                                    global_step = global_step,
                                    saver = saver,
                                    save_model_secs = self.save_model_secs)

                tf.logging.info('%s Supervisor' % datetime.now())

                sess_config = tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=self.log_device_placement)

                # Get a session.
                with sv.prepare_or_wait_for_session(server.target, config = sess_config) as sess:
                    # Start the queue runners.
                    queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
                    sv.start_queue_runners(sess, queue_runners)
                    tf.logging.info('Started %d queues for processing input data.',
                                len(queue_runners))

                    if is_chief:
                        sv.start_queue_runners(sess, chief_queue_runners)
                        sess.run(init_tokens_op)

                    # Train, checking for Nans. Concurrently run the summary operation at a
                    # specified interval. Note that the summary_op and train_op never run
                    # simultaneously in order to prevent running out of GPU memory.
                    next_summary_time = time.time() + self.save_summaries_secs
                    while not sv.should_stop():
                        try:
                            start_time = time.time()
                            loss_value, step = sess.run([train_op, global_step])
                            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                            if step > self.max_steps:
                                break
                            duration = time.time() - start_time

                            if step % 100 == 0:
                                examples_per_sec = self.batch_size / float(duration)
                                format_str = ('Worker %d: %s: step %d, loss = %.2f'
                                        '(%.1f examples/sec; %.3f  sec/batch)')
                                tf.logging.info(format_str %
                                            (int(os.environ['TASK_INDEX']), datetime.now(), step, loss_value,
                                            examples_per_sec, duration))

                            # Determine if the summary_op should be run on the chief worker.
                            if is_chief and next_summary_time < time.time():
                                tf.logging.info('Running Summary operation on the chief.')
                                summary_str = sess.run(summary_op)
                                sv.summary_computed(sess, summary_str)
                                tf.logging.info('Finished running Summary operation.')

                                # Determine the next time for running the summary.
                                next_summary_time += self.save_summaries_secs
                        except Exception as e:
                            tf.logging.error(e)
                            if is_chief:
                                tf.logging.info('About to execute sync_clean_up_op!')
                                sess.run(clean_up_op)
                            raise

                    # Stop the supervisor.  This also waits for service threads to finish.
                    sv.stop()

                    # Save after the training ends.
                    if is_chief:
                        saver.save(sess,
                            os.path.join(self.logdir, 'model.ckpt'),
                            global_step = global_step)



