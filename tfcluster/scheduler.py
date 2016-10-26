import os
import sys
import math
import logging
# from collections import OrderedDict
from kazoo.client import KazooClient
from mesos.interface import mesos_pb2, Scheduler
from tfcluster.utils import setup_logger
from tfcluster.job import Job

FOREVER = 0xFFFFFFFF
logger = logging.getLogger(__name__)


class TFClusterScheduler(Scheduler):
    def __init__(self, jobs_def, zk_master = None,
                 name = None, quiet = False):
        if not quiet:
            global logger
            setup_logger(logger)

        self.name = name or '[tensorflow] %s %s' % (
            os.path.abspath(sys.argv[0]), ' '.join(sys.argv[1:]))

        self.jobs = Job(jobs_def)
        logger.info('job uid: {0}'.format(self.jobs.uid))

        self.zk = KazooClient(hosts = zk_master)
        self.zk.start()
        logger.info('zookeeper is in {0}'.format(self.zk.state))

        self.job_zk_dir = '/' + self.jobs.uid
        self.zk.ensure_path(self.job_zk_dir + '/member')
        self.zk.ensure_path(self.job_zk_dir + '/status')

    def __del__(self):
        logger.info('Deleting node {0} in Zookeeper'.format(self.job_zk_dir))
        self.zk.delete(self.job_zk_dir, recursive = True)

        self.zk.stop()

    def resourceOffers(self, driver, offers):
        '''
        Offer resources and launch tasks
        '''

        for offer in offers:
            if all(task.offered for _, task in self.jobs.tasks.iteritems()):
                driver.declineOffer(offer.id, mesos_pb2.Filters(refuse_seconds = FOREVER))
                continue

            offer_ip = offer.url.address.ip
            offered_cpus = offered_mem = offered_gpus = 0.0
            offered_port_begin = 0
            offered_port_end = 0

            offered_tasks = []

            for resource in offer.resources:
                if resource.name == 'cpus':
                    offered_cpus = resource.scalar.value
                elif resource.name == 'mem':
                    offered_mem = resource.scalar.value
                elif resource.name == 'gpus':
                    offered_gpus = int(resource.scalar.value)
                elif resource.name == 'ports':
                    offered_port_begin = int(resource.ranges.range[0].begin)
                    offered_port_end = int(resource.ranges.range[0].end)

            logger.info('Offered cpus: {0} offered mem: {1} offered gpus: {2} offerd port: {3}~{4} on {5}'
                        .format(offered_cpus, offered_mem, offered_gpus, offered_port_begin, offered_port_end, offer.url.address.hostname))

            for _, task in self.jobs.tasks.iteritems():
                if task.offered:
                    continue

                if not (task.cpus <= offered_cpus and
                        task.mem <= offered_mem and
                        int(math.ceil(task.gpus)) <= offered_gpus and
                        offered_port_begin <= offered_port_end and
                        offered_port_begin != 0):
                    logger.info('/job:{0}/task{1} does not get all needed resources on offer: {2}'
                                .format(task.job_name, task.task_index, offer.id.value))
                    continue

                offered_cpus -= task.cpus
                offered_mem -= task.mem
                offered_gpus -= math.ceil(task.gpus)
                offered_tasks.append(task.to_task_info(offer.slave_id.value, offered_port_begin, True))

                node = '/job:{0}/task:{1}'.format(task.job_name, task.task_index).replace('/', '_').replace(':', '_')
                addr = str(offer_ip) + ':' + str(offered_port_begin)
                offered_port_begin += 1

                logger.info('Registering in zookeeper {0}, value: {1}' .format(self.job_zk_dir + '/member/' + node, addr))
                self.zk.create(self.job_zk_dir + '/member/' + node, addr)

                if task.job_name == 'ps':
                    self.zk.create(self.job_zk_dir + '/status/' + node, 'false')

                task.offered = True
                logger.info('Allocating resource for /job:{0}/task:{1} successful'.format(task.job_name, task.task_index))

            driver.launchTasks(offer.id, offered_tasks, mesos_pb2.Filters())


    def registered(self, driver, framework_id, master_info):
        logger.info(
            "Cluster registered. "
            "( http://%s:%s/#/frameworks/%s )",
            master_info.hostname, master_info.port, framework_id.value
        )

    def statusUpdate(self, driver, update):
        mesos_task_id = update.task_id.value
        task = self.jobs.tasks[mesos_task_id]

        if update.state == mesos_pb2.TASK_RUNNING:
            logger.info('Task /job:{0}/task:{1}: {2}'
                    .format(task.job_name, task.task_index, mesos_pb2.TaskState.Name(update.state)))
        elif update.state == mesos_pb2.TASK_FINISHED:
            logger.info('Task /job:{0}/task:{1}: {2}'
                    .format(task.job_name, task.task_index, mesos_pb2.TaskState.Name(update.state)))
            driver.stop()
        elif update.state == mesos_pb2.TASK_FAILED:
            logger.warn('Task /job:{0}/task:{1}: {2}'
                    .format(task.job_name, task.task_index, update.message))
            driver.stop()
        elif update.state == mesos_pb2.TASK_KILLED:
            logger.warn('Task /job:{0}/task:{1}: {2}'
                    .format(task.job_name, task.task_index, update.message))
            driver.stop()
        elif update.state == mesos_pb2.TASK_LOST:
            logger.warn('Task /job:{0}/task:{1}: {2}'
                    .format(task.job_name, task.task_index, update.message))
            driver.stop()
        elif update.state == mesos_pb2.TASK_ERROR:
            logger.error('Task /job:{0}/task:{1}: {2}'
                    .format(task.job_name, task.task_index, update.message))
            driver.stop()
        else:
            logger.info('Task /job:{0}/task:{1}: {2}'
                    .format(task.job_name, task.task_index, mesos_pb2.TaskState.Name(update.state)))
            driver.stop()


    def slaveLost(self, driver, slaveId):
        logger.error("Slave %s lost:", slaveId.value)

    def executorLost(self, driver, executorId, slaveId, status):
        logger.error("Executor %s lost:", executorId.value)

    def error(self, driver, message):
        logger.error("Mesos error: %s", message)

