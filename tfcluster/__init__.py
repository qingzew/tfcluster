from contextlib import contextmanager
from tfcluster.scheduler import Job, TFClusterScheduler

__VERSION__ = '0.0.1'


@contextmanager
def cluster(jobs, master = None, name = None,
            return_target = True, quiet = False):
    if isinstance(jobs, dict):
        jobs = [Job(**jobs)]

    if isinstance(jobs, Job):
        jobs = [jobs]

    jobs = [job if isinstance(job, Job) else Job(**job)
            for job in jobs]
    s = TFClusterScheduler(jobs, master = master, name = name,
                           return_target = return_target, quiet = quiet)
    yield s.start()
    s.stop()
