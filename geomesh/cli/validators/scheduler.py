from copy import deepcopy
# from functools import partial
from multiprocessing import cpu_count

from geomesh.cli import schedulers


class SchedulerConfigValidator:

    def __init__(self, config: dict):
        self.config = config

    @property
    def Scheduler(self):
        # return partial(self._Scheduler, **self.scheduler_requests)
        return self._Scheduler

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: dict):
        self._config = deepcopy(config)
        scheduler_request = self._config.get('scheduler', {'mpi': {'ntasks': cpu_count()}})

        if not isinstance(scheduler_request, dict):
            raise ValueError('Argument `server.scheduler` must be a dictionary.')
        if len(scheduler_request) > 1 or len(scheduler_request) == 0:
            raise ValueError('Argument `server.scheduler` must contain a single key.')

        scheduler_type = list(scheduler_request.keys()).pop()
        # if scheduler_type not in self.allowed_scheduler_types:
        #     raise ValueError(
        #             'Argument `server.scheduler.type` must be one of '
        #             f'{", ".join(self.allowed_scheduler_types)}, not {scheduler_type}'
        #             )
        if scheduler_type.lower() == 'slurm':
            # self._Scheduler = partial(schedulers.SLURMCluster, **scheduler_request[scheduler_type])
            self._Scheduler = schedulers.SLURMCluster
        elif scheduler_type.lower() == 'mpi':
            self._Scheduler = schedulers.MPICluster
        else:
            raise NotImplementedError(
                    'Argument `server.scheduler.type` must be one of '
                    f'{", ".join(self.allowed_scheduler_types)}, not {scheduler_type}'
                    )
        self.scheduler_requests = scheduler_request[scheduler_type]

    @property
    def allowed_scheduler_types(self):
        return ['slurm', 'mpi']
