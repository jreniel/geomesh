from copy import deepcopy
import numpy as np

class BoundariesConfigValidator:

    def __init__(self, config: dict):
        self.config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: dict):
        config = deepcopy(config)
        # config.setdefault('hfun', None)
        config.setdefault('boundaries', None)
        self._config = config

    @property
    def boundaries_config(self):
        return self.config['boundaries']

    # @property
    # def ntasks(self):
    #     return self.config['hfun']['ntasks']

    # def _init_cpus_per_task(self, hfun_config):
    #     cpus_per_task = hfun_config.get('cpus_per_task')
    #     max_cpus_per_task = hfun_config.get('max_cpus_per_task')
    #     if np.all([bool(cpus_per_task), bool(max_cpus_per_task)]):
    #         raise ValueError('Arguments cpus_per_task and max_cpus_per_task are mutually exclusive.')
