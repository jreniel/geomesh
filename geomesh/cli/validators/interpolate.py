from copy import deepcopy
import numpy as np

class InterpolateConfigValidator:

    def __init__(self, config: dict):
        self.config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: dict):
        config = deepcopy(config)
        # config.setdefault('hfun', None)
        config.setdefault('interpolate', None)
        self._config = config
        # hfun_config = config['hfun']
        # if hfun_config is None:
        #     return
        # ntasks = hfun_config.pop('ntasks', None)
        # if ntasks is None or (isinstance(ntasks, int) and ntasks < 1):
        #     raise ValueError(f'Argument `hfun` must have a `ntasks` key with a postive int value.')
        # hfun_config['ntasks'] = ntasks

        # hfun_config.setdefault('dst_crs', 'epsg:4326')
        # self._init_cpus_per_task(hfun_config)
        # self._init_hfun_raster_opts(hfun_config)
        # self._init_cluster_opts(hfun_config)
        # hfun_config.setdefault('sieve', False)
        # if hfun_config.get('sieve'):
        #     if hfun_config.get('sieve') is True:
        #         hfun_config.update(sieve=partial(sieve_gdf))
        # hfun_config.setdefault('to_file', None)
        # hfun_config.setdefault('to_feather', None)

    @property
    def interpolate_config(self):
        return self.config['interpolate']

    @property
    def ntasks(self):
        return self.config['interpolate']['ntasks']

    # def _init_cpus_per_task(self, hfun_config):
    #     cpus_per_task = hfun_config.get('cpus_per_task')
    #     max_cpus_per_task = hfun_config.get('max_cpus_per_task')
    #     if np.all([bool(cpus_per_task), bool(max_cpus_per_task)]):
    #         raise ValueError('Arguments cpus_per_task and max_cpus_per_task are mutually exclusive.')
