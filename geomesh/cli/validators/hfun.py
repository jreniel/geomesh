from copy import deepcopy
import numpy as np


class HfunConfigValidator:

    def __init__(self, config: dict):
        self.config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: dict):
        config = deepcopy(config)
        config.setdefault('hfun', None)
        self._config = config
        hfun_config = config['hfun']
        if hfun_config is None:
            return
        # ntasks = hfun_config.pop('ntasks', None)
        # if ntasks is None or (isinstance(ntasks, int) and ntasks < 1):
        #     raise ValueError('Argument `hfun` must have a `ntasks` key with a postive int value.')
        # hfun_config['ntasks'] = ntasks

        hfun_config.setdefault('dst_crs', 'epsg:4326')
        self._init_cpus_per_task(hfun_config)
        # self._init_hfun_raster_opts(hfun_config)
        # self._init_cluster_opts(hfun_config)
        # hfun_config.setdefault('sieve', False)
        # if hfun_config.get('sieve'):
        #     if hfun_config.get('sieve') is True:
        #         hfun_config.update(sieve=partial(sieve_gdf))
        # hfun_config.setdefault('to_file', None)
        # hfun_config.setdefault('to_feather', None)

    @property
    def hfun_config(self):
        return self.config['hfun']

    @property
    def ntasks(self):
        return self.config['hfun']['ntasks']

    def _init_cpus_per_task(self, hfun_config):
        cpus_per_task = hfun_config.get('cpus_per_task')
        max_cpus_per_task = hfun_config.get('max_cpus_per_task')
        if np.all([bool(cpus_per_task), bool(max_cpus_per_task)]):
            raise ValueError('Arguments cpus_per_task and max_cpus_per_task are mutually exclusive.')

    def _init_hfun_raster_opts(self, hfun_config):
        hfun_config.setdefault('rasters', [])
        rasters_config = hfun_config['rasters']
        if not isinstance(rasters_config, list):
            raise ValueError('hfun.rasters must be a list.')
        # self._init_raster_contour_requests(hfun_config)
        # _init_constant_value_requests(
        #         # self,
        #         config_dict)
        # _init_gradient_delimiter_request(
        #         # self,
        #         config_dict)
        # _init_raster_features_requests(
        #         # self,
        #         config_dict)
        # _init_raster_narrow_channel_anti_aliasing(
        #         # self,
        #         config_dict)

    # def _init_raster_contour_requests(hfun_config):
    #     for raster_path, hfun_request in iter_raster_requests(config_dict['hfun']):
    #         hfun_request.setdefault('contours', [])
    #         contour_requests = hfun_request['contours']
    #         if not isinstance(contour_requests, list):
    #             raise AttributeError('hfun.contours must be a list.')
    #     pass


