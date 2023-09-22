from copy import deepcopy
import numpy as np


class QuadsConfigValidator:

    def __init__(self, config: dict):
        self.config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: dict):
        config = deepcopy(config)
        config.setdefault('quads', None)
        self._config = config
        quads_config = config['quads']
        if quads_config is None:
            return

    @property
    def quads_config(self):
        return self.config['quads']

    @property
    def ntasks(self):
        return self.config['quads']['ntasks']

    def _init_quads_raster_opts(self, quads_config):
        quads_config.setdefault('rasters', [])
        rasters_config = quads_config['rasters']
        if not isinstance(rasters_config, list):
            raise ValueError('quads.rasters must be a list.')
        # self._init_raster_contour_requests(quads_config)
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


