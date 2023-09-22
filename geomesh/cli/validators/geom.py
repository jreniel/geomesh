from copy import deepcopy
from functools import partial

import numpy as np
from psutil import cpu_count

from geomesh.cli.schedulers.cluster_types import ClusterTypes


class GeomConfigValidator:

    def __init__(self, config: dict):
        self.config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: dict):
        config = deepcopy(config)
        config.setdefault('geom', None)
        geom_config = config['geom']
        if geom_config is None:
            return

        # ntasks = geom_config.pop('ntasks', None)
        # if ntasks is None or (isinstance(ntasks, int) and ntasks < 1):
        #     raise ValueError(f'Argument `geom` must have a `ntasks` key with a postive int value.')
        # geom_config['ntasks'] = ntasks
        self._init_geom_raster_opts(geom_config)
        geom_config.setdefault('dst_crs', 'epsg:4326')
        geom_config.setdefault('sieve', False)
        # if geom_config.get('sieve'):
        #     if geom_config.get('sieve') is True:
        #         geom_config.update(sieve=partial(sieve_gdf))
        geom_config.setdefault('to_file', None)
        geom_config.setdefault('to_feather', None)

        self._config = config

    @property
    def ntasks(self):
        return self.config['geom']['ntasks']

    @property
    def geom_config(self):
        return self.config['geom']

    # def _init_ntasks(self, geom_config):
    #     ntasks = geom_config.get('ntasks')

    #     if ntasks is None:
    #         return False

    #     if not isinstance(ntasks, int):
    #         raise ValueError('Argument ntasks must be int >= 1 or None.')

    #     if ntasks < 1:
    #         raise ValueError('Argument ntasks must be int >= 1 or None.')

    #     return ntasks

    def _init_geom_raster_opts(self, geom_config):
        # string is most likely a geometry at rest.
        if isinstance(geom_config, str):
            raise NotImplementedError('Argument config.geom cannot be a string (yet).')

        if 'rasters' not in geom_config and 'features' not in geom_config:
            raise ValueError('`geom` key must contain at least one of `rasters` or `features` keys.')

        elif 'rasters' in geom_config and 'features' not in geom_config:
            geom_request_iter_order = ['rasters']
        elif 'rasters' not in geom_config and 'features' in geom_config:
            geom_request_iter_order = ['features']
        else:
            if list(geom_config.keys()).index('rasters') < list(geom_config.keys()).index('features'):
                geom_request_iter_order = ['rasters', 'features']
            else:
                geom_request_iter_order = ['features', 'rasters']

        geom_config['iter_order'] = geom_request_iter_order

        # if 'rasters' in config_dict:

        # logger.info('Validating user geom rasters requests...')
        # config_dict['geom']['raster_window_requests'] = list(iter_raster_window_requests(config_dict['geom'] , cache_directory=config_dict['cache_directory']))
        # _init_geom_raster_opts(config_dict)







