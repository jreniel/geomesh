from copy import deepcopy
from functools import partial

import numpy as np
from psutil import cpu_count
from jigsawpy import jigsaw_jig_t

from geomesh.cli.schedulers.cluster_types import ClusterTypes


class MshtConfigValidator:

    def __init__(self, config: dict):
        self.config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: dict):
        config = deepcopy(config)
        config.setdefault('msh_t', {})
        msh_t_config = config['msh_t']
        self._init_opts(msh_t_config)
        # ntasks = msh_t_config.pop('ntasks', None)
        # if ntasks is None or (isinstance(ntasks, int) and ntasks < 1):
        #     raise ValueError(f'Argument `msh_t` must have a `ntasks` key with a postive int value.')
        # msh_t_config['ntasks'] = ntasks
        # self._init_msh_t_raster_opts(msh_t_config)
        # msh_t_config.setdefault('dst_crs', 'epsg:4326')
        # msh_t_config.setdefault('sieve', False)
        # # if msh_t_config.get('sieve'):
        # #     if msh_t_config.get('sieve') is True:
        # #         msh_t_config.update(sieve=partial(sieve_gdf))
        # msh_t_config.setdefault('to_file', None)
        # msh_t_config.setdefault('to_feather', None)

        self._config = config

    def _init_opts(self, msh_t_config):
        opts_config = msh_t_config.setdefault('opts', {})
        # opts_config = msh_t_config['opts']
        # sample_jcfg = jigsaw_jig_t()
        # for key, val in sample_jcfg.__dict__.items():
        #     opts_config.setdefault(key, opts_config.get(key, val))

    @property
    def msh_t_config(self):
        return self.config['msh_t']

