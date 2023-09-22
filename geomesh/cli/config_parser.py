from copy import deepcopy
from functools import cached_property
from pathlib import Path
import os

import yaml

from . import validators


class ConfigParser:

    def __init__(self, config_path):
        self.config_path = config_path

    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, config_path):
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f'No file with path {config_path} exists.')
        self._config_path = config_path

    @property
    def config_dict(self):
        if not hasattr(self, '_raw'):
            if str(self.config_path).endswith('.yml') or str(self.config_path).endswith('.yaml'):
                with open(self.config_path) as fh:
                    self._raw = yaml.load(fh, Loader=yaml.SafeLoader)
            else:
                # TODO: Implement json/other config file readers
                raise NotImplementedError("Only yaml is supported for now.")
        # ensures immutability for self.config
        return deepcopy(self._raw)

    @cached_property
    def cache_directory(self):
        cache_directory = self.config_dict.get('cache_directory', Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp')))
        # self.config_dict.setdefault("cache_directory", cache_directory)
        return Path(cache_directory)

    @cached_property
    def geom(self):
        return validators.GeomConfigValidator(self.config_dict)

    @cached_property
    def hfun(self):
        return validators.HfunConfigValidator(self.config_dict)

    @cached_property
    def quads(self):
        return validators.QuadsConfigValidator(self.config_dict)

    @cached_property
    def msh_t(self):
        return validators.MshtConfigValidator(self.config_dict)

    @cached_property
    def scheduler(self):
        return validators.SchedulerConfigValidator(self.config_dict)


    @cached_property
    def boundaries(self):
        return validators.BoundariesConfigValidator(self.config_dict)

    @cached_property
    def interpolate(self):
        return validators.InterpolateConfigValidator(self.config_dict)

    @property
    def Scheduler(self):
        return self.scheduler.Scheduler
