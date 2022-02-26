from abc import ABC, abstractmethod
from functools import cached_property
import pathlib
import logging

import yaml

from ... import db


logger = logging.getLogger(__name__)


class YamlComponentParser(ABC):

    def __init__(self, config: "YamlParser"):
        self.config = config

    @property
    @abstractmethod
    def key(self):
        raise NotImplementedError("Attribute `key` must be implemented by subclass.")

class YamlParser:

    VERSION = 0

    def __init__(self, path, cache: db.Cache = None, skip_raster_checks=False):
        self._path = pathlib.Path(path)
        self._cache = cache
        self._features = FeatureConfig(self)
        self._rasters = RasterConfig(self)
        self._geom = GeomConfig(self)
        # self._hfun = HfunConfig(self)


    @property
    def geom(self) -> "GeomConfig":
        return self._geom

    # @property
    # def hfun(self) -> HfunConfig:
    #     return self._hfun

    @property
    def rasters(self) -> "RasterConfig":
        return self._rasters

    @property
    def features(self) -> "FeatureConfig":
        return self._features

    @property
    def cache(self) -> db.Cache:
        return self._cache

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @cached_property
    def yaml(self):
        with open(self.path) as fh:
            logger.info(f"Loading configuration file: {self.path}")
            return yaml.load(fh, Loader=yaml.SafeLoader)

    @property
    def yml(self):
        return self.yaml

# Keep at bottom to avoid circular dep issue.
from .feature import FeatureConfig
from .raster import RasterConfig
from .geom import GeomConfig
