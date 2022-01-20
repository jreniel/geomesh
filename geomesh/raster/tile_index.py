import logging
import pathlib

import geopandas as gpd

from geomesh.raster.cache import RasterCache
from geomesh.raster.raster import Raster


logger = logging.getLogger(__name__)


class RasterTileIndex:

    def __init__(self, path, cache=None, **kwargs) -> None:
        self._gdf = gpd.read_file(path, **kwargs)
        self._path = path
        self._cache = cache

    def __iter__(self):
        for row in self.gdf.itertuples():
            yield Raster(row.URL if self.cache is None else self.cache.fetch(row.URL))

    @property
    def gdf(self):
        return self._gdf

    @property
    def _gdf(self):
        return self.__gdf

    @_gdf.setter
    def _gdf(self, gdf):
        self.__gdf = gdf

    @property
    def path(self):
        return self._path

    @property
    def cache(self):
        return self._cache

    @property
    def _cache(self):
        return self.__cache

    @_cache.setter
    def _cache(self, cache):
        if cache is False:
            self.__cache = None
        else:
            self.__cache = RasterCache()
