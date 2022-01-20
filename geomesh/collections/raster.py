from collections import UserDict
from typing import Dict

from geomesh import Raster


class RasterConfig(UserDict):

    def add_raster(
            self,
            raster: Raster,
            geom_config: Dict = None,
            hfun_config: Dict = None
    ):
        assert isinstance(raster, Raster)
        md5 = raster.md5
        if md5 in self:
            raise ValueError('repeated raster')
        self[md5] = {
            'raster': raster,
            'geom_config': geom_config,
            'hfun_config': hfun_config,
        }

    def __iter__(self):
        for data in self.values():
            yield data
