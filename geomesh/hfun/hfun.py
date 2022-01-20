from enum import Enum
from typing import Union

# from shapely.geometry import MultiPolygon

from geomesh.hfun.base import BaseHfun
from geomesh.hfun.mesh import MeshHfun
from geomesh.hfun.raster import RasterHfun
from geomesh.hfun.raster_tile_index import RasterTileIndexHfun
# from geomesh.mesh import Mesh
from geomesh.raster import Raster
# from geomesh.hfun.shapely import PolygonHfun, MultiPolygonHfun


class HfunInputType(Enum):

    EuclideanMesh2D = MeshHfun
    Raster = RasterHfun
    RasterTileIndex = RasterTileIndexHfun

    @classmethod
    def _missing_(cls, name):
        raise TypeError(f'Unhandled type {name} for argument hfun.')


class Hfun(BaseHfun):
    """
    Factory class that creates concrete instances (subclasses) of Hfun from different sources.
    """

    def __new__(
            cls,
            hfun: Union[Raster],
            **kwargs
    ):
        """
        :param hfun: Object to use as input to compute the output mesh hull.
        """
        return HfunInputType[hfun.__class__.__name__].value(hfun, **kwargs)
