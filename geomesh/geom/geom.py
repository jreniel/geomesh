from enum import Enum
from typing import Union

# from shapely.geometry import MultiPolygon

from geomesh.geom.base import BaseGeom
from geomesh.geom.mesh import MeshGeom
from geomesh.geom.raster import RasterGeom
from geomesh.mesh import Mesh
from geomesh.raster import Raster
# from geomesh.geom.shapely import PolygonGeom, MultiPolygonGeom


class GeomInputType(Enum):

    Mesh = MeshGeom
    Raster = RasterGeom

    @classmethod
    def _missing_(cls, name):
        raise TypeError(f'Unhandled type {name} for argument geom.')


class Geom(BaseGeom):
    """
    Factory class that creates concrete instances (subclasses) of Geom from different sources.
    """

    def __new__(
            cls,
            geom: Union[Raster, Mesh],
            **kwargs
    ):
        """
        :param geom: Object to use as input to compute the output mesh hull.
        """
        return GeomInputType[geom.__class__.__name__].value(geom, **kwargs)
