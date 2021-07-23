from enum import Enum
from typing import Union

from shapely.geometry import MultiPolygon, Polygon

from .base import BaseGeom
from .mesh import MeshGeom
from .raster import RasterGeom
from geomesh.mesh.mesh import Mesh
from geomesh.raster import Raster

from geomesh.geom.shapely import PolygonGeom, MultiPolygonGeom


class GeomInputType(Enum):

    Mesh = MeshGeom
    MultiPolygon = MultiPolygonGeom
    Polygon = PolygonGeom
    Raster = RasterGeom

    @classmethod
    def _missing_(cls, name):
        raise TypeError(f"Unhandled type {name} for argument geom.")


class Geom(BaseGeom):
    """
    Factory class that creates concrete instances (subclasses) of Geom from different sources.
    """

    def __new__(cls, geom: Union[Mesh, MultiPolygon, Polygon, Raster], **kwargs):
        """
        :param geom: Object to use as input to compute the output mesh hull.
        """
        return GeomInputType[geom.__class__.__name__].value(geom, **kwargs)
