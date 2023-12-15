from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import Polygon, MultiPolygon
from typing import Union

from geomesh.geom.base import BaseGeom

from enum import Enum



class ShapelyGeom(BaseGeom):
    """
    Factory class that creates concrete instances (subclasses) of Geom from different sources.
    """

    def __new__(cls, geom, **kwargs):
        """
        :param geom: Object to use as input to compute the output mesh hull.
        """
        return ShapelyGeomInputType[geom.__class__.__name__].value(geom, **kwargs)


class PolygonGeom(BaseGeom):

    def __init__(self, polygon: Polygon, crs: Union[CRS, str]):
        assert isinstance(polygon, Polygon)
        self._polygon = polygon
        super().__init__(crs)

    def get_multipolygon(self):
        return MultiPolygon([self._polygon])

    @property
    def polygon(self):
        return self._polygon

    @property
    def crs(self):
        return self._crs


class MultiPolygonGeom(BaseGeom):

    def __init__(self, multipolygon: MultiPolygon, crs: Union[CRS, str]):
        assert isinstance(multipolygon, MultiPolygon)
        self._multipolygon = multipolygon
        super().__init__(crs)

    def get_multipolygon(self, dst_crs=None):
        if dst_crs is not None:
            dst_crs = CRS.from_user_input(dst_crs)
            if not self.crs.equals(dst_crs):
                return ops.transform(
                    Transformer.from_crs(
                        self.crs,
                        dst_crs,
                        always_xy=True
                    ).transform,
                    self._multipolygon
                )
        return self._multipolygon

    @property
    def multipolygon(self):
        return self._multipolygon

class ShapelyGeomInputType(Enum):
    
    MultiPolygon = MultiPolygonGeom
    Polygon = PolygonGeom

    @classmethod
    def _missing_(cls, name):
        raise TypeError(f"Unhandled type {name} for argument geom.")
