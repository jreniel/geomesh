import logging
import os
from typing import Union

from jigsawpy import jigsaw_msh_t
from shapely import ops
from shapely.geometry import Polygon, MultiPolygon

from geomesh.figures import figure
from geomesh.geom.base import BaseGeom
from geomesh.mesh.base import BaseMesh
from geomesh.mesh.mesh import Mesh


logger = logging.getLogger(__name__)


class MeshGeom(BaseGeom):
    def __init__(self, mesh: Union[BaseMesh, str, os.PathLike, jigsaw_msh_t]):
        """
        Input parameters
        ----------------
        mesh:
            Input object used to compute the output mesh hull.
        """
        self._mesh = mesh  # type: ignore[assignment]

    def get_multipolygon(self):
        return self.mesh.hull.multipolygon()

    @figure
    def make_plot(self, axes=None, **kwargs):
        for polygon in self.multipolygon.geoms:
            axes.plot(*polygon.exterior.xy, color="k")
            for interior in polygon.interiors:
                axes.plot(*interior.xy, color="r")
        return axes

    @property
    def mesh(self) -> BaseMesh:
        return self._mesh

    @property
    def _mesh(self):
        return self.__mesh

    @_mesh.setter
    def _mesh(self, mesh: Union[BaseMesh, str, os.PathLike, jigsaw_msh_t]):

        if isinstance(mesh, (str, os.PathLike)):
            mesh = Mesh.open(mesh)

        elif isinstance(mesh, jigsaw_msh_t):
            mesh = Mesh(mesh)

        if not isinstance(mesh, BaseMesh):
            raise TypeError(
                f"Argument mesh must be of type {Mesh}, {str} "
                f"or {os.PathLike}, not type {type(mesh)}"
            )

        self.__mesh = mesh

    @property
    def crs(self):
        return self.mesh.crs

    @property
    def freeze_quads(self) -> bool:
        if not hasattr(self, '_freeze_quads'):
            self._freeze_quads = False
        return self._freeze_quads

    @freeze_quads.setter
    def freeze_quads(self, freeze_quads: bool):
        assert isinstance(freeze_quads, bool)
        self._freeze_quads = freeze_quads

    @property
    def _frozen_polygons(self):
        if not hasattr(self, '__frozen_polygons'):
            self.__frozen_polygons = []
        return self.__frozen_polygons

    @property
    def multipolygon(self) -> MultiPolygon:
        '''Returns a :class:shapely.geometry.MultiPolygon object representing
        the configured geometry.'''
        mp = self.get_multipolygon()
        if self.freeze_quads:
            poly_coll = []
            quad_interiors = []
            quad_mp = ops.unary_union(MultiPolygon(
                [Polygon(coords) for coords in self.mesh.coord[self.mesh.elements.quads()]]))
            for poly in mp.geoms:
                for quad_poly in quad_mp.geoms:
                    if poly.intersects(quad_poly.exterior):
                        poly_coll.append(Polygon(poly.exterior, [*poly.interiors, quad_poly.exterior]))
                        quad_interiors.extend([Polygon(geom) for geom in quad_poly.interiors])
            poly_coll.extend(quad_interiors)
            mp = MultiPolygon(poly_coll)
        return mp
