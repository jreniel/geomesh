from abc import ABC, abstractmethod
import hashlib
import logging
from typing import List, Tuple, Union

from jigsawpy import jigsaw_msh_t
import numpy as np
from pyproj import CRS  # , Transformer
# from shapely import ops
from shapely.geometry import MultiPolygon
# import utm

from geomesh import utils
from geomesh.figures import figure


logger = logging.getLogger(__name__)


class BaseGeom(ABC):
    '''Abstract base class used to construct geomesh "geom" objects.

    More concretely, a "geom" object can be visualized as a collection of
    polygons. In terms of data structures, a collection of polygons can be
    represented as a :class:shapely.geometry.MultiPolygon object, or as a
    :class:jigsawpy.jigsaw_msh_t object.

    A 'geom' object can be visualized as the "hull" of the input object, but
    this should not be confused with the convex hull (the geom object does not
    have to be convex).

    Derived classes from :class:`geomesh.geom.BaseGeom` expose the concrete
    implementation of how to compute this hull based on inputs provided by the
    users.
    '''

    def __init__(self, crs):
        self._crs = crs

    @figure
    def make_plot(self, axes=None, **kwargs):
        for polygon in self.multipolygon.geoms:
            axes.plot(*polygon.exterior.xy, color="k")
            for interior in polygon.interiors:
                axes.plot(*interior.xy, color="r")
        return axes

    @property
    def multipolygon(self) -> MultiPolygon:
        '''Returns a :class:shapely.geometry.MultiPolygon object representing
        the configured geometry.'''
        return self.get_multipolygon()

    def msh_t(self, *args, dst_crs=None, **kwargs) -> jigsaw_msh_t:
        '''Returns a :class:jigsawpy.jigsaw_msh_t object representing the
        geometry constrained by the arguments.'''
        return multipolygon_to_jigsaw_msh_t(
            self.get_multipolygon(*args, **kwargs),
            self.crs,
            dst_crs,
        )

    @abstractmethod
    def get_multipolygon(self, *args, **kwargs) -> MultiPolygon:
        '''Returns a :class:shapely.geometry.MultiPolygon object representing
        the geometry constrained by the arguments.'''
        raise NotImplementedError

    @property
    def md5(self):
        return hashlib.md5(self.multipolygon.wkt.encode('utf-8')).hexdigest()

    @property
    def crs(self) -> Union[CRS, None]:
        return self._crs

    @property
    def _crs(self) -> Union[CRS, None]:
        return self.__crs

    @_crs.setter
    def _crs(self, crs: Union[CRS, str, None]):
        if isinstance(crs, str):
            crs = CRS.from_user_input(crs)

        if not isinstance(crs, (CRS, type(None))):
            raise TypeError(f'Argument crs must be of type {str} or {CRS},'
                            f' not type {type(crs)}.')

        self.__crs = crs


def multipolygon_to_jigsaw_msh_t(
        multipolygon: MultiPolygon,
        src_crs: Union[str, CRS] = None,
        dst_crs: Union[str, CRS] = None,
) -> jigsaw_msh_t:
    '''Casts shapely.geometry.MultiPolygon to jigsawpy.jigsaw_msh_t'''

    logger.info('Generating geom jigsaw_msh_t.')

    if isinstance(src_crs, str):
        src_crs = CRS.from_user_input(src_crs)

    # vert2: List[Tuple[Tuple[float, float], int]] = list()
    # for polygon in multipolygon.geoms:
    #     for x, y in polygon.exterior.coords[:-1]:
    #         vert2.append(((x, y), 0))
    #     for interior in polygon.interiors:
    #         for x, y in interior.coords[:-1]:
    #             vert2.append(((x, y), 0))
    vert2 = [((x, y), 0) for polygon in multipolygon.geoms
             for ring in [polygon.exterior, *polygon.interiors]
             for x, y in ring.coords[:-1]]
    edge2 = []
    offset = 0
    for polygon in multipolygon.geoms:
        for linear_ring in [polygon.exterior, *polygon.interiors]:
            n = len(linear_ring.coords) - 1  # Adjust for the looped back coordinate
            edges = [((i + offset, (i + 1) % n + offset), 0) for i in range(n)]
            edge2.extend(edges)
            offset += n

#     # edge2
#     edge2: List[Tuple[int, int]] = list()
#     for polygon in multipolygon.geoms:
#         polygon = [polygon.exterior, *polygon.interiors]
#         for linear_ring in polygon:
#             _edge2 = list()
#             for i in range(len(linear_ring.coords)-2):
#                 _edge2.append(((i, i+1), 0))
#             _edge2.append((_edge2[-1][1], _edge2[0][0]))
#             edge2.extend(
#                 [((e0+len(edge2), e1+len(edge2), 0))
#                     for e0, e1 in _edge2])
    # geom
    msh_t = jigsaw_msh_t()
    msh_t.ndims = +2
    msh_t.mshID = 'euclidean-mesh'
    # TODO: Consider ellipsoidal case.
    # msh_t.mshID = 'euclidean-mesh' if self._ellipsoid is None \
    #     else 'ellipsoidal-mesh'
    msh_t.vert2 = np.array(vert2, dtype=jigsaw_msh_t.VERT2_t)
    msh_t.edge2 = np.array(edge2, dtype=jigsaw_msh_t.EDGE2_t)
    msh_t.crs = src_crs
    if src_crs is not None and dst_crs is not None:
        if not src_crs.equals(dst_crs):
            utils.reproject(msh_t, dst_crs)
    return msh_t


# def geodetic_to_geocentric(ellipsoid, latitude, longitude, height):
#     """Return geocentric (Cartesian) Coordinates x, y, z corresponding to
#     the geodetic coordinates given by latitude and longitude (in
#     degrees) and height above ellipsoid. The ellipsoid must be
#     specified by a pair (semi-major axis, reciprocal flattening).
#     https://codereview.stackexchange.com/questions/195933/convert-geodetic-coordinates-to-geocentric-cartesian
#     """
#     φ = np.deg2rad(latitude)
#     λ = np.deg2rad(longitude)
#     sin_φ = np.sin(φ)
#     a, rf = ellipsoid           # semi-major axis, reciprocal flattening
#     e2 = 1 - (1 - 1 / rf) ** 2  # eccentricity squared
#     n = a / np.sqrt(1 - e2 * sin_φ ** 2)  # prime vertical radius
#     r = (n + height) * np.cos(φ)   # perpendicular distance from z axis
#     x = r * np.cos(λ)
#     y = r * np.sin(λ)
#     z = (n * (1 - e2) + height) * sin_φ
#     return x, y, z
