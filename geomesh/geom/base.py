from abc import ABC, abstractmethod
import hashlib
import logging
from typing import List, Tuple, Union

from jigsawpy import jigsaw_msh_t
import numpy as np
from pyproj import CRS, Transformer
# from shapely import ops
from shapely.geometry import MultiPolygon
# import utm


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

    @property
    def multipolygon(self) -> MultiPolygon:
        '''Returns a :class:shapely.geometry.MultiPolygon object representing
        the configured geometry.'''
        return self.get_multipolygon()

    def msh_t(self, dst_crs=None, **kwargs) -> jigsaw_msh_t:
        '''Returns a :class:jigsawpy.jigsaw_msh_t object representing the
        geometry constrained by the arguments.'''
        return multipolygon_to_jigsaw_msh_t(
            self.get_multipolygon(**kwargs),
            self.crs,
            dst_crs,
        )

    @abstractmethod
    def get_multipolygon(self) -> MultiPolygon:
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

    vert2: List[Tuple[Tuple[float, float], int]] = list()
    for polygon in multipolygon:
        if np.all(
                np.asarray(
                    polygon.exterior.coords).flatten() == float('inf')):
            raise NotImplementedError("ellispoidal-mesh")
        for x, y in polygon.exterior.coords[:-1]:
            vert2.append(((x, y), 0))
        for interior in polygon.interiors:
            for x, y in interior.coords[:-1]:
                vert2.append(((x, y), 0))

    # edge2
    edge2: List[Tuple[int, int]] = list()
    for polygon in multipolygon:
        polygon = [polygon.exterior, *polygon.interiors]
        for linear_ring in polygon:
            _edge2 = list()
            for i in range(len(linear_ring.coords)-2):
                _edge2.append((i, i+1))
            _edge2.append((_edge2[-1][1], _edge2[0][0]))
            edge2.extend(
                [(e0+len(edge2), e1+len(edge2))
                    for e0, e1 in _edge2])
    # geom
    mesh_t = jigsaw_msh_t()
    mesh_t.ndims = +2
    mesh_t.mshID = 'euclidean-mesh'
    # TODO: Consider ellipsoidal case.
    # mesh_t.mshID = 'euclidean-mesh' if self._ellipsoid is None \
    #     else 'ellipsoidal-mesh'
    mesh_t.vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)
    mesh_t.edge2 = np.asarray(
        [((e0, e1), 0) for e0, e1 in edge2],
        dtype=jigsaw_msh_t.EDGE2_t)
    mesh_t.crs = src_crs
    if src_crs is not None and dst_crs is not None:
        if not src_crs.equals(dst_crs):
            raise NotImplementedError('transform msh_t with utils')
    return mesh_t
