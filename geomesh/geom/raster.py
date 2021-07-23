import logging
import os
from typing import Union

from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer, CRS
from shapely import ops
from shapely.geometry import Polygon, Point, MultiPolygon, LinearRing

from geomesh.figures import figure
from geomesh.geom.base import BaseGeom
from geomesh.raster import Raster

logger = logging.getLogger(__name__)


class RasterGeom(BaseGeom):
    def __init__(
        self,
        raster: Union[Raster, str, os.PathLike],
        zmin=None,
        zmax=None,
    ):
        """
        Input parameters
        ----------------
        raster:
            Input object used to compute the output mesh hull.
        """
        self.raster = raster  # type: ignore[assignment]
        self.zmin = zmin
        self.zmax = zmax

    def get_multipolygon(self, zmin: float = None, zmax: float = None, dst_crs=None) -> MultiPolygon:  # type: ignore[override]
        """Returns the shapely.geometry.MultiPolygon object that represents
        the hull of the raster given optional zmin and zmax contraints.
        """

        logger.debug("Generate multipolygon from raster")
        zmin = self.zmin if zmin is None else zmin
        zmax = self.zmax if zmax is None else zmax

        if zmin is None and zmax is None:
            return self.raster.get_multipolygon()

        windows = list(self.raster.iter_windows())
        polygon_collection = []
        for window in windows:
            x, y, z = self.raster.get_window_data(window, band=1)
            new_mask = np.full(z.mask.shape, 0)
            new_mask[np.where(z.mask)] = -1
            new_mask[np.where(~z.mask)] = 1

            if zmin is not None:
                new_mask[np.where(z < zmin)] = -1

            if zmax is not None:
                new_mask[np.where(z > zmax)] = -1

            if np.all(new_mask == -1):  # or not new_mask.any():
                continue

            else:
                plt.ioff()
                fig, ax = plt.subplots()
                ax.contourf(x, y, new_mask, levels=[0, 1])
                plt.close(fig)
                plt.ion()
                polygon_collection.extend(get_multipolygon_from_axes(ax).geoms)

        logger.debug("Multipolygon unary_union from raster")
        union_result = ops.unary_union(polygon_collection)

        if isinstance(union_result, Polygon):
            union_result = MultiPolygon([union_result])

        if dst_crs is not None:
            if not isinstance(dst_crs, CRS):
                dst_crs = CRS.from_user_input(dst_crs)
            if not dst_crs.equals(self.crs):
                transformer = Transformer.from_crs(self.crs, dst_crs, always_xy=True)
                union_result = ops.transform(transformer.transform, union_result)
        return union_result

    @property
    def raster(self) -> Raster:
        return self._raster

    @raster.setter
    def raster(self, raster: Union[Raster, str, os.PathLike]):
        if not isinstance(raster, Raster):
            raster = Raster(raster)
        if not isinstance(raster, Raster):
            raise TypeError(
                f"Argument raster must be of type {Raster}, {str} or {os.PathLike}, not type {type(raster)}."
            )
        self._raster = raster

    @property
    def crs(self):
        return self.raster.crs

    @figure
    def make_plot(self, axes=None, **kwargs):
        # TODO: Consider the ellipsoidal case. Refer to commit
        # dd087257c15692dd7d8c8e201d251ab5e66ff67f on main branch for
        # ellipsoidal ploting routing (removed).
        for polygon in self.multipolygon.geoms:
            axes.plot(*polygon.exterior.xy, color="k")
            for interior in polygon.interiors:
                axes.plot(*interior.xy, color="r")
        return axes

    def triplot(self, show=False, linewidth=0.07, color="black", alpha=0.5, **kwargs):
        plt.triplot(
            self.triangulation, linewidth=linewidth, color=color, alpha=alpha, **kwargs
        )
        if show:
            plt.axis("scaled")
            plt.show()


def get_multipolygon_from_axes(ax):
    from time import time

    start = time()
    logger.debug("start get mp from axes")
    # extract linear_rings from plot
    linear_ring_collection = list()
    for path_collection in ax.collections:
        for path in path_collection.get_paths():
            polygons = path.to_polygons(closed_only=True)
            for linear_ring in polygons:
                if linear_ring.shape[0] > 3:
                    linear_ring_collection.append(LinearRing(linear_ring))
    if len(linear_ring_collection) > 1:
        # reorder linear rings from above
        areas = [Polygon(linear_ring).area for linear_ring in linear_ring_collection]
        test_points = [linear_ring.coords[0] for linear_ring in linear_ring_collection]
        idx = np.where(areas == np.max(areas))[0][0]
        polygon_collection = list()
        outer_ring = linear_ring_collection.pop(idx)
        areas.pop(idx)
        test_points.pop(idx)
        path = Path(np.array(outer_ring.coords), closed=True)
        while len(linear_ring_collection) > 0:
            contained_paths = np.where(path.contains_points(np.array(test_points)))[0]
            inner_rings = list()
            for idx in contained_paths[::-1]:
                inner_rings.append(linear_ring_collection.pop(idx))
                areas.pop(idx)
                test_points.pop(idx)
            polygon_collection.append(Polygon(outer_ring, inner_rings))
            if len(linear_ring_collection) > 0:
                idx = np.where(areas == np.max(areas))[0][0]
                outer_ring = linear_ring_collection.pop(idx)
                areas.pop(idx)
                test_points.pop(idx)
                path = Path(np.array(outer_ring.coords), closed=True)
        multipolygon = MultiPolygon(polygon_collection)
    else:
        multipolygon = MultiPolygon([Polygon(linear_ring_collection.pop())])
    logger.debug(f"get mp from axes took {time() - start}")
    return multipolygon
