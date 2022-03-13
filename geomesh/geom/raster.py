from ast import Mult
from audioop import mul
import logging
import os
from typing import List, Union

import geopandas as gpd
from matplotlib.path import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from pyproj import Transformer, CRS
from shapely import ops
from shapely.geometry import Polygon, MultiPolygon, LinearRing, box
from shapely.validation import make_valid

from geomesh.figures import figure
from geomesh.geom.base import BaseGeom
from geomesh.raster import Raster
# from geomesh import utils

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

    def get_multipolygon(self, zmin: float = None, zmax: float = None, dst_crs=None, nprocs: int = None) -> MultiPolygon:  # type: ignore[override]
        """Returns the shapely.geometry.MultiPolygon object that represents
        the hull of the raster given optional zmin and zmax contraints.
        """

        logger.debug("Generate multipolygon from raster")
        zmin = self.zmin if zmin is None else zmin
        zmax = self.zmax if zmax is None else zmax

        if zmin is None and zmax is None:
            return box(*self.raster.get_bbox().extents)

        if nprocs is not None:
            multipolygon = self._get_multipolygon_parallel(nprocs, zmin, zmax, dst_crs)

        else:
            multipolygon = self._get_multipolygon_serial(zmin, zmax, dst_crs)

        return multipolygon
        
    def _get_multipolygon_serial(self, zmin, zmax, dst_crs) -> MultiPolygon:
        windows = list(self.raster.iter_windows())
        multipolygon_collection: List[MultiPolygon] = []
        for window in windows:
            x, y, z = self.raster.get_window_data(window, band=1)
            multipolygon_collection.append(
                self._build_multipolygon_from_window(x, y, z, zmin, zmax, self.crs, dst_crs)
                )

        if len(windows) > 1:
            logger.info('Calling geopandas unary_union for serial mp generation.')
            return gpd.GeoDataFrame(
                [{'geometry': mp} for mp in multipolygon_collection],
                crs=dst_crs).unary_union
        else:
            return multipolygon_collection.pop()


    def _get_multipolygon_parallel(self, nprocs, zmin, zmax, dst_crs):
        job_args = []
        windows_multipolygons = []
        with Pool(processes=nprocs) as pool:
            for i, window in enumerate(self.raster.iter_windows()):
                logger.info(f"Processing geom for {self.raster.path} in parallel.")
                x, y, z = self.raster.get_window_data(window=window, band=1)
                job_args.append([
                    x,
                    y,
                    z,
                    zmin,
                    zmax,
                    self.crs,
                    dst_crs,


                ])
                if (i+1)%nprocs == 0:
                    results = pool.starmap(
                            self._build_multipolygon_from_window,
                            job_args
                        )
                    windows_multipolygons.extend([result for result in results if result is not None])
                    job_args = []

        pool.join()
        if len(job_args) > 0:
            with Pool(processes=len(job_args)) as pool:

                results = pool.starmap(
                        self._build_multipolygon_from_window,
                        job_args
                    )
                windows_multipolygons.extend([result for result in results if result is not None])
                job_args = []
        pool.join()
        if len(windows_multipolygons) > 0:
            logger.info('Serial unary union for raster mp combine (using ops).')
            return ops.unary_union(windows_multipolygons)
            # logger.info('Serial unary union for raster mp combine (using geopandas).')
            # return gpd.GeoDataFrame(
            #     [{'geometry': mp} for mp in windows_multipolygons],
            #     crs=dst_crs).unary_union
            # return utils.parallel_unary_union(windows_multipolygons, nprocs)
        else:
            return MultiPolygon(Polygon([]))
        # return utils.parallel_unary_union(windows_multipolygons, nprocs)
        # raise NotImplementedError("Create parallelization interface for polygon collection")
        # windows_multipolygons
        # union_result = ops.unary_union(polygon_collection)

        # if isinstance(union_result, Polygon):
        #     union_result = MultiPolygon([union_result])

        # return union_result


    @staticmethod
    def _build_multipolygon_from_window(x, y, z, zmin, zmax, src_crs, dst_crs):

        if zmin is None:
            zmin = -1.e16
        if zmax is None:
            zmax = 1.e16

        if zmin <= np.min(z) and zmax >= np.max(z):
            multipolygon = MultiPolygon([box(np.min(x), np.min(y), np.max(x), np.max(y))])
        elif zmax < np.min(z) or zmin > np.max(z):
            return

        else:
            plt.ioff()
            original_backend = plt.get_backend()
            plt.switch_backend('agg')
            fig, ax = plt.subplots()
            multipolygon = get_multipolygon_from_axes(ax.contourf(x, y, z, levels=[zmin, zmax]))
            plt.close(fig)
            plt.switch_backend(original_backend)
            plt.ion()

        if dst_crs is not None:
            if not isinstance(dst_crs, CRS):
                dst_crs = CRS.from_user_input(dst_crs)
            if not dst_crs.equals(src_crs):
                transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                multipolygon = ops.transform(transformer.transform, multipolygon)
        if isinstance(multipolygon, Polygon):
            multipolygon = MultiPolygon([multipolygon])

        return multipolygon



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
        for polygon in self.get_multipolygon(nprocs=kwargs.get('nprocs')).geoms:
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

    elif len(linear_ring_collection) == 1:
        multipolygon = MultiPolygon([Polygon(linear_ring_collection.pop())])

    else:
        raise Exception(f'unhandled linear_ring_collection output: {linear_ring_collection}')

    if not multipolygon.is_valid:
        final_geoms = []
        for fixed_geom in make_valid(multipolygon).geoms:
            if isinstance(fixed_geom, Polygon):
                final_geoms.append(fixed_geom)
            elif isinstance(fixed_geom, MultiPolygon):
                for polygon in fixed_geom.geoms:
                    final_geoms.append(polygon)
        multipolygon = MultiPolygon(final_geoms)
        
    if isinstance(multipolygon, Polygon):
        multipolygon = MultiPolygon([multipolygon])
            
    logger.debug(f"get mp from axes took {time() - start}")
    return multipolygon
