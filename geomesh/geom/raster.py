from typing import List, Union, Tuple
import logging
import os
import warnings

from jigsawpy import jigsaw_msh_t
from matplotlib.path import Path
from multiprocessing import Pool
from pyproj import Transformer, CRS
from shapely import ops
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import LinearRing
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.validation import make_valid
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from geomesh import utils
from geomesh.geom import quadgen
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
        self._quads_gdf = gpd.GeoDataFrame(
                columns=[
                    'geometry',
                    'group_id',
                    'quad_id',
                    'group_area',
                    'centerline_id',
                    ],
                crs=CRS.from_epsg(4326)
                )

    def get_multipolygon(
            self,
            zmin: float = None,
            zmax: float = None,
            dst_crs=None,
            nprocs: int = None,
            unary_union: bool = True,
            quad_holes: bool = True,
            ) -> Union[MultiPolygon, List[MultiPolygon]]:  # type: ignore[override]
        """Returns the shapely.geometry.MultiPolygon object that represents
        the hull of the raster given optional zmin and zmax contraints.
        """

        logger.debug("Generate multipolygon from raster")
        zmin = self.zmin if zmin is None else zmin
        zmax = self.zmax if zmax is None else zmax

        dst_crs = CRS.from_user_input(dst_crs) if dst_crs else self.crs

        if zmin is None and zmax is None:
            return MultiPolygon([box(*self.raster.get_bbox(dst_crs=dst_crs).extents)])
        if nprocs is not None and nprocs > 1:
            logger.debug('Using parallel multipolygon build')
            # TODO: AttributeError: 'Raster' object has no attribute 'get_window_data'
            multipolygon = self._get_multipolygon_parallel(nprocs, zmin, zmax, dst_crs, unary_union)

        else:
            logger.debug('Using serial multipolygon build')
            multipolygon = self._get_multipolygon_serial(zmin, zmax, dst_crs, unary_union)

        if not quad_holes:
            return multipolygon

        if len(self._quads_gdf) == 0:
            return multipolygon

        quads_gdf = gpd.GeoDataFrame(
                self._quads_gdf.to_crs(dst_crs).groupby('group_id')['geometry'].apply(lambda x: x.unary_union)
                ).reset_index()

        modified_mp = self._remove_interiors_for_quads(multipolygon, quads_gdf)
        return self._add_quad_holes_to_mp(quads_gdf, modified_mp)

    @staticmethod
    def _remove_interiors_for_quads(multipolygon, quads_gdf):
        # Convert the MultiPolygon to a list of Polygons
        polygons = list(multipolygon.geoms)

        # This will hold the updated list of polygons
        new_polygons = []

        # For each Polygon in the MultiPolygon
        for polygon in polygons:
            # Get the exterior of the polygon
            exterior = polygon.exterior

            # This will hold the updated list of interiors (holes) for the current polygon
            new_interiors = []

            # For each interior (hole) in the polygon
            for interior in polygon.interiors:
                # Convert the interior to a Polygon
                hole = Polygon(interior)

                # For each row in quads_gdf
                for _, row in quads_gdf.iterrows():
                    # Get the polygon from the row
                    quad = row.geometry

                    # If the quad intersects the hole, we will exclude this hole
                    # and move on to the next hole
                    if quad.intersects(hole):
                        break
                else:
                    # If we've gone through all the quads and none of them intersect the hole,
                    # we can include this hole in the new list of interiors
                    new_interiors.append(interior)

            # Create a new Polygon with the same exterior but with the updated interiors
            new_polygon = Polygon(exterior, new_interiors)

            # Add the new Polygon to the list of new Polygons
            new_polygons.append(new_polygon)

        # Create a new MultiPolygon from the list of new Polygons
        return MultiPolygon(new_polygons)

    def _add_quad_holes_to_mp(self, quads_gdf, new_mp):
        # Now that we have a MultiPolygon without any intersecting holes,
        # we can apply the difference operation for each quad in quads_gdf
        for _, row in quads_gdf.iterrows():
            quad = row.geometry
            if not quad.intersects(new_mp.boundary):
                new_mp = new_mp.difference(quad)

        return new_mp


    def msh_t(self, *args, dst_crs=None, **kwargs) -> jigsaw_msh_t:
        '''Returns a :class:jigsawpy.jigsaw_msh_t object representing the
        geometry constrained by the arguments.'''
        if len(self._quads_gdf) == 0:
            return super().msh_t(*args, dst_crs=dst_crs, **kwargs)
        logger.info('Generating geom jigsaw_msh_t.')
        src_crs = self.crs
        # dst_crs = kwargs.pop('dst_crs', None) or self.crs
        multipolygon = self.get_multipolygon(
                *args,
                quad_holes=False,
                **kwargs
                )
        quads_gdf = gpd.GeoDataFrame(
                self._quads_gdf.to_crs(dst_crs).groupby('group_id')['geometry'].apply(lambda x: x.unary_union)
                ).reset_index()
        # print(quads_gdf)
        multipolygon = self._remove_interiors_for_quads(multipolygon, quads_gdf)
        multipolygon = interpolate_multipolygon(multipolygon, 100.)
        vert2: List[Tuple[Tuple[float, float], int]] = list()
        edge2: List[Tuple[int, int]] = list()

        for polygon in multipolygon.geoms:
            if np.all(np.asarray(polygon.exterior.coords).flatten() == float('inf')):
                raise NotImplementedError("ellipsoidal-mesh")
            
            # add exterior vertices and edges
            for x, y in polygon.exterior.coords[:-1]:
                vert2.append(((x, y), 0))
            
            _edge2 = list()
            for i in range(len(polygon.exterior.coords)-2):
                _edge2.append((i, i+1))
            _edge2.append((_edge2[-1][1], _edge2[0][0]))
            edge2.extend([(((e0+len(edge2), e1+len(edge2)), 0)) for e0, e1 in _edge2])

            # find corresponding holes in quads_gdf
            hole_gdf = quads_gdf[quads_gdf.within(polygon)]
            # print(hole_gdf)
            for _, hole in hole_gdf.iterrows():
                # Check if the hole is a MultiPolygon
                if hole['geometry'].geom_type == 'MultiPolygon':
                    for poly in hole['geometry'].geoms:
                        for x, y in poly.exterior.coords[:-1]:
                            vert2.append(((x, y), -1))
                        
                        _edge2 = list()
                        for i in range(len(poly.exterior.coords)-2):
                            _edge2.append((i, i+1))
                        _edge2.append((_edge2[-1][1], _edge2[0][0]))
                        edge2.extend([((e0+len(edge2), e1+len(edge2)), -1) for e0, e1 in _edge2])
                else:  # if it's a single Polygon
                    for x, y in hole['geometry'].exterior.coords[:-1]:
                        vert2.append(((x, y), -1))
                    
                    _edge2 = list()
                    for i in range(len(hole['geometry'].exterior.coords)-2):
                        _edge2.append((i, i+1))
                    _edge2.append((_edge2[-1][1], _edge2[0][0]))
                    edge2.extend([((e0+len(edge2), e1+len(edge2)), -1) for e0, e1 in _edge2])

            for interior in polygon.interiors:
                # add interior vertices and edges
                for x, y in interior.coords[:-1]:
                    vert2.append(((x, y), 0))

                _edge2 = list()
                for i in range(len(interior.coords)-2):
                    _edge2.append((i, i+1))
                _edge2.append((_edge2[-1][1], _edge2[0][0]))
                edge2.extend([((e0+len(edge2), e1+len(edge2)), 0) for e0, e1 in _edge2])
        # geom
        msh_t = jigsaw_msh_t()
        msh_t.ndims = +2
        msh_t.mshID = 'euclidean-mesh'
        # TODO: Consider ellipsoidal case.
        # msh_t.mshID = 'euclidean-mesh' if self._ellipsoid is None \
        #     else 'ellipsoidal-mesh'
        msh_t.vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)
        msh_t.edge2 = np.asarray(edge2, dtype=jigsaw_msh_t.EDGE2_t)
        msh_t.crs = src_crs
        if src_crs is not None and dst_crs is not None:
            if not src_crs.equals(dst_crs):
                utils.reproject(msh_t, dst_crs)
        return msh_t

    # def msh_t(self, *args, dst_crs=None, **kwargs) -> jigsaw_msh_t:
    def _get_initial_msh_t(self, *args, dst_crs=None, quads_only=True, **kwargs):
        '''Returns a :class:jigsawpy.jigsaw_msh_t object representing the
        geometry constrained by the arguments.'''
        logger.info('Generating geom jigsaw_msh_t.')
        src_crs = self.crs
        dst_crs = kwargs.pop('dst_crs', None) or self.crs
        multipolygon = self.get_multipolygon(
                *args,
                quad_holes=False,
                **kwargs
                )
        quads_gdf = gpd.GeoDataFrame(
                self._quads_gdf.to_crs(dst_crs).groupby('group_id')['geometry'].apply(lambda x: x.unary_union)
                ).reset_index()
        # print(quads_gdf)
        multipolygon = self._remove_interiors_for_quads(multipolygon, quads_gdf)

        vert2: List[Tuple[Tuple[float, float], int]] = list()
        edge2: List[Tuple[int, int]] = list()

        for polygon in multipolygon.geoms:
            if np.all(np.asarray(polygon.exterior.coords).flatten() == float('inf')):
                raise NotImplementedError("ellipsoidal-mesh")

            # add exterior vertices and edges
            if quads_only is False:
                for x, y in polygon.exterior.coords[:-1]:
                    vert2.append(((x, y), 0))

                _edge2 = list()
                for i in range(len(polygon.exterior.coords)-2):
                    _edge2.append((i, i+1))
                _edge2.append((_edge2[-1][1], _edge2[0][0]))
                edge2.extend([(((e0+len(edge2), e1+len(edge2)), 0)) for e0, e1 in _edge2])

            # find corresponding holes in quads_gdf
            hole_gdf = quads_gdf[quads_gdf.within(polygon)]
            # print(hole_gdf)
            for _, hole in hole_gdf.iterrows():
                # Check if the hole is a MultiPolygon
                if hole['geometry'].geom_type == 'MultiPolygon':
                    for poly in hole['geometry'].geoms:
                        for x, y in poly.exterior.coords[:-1]:
                            vert2.append(((x, y), -1))

                        _edge2 = list()
                        for i in range(len(poly.exterior.coords)-2):
                            _edge2.append((i, i+1))
                        _edge2.append((_edge2[-1][1], _edge2[0][0]))
                        edge2.extend([((e0+len(edge2), e1+len(edge2)), -1) for e0, e1 in _edge2])
                else:  # if it's a single Polygon
                    for x, y in hole['geometry'].exterior.coords[:-1]:
                        vert2.append(((x, y), -1))

                    _edge2 = list()
                    for i in range(len(hole['geometry'].exterior.coords)-2):
                        _edge2.append((i, i+1))
                    _edge2.append((_edge2[-1][1], _edge2[0][0]))
                    edge2.extend([((e0+len(edge2), e1+len(edge2)), -1) for e0, e1 in _edge2])

            if quads_only is False:
                for interior in polygon.interiors:
                    # add interior vertices and edges
                    for x, y in interior.coords[:-1]:
                        vert2.append(((x, y), 0))

                    _edge2 = list()
                    for i in range(len(interior.coords)-2):
                        _edge2.append((i, i+1))
                    _edge2.append((_edge2[-1][1], _edge2[0][0]))
                    edge2.extend([((e0+len(edge2), e1+len(edge2)), 0) for e0, e1 in _edge2])
        # geom
        msh_t = jigsaw_msh_t()
        msh_t.ndims = +2
        msh_t.mshID = 'euclidean-mesh'
        # TODO: Consider ellipsoidal case.
        # msh_t.mshID = 'euclidean-mesh' if self._ellipsoid is None \
        #     else 'ellipsoidal-mesh'
        msh_t.vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)
        msh_t.edge2 = np.asarray(edge2, dtype=jigsaw_msh_t.EDGE2_t)
        msh_t.crs = src_crs
        if src_crs is not None and dst_crs is not None:
            if not src_crs.equals(dst_crs):
                utils.reproject(msh_t, dst_crs)
        return msh_t

    # def _get_initial_msh_t(self):
    #     points = [list(quad.exterior.coords) for quad in self._quads_gdf['geometry']]
    #     points = np.array([item for sublist in points for item in sublist])  # flatten
    #     tri = Delaunay(points)

    def _get_multipolygon_serial(self, zmin, zmax, dst_crs, unary_union) -> MultiPolygon:
        windows = list(self.raster.iter_windows())
        multipolygon_collection: List[MultiPolygon] = []
        for xvals, yvals, zvals in self.raster:
            # x, y, z = self.raster.get_window_data(window, band=1)
            multipolygon_collection.append(
                    self._build_multipolygon_from_window(xvals, yvals, zvals[0, :], zmin, zmax, self.crs, dst_crs)
                )

        if len(windows) > 1:
            if unary_union:
                logger.info('Calling geopandas unary_union for serial mp generation.')
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore",
                            "`unary_union` returned None due to all-None GeoSeries. "
                            "In future, `unary_union` will return 'GEOMETRYCOLLECTION EMPTY' instead.",
                            )
                    return gpd.GeoDataFrame(
                        [{'geometry': mp} for mp in multipolygon_collection],
                        crs=dst_crs).unary_union
            else:
                return multipolygon_collection
        else:
            return multipolygon_collection.pop()

    def _get_multipolygon_parallel(self, nprocs, zmin, zmax, dst_crs, unary_union):
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
                if (i+1) % nprocs == 0:
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
            if unary_union:
                logger.info('Serial unary union for raster mp combine (using ops).')
                return ops.unary_union(windows_multipolygons)
            else:
                return windows_multipolygons
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
        if not np.ma.is_masked(z):
            z = np.ma.masked_array(z)
        if not np.any(z.mask):
            if zmin <= np.min(z) and zmax >= np.max(z):
                return MultiPolygon([box(np.min(x), np.min(y), np.max(x), np.max(y))])
            elif zmax < np.min(z) or zmin > np.max(z):
                return

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

    def make_plot(self, ax=None, **kwargs):
        # TODO: Consider the ellipsoidal case. Refer to commit
        # dd087257c15692dd7d8c8e201d251ab5e66ff67f on main branch for
        # ellipsoidal ploting routing (removed).
        ax = ax or plt.gca()
        for polygon in self.get_multipolygon(nprocs=kwargs.get('nprocs')).geoms:
            ax.plot(*polygon.exterior.xy, color="k")
            for interior in polygon.interiors:
                ax.plot(*interior.xy, color="r")
        return ax

    def triplot(self, show=False, linewidth=0.07, color="black", alpha=0.5, **kwargs):
        plt.triplot(
            self.triangulation, linewidth=linewidth, color=color, alpha=alpha, **kwargs
        )
        if show:
            plt.axis("scaled")
            plt.show()

    @staticmethod
    def resample_linear_ring(linear_ring, segment_length):
        return quadgen.resample_linear_ring(linear_ring, segment_length)

    @staticmethod
    def resample_polygon(polygon, segment_length):
        return quadgen.resample_polygon(polygon, segment_length)

    @staticmethod
    def resample_multipolygon(mp, segment_length):
        return quadgen.resample_multipolygon(mp, segment_length)


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
        return MultiPolygon([Polygon(linear_ring_collection)])
        # raise Exception(f'unhandled linear_ring_collection output: {linear_ring_collection}')
    return make_multipolygon_valid(multipolygon)


def interpolate_polygon(poly, distance):
    # This function interpolates the polygon and returns a new polygon.
    if poly.is_empty:
        return poly
    # Interpolate the exterior ring
    ext = interpolate_linestring(poly.exterior, distance)
    # Interpolate each interior ring
    ints = [interpolate_linestring(ring, distance) for ring in poly.interiors]
    # Return a new polygon with the interpolated rings
    return Polygon(ext, ints)


def interpolate_linestring(line, distance):
    # This function interpolates the linestring and returns a new linestring.
    num_points = int(np.ceil(line.length / distance))
    if num_points <= 1:
        return line
    return LineString([line.interpolate(float(i)/num_points, normalized=True) for i in range(num_points+1)])


def interpolate_multipolygon(mpoly, distance):
    # This function interpolates the multipolygon and returns a new multipolygon.
    if mpoly.is_empty:
        return mpoly
    # Interpolate each polygon
    polys = [interpolate_polygon(poly, distance) for poly in mpoly.geoms]
    # Return a new multipolygon with the interpolated polygons
    return MultiPolygon(polys)


def make_multipolygon_valid(multipolygon):
    if not multipolygon.is_valid:
        final_geoms = []
        fixed_geom = make_valid(multipolygon)
        if isinstance(fixed_geom, Polygon):
            final_geoms.append(fixed_geom)
        elif isinstance(fixed_geom, (GeometryCollection, MultiPolygon)):
            for _fixed_geom in fixed_geom.geoms:
                if isinstance(_fixed_geom, Polygon):
                    final_geoms.append(_fixed_geom)
                elif isinstance(_fixed_geom, MultiPolygon):
                    for polygon in _fixed_geom.geoms:
                        final_geoms.append(polygon)
        elif isinstance(fixed_geom, (LineString, MultiLineString)):
            # sometimes make_valid will yield linestrings, specify ops.unary_union as fix method in this case.
            fixed_geom = ops.unary_union(multipolygon)
            final_geoms.append(fixed_geom)
        elif isinstance(fixed_geom, Point):
            final_geoms.append(MultiPolygon([]))
        else:
            raise ValueError(f'Unhandled geom type: {type(fixed_geom)}')
        multipolygon = MultiPolygon(final_geoms)

    if isinstance(multipolygon, Polygon):
        multipolygon = MultiPolygon([multipolygon])
            
    return multipolygon


# def test_generate_quads():
#     from geomesh import Geom, Raster
#     from appdirs import user_data_dir
#     rootdir = user_data_dir('geomesh')
#     raster = Raster(
#             f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/ncei19_n41x00_w074x00_2015v1.tif',
#             resampling_factor=0.2,
#             )
#     geom = Geom(
#             raster,
#             zmax=20.
#             )
#     geom.generate_quads(
#             # resample_distance=100.,
#             zmax=0.
#             )
#     geom.generate_quads(
#             zmin=0.,
#             zmax=20.
#             # resample_distance=500.,
#             # interpolation_distance=250.,
#             )
#     # geom.generate_quads(
#     #         resample_distance=100.,
#     #         interpolation_distance=50.,
#     #         zmin=0.,
#     #         zmax=20.,
#     #         )

#     from geomesh import utils
#     # geom.make_plot()
#     final_mp = utils.geom_to_multipolygon(geom.msh_t())
#     # for polygon in geom.get_multipolygon(nprocs=None).geoms:
#     for polygon in final_mp.geoms:
#         plt.gca().plot(*polygon.exterior.xy, color="k")
#         for interior in polygon.interiors:
#             plt.gca().plot(*interior.xy, color="r")
#     plt.gca().axis('scaled')
#     plt.show(block=True)
#     # from shapely.geometry import box

#     # Let's assume your GeoDataFrame is named 'gdf'
#     # ... your gdf should already be loaded here ...

#     # Define boundaries for Roosevelt Island (you might want to refine these coordinates)
#     # minx, miny, maxx, maxy = -73.987, 40.730, -73.923, 40.790

#     # # Create a box geometry using these coordinates
#     # roi = box(minx, miny, maxx, maxy)

#     # # Create a GeoDataFrame from the box and set the same CRS as the original gdf
#     # roi_gdf = gpd.GeoDataFrame(gpd.GeoSeries(roi), columns=['geometry'], crs=geom._quads_gdf.crs)

#     # # Find the intersection between your data and the roi
#     # gdf_roi = gpd.overlay(geom._quads_gdf, roi_gdf, how='intersection')

#     # # Now, plot just the data that falls within the roi
#     # fig, ax = plt.subplots()
#     # gdf_roi.plot(ax=ax, facecolor='none', edgecolor='blue')
#     # import contextily as cx
#     # cx.add_basemap(
#     #     plt.gca(),
#     #     source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
#     #     crs=geom._quads_gdf.crs,
#     # )
#     # # geom._quads_gdf.plot(ax=plt.gca(), facecolor='none')


# if __name__ == "__main__":
#     test_generate_quads()
