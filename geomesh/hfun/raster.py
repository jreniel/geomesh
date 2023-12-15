from multiprocessing import cpu_count, Pool
from functools import lru_cache
from copy import copy
from time import time
from typing import Union, List
import logging
import os
import tempfile
import warnings

from centerline.geometry import Centerline
from jigsawpy import jigsaw_msh_t, jigsaw_jig_t
from jigsawpy import libsaw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS, Transformer
from rasterio.io import MemoryFile
from scipy.spatial import KDTree
from shapely import ops
from shapely.geometry import LineString
from shapely.geometry import LinearRing
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import centerline
import fiona
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import rasterio
import scipy

from geomesh import utils, Geom
from geomesh.figures import figure
from geomesh.geom.shapely_geom import PolygonGeom
from geomesh.hfun.base import BaseHfun
from geomesh.raster import Raster, get_iter_windows, get_window_data


logger = logging.getLogger(__name__)


warnings.filterwarnings(
        'ignore',
        message='converting a masked element to nan.'
        )


class RasterHfun(BaseHfun, Raster):
    def __init__(
            self,
            raster: Raster,
            hmin: float = None,
            hmax: float = None,
            verbosity=0,
            geom=None,
            marche=False,
            nprocs=True,
            init=None,
            ):
        self._raster = raster
        self._hmin = float(hmin) if hmin is not None else hmin
        self._hmax = float(hmax) if hmax is not None else hmax
        self._verbosity = int(verbosity)
        self.geom = geom
        self.marche = marche
        self.nprocs = nprocs
        self.init = init

    def __iter__(self):
        iter_windows = list(self.iter_windows())
        for i, (xvals, yvals, zvals) in enumerate(self.raster):
            window = iter_windows[i]
            irange, jrange = rasterio.windows.toranges(iter_windows[i])
            zvals = self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])]
        #     x, y, zvals = get_window_data(
        #             self.src,
        #             window,
        #             band=None,
        #             masked=True,
        #             resampling_method=self.resampling_method,
        #             resampling_factor=self.resampling_factor,
        #             clip=self.clip,
        #             )
            # irange, jrange = rasterio.windows.toranges(window)
            # self._fp.flush()
            # zvals = self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])]
            # self._fp.flush()
            yield xvals, yvals, zvals

    @lru_cache
    def msh_t(
        self,
        # window: rasterio.windows.Window = None,
        marche: bool = None,
        verbosity=None,
        dst_crs=None,
        init=None,
        nprocs: int = None,
        # opts: jigsaw_jig_t = None,
        **opts_kwargs,
    ) -> jigsaw_msh_t:

        logger.debug("Begin generation of hfun msh_t")
        marche = bool(self.marche) if marche is None else marche
        nprocs = nprocs if nprocs is not None else self.nprocs
        verbosity = verbosity if verbosity is not None else self.verbosity

        if self.geom is not None:
            if hasattr(self.geom, '_quads_gdf'):
                if len(self.geom._quads_gdf) > 0:
                    self._add_quad_sizes(self.geom._quads_gdf)
        meshed_windows = []

        for (xvals, yvals, zvals), (rx, ry, rvals) in zip(self, self.raster):
            if np.all(rvals.mask):
                continue
            if np.any(np.isnan(zvals)):
                outshape = zvals.shape
                x, y = np.meshgrid(xvals, yvals)
                x = x.flatten()
                y = y.flatten()
                xy = np.vstack([x, y]).T
                zvals = zvals.flatten()
                nan_idxs = np.where(np.isnan(zvals))[0]
                q_non_nan = np.where(~np.isnan(zvals))[0]
                zvals[nan_idxs] = scipy.interpolate.griddata(
                    xy[q_non_nan, :],
                    zvals[q_non_nan],
                    xy[nan_idxs, :],
                    method='linear',
                    fill_value=np.nan,
                )
                zvals = zvals.reshape(outshape)
                del x, y, xy, nan_idxs, q_non_nan, outshape

            hfun = jigsaw_msh_t()
            hfun.ndims = +2
            if self.crs.is_geographic:
                hfun.mshID = 'euclidean-mesh'
                local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={np.median(yvals)} +lon_0={np.median(xvals)}"
                local_crs = CRS.from_user_input(local_azimuthal_projection)
                transformer = Transformer.from_crs(self.crs, local_crs, always_xy=True)
                geographic_to_local = transformer.transform

                dim1, dim2 = zvals.shape
                tria3 = np.empty((2, dim1 - 1, dim2 - 1, 3), dtype=int)

                # Generate a 2D array of indices
                indices = np.arange(dim1 * dim2).reshape(dim1, dim2)

                # Define the two triangles in each grid cell
                tria3[0, :, :, 0] = indices[:-1, :-1]
                tria3[0, :, :, 1] = indices[1:, :-1]
                tria3[0, :, :, 2] = indices[1:, 1:]
                
                tria3[1, :, :, 0] = indices[:-1, :-1]
                tria3[1, :, :, 1] = indices[1:, 1:]
                tria3[1, :, :, 2] = indices[:-1, 1:]

                hfun.tria3 = np.array([(_, 0) for _ in tria3.reshape(-1, 3)], dtype=jigsaw_msh_t.TRIA3_t)
                hfun.vert2 = np.empty(
                    zvals.size, dtype=jigsaw_msh_t.VERT2_t
                )
                hfun.vert2["coord"][:] = np.array(
                    np.vstack(
                        geographic_to_local(*[_.flatten() for _ in np.meshgrid(xvals, yvals)])
                        ).T
                )
                hfun.value = np.array(
                    zvals
                    .flatten()
                    .reshape((zvals.shape[1] * zvals.shape[0], 1)),
                    dtype=jigsaw_msh_t.REALS_t,
                )

                # verify
                # utils.tricontourf(hfun)
                # utils.triplot(hfun)
                # plt.show(block=True)

                logger.debug("Done building hfun.value...")

                # Build Geom
                logger.debug("Building initial geom...")

                if self.geom is None:
                    bbox = [
                        (np.min(xvals), np.min(yvals)),
                        (np.max(xvals), np.min(yvals)),
                        (np.max(xvals), np.max(yvals)),
                        (np.min(xvals), np.max(yvals)),
                        (np.min(xvals), np.min(yvals)),
                    ]

                    geom = PolygonGeom(Polygon(bbox), self.crs).msh_t(dst_crs=CRS.from_user_input(local_azimuthal_projection))
                else:
                    # import contextily as cx
                    # self.geom.make_plot(axes=plt.gca())
                    # cx.add_basemap(plt.gca(), crs=self.geom.crs)
                    # import os
                    # plt.title(os.getpid())
                    # plt.show()
                    # TODO: Hardcided IDtag = -1
                    geom = self.geom.msh_t(dst_crs=local_crs)
                    # print(geom.crs)
                    # print(geom.vert2)
                    # print(geom.edge2)
                    # raise
                    # geom.vert2['IDtag'] = -1
                # utils.reproject(geom, )

                logger.debug("Building initial geom done.")
                kwargs = {"method": "nearest"}

            else:
                # TODO: undefined variables below
                raise NotImplementedError('fix hfun for euclidean-grid case')
                # logger.debug("Forming initial hmat (euclidean-grid).")
                # hfun.mshID = "euclidean-grid"
                # hfun.xgrid = np.array(
                #     np.array(self.get_x(window=window)), dtype=jigsaw_msh_t.REALS_t
                # )
                # hfun.ygrid = np.array(
                #     np.flip(self.get_y(window=window)), dtype=jigsaw_msh_t.REALS_t
                # )
                # hfun.value = np.array(
                #     np.flipud(self.get_values(window=window, band=1)),
                #     dtype=jigsaw_msh_t.REALS_t,
                # )
                # kwargs = {"kx": 1, "ky": 1}  # type: ignore[dict-item]
                # geom = PolygonGeom(box(x0, y0, x1, y1), self.crs).msh_t()

            logger.debug("Configuring jigsaw...")
            opts = copy(opts_kwargs.get("opts", None)) or jigsaw_jig_t()
            opts.mesh_dims = getattr(opts, 'mesh_dims', None) or opts_kwargs.get('mesh_dims', 2)
            opts.numthread = getattr(opts, 'numthread', None) or opts_kwargs.get('numthread', nprocs)
            opts.hfun_scal = getattr(opts, 'hfun_scal', None) or opts_kwargs.get('hfun_scal', 'absolute')
            opts.optm_tria = getattr(opts, 'optm_tria', None) or opts_kwargs.get('optm_tria', None)
            opts.mesh_iter = getattr(opts, 'mesh_iter', None) or opts_kwargs.get('mesh_iter', None)
            opts.mesh_rad2 = getattr(opts, 'mesh_rad2', None) or opts_kwargs.get('mesh_rad2', None)  # should be float or None
            opts.verbosity = getattr(opts, 'verbosity', None) or opts_kwargs.get('verbosity', verbosity)  # should be float or None

            opts.hfun_hmin = opts_kwargs.get("hfun_hmin", self.hmin) if not hasattr(opts, "hfun_hmin") else opts.hfun_hmin or np.min(hfun.value)
            if self.hmin is not None:
                hfun.value[np.where(hfun.value < self.hmin)] = self.hmin

            opts.hfun_hmax = opts_kwargs.get("hfun_hmax", self.hmax) if not hasattr(opts, "hfun_hmax") else opts.hfun_hmax or np.max(hfun.value)
            if self.hmax is not None:
                hfun.value[np.where(hfun.value > self.hmax)] = self.hmax

            window_mesh = jigsaw_msh_t()
            window_mesh.mshID = "euclidean-mesh"
            window_mesh.ndims = +2

            if marche is True:
                logger.info("Launching libsaw.marche...")
                libsaw.marche(opts, hfun)

            logger.info("Launching libsaw.jigsaw...")
            # from jigsawpy import savemsh, savejig
            # pathlib.Path('jigsawpy_debug').mkdir(exist_ok=True)
            # savemsh('jigsawpy_debug/hfun.msh', hfun)
            # savemsh('jigsawpy_debug/geom.msh', geom)
            # savejig('jigsawpy_debug/opts.jig', opts)
            libsaw.jigsaw(
                    opts=opts,
                    geom=geom,
                    mesh=window_mesh,
                    init=init or self.init,
                    hfun=hfun
                    )

            # do post processing
            utils.interpolate(hfun, window_mesh, **kwargs)
            # utils.tricontourf(window_mesh, axes=ax, show=True)
            if local_azimuthal_projection is not None:
                window_mesh.crs = CRS.from_user_input(local_azimuthal_projection)
                # logger.debug(
                #     f"Projecting mesh window {i+1} of {tot} back to original CRS..."
                # )
                utils.reproject(window_mesh, self.crs)
                logger.debug("Done reprojecting.")
            else:
                window_mesh.crs = self.crs

            # import matplotlib.pyplot as plt
            # ax = utils.triplot(window_mesh, axes=plt.gca())
            # ax.set_title(self.raster.path)
            # import contextily as cx
            # print(self.crs)
            # cx.add_basemap(
            #         ax,
            #         source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            #     )
            # plt.show()
            meshed_windows.append(window_mesh)

            # logger.debug(
            #     f"Finished processing window {i+1} of {tot}. Took {time()-window_start}."
            # )

        if len(meshed_windows) == 1:
            output_mesh = meshed_windows.pop()
            utils.reproject(output_mesh, self.crs)
        else:
            output_mesh = jigsaw_msh_t()
            output_mesh.mshID = "euclidean-mesh"
            output_mesh.ndims = +2
            output_mesh.crs = self.crs
            for window_mesh in meshed_windows:
                window_mesh.tria3["index"] += len(output_mesh.vert2)
                output_mesh.tria3 = np.append(
                    output_mesh.tria3, window_mesh.tria3, axis=0
                )
                # x, y = Transformer.from_crs(window_mesh.crs, self.crs, always_xy=True).transform(window_mesh.vert2['coord'][:, 0], window_mesh.vert2['coord'][:, 1])
                utils.reproject(window_mesh, self.crs)
                output_mesh.vert2 = np.append(
                    output_mesh.vert2,
                    window_mesh.vert2,
                    # np.array(np.vstack([x, y]).T, dtype=jigsaw_msh_t.VERT2_t),

                    axis=0
                )
                output_mesh.value = np.append(output_mesh.value, window_mesh.value)
        # output_mesh.crs = self.crs
        if dst_crs is not None:
            dst_crs = CRS.from_user_input(dst_crs)
            if not dst_crs.equals(self.crs):
                utils.reproject(output_mesh, dst_crs)
        # verify
        # ax = utils.triplot(output_mesh, axes=plt.gca())
        # ax.set_title(self.raster.path)
        # import contextily as cx
        # # print(self.crs)
        # cx.add_basemap(
        #         ax,
        #         source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        #         crs=self.crs,
        #     )
        # plt.show(block=True)
        return output_mesh

    def add_contour(
        self,
        level: Union[List[float], float],
        expansion_rate: float,
        target_size: float = None,
        nprocs: int = None,
    ):
        """See https://outline.com/YU7nSM for an excellent explanation about
        tree algorithms.
        """
        contours = []
        for xvals, yvals, zvals in self.raster:
            zvals = zvals[0, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                logger.debug("Computing contours...")
                start = time()
                plt.ioff()
                fig, ax = plt.subplots()
                ax.contour(xvals, yvals, zvals, levels=[level])
                logger.debug(f"Took {time()-start}...")
                plt.close(fig)
                plt.ion()
            # features = []
            for path_collection in ax.collections:
                for path in path_collection.get_paths():
                    try:
                        # features.append(LineString(path.vertices))
                        contours.append(LineString(path.vertices))
                    except ValueError:
                        # LineStrings must have at least 2 coordinate tuples
                        pass
            # contours.append(ops.linemerge(features))
        self.add_feature(MultiLineString(contours), expansion_rate, target_size, nprocs)

    def add_feature(
        self,
        feature: Union[LineString, MultiLineString],
        expansion_rate: float,
        target_size: float = None,
        nprocs=None,
    ):
        """Adds a linear distance size function constraint to the mesh.

        Arguments:
            feature: shapely.geometryLineString or MultiLineString
        """
        nprocs = nprocs if nprocs is not None else self.nprocs
        if not isinstance(feature, (LineString, MultiLineString)):
            raise TypeError(
                f"Argument feature must be of type {LineString} or "
                f"{MultiLineString}, not type {type(feature)}."
            )

        if isinstance(feature, LineString):
            feature = [feature]

        elif isinstance(feature, MultiLineString):
            feature = [linestring for linestring in feature.geoms]

        _gdf = gpd.GeoDataFrame([
            {"geometry": geometry} for geometry in feature
            ], crs=self.crs)

        target_size = self.hmin if target_size is None else target_size
        if target_size is None:
            raise ValueError(
                "Argument target_size must be specified if no "
                "global hmin has been set."
            )
        if target_size <= 0:
            raise ValueError("Argument target_size must be greater than zero.")

        local_azimuthal_projection = None
        iter_windows = list(self.iter_windows())
        tot = len(iter_windows)

        # for i, window in enumerate(iter_windows):
        for i, (xvals, yvals, zvals) in enumerate(self.raster):
            window = iter_windows[i]
            irange, jrange = rasterio.windows.toranges(iter_windows[i])
            zvals = self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])]
            # logger.debug(f'iter: {i}, window={window}')
            # print(f'iter: {i}, window={window}')
            # logger.info(f"Processing window {i+1}/{len(iter_windows)}.")
            if self.crs.is_geographic:
                # x0, y0, x1, y1 = self.get_window_bounds(window)
                local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m ' \
                        '+lat_0={np.median(yvals)} +lon_0={np.median(xvals)}"
                local_crs = CRS.from_user_input(local_azimuthal_projection)
                transformer = Transformer.from_crs(self.crs, local_crs, always_xy=True)
                geographic_to_local = transformer.transform
                _gdf = _gdf.to_crs(local_crs)

            # feature = _gdf.clip(box(*[
            #     np.min(xvals) - 0.5*(np.max(xvals) - np.min(xvals)),
            #     np.min(yvals) - 0.5*(np.max(yvals) - np.min(yvals)),
            #     np.max(xvals) + 0.5*(np.max(xvals) - np.min(xvals)),
            #     np.max(yvals) + 0.5*(np.max(yvals) - np.min(yvals)),
            #     ]))

            # if feature is None:
            #     continue

            # feature.plot()
            # plt.show()
            feature = [row.geometry for row in _gdf.itertuples()]
            if len(feature) == 0:
                continue
            # logger.info("Repartitioning features...")
            # max_verts = 200
            # start = time()
            # if nprocs == 1:
            #     res = [repartition_features(linestring, max_verts) for linestring in feature]
            # else:
            #     with Pool(processes=nprocs) as pool:
            #         res = pool.starmap(
            #             repartition_features,
            #             [(linestring, max_verts) for linestring in feature],
            #         )
            #     pool.join()
            # logger.debug(f"Repartitioning features took {time()-start}.")

            # feature = reduce(operator.iconcat, res, [])

            logger.info("Resampling features...")
            start = time()
            if nprocs == 1:
                transformed_features = [transform_linestring(linestring, target_size) for linestring in feature]
            else:
                with Pool(processes=nprocs) as pool:
                    transformed_features = pool.starmap(
                        transform_linestring,
                        [
                            (
                                linestring, target_size,
                                # self.src.crs, local_crs
                            )
                            for linestring in feature
                        ],
                    )
                pool.join()
            logger.debug(f"Resampling features took {time()-start} on {nprocs=}.")

            # logger.info("Concatenating points...")
            # start = time()
            points = []
            for geom in transformed_features:
                if isinstance(geom, LineString):
                    points.extend(geom.coords)
                elif isinstance(geom, MultiLineString):
                    for linestring in geom.geoms:
                        points.extend(linestring.coords)
            # logger.debug(f"Point concatenation took {time()-start}.")

            start = time()
            logger.info("Generate mesh grid...")
            if self.crs.is_geographic:
                x, y = geographic_to_local(*[_.flatten() for _ in np.meshgrid(xvals, yvals)])
            else:
                x, y = np.meshgrid(xvals, yvals)
            xy = np.vstack([x, y]).T
            logger.debug(f"Generate meshgrid took {time()-start}.")

            logger.debug("Querying KDTree...")
            start = time()
            points = np.array(points)
            distances, _ = KDTree(np.array(points)).query(xy, workers=nprocs)
            logger.debug(f'Querying KDTree took {time()-start} for {len(points)=} on {nprocs=}.')

            values = expansion_rate * target_size * distances + target_size

            values = values.reshape(
                    *zvals.shape,
                    # window.height, window.width
                    )
            if self.hmin is not None:
                values[np.where(values < self.hmin)] = self.hmin
            if self.hmax is not None:
                values[np.where(values > self.hmax)] = self.hmax
            values = np.minimum(zvals.data, values)
            irange, jrange = rasterio.windows.toranges(iter_windows[i])
            self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])] = values
            self._fp.flush()


    # def add_quads_sizes(self, quads_gdf):



    def add_narrow_channel_anti_aliasing(
            self,
            threshold_size=None,
            resample_distance=None,
            simplify_tolerance=None,
            interpolation_distance=None,
            cross_section_node_count=4,
            min_ratio=0.1,
            min_area=np.finfo(np.float64).min,
            lower_bound=None,
            upper_bound=None,
            zmin=None,
            zmax=None,
            hmin=None,
            hmax=None,
            nprocs=None,
            exclude_gdf=None,
            ):

        # old_resampling_factor = self.raster.resampling_factor
        # self.raster.resampling_factor = 1
        threshold_size = float(threshold_size) if threshold_size is not None else np.finfo(np.float32).max
        nprocs = nprocs if nprocs is not None else self.nprocs

        if zmax is None:
            zmax = 0 if zmin is None else np.finfo(np.float64).max

        geom = Geom(self.raster, zmin=zmin, zmax=zmax)
        mp = geom.get_multipolygon()
        # self.raster.resampling_factor = old_resampling_factor

        if mp is None:
            return

        centroid = np.array(mp.centroid.coords).flatten()
        local_azimuthal_projection = CRS.from_user_input(
            f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
            )
        gdf = gpd.GeoDataFrame([{'geometry': mp}], crs=self.raster.crs)
        original_gdf = gdf.to_crs(local_azimuthal_projection)
        # original_gdf = original_gdf.simplify(tolerance=0.1, preserve_topology=True)
        buffer_size = threshold_size * cross_section_node_count
        buffered_geom = original_gdf.unary_union.buffer(-buffer_size).buffer(buffer_size)
        if not buffered_geom.is_empty:
            final_patches = original_gdf.unary_union.difference(
                gpd.GeoDataFrame([{'geometry': buffered_geom}], crs=original_gdf.crs).unary_union
                    )
        else:
            final_patches = original_gdf.unary_union
        if final_patches.is_empty:
            return
        from shapely.geometry import GeometryCollection
        if isinstance(final_patches, GeometryCollection):
            return

        elif isinstance(final_patches, Polygon):
            final_patches = [final_patches]
        elif isinstance(final_patches, MultiPolygon):
            final_patches = [patch for patch in final_patches.geoms]
        final_patches = [
                patch for patch in final_patches
                if patch.is_valid and not patch.is_empty
                and patch.area > min_area
                and (patch.length / patch.area) < min_ratio
                and isinstance(patch, (Polygon, MultiPolygon))
            ]
        from time import time
        print('will compute centerlines')
        start = time()
        with Pool(nprocs) as pool:
            if simplify_tolerance is not None:
                final_patches = pool.starmap(
                    simplify_geometry,
                    [(patch, simplify_tolerance, True) for patch in final_patches]
                    )
            if resample_distance is not None:
                final_patches = pool.starmap(
                    resample_polygon,
                    [(patch, resample_distance) for patch in final_patches]
                    )
            final_patches = [patch for patch in final_patches if isinstance(patch, (Polygon, MultiPolygon))]
            all_centerlines = pool.starmap(
                    get_centerlines,
                    [(patch, interpolation_distance) for patch in final_patches]
                    )
        print(f'computing centerlines took {time()-start}')

        patches_and_lines = [(patch, cline) for patch, cline in zip(final_patches, all_centerlines) if cline is not None]
        final_patches = [patch for patch, cline in patches_and_lines]
        if len(final_patches) == 0:
            return
        all_centerlines = [cline for patch, cline in patches_and_lines]
        del patches_and_lines
        print('will get mp_points_values')
        start = time()
        mp_points, mp_values = get_mp_points_values(
                self.raster,
                final_patches,
                zmin,
                zmax,
                local_azimuthal_projection,
                nprocs,
                cross_section_node_count,
                all_centerlines
                )
        if len(mp_points) == 0:
            return
        print(f'getting mp_points_values took {time()-start}')
        kwargs = self.src.profile.copy()
        kwargs.update(
                dtype=rasterio.int8,
                nodata=0,
                )
        final_patches = gpd.GeoDataFrame([{'geometry': patch} for patch in final_patches], crs=original_gdf.crs)
        # eliminate invalid patches
        if exclude_gdf is not None:
            exclude_gdf = exclude_gdf.to_crs(original_gdf.crs)
            # Perform difference operation for each polygon
            for idx, row in exclude_gdf.iterrows():
                try:
                    final_patches['geometry'] = final_patches.geometry.apply(lambda x: x.difference(row.geometry))
                except:
                    pass
            # Remove empty geometries
            final_patches = final_patches.loc[~final_patches.geometry.is_empty]
            final_patches = final_patches.loc[final_patches.geometry.is_valid]
        final_patches.to_crs(self.raster.crs, inplace=True)
        final_patches = [row.geometry for row in final_patches.itertuples()]
        # final_patches = final_patches.unary_union
        # if isinstance(final_patches, Polygon):
        #     final_patches = [final_patches]
        # elif isinstance(final_patches, MultiPolygon):
        #     final_patches = [p for p in final_patches.geoms]

        with MemoryFile() as memfile:
            with memfile.open(**kwargs) as dataset:
                new_mask, _, _ = rasterio.mask.raster_geometry_mask(
                        dataset, final_patches, crop=False)
        iter_windows = list(self.iter_windows())
        # plt.scatter(mp_points[:, 0], mp_points[:, 1], c=mp_values)
        # plt.show(block=True)
        # exit()

        for i, (xvals, yvals, zvals) in enumerate(self.raster):
            zvals = zvals[0, :]
            window = iter_windows[i]
            irange, jrange = rasterio.windows.toranges(window)
            iidxs = (int(irange[0]), int(irange[1]))
            jidxs = (int(jrange[0]), int(jrange[1]))
            window_mask = new_mask[iidxs[0]:iidxs[1], jidxs[0]:jidxs[1]]
            if np.all(window_mask):
                continue
            if upper_bound is not None:
                window_mask[np.where(zvals >= upper_bound)] = True
            if lower_bound is not None:
                window_mask[np.where(zvals <= lower_bound)] = True
            idxs = np.where(~window_mask)
            x, y = np.meshgrid(xvals, yvals)
            x = x[idxs]
            y = y[idxs]
            x = x.flatten()
            y = y.flatten()
            xy = np.vstack([x, y]).T
            outvals = scipy.interpolate.griddata(
                    mp_points,
                    mp_values,
                    xy,
                    method='linear',
                    # method='cubic',
                    fill_value=np.nan
                    )

            # if np.all(np.isnan(outvals)):
            #     outvals = scipy.interpolate.griddata(
            #             mp_points,
            #             mp_values,
            #             xy,
            #             method='nearest',
            #             )
            # if np.any(np.isnan(outvals)):
            #     nan_idxs = np.where(np.isnan(outvals))
            #     # q_non_nan = np.where(~np.isnan(outvals))
            #     outvals[nan_idxs] = scipy.interpolate.griddata(
            #         xy,
            #         # outvals[q_non_nan],
            #         self._fp[iidxs[0]:iidxs[1], jidxs[0]:jidxs[1]][idxs].flatten(),
            #         xy[nan_idxs],
            #         # method='linear',
            #         method='nearest',
            #         fill_value=np.nan,
            #     )
            # if np.any(np.isnan(outvals)):
            #     nan_idxs = np.where(np.isnan(outvals))
            #     q_non_nan = np.where(~np.isnan(outvals))
            #     outvals[nan_idxs] = scipy.interpolate.griddata(
            #         xy[q_non_nan],
            #         outvals[q_non_nan],
            #         xy[nan_idxs],
            #         # method='linear',
            #         method='nearest',
            #     )
            if hmin is not None:
                outvals[np.where(outvals < hmin)] = hmin
            if hmax is not None:
                outvals[np.where(outvals > hmax)] = hmax
            # breakpoint()
            self._fp[iidxs[0]:iidxs[1], jidxs[0]:jidxs[1]][idxs] = np.fmin(
                    outvals,
                    self._fp[iidxs[0]:iidxs[1], jidxs[0]:jidxs[1]][idxs]
                    )
            self._fp.flush()
            # plt.contourf(xvals, yvals, self._fp[iidxs[0]:iidxs[1], jidxs[0]:jidxs[1]])
            # plt.show(block=True)
            # exit()
        # self.marche = True

    def add_gradient_delimiter(
        self,
        multiplier: float = 1.0 / 3.0,
        hmin=None,
        hmax=None,
        upper_bound=None,
        lower_bound=None,
    ):

        hmin = float(hmin) if hmin is not None else hmin
        hmax = float(hmax) if hmax is not None else hmax

        # tmpfile = tempfile.NamedTemporaryFile()
        # utm_crs: Union[CRS, None] = None
        # with rasterio.open(tmpfile.name, "w", **self.src.meta) as dst:

        iter_windows = list(self.iter_windows())
        tot = len(iter_windows)

        # for i, window in enumerate(iter_windows):
        for i, (xvals, yvals, zvals) in enumerate(self.raster):
            window = iter_windows[i]
            topobathy = zvals[0, :]

            logger.debug(f"Processing window {i+1}/{tot}.")

            if self.crs.is_geographic:
                logger.debug(
                    "CRS is geographic, transforming points to local projection."
                )
                local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m "\
                    f"+lat_0={np.median(yvals)} +lon_0={np.median(xvals)}"
                local_crs = CRS.from_user_input(local_azimuthal_projection)
                transformer = Transformer.from_crs(self.crs, local_crs, always_xy=True)
                geographic_to_local = transformer.transform
                x0, x1 = np.min(xvals), np.max(xvals)
                y0, y1 = np.min(yvals), np.max(yvals)
                (x0, x1), (y0, y1) = geographic_to_local([x0, x1], [y0, y1])
                dx = np.diff(np.linspace(x0, x1, window.width))[0]
                dy = np.diff(np.linspace(y0, y1, window.height))[0]
            else:
                dx = self.dx
                dy = self.dy
            logger.debug("Loading topobathy values from raster.")
            # topobathy = self.raster.get_values(band=1, window=window)
            # topobathy = self.raster.get_values(band=1, window=window)
            dx, dy = np.gradient(topobathy, dx, dy)
            with warnings.catch_warnings():
                # in case self._src.values is a masked array
                warnings.simplefilter("ignore", category=RuntimeWarning)
                dh = np.sqrt(dx ** 2 + dy ** 2)
            dh = np.ma.masked_equal(dh, 0.0)
            logger.debug("Loading hfun_values.")
            hfun_values = np.abs((multiplier) * (topobathy / dh))
            del dh

            if hmin is not None:
                logger.debug("Apply hmin.")
                hfun_values[np.where(hfun_values < hmin)] = hmin

            if hmax is not None:
                logger.debug("Apply hmax.")
                hfun_values[np.where(hfun_values > hmax)] = hmax

            if self.hmin is not None:
                hfun_values[np.where(hfun_values < self.hmin)] = self.hmin

            if self.hmax is not None:
                hfun_values[np.where(hfun_values > self.hmax)] = self.hmax

            irange, jrange = rasterio.windows.toranges(iter_windows[i])
            current_hfun_values = np.array(self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])])

            if upper_bound is not None:
                logger.debug("Apply upper_bound.")
                idxs = np.where(topobathy > upper_bound)
                hfun_values[idxs] = current_hfun_values[idxs]

            if lower_bound is not None:
                logger.debug("Apply lower_bound.")
                idxs = np.where(topobathy < lower_bound)
                hfun_values[idxs] = current_hfun_values[idxs]

            del topobathy

            logger.debug("Apply np.minimum")
            hfun_values = np.minimum(
                current_hfun_values, hfun_values
            )  # .astype(self.dtype(1))
            logger.debug("Write band")
            # dst.write_band(1, hfun_values, window=window)

            self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])] = hfun_values
            self._fp.flush()

        logger.debug("Done with add_gradient_delimiter")
        # self._tmpfile = tmpfile

    def add_constant_value(self, value, lower_bound=None, upper_bound=None, blend_method=np.minimum):

        lower_bound = -float("inf") if lower_bound is None else float(lower_bound)

        upper_bound = float("inf") if upper_bound is None else float(upper_bound)
        # tmpfile = tempfile.NamedTemporaryFile()

        # with rasterio.open(tmpfile.name, "w", **self.src.meta) as dst:
        iter_windows = list(self.iter_windows())
        # tot = len(iter_windows)
        for i, ((xvals, yvals, hfun_values), (rx, ry, rast_values)) in enumerate(zip(self, self.raster)):
            rast_values = rast_values[0, :]
            irange, jrange = rasterio.windows.toranges(iter_windows[i])
            hfun_values = np.array(self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])])
            hfun_values[
                np.where(
                    np.logical_and(
                        rast_values > lower_bound, rast_values < upper_bound
                    )
                )
            ] = value
            # if blend_method is not None:
            hfun_values = blend_method(
                self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])],
                hfun_values,
            )
            # irange, jrange = rasterio.windows.toranges(iter_windows[i])
            self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])] = hfun_values
            self._fp.flush()
            # dst.write_band(1, hfun_values, window=window)
                # del rast_values
                # gc.collect()
        # self._tmpfile = tmpfile

    def _add_quad_sizes(self, quads_gdf):
        from rasterstats.io import bounds_window
        from rasterio.features import rasterize
        from rasterio.windows import Window, from_bounds, toranges
        for i, ((xvals, yvals, hfun_values), (rx, ry, rast_values)) in enumerate(zip(self, self.raster)):
            outarray = np.full(hfun_values.shape, np.nan)
            local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={np.median(yvals)} +lon_0={np.median(xvals)}"
            transformer = Transformer.from_crs(self.raster.crs, local_azimuthal_projection, always_xy=True)
            for _, row in quads_gdf.to_crs(self.raster.crs).iterrows():
                exterior_ring = row.geometry.exterior
                # Iterate over each LineString segment of the exterior ring
                for i in range(len(exterior_ring.coords) - 1):
                    linestring = LineString([exterior_ring.coords[i], exterior_ring.coords[i + 1]])
                    geom_window = bounds_window(linestring.bounds, self.raster.transform)
                    geom_window = Window(col_off=geom_window[1][0], row_off=[0][0],
                                         width=geom_window[1][1] - geom_window[1][0],
                                         height=geom_window[0][1] - geom_window[0][0])
                    col_off = rx[0] - self.raster.src.bounds.left
                    row_off = ry[0] - self.raster.src.bounds.top
                    raster_window = Window(col_off=col_off, row_off=row_off,
                                           width=hfun_values.shape[1], height=hfun_values.shape[0])
                    window_intersection = raster_window.intersection(geom_window)
                    
                    if window_intersection.width > 0 and window_intersection.height > 0:
                        # Burn-in the length value for each line segment
                        local_linestring = ops.transform(transformer.transform, linestring)
                        length_raster = rasterize([(linestring, local_linestring.length)],
                                                  out_shape=hfun_values.shape,
                                                  transform=self.raster.transform,
                                                  fill=np.nan,
                                                  default_value=local_linestring.length)
                        data_in_window = length_raster[
                            int(window_intersection.row_off):int(window_intersection.row_off + window_intersection.height),
                            int(window_intersection.col_off):int(window_intersection.col_off + window_intersection.width)
                            ]
                        outarray[
                                int(window_intersection.row_off):int(window_intersection.row_off + window_intersection.height),
                                int(window_intersection.col_off):int(window_intersection.col_off + window_intersection.width)
                                ] = data_in_window
            window = from_bounds(min(rx), min(ry), max(rx), max(ry), self.raster.transform)
            row_range, col_range = toranges(window)
            print(outarray[~np.isnan(outarray)])
            self._fp[
                int(row_range[0]):int(row_range[-1]),
                int(col_range[0]):int(col_range[-1])
                ][~np.isnan(outarray)] = outarray[~np.isnan(outarray)]
            self._fp.flush()

    @property
    def values(self):
        return self._fp[:]

    def make_plot(
        self,
        band=1,
        window=None,
        axes=None,
        vmin=None,
        vmax=None,
        cmap="jet",
        levels=None,
        show=False,
        title=None,
        figsize=None,
        colors=256,
        cbar_label=None,
        norm=None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Generates a "standard" topobathy plot.
        """
        logger.debug(f"Read raster values for band {band}")

        # x, y, values = get_window_data(
        #             self.src,
        #             self.window,
        #             band=band,
        #             masked=True,
        #             resampling_method=self.resampling_method,
        #             resampling_factor=self.resampling_factor,
        #             clip=self.clip,
        #             mask=self.mask,
        #             )
        # x = []
        # y = []
        minvals = []
        maxvals = []
        for xvals, yvals, hvals in self:
            # x.append(xvals)
            # y.append(yvals)
            minvals.append(np.nanmin(hvals))
            maxvals.append(np.nanmax(hvals))

        vmin = np.min(minvals) if vmin is None else float(vmin)
        vmax = np.max(maxvals) if vmax is None else float(vmax)
        logger.debug(f"Generating contourf plot with {levels}")
        for xvals, yvals, hvals in self:
            ax = axes.contourf(
                xvals,
                yvals,
                hvals,
                levels=levels,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
        logger.debug("Done with plot.")
        axes.axis("scaled")
        if title is not None:
            axes.set_title(title)
        # mappable = ScalarMappable(cmap=cmap)
        # mappable.set_array([])
        # mappable.set_clim(vmin, vmax)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("bottom", size="2%", pad=0.5)

        logging.debug(f"Generating colorbar plot with {levels}")
        plt.colorbar(
            ax,
            # mappable,
            cax=cax,
            # extend=cmap_extend,
            orientation="horizontal",
        )
        # if col_val != 0:
        #     cbar.set_ticks([vmin, vmin + col_val * (vmax-vmin), vmax])
        #     cbar.set_ticklabels([np.around(vmin, 2), 0.0, np.around(vmax, 2)])
        # else:
        #     cbar.set_ticks([vmin, vmax])
        #     cbar.set_ticklabels([np.around(vmin, 2), np.around(vmax, 2)])
        # if cbar_label is not None:
        #     cbar.set_label(cbar_label)
        return axes

    def tricontourf(
        self,
        marche: bool = False,
        verbosity=None,
        ax=None,
        elements=True,
        color="k",
        linewidth=0.07,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        ax = ax or plt.gca()
        msh_t = self.msh_t(marche=marche, verbosity=verbosity)
        ax.tricontourf(
            msh_t.vert2["coord"][:, 0],
            msh_t.vert2["coord"][:, 1],
            msh_t.tria3["index"],
            msh_t.value.flatten(),
            **kwargs,
        )
        if elements is True:
            ax.triplot(
                msh_t.vert2["coord"][:, 0],
                msh_t.vert2["coord"][:, 1],
                msh_t.tria3["index"],
                color=color,
                linewidth=linewidth,
            )
        return ax

    def triplot(
        self,
        window: rasterio.windows.Window = None,
        marche: bool = False,
        verbosity=None,
        axes=None,
        show=False,
        figsize=None,
        color="k",
        linewidth=0.07,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        msh_t = self.msh_t(window=window, marche=marche, verbosity=verbosity)
        axes.triplot(
            msh_t.vert2["coord"][:, 0],
            msh_t.vert2["coord"][:, 1],
            msh_t.tria3["index"],
            color=color,
            linewidth=linewidth,
            **kwargs,
        )
        return axes

    @property
    def raster(self):
        return self._raster

    @property
    def _raster(self):
        return self.__raster

    @_raster.setter
    def _raster(self, raster: Raster):
        logger.debug("Generating base raster for hfun...")
        if not isinstance(raster, Raster):
            raise TypeError(
                f"Argument raster must be of type {Raster}, not "
                f"type {type(raster)}."
            )

        window = raster.window
        width = window.width
        height = window.height
        # col_off = window.col_off
        # row_off = window.row_off
        out_shape = (int(np.around(height)), int(np.around(width)))
        out_transform = rasterio.windows.transform(raster.window, raster.transform)
        chunk_size = raster.chunk_size if raster.chunk_size is not None else np.max([window.width, window.height])
        if raster.resampling_factor is not None:
            out_shape = (
                    int(np.around(out_shape[0] * raster.resampling_factor)),
                    int(np.around(out_shape[1] * raster.resampling_factor))
                    )
        _tmpfile = tempfile.NamedTemporaryFile(dir=os.getenv('GEOMESH_TEMPDIR'))
        profile = raster.src.profile.copy()
        profile.update({
            'transform': out_transform,
            'width': out_shape[1],
            'dtype': np.float32,
            'height': out_shape[0],
            'count': 1,
            'nodata': np.finfo(np.float32).max,
            })
        self._fp_tmpfile = tempfile.NamedTemporaryFile(dir=os.getenv('GEOMESH_TEMPDIR'))
        self._fp = np.memmap(self._fp_tmpfile, dtype="float32", mode="w+", shape=out_shape)
        self._fp_out_shape = out_shape
        self._fp_dtype = "float32"
        with rasterio.open(_tmpfile.name, 'w', **profile) as _:
            for i, window in enumerate(
                get_iter_windows(
                    out_shape[1],
                    out_shape[0],
                    chunk_size=chunk_size,
                    )):
                _out_shape = (int(np.around(window.height)), int(np.around(window.width)))
                irange, jrange = rasterio.windows.toranges(window)
                self._fp[int(irange[0]):int(irange[1]), int(jrange[0]):int(jrange[1])] = np.full(_out_shape, profile['nodata'])
                self._fp.flush()
        self.__raster = raster
        self._path = _tmpfile.name
        self._tmpfile = _tmpfile
        self._window = rasterio.windows.Window(0, 0, out_shape[1], out_shape[0])
        self._resampling_factor = None
        self._resampling_method = raster.resampling_method
        self.chunk_size = raster.chunk_size
        self.overlap = raster.overlap
        self.clip = raster.clip
        self.mask = raster.mask

    @property
    def output(self):
        return self

    @property
    def nprocs(self):
        return self._nprocs

    @nprocs.setter
    def nprocs(self, nprocs: Union[int, None]):

        _err = 'Argument nprocs must be a positive int, -1 (all threads) or ' \
               f'None (1 thread) but got {nprocs=} {type(nprocs)=} .'

        if nprocs is None:
            # nprocs = cpu_count(logical=True)
            nprocs = 1

        if not isinstance(nprocs, (int, np.integer)):
            raise ValueError(_err)

        if isinstance(nprocs, np.signedinteger) and nprocs < 0:
            if nprocs == -1:
                nprocs = cpu_count(logical=True)
            else:
                raise ValueError(_err)

        elif isinstance(nprocs, np.unsignedinteger):
            if not 1 <= nprocs <= cpu_count(logical=True):
                raise ValueError(f'Argument nprocs must be > 0 and <= {cpu_count(logical=True)=}')

        self._nprocs = nprocs


    @property
    def hmin(self):
        return self._hmin

    @property
    def hmax(self):
        return self._hmax

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity: int):
        self._verbosity = verbosity

    @property
    def crs(self):
        return self.raster.crs

    @property
    def geom(self):
        return self._geom

    @geom.setter
    def geom(self, geom):
        from geomesh.geom import Geom
        from geomesh.geom.base import BaseGeom
        if geom is None:
            pass
        elif isinstance(geom, BaseGeom):
            pass
        elif isinstance(geom, str):
            for func in [gpd.read_file, gpd.read_feather]:
                try:
                    gdf = func(geom)
                except fiona.errors.DriverError as err:
                    if 'not recognized as a supported file format.' in str(err):
                        continue
                    raise
                except pyarrow.lib.ArrowInvalid as err:
                    if 'Not a Feather V1 or Arrow IPC file' in str(err):
                        continue
                    raise
            geom = Geom(gdf.unary_union, crs=gdf.crs)
        else:
            geom = Geom(geom)
        self._geom = geom


def transform_point(x, y, src_crs, utm_crs):
    transformer = Transformer.from_crs(src_crs, utm_crs, always_xy=True)
    return transformer.transform(x, y)


def transform_linestring(
    linestring: LineString,
    target_size: float,
    # src_crs: CRS = None,
    # utm_crs: CRS = None
):
    distances = [0.0]
    # if utm_crs is not None:
    #     transformer = Transformer.from_crs(src_crs, utm_crs, always_xy=True)
    #     linestring = ops.transform(transformer.transform, linestring)
    while distances[-1] + target_size < linestring.length:
        distances.append(distances[-1] + target_size)
    distances.append(linestring.length)
    linestring = LineString(
        [linestring.interpolate(distance) for distance in distances]
    )
    return linestring


def get_contours_linestring_collections(zmin, zmax, raster, local_crs):
    levels = [zmin] if zmin is not None else []
    levels.append(zmax)
    pad_width = 20
    pad_width_2d = ((pad_width, pad_width), (pad_width, pad_width))
    contours_mls = []

    for xval, yval, zvals in raster:
        zvals = zvals[0, :]
        # Compute the step size at the edges
        xval_diff_before = xval[1] - xval[0]
        xval_diff_after = xval[-1] - xval[-2]
        yval_diff_before = yval[1] - yval[0]
        yval_diff_after = yval[-1] - yval[-2]

        # Generate padding for xval and yval
        xval_pad_before = xval[0] - np.array(range(1, pad_width+1)) * xval_diff_before
        xval_pad_after = xval[-1] + np.array(range(1, pad_width+1)) * xval_diff_after
        yval_pad_before = yval[0] - np.array(range(1, pad_width+1)) * yval_diff_before
        yval_pad_after = yval[-1] + np.array(range(1, pad_width+1)) * yval_diff_after

        # Concatenate the padding and the original arrays
        xval_padded = np.concatenate([xval_pad_before[::-1], xval, xval_pad_after])
        yval_padded = np.concatenate([yval_pad_before[::-1], yval, yval_pad_after])

        # Pad zvals as before
        zvals_padded = np.pad(zvals, pad_width_2d, mode='edge')
        # Continue with padded arrays
        plt.ioff()
        previous_backend = matplotlib.get_backend()
        plt.switch_backend('agg')
        axes = plt.contour(xval_padded, yval_padded, zvals_padded, levels=levels)
        plt.close(plt.gcf())
        plt.switch_backend(previous_backend)
        plt.ion()

#         plt.ioff()
#         previous_backend = matplotlib.get_backend()
#         plt.switch_backend('agg')
#         axes = plt.contour(xval, yval, zvals, levels=levels)
#         plt.close(plt.gcf())
#         plt.switch_backend(previous_backend)
#         plt.ion()

        for level, path_collection in zip(axes.levels, axes.collections):
            for path in path_collection.get_paths():
                try:
                    contours_mls.append(LineString(path.vertices))
                except ValueError as e:
                    if "LineStrings must have at least 2 coordinate tuples" not in str(e):
                        raise e

    if len(contours_mls) == 0:
        return

    return gpd.GeoDataFrame([{'geometry': ls} for ls in contours_mls], crs=raster.crs).to_crs(local_crs)


from shapely.geometry import Point


def calculate_distances_and_points(centerlines, polygon, cross_section_node_count):

    midpoints_points = []
    midpoints_distances = []
    # Find midpoint of each line in centerlines
    for line in centerlines.geometry.geoms:
        for segment in segments(line):
            segment_midpoint = segment.interpolate(0.5, normalized=True)
            midpoints_points.append(np.array(segment_midpoint.coords).flatten())
            midpoints_distances.append((2/cross_section_node_count)*polygon.boundary.distance(segment_midpoint))

    # Calculate the midpoints of the polygon's edges (exterior and interiors)
    for ring in [polygon.exterior] + list(polygon.interiors):
        ring_coords = list(ring.coords)
        ring_midpoints = [np.array(LineString(ring_coords).interpolate((i+0.5)/len(ring_coords), normalized=True).coords).flatten() for i in range(len(ring_coords)-1)]
        
        # Calculate distance from each midpoint of polygon edges to the centerlines
        midpoints_distances.extend([(2/cross_section_node_count)*centerlines.geometry.distance(Point(midpoint)) for midpoint in ring_midpoints])
        midpoints_points.extend(ring_midpoints)

    return np.array(midpoints_points), np.array(midpoints_distances)


# def get_mp_points_values(raster, final_patches, zmin, zmax, local_crs, nprocs, cross_section_node_count, all_centerlines):

#     # Create a multiprocessing Pool
#     with Pool(nprocs) as pool:
#         # Use the pool to map the function across all inputs
#         results = pool.starmap(
#                 calculate_distances_and_points,
#                 zip(all_centerlines, final_patches, len(all_centerlines)*[cross_section_node_count])
#                 )

#     # Concatenate all results
#     midpoints_points = np.concatenate([res[0] for res in results])
#     midpoints_distances = np.concatenate([res[1] for res in results])

#     return midpoints_points, midpoints_distances

def process_contours(centerlines, local_contours, cross_section_node_count):
    interior_segments = gpd.GeoDataFrame(
            [{'geometry': ls} for curve in centerlines.geometry.geoms for ls in segments(curve)],
            crs=local_contours.crs
            )
    centerline_midpoints = np.vstack([np.array(p.coords) for p in interior_segments.centroid])
    points = [np.vstack(p.geometry.coords) for p in local_contours.itertuples()]
    if len(points) == 0:
        return [None, None]
    tree = KDTree(np.vstack(points))
    dist, idxs = tree.query(
            centerline_midpoints,
            # workers=nprocs
            )
    interior_segments['target_size'] = [(2/cross_section_node_count)*d for d in dist]
    tree = KDTree(centerline_midpoints)
    local_contours_segments_gdf = gpd.GeoDataFrame(
            [{'geometry': segment} for row in local_contours.itertuples() for segment in segments(row.geometry)],
            crs=local_contours.crs
            )
    dist, idxs = tree.query(
            np.vstack([np.array(p.geometry.centroid.coords) for p in local_contours_segments_gdf.itertuples()]),
            # workers=nprocs
            )
    local_contours_segments_gdf['target_size'] = [(2/cross_section_node_count)*d for d in dist]
    return pd.concat([interior_segments, local_contours_segments_gdf])


def get_mp_points_values(raster, final_patches, zmin, zmax, local_crs, nprocs, cross_section_node_count, all_centerlines):
    contours_gdf = get_contours_linestring_collections(zmin, zmax, raster, local_crs)
    if contours_gdf is None:
        return [], []
    job_args = []
    for i, patch in enumerate(final_patches):
        centerlines = all_centerlines[i]
        if centerlines is None:
            continue
        selection = contours_gdf.iloc[np.where(contours_gdf.intersects(patch))]
        job_args.append((centerlines, selection, cross_section_node_count))

    with Pool(nprocs) as pool:
        results = pool.starmap(process_contours, job_args)
    results = pd.concat(results)
    midpoints_gdf = gpd.GeoDataFrame([{'geometry': row.geometry.centroid} for row in results.itertuples()], crs=local_crs)
    midpoints_gdf.to_crs(raster.crs, inplace=True)
    mp_points = np.vstack([np.array(row.geometry.coords) for row in midpoints_gdf.itertuples()])
    mp_values = np.array(results['target_size'])
    return mp_points, mp_values


def repartition_features(linestring, max_verts):
    features = []
    if isinstance(linestring, MultiLineString):
        _mls = linestring
        for linestring in _mls:
            if len(linestring.coords) > max_verts:
                new_feat = []
                for segment in list(
                    map(LineString, zip(linestring.coords[:-1], linestring.coords[1:]))
                ):
                    new_feat.append(segment)
                    if len(new_feat) == max_verts - 1:
                        features.append(ops.linemerge(new_feat))
                        new_feat = []
                if len(new_feat) != 0:
                    features.append(ops.linemerge(new_feat))
            else:
                features.append(linestring)
    elif isinstance(linestring, LineString):
        if len(linestring.coords) > max_verts:
            new_feat = []
            for segment in list(
                map(LineString, zip(linestring.coords[:-1], linestring.coords[1:]))
            ):
                new_feat.append(segment)
                if len(new_feat) == max_verts - 1:
                    features.append(ops.linemerge(new_feat))
                    new_feat = []
            if len(new_feat) != 0:
                features.append(ops.linemerge(new_feat))
        else:
            features.append(linestring)
    return features


def segments(curve):
    return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))


def simplify_geometry(geometry, tolerance, preserve_topology: bool = True):
    return geometry.simplify(
        tolerance=float(tolerance),
        preserve_topology=bool(preserve_topology),
        )


def get_centerlines(patch, interpolation_distance):
    try:
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        # TODO: We can accomodate a much larger range of kwargs easily
        kwargs = {}
        if interpolation_distance is not None:
            kwargs['interpolation_distance'] = interpolation_distance
        return Centerline(patch, **kwargs)
    except scipy.spatial._qhull.QhullError:
        pass
    except centerline.exceptions.TooFewRidgesError:
        pass
    except centerline.exceptions.InputGeometryEnvelopeIsPointError:
        pass


def resample_linear_ring(linear_ring, segment_length):
    total_length = linear_ring.length
    num_segments = int(np.ceil(total_length / segment_length))
    resampled_points = [
        linear_ring.interpolate(i * total_length / num_segments)
        for i in range(num_segments)
    ]
    try:
        return LinearRing([point.coords[0] for point in resampled_points])
    except ValueError as err:
        if str(err) == "A linearring requires at least 4 coordinates.":
            return LinearRing()
        else:
            raise err


def resample_polygon(polygon, segment_length):
    exterior = polygon.exterior
    interiors = polygon.interiors

    # Resample the exterior linear ring
    resampled_exterior = resample_linear_ring(exterior, segment_length)
    if resampled_exterior is None:
        return Polygon()

    # Resample each interior linear ring
    resampled_interiors = [resample_linear_ring(ring, segment_length) for ring in interiors]
    resampled_interiors = [ring for ring in resampled_interiors if not ring.is_empty]
    return Polygon(shell=resampled_exterior, holes=resampled_interiors)


def test_aa_on_heavy_raster():
    from appdirs import user_data_dir
    from pathlib import Path
    from geomesh import Raster, utils
    rootdir = Path(user_data_dir('geomesh'))
    # heavy_raster = rootdir / 'raster_cache/chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/chesapeake_bay/ncei19_n38x00_w076x75_2019v1.tif'
    # raster = Raster(
    #         heavy_raster,
    #         resampling_factor=0.1,
    #         )
    raster = Raster(
            f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/ncei19_n41x00_w074x00_2015v1.tif',
            resampling_factor=0.2,
            )
    hfun = RasterHfun(
            raster,
            nprocs=cpu_count(),
            hmin=10.,
            verbosity=1.
            )
    # print('adding contour')
    # hfun.add_contour(0., target_size=100., expansion_rate=0.01)
    hfun.add_narrow_channel_anti_aliasing(
            zmax=0.,
            # resample_distance=100.,
            )
    hfun.add_narrow_channel_anti_aliasing(
            # resample_distance=100.,
            zmin=0.,
            zmax=20.
            )
    hfun.add_narrow_channel_anti_aliasing(
            # resample_distance=100.,
            zmin=20.,
            )
    # for xvals, yvals, hfun_vals in hfun:
    #     plt.gca().contourf(xvals, yvals, hfun_vals)
    #     plt.show(block=True)
    # exit()
    print('will call msh_t()')
    msh_t = hfun.msh_t()
    print('done with calling msh_t')
    from geomesh.cli.mpi.hgrid_build import interpolate_raster_to_mesh
    raster.resampling_factor = None
    idxs, values = interpolate_raster_to_mesh(msh_t, raster, use_aa=False)
    msh_t.value[idxs] = values
    from geomesh import Mesh
    ax = Mesh(msh_t).make_plot()
    utils.triplot(msh_t, axes=ax)
    plt.title(f'node count: {len(msh_t.vert2["coord"])}')
    plt.gca().axis('scaled')
    plt.show(block=True)


def test_raster_window_read():
    import matplotlib.pyplot as plt
    from rasterio.windows import Window
    raster_path="/sciclone/pscr/jrcalzada/thesis/.git/modules/runs/tropicald-validations/modules/data/prusvi_19_topobathy_2019/annex/objects/36/xM/MD5E-s165602212--c49959a2e85389cdeba86045b080a6e4.tif/MD5E-s165602212--c49959a2e85389cdeba86045b080a6e4.tif"
    window = Window(col_off=4447, row_off=0, width=3665, height=7297)
    # window=None
    raster = Raster(raster_path, window=window, resampling_factor=0.2)
    # ax1 = raster.make_plot()
    # plt.show(block=False)
    hfun = RasterHfun(raster)
    hfun.verbosity = 1
    hfun.add_contour(0., expansion_rate=0.007, target_size=100.)
    hfun.add_contour(10., expansion_rate=0.007, target_size=100.)
    # hfun.add_gradient_delimiter(hmin=30., hmax=500.)
    ax2 = hfun.tricontourf(elements=True)
    # hfun.triplot(ax=ax2)
    # ax2 = hfun.contourf()
    ax2.axis("scaled")
    plt.show(block=True)

if __name__ == "__main__":
    test_aa_on_heavy_raster()
