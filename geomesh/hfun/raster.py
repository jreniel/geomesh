import functools
import gc
import logging
from multiprocessing import cpu_count, Pool
import operator
import pathlib
import tempfile
from time import time
from typing import Union, List
import warnings

import geopandas as gpd
from jigsawpy import jigsaw_msh_t, jigsaw_jig_t
from jigsawpy import libsaw
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pyproj
from pyproj import CRS, Transformer
import rasterio
from scipy.spatial import cKDTree
from shapely import ops
from shapely.geometry import (
    LineString,
    MultiLineString,
    box,
    GeometryCollection,
    Polygon,
)

from geomesh import utils
from geomesh.figures import figure
from geomesh.geom.shapely import PolygonGeom
from geomesh.hfun.base import BaseHfun
from geomesh.raster import Raster, get_iter_windows


logger = logging.getLogger(__name__)


class RasterHfun(BaseHfun, Raster):
    def __init__(
        self, raster: Raster, hmin: float = None, hmax: float = None, verbosity=0
    ):
        self._raster = raster
        self._hmin = float(hmin) if hmin is not None else hmin
        self._hmax = float(hmax) if hmax is not None else hmax
        self._verbosity = int(verbosity)

    @functools.lru_cache(maxsize=None)
    def msh_t(
        self,
        window: rasterio.windows.Window = None,
        marche: bool = False,
        verbosity=None,
        dst_crs=None,
    ) -> jigsaw_msh_t:

        logger.debug("Begin generation of hfun msh_t")

        if window is None:
            iter_windows = list(self.iter_windows())
        else:
            iter_windows = [window]
        tot = len(iter_windows)
        meshed_windows = []
        for i, window in enumerate(iter_windows):
            window_start = time()
            hfun = jigsaw_msh_t()
            hfun.ndims = +2

            x0, y0, x1, y1 = self.get_window_bounds(window)

            local_azimuthal_projection = None

            if self.crs.is_geographic:
                hfun.mshID = "euclidean-mesh"
                logger.debug(
                    "CRS is geographic, transforming points to local projection."
                )
                local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={(y0 + y1)/2} +lon_0={(x0 + x1)/2}"
                local_crs = CRS.from_user_input(local_azimuthal_projection)
                transformer = Transformer.from_crs(self.crs, local_crs, always_xy=True)
                geographic_to_local = transformer.transform
                # get bbox data
                xgrid = self.get_x(window=window)
                ygrid = np.flip(self.get_y(window=window))
                xgrid, ygrid = np.meshgrid(xgrid, ygrid)
                bottom = xgrid[0, :]
                top = xgrid[1, :]
                del xgrid
                left = ygrid[:, 0]
                right = ygrid[:, 1]
                del ygrid

                dim1 = window.width
                dim2 = window.height

                logger.debug("Building hfun.tria3...")

                tria3 = np.empty(((dim1 - 1), (dim2 - 1)), dtype=jigsaw_msh_t.TRIA3_t)
                index = tria3["index"]
                helper_ary = (
                    np.ones(
                        ((dim1 - 1), (dim2 - 1)), dtype=jigsaw_msh_t.INDEX_t
                    ).cumsum(1)
                    - 1
                )
                index[:, :, 0] = np.arange(
                    0, dim1 - 1, dtype=jigsaw_msh_t.INDEX_t
                ).reshape(dim1 - 1, 1)
                index[:, :, 0] += (helper_ary + 0) * dim1

                index[:, :, 1] = np.arange(
                    1, dim1 - 0, dtype=jigsaw_msh_t.INDEX_t
                ).reshape(dim1 - 1, 1)
                index[:, :, 1] += (helper_ary + 0) * dim1

                index[:, :, 2] = np.arange(
                    1, dim1 - 0, dtype=jigsaw_msh_t.INDEX_t
                ).reshape(dim1 - 1, 1)
                index[:, :, 2] += (helper_ary + 1) * dim1

                hfun.tria3 = tria3.ravel()
                del tria3, helper_ary
                gc.collect()
                logger.debug("Done building hfun.tria3...")

                # BUILD VERT2_t. this one comes from the memcache array
                logger.debug("Building hfun.vert2...")
                hfun.vert2 = np.empty(
                    window.width * window.height, dtype=jigsaw_msh_t.VERT2_t
                )
                hfun.vert2["coord"][:] = np.array(
                    self.get_xy_memcache(window, geographic_to_local)
                )

                logger.debug("Done building hfun.vert2...")

                # Build REALS_t: this one comes from hfun raster
                logger.debug("Building hfun.value...")
                hfun.value = np.array(
                    self.get_values(window=window, band=1)
                    .flatten()
                    .reshape((window.width * window.height, 1)),
                    dtype=jigsaw_msh_t.REALS_t,
                )
                logger.debug("Done building hfun.value...")

                # Build Geom
                logger.debug("Building initial geom...")
                bbox = [
                    *[(x, left[0]) for x in bottom],
                    *[(bottom[-1], y) for y in reversed(right)],
                    *[(x, right[-1]) for x in reversed(top)],
                    *[(bottom[0], y) for y in reversed(left)],
                ]

                geom = PolygonGeom(Polygon(bbox), self.crs).msh_t()
                utils.reproject(geom, CRS.from_user_input(local_azimuthal_projection))
                logger.debug("Building initial geom done.")
                kwargs = {"method": "nearest"}

            else:
                logger.debug("Forming initial hmat (euclidean-grid).")
                hfun.mshID = "euclidean-grid"
                hfun.xgrid = np.array(
                    np.array(self.get_x(window=window)), dtype=jigsaw_msh_t.REALS_t
                )
                hfun.ygrid = np.array(
                    np.flip(self.get_y(window=window)), dtype=jigsaw_msh_t.REALS_t
                )
                hfun.value = np.array(
                    np.flipud(self.get_values(window=window, band=1)),
                    dtype=jigsaw_msh_t.REALS_t,
                )
                kwargs = {"kx": 1, "ky": 1}  # type: ignore[dict-item]
                geom = PolygonGeom(box(x0, y0, x1, y1), self.crs).msh_t()

            logger.debug("Configuring jigsaw...")

            opts = jigsaw_jig_t()

            # additional configuration options
            opts.mesh_dims = +2
            opts.hfun_scal = "absolute"
            # no need to optimize for size function generation
            opts.optm_tria = False

            opts.hfun_hmin = np.min(hfun.value) if self.hmin is None else self.hmin
            opts.hfun_hmax = np.max(hfun.value) if self.hmax is None else self.hmax
            opts.verbosity = self.verbosity if verbosity is None else verbosity

            # output mesh
            window_mesh = jigsaw_msh_t()
            window_mesh.mshID = "euclidean-mesh"
            window_mesh.ndims = +2

            if marche is True:
                logger.debug("Launching libsaw.marche...")
                libsaw.marche(opts, hfun)

            logger.debug("Launching libsaw.jigsaw...")
            libsaw.jigsaw(opts, geom, window_mesh, hfun=hfun)

            del geom

            # do post processing
            utils.interpolate(hfun, window_mesh, **kwargs)

            if local_azimuthal_projection is not None:
                window_mesh.crs = CRS.from_user_input(local_azimuthal_projection)
                logger.debug(
                    f"Projecting mesh window {i+1} of {tot} back to original CRS..."
                )
                utils.reproject(window_mesh, self.crs)
                logger.debug("Done reprojecting.")
            else:
                window_mesh.crs = self.crs

            meshed_windows.append(window_mesh)

            logger.debug(
                f"Finished processing window {i+1} of {tot}. Took {time()-window_start}."
            )

        if len(meshed_windows) == 1:
            output_mesh = meshed_windows.pop()

        else:

            output_mesh = jigsaw_msh_t()
            output_mesh.mshID = "euclidean-mesh"
            output_mesh.ndims = +2
            for window_mesh in meshed_windows:
                window_mesh.tria3["index"] += len(output_mesh.vert2)
                output_mesh.tria3 = np.append(
                    output_mesh.tria3, window_mesh.tria3, axis=0
                )
                output_mesh.vert2 = np.append(
                    output_mesh.vert2, window_mesh.vert2, axis=0
                )
                output_mesh.value = np.append(output_mesh.value, window_mesh.value)
        output_mesh.crs = self.crs
        if dst_crs is not None:
            dst_crs = CRS.from_user_input(dst_crs)
            if not dst_crs.equals(self.crs):
                utils.reproject(output_mesh, dst_crs)
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
        logger.debug(f"Adding contour at level={level} with nprocs={nprocs}")
        if not isinstance(level, list):
            level = [level]

        contours = []
        for _level in level:
            _contours = self.get_contour(_level)
            if isinstance(_contours, GeometryCollection):
                continue
            elif isinstance(_contours, LineString):
                contours.append(_contours)
            elif isinstance(_contours, MultiLineString):
                for _cont in _contours.geoms:
                    contours.append(_cont)

        if len(contours) == 0:
            logger.debug("No contours found!")
            return

        contours = MultiLineString(contours)

        logger.debug("Adding contours as features...")
        self.add_feature(contours, expansion_rate, target_size, nprocs)
        self.msh_t.cache_clear()

    def add_feature(
        self,
        feature: Union[LineString, MultiLineString],
        expansion_rate: float,
        target_size: float = None,
        nprocs=None,
        max_verts=200,
    ):
        """Adds a linear distance size function constraint to the mesh.

        Arguments:
            feature: shapely.geometryLineString or MultiLineString
        """

        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs
        logger.debug(f"Using nprocs={nprocs}")
        if not isinstance(feature, (LineString, MultiLineString)):
            raise TypeError(
                f"Argument feature must be of type {LineString} or "
                f"{MultiLineString}, not type {type(feature)}."
            )

        if isinstance(feature, LineString):
            feature = [feature]

        elif isinstance(feature, MultiLineString):
            feature = [linestring for linestring in feature.geoms]

        # check target size
        target_size = self.hmin if target_size is None else target_size
        if target_size is None:
            raise ValueError(
                "Argument target_size must be specified if no "
                "global hmin has been set."
            )
        if target_size <= 0:
            raise ValueError("Argument target_size must be greater than zero.")
        tmpfile = tempfile.NamedTemporaryFile()
        meta = self.src.meta.copy()
        meta.update({"driver": "GTiff"})
        local_azimuthal_projection = None
        with rasterio.open(
            tmpfile,
            "w",
            **meta,
        ) as dst:
            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)
            for i, window in enumerate(iter_windows):
                logger.debug(f"Processing window {i+1}/{tot}.")
                if self.crs.is_geographic:
                    x0, y0, x1, y1 = self.get_window_bounds(window)
                    local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={(y0 + y1)/2} +lon_0={(x0 + x1)/2}"
                    # geographic_to_local = functools.partial(
                    #     pyproj.transform,
                    #     pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
                    #     pyproj.Proj(local_azimuthal_projection),
                    # )
                    local_crs = CRS.from_user_input(local_azimuthal_projection)
                    transformer = Transformer.from_crs(
                        self.crs, local_crs, always_xy=True
                    )
                    geographic_to_local = transformer.transform
                logger.debug("Repartitioning features...")
                start = time()
                with Pool(processes=nprocs) as pool:
                    res = pool.starmap(
                        repartition_features,
                        [(linestring, max_verts) for linestring in feature],
                    )
                pool.join()
                feature = functools.reduce(operator.iconcat, res, [])
                logger.debug(f"Repartitioning features took {time()-start}.")

                logger.debug(f"Resampling features on nprocs {nprocs}...")
                start = time()
                with Pool(processes=nprocs) as pool:
                    transformed_features = pool.starmap(
                        transform_linestring,
                        [
                            (linestring, target_size, self.src.crs, local_crs)
                            for linestring in feature
                        ],
                    )
                pool.join()
                logger.debug(f"Resampling features took {time()-start}.")
                logger.debug("Concatenating points...")
                start = time()
                points = []
                for geom in transformed_features:
                    if isinstance(geom, LineString):
                        points.extend(geom.coords)
                    elif isinstance(geom, MultiLineString):
                        for linestring in geom.geoms:
                            points.extend(linestring.coords)
                logger.debug(f"Point concatenation took {time()-start}.")

                logger.debug("Generating KDTree...")
                start = time()
                tree = cKDTree(np.array(points))
                logger.debug(f"Generating KDTree took {time()-start}.")
                if local_azimuthal_projection is not None:
                    xy = self.get_xy_memcache(window, geographic_to_local)
                else:
                    xy = self.get_xy(window)

                logger.debug(f"Transforming points took {time()-start}.")
                logger.debug("Querying KDTree...")
                start = time()
                distances, _ = tree.query(xy, workers=nprocs)
                logger.debug(f"Querying KDTree took {time()-start}.")
                values = expansion_rate * target_size * distances + target_size
                values = values.reshape(window.height, window.width).astype(
                    self.dtype(1)
                )
                if self.hmin is not None:
                    values[np.where(values < self.hmin)] = self.hmin
                if self.hmax is not None:
                    values[np.where(values > self.hmax)] = self.hmax
                values = np.minimum(self.get_values(window=window), values)
                logger.debug(f"Write array to file {tmpfile.name}...")
                start = time()
                dst.write_band(1, values, window=window)
                logger.debug(f"Write array to file took {time()-start}.")
        self._tmpfile = tmpfile
        self.msh_t.cache_clear()

    def get_xy_memcache(self, window, transformer):
        if not hasattr(self, "_xy_cache"):
            self._xy_cache = {}
        tmpfile = self._xy_cache.get(f"{window}")
        if tmpfile is None:
            logger.debug("Transform points to local CRS...")
            tmpfile = tempfile.NamedTemporaryFile()
            xy = self.get_xy(window)
            fp = np.memmap(tmpfile, dtype="float32", mode="w+", shape=xy.shape)
            fp[:] = np.vstack(transformer(xy[:, 0], xy[:, 1])).T
            logger.debug("Saving values to memcache...")
            fp.flush()
            logger.debug("Done!")
            self._xy_cache[f"{window}"] = tmpfile
            return fp[:]
        else:
            logger.debug("Loading values from memcache...")
            return np.memmap(
                tmpfile,
                dtype="float32",
                mode="r",
                shape=((window.width * window.height), 2),
            )[:]

    def add_gradient_delimiter(
        self,
        hmin=None,
        hmax=None,
        upper_bound=None,
        lower_bound=None,
        multiplier: float = 1.0 / 3.0,
    ):

        hmin = float(hmin) if hmin is not None else hmin
        hmax = float(hmax) if hmax is not None else hmax

        tmpfile = tempfile.NamedTemporaryFile()
        # utm_crs: Union[CRS, None] = None
        with rasterio.open(tmpfile.name, "w", **self.src.meta) as dst:

            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)

            for i, window in enumerate(iter_windows):

                logger.debug(f"Processing window {i+1}/{tot}.")
                x0, y0, x1, y1 = self.get_window_bounds(window)

                if self.crs.is_geographic:
                    logger.debug(
                        "CRS is geographic, transforming points to local projection."
                    )
                    local_azimuthal_projection = (
                        "+proj=aeqd +R=6371000 +units=m "
                        f"+lat_0={(y0 + y1)/2} +lon_0={(x0 + x1)/2}"
                    )
                    # wgs84_to_aeqd = functools.partial(
                    #     pyproj.transform,
                    #     pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
                    #     pyproj.Proj(local_azimuthal_projection),
                    # )
                    local_crs = CRS.from_user_input(local_azimuthal_projection)
                    transformer = Transformer.from_crs(
                        self.crs, local_crs, always_xy=True
                    )
                    wgs84_to_aeqd = transformer.transform
                    (x0, x1), (y0, y1) = wgs84_to_aeqd([x0, x1], [y0, y1])
                    dx = np.diff(np.linspace(x0, x1, window.width))[0]
                    dy = np.diff(np.linspace(y0, y1, window.height))[0]
                else:
                    dx = self.dx
                    dy = self.dy
                logger.debug("Loading topobathy values from raster.")
                topobathy = self.raster.get_values(band=1, window=window)
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

                if upper_bound is not None:
                    logger.debug("Apply upper_bound.")
                    idxs = np.where(topobathy > upper_bound)
                    hfun_values[idxs] = self.get_values(band=1, window=window)[idxs]

                if lower_bound is not None:
                    logger.debug("Apply lower_bound.")
                    idxs = np.where(topobathy < lower_bound)
                    hfun_values[idxs] = self.get_values(band=1, window=window)[idxs]

                del topobathy

                logger.debug("Apply np.minimum")
                hfun_values = np.minimum(
                    self.get_values(band=1, window=window), hfun_values
                ).astype(self.dtype(1))
                logger.debug("Write band")
                dst.write_band(1, hfun_values, window=window)
                del hfun_values
        logger.debug("Done with add_gradient_delimiter")
        self._tmpfile = tmpfile
        self.msh_t.cache_clear()

    def add_constant_value(self, value, lower_bound=None, upper_bound=None):
        lower_bound = -float("inf") if lower_bound is None else float(lower_bound)
        upper_bound = float("inf") if upper_bound is None else float(upper_bound)
        tmpfile = tempfile.NamedTemporaryFile()

        with rasterio.open(tmpfile.name, "w", **self.src.meta) as dst:

            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)

            for i, window in enumerate(iter_windows):

                logger.debug(f"Processing window {i+1}/{tot}.")
                hfun_values = self.get_values(band=1, window=window)
                rast_values = self.raster.get_values(band=1, window=window)
                hfun_values[
                    np.where(
                        np.logical_and(
                            rast_values > lower_bound, rast_values < upper_bound
                        )
                    )
                ] = value
                hfun_values = np.minimum(
                    self.get_values(band=1, window=window),
                    hfun_values.astype(self.dtype(1)),
                )
                dst.write_band(1, hfun_values, window=window)
                del rast_values
                gc.collect()
        self._tmpfile = tmpfile
        self.msh_t.cache_clear()

    @figure
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
        values = self.get_values(band=band, masked=True, window=window)
        vmin = np.min(values) if vmin is None else float(vmin)
        vmax = np.max(values) if vmax is None else float(vmax)
        logger.debug(f"Generating contourf plot with {levels}")
        ax = axes.contourf(
            self.get_x(window),
            self.get_y(window),
            values,
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

    @figure
    def tricontourf(
        self,
        window: rasterio.windows.Window = None,
        marche: bool = False,
        verbosity=None,
        axes=None,
        show=False,
        figsize=None,
        extend="both",
        elements=True,
        cmap="jet",
        color="k",
        linewidth=0.07,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        msh_t = self.msh_t(window=window, marche=marche, verbosity=verbosity)
        axes.tricontourf(
            msh_t.vert2["coord"][:, 0],
            msh_t.vert2["coord"][:, 1],
            msh_t.tria3["index"],
            msh_t.value.flatten(),
            **kwargs,
        )
        if elements is True:
            axes.triplot(
                msh_t.vert2["coord"][:, 0],
                msh_t.vert2["coord"][:, 1],
                msh_t.tria3["index"],
                color=color,
                linewidth=linewidth,
            )
        return axes

    @figure
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
        # init output raster file
        tmpfile = tempfile.NamedTemporaryFile()
        with rasterio.open(raster.tmpfile) as src:
            if raster.chunk_size is not None:
                windows = get_iter_windows(
                    src.width, src.height, chunk_size=raster.chunk_size
                )
            else:
                windows = [rasterio.windows.Window(0, 0, src.width, src.height)]
            meta = src.meta.copy()
            meta.update({"driver": "GTiff", "dtype": np.float32})
            with rasterio.open(
                tmpfile,
                "w",
                **meta,
            ) as dst:
                for window in windows:
                    values = src.read(window=window).astype(np.float32)
                    values[:] = np.finfo(np.float32).max
                    dst.write(values, window=window)
        self.__raster = raster
        self._path = tmpfile.name
        self._tmpfile = tmpfile
        self.chunk_size = raster.chunk_size
        self.overlap = raster.overlap

    @property
    def output(self):
        return self

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

    def get_contour(self, level: float, window: rasterio.windows.Window = None):
        logger.debug(f"RasterHfun.get_raster_contours(level={level}, window={window})")
        if window is None:
            iter_windows = list(self.iter_windows())
        else:
            iter_windows = [window]
        if len(iter_windows) > 1:
            return self._get_raster_contour_feathered(level, iter_windows)
        else:
            return self._get_raster_contour_windowed(level, window)

    def _get_raster_contour_windowed(self, level, window):
        x, y = self.get_x(), self.get_y()
        features = []
        values = self.raster.get_values(band=1, window=window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            logger.debug("Computing contours...")
            start = time()
            plt.ioff()
            fig, ax = plt.subplots()
            ax.contour(x, y, values, levels=[level])
            logger.debug(f"Took {time()-start}...")
            plt.close(fig)
            plt.ion()
        for path_collection in ax.collections:
            for path in path_collection.get_paths():
                try:
                    features.append(LineString(path.vertices))
                except ValueError:
                    # LineStrings must have at least 2 coordinate tuples
                    pass
        return ops.linemerge(features)

    def _get_raster_contour_feathered(self, level, iter_windows):
        feathers = []
        total_windows = len(iter_windows)
        logger.debug(f"Total windows to process: {total_windows}.")
        for i, window in enumerate(iter_windows):
            x, y = self.get_x(window), self.get_y(window)
            logger.debug(f"Processing window {i+1}/{total_windows}.")
            features = []
            values = self.get_values(band=1, window=window)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                logger.debug("Computing contours...")
                plt.ioff()
                start = time()
                fig, ax = plt.subplots()
                ax.contour(x, y, values, levels=[level])
                logger.debug(f"Took {time()-start}...")
                plt.close(fig)
                plt.ion()
            for path_collection in ax.collections:
                for path in path_collection.get_paths():
                    if (
                        path.vertices.shape[0] >= 2
                    ):  # LineStrings must have at least 2 coordinate tuples
                        features.append(LineString(path.vertices))
            if len(features) > 0:
                tmpfile = (
                    pathlib.Path(self.tmpdir)
                    / pathlib.Path(
                        tempfile.NamedTemporaryFile(suffix=".feather").name
                    ).name
                )
                logger.debug("Saving feather.")
                features = ops.linemerge(features)
                gpd.GeoDataFrame([{"geometry": features}]).to_feather(tmpfile)
                feathers.append(tmpfile)
        logger.debug("Concatenating feathers.")
        features = []
        out = gpd.GeoDataFrame()
        for feather in feathers:
            out = out.append(gpd.read_feather(feather), ignore_index=True)
            feather.unlink()
            for geometry in out.geometry:
                if isinstance(geometry, LineString):
                    geometry = MultiLineString([geometry])
            for linestring in geometry:
                features.append(linestring)
        logger.debug("Merging features.")
        return ops.linemerge(features)


def transform_point(x, y, src_crs, utm_crs):
    transformer = Transformer.from_crs(src_crs, utm_crs, always_xy=True)
    return transformer.transform(x, y)


def transform_linestring(
    linestring: LineString, target_size: float, src_crs: CRS = None, utm_crs: CRS = None
):
    distances = [0.0]
    if utm_crs is not None:
        transformer = Transformer.from_crs(src_crs, utm_crs, always_xy=True)
        linestring = ops.transform(transformer.transform, linestring)
    while distances[-1] + target_size < linestring.length:
        distances.append(distances[-1] + target_size)
    distances.append(linestring.length)
    linestring = LineString(
        [linestring.interpolate(distance) for distance in distances]
    )
    return linestring


def repartition_features(linestring, max_verts):
    features = []
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
