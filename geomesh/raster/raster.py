import hashlib
import logging
import multiprocessing
import os
import pathlib
import tempfile
from typing import Union

import matplotlib
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio import warp
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.transform import array_bounds
from rasterio import windows
import requests
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPolygon, Polygon

from geomesh.figures import figure, get_topobathy_kwargs

logger = logging.getLogger(__name__)


class Raster:
    """
    Class to read a raster file from a path.

    This class uses rasterio as the backend, so any path or file format readable by rasterio is valid.

    The raster data is not loaded automatically to the RAM, instead this class will hold a
    pointer to provided file, and will load the array data into memory only when it is requested
    by some method or operation. The class also supports processing raster data in "chunks", so that
    large rasters can be processed via windows, since the entire raster may not always fit into memory.

    Another feature of this class is that it always behaves as "copy-on-write". What this means is that,
    if the user requests a destructive operation such as applying a gaussian filter, or adding a data band,
    the source raster file will be copied first to a temporary location, keeping the original file unmodified.
    If the user wishes to save their changes to file, the :func:`geomesh.raster.Raster.write` method is available
    for writing the intermediary steps to a file.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        crs: Union[str, CRS] = None,
        chunk_size: int = None,
        overlap: int = None,
    ):
        """
        :param path: Path to the raster.

        :param crs: Used to specify coordinate reference system for files which are not georeference. If used, it will override
            the referencing in the metadata.

        :param chunk_size: Used to specify maximum pixel window size to use when performing operations over data. This
            parameter is useful for processing large rasters that don't fit into memory. If chunk_size is `None`, the entire
            array will be loaded, otherwise it will use windows of *maximum* size of chunk_size on each dimension:
            (chunk_size x chunk_size).

        :param overlap: Used in conjunction with `chunk_size`, this parameter controls the overlap (in number of pixels)
            between two adjacent processing windows. This is because, for some operations having no overlap may cause the
            combined geometry to be piecewise disconnected.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._path = path
        self._crs = crs

    def __iter__(self, chunk_size=None, overlap=None):
        for window in self.iter_windows(chunk_size, overlap):
            yield window, self.get_window_bounds(window)

    def get_x(self, window: windows.Window = None) -> np.ndarray:
        """
        Returns linear space over x-coordinate of raster.

        :param window: A :class:`rasterio.windows.Window` instance, used for sub-sampling the data.
        """
        window = (
            windows.Window(0, 0, self.src.shape[1], self.src.shape[0])
            if window is None
            else window
        )
        if window is not None:
            assert isinstance(window, windows.Window)
            width = window.width
        else:
            width = self.shape[0]
        x0, y0, x1, y1 = self.get_window_bounds(window)
        return np.linspace(x0, x1, width)

    def copy(self):
        return Raster(self._tmpfile)

    def get_y(self, window: windows.Window = None) -> np.ndarray:
        """
        Returns linear space over y-coordinate of raster.

        :param window: A :class:`rasterio.windows.Window` instance, used for sub-sampling the data.
        """
        window = (
            windows.Window(0, 0, self.src.shape[1], self.src.shape[0])
            if window is None
            else window
        )
        if window is not None:
            assert isinstance(window, windows.Window)
            height = window.height
        else:
            height = self.shape[1]
        x0, y0, x1, y1 = self.get_window_bounds(window)
        return np.linspace(y1, y0, height)

    def get_xy(self, window: windows.Window = None) -> np.ndarray:
        """
        Returns a scattered list of coordinate tuples in the form of an array [M x 2]
        where M = :attr:`geomesh.Raster.height` * :attr:`geomesh.Raster.width`.

        :param window: A :class:`rasterio.windows.Window` instance, used for sub-sampling the data.
        """
        x, y = np.meshgrid(self.get_x(window), self.get_y(window))
        return np.vstack([x.flatten(), y.flatten()]).T

    def get_values(
        self, window: windows.Window = None, band: int = None, **kwargs
    ) -> np.ndarray:
        """
        Returns a :class:`np.ndarray` of band values. Return array may be multidimensional.

        :param window: A :class:`rasterio.windows.Window` instance, used for sub-sampling the data.
        :param band: Used for selecting band (first band is 1). If omitted, will return a multidimensional array containing
            all raster bands (e.g. will load all the data into RAM).
        :param kwargs: Passed directly to :func:`rasterio.Dataset.read`.
        """
        i = 1 if band is None else band
        window = (
            windows.Window(0, 0, self.src.shape[1], self.src.shape[0])
            if window is None
            else window
        )
        if window is not None:
            assert isinstance(window, windows.Window)
        return self.src.read(i, window=window, **kwargs)

    def get_xyz(self, window: windows.Window = None, band: int = 1) -> np.ndarray:
        """
        Returns a scattered list "xyz" of coordinate tuples in the form of an array [M x 3]
        where M = :attr:`geomesh.Raster.height` * :attr:`geomesh.Raster.width`.

        :param window: A :class:`rasterio.windows.Window` instance, used for sub-sampling the data.
        :param band: Used for selecting band (first band is 1). Default value is first band.
        """
        xy = self.get_xy(window)
        values = self.get_values(window=window, band=band).reshape((xy.shape[0], 1))
        return np.hstack([xy, values])

    def get_bbox(
        self,
        crs: Union[str, CRS] = None,
    ) -> Bbox:
        """
        Returns the bounding box of the data transform.

        :param crs: Target CRS at which to return the bounding box. If `None`, the return bbox will
            match the current CRS of the DEM.
        """
        xmin, xmax = np.min(self.x), np.max(self.x)
        ymin, ymax = np.min(self.y), np.max(self.y)
        crs = self.crs if crs is None else crs
        if crs is not None:
            if not self.crs.equals(crs):
                transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
                (xmin, xmax), (ymin, ymax) = transformer.transform(
                    (xmin, xmax), (ymin, ymax)
                )
        return Bbox([[xmin, ymin], [xmax, ymax]])

    @figure
    def make_plot(
        self,
        band=1,
        window=None,
        axes=None,
        vmin=None,
        vmax=None,
        cmap="topobathy",
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
        logger.debug(f"Raster.make_plot: Read raster values for band {band}")
        values = self.get_values(band=band, masked=True, window=window)
        vmin = np.min(values) if vmin is None else float(vmin)
        vmax = np.max(values) if vmax is None else float(vmax)
        kwargs.update(get_topobathy_kwargs(values, vmin, vmax))
        col_val = kwargs.pop("col_val")
        logger.debug(f"Generating contourf plot with {levels}")
        axes.contourf(
            self.get_x(window),
            self.get_y(window),
            values,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        axes.axis("scaled")
        if title is not None:
            axes.set_title(title)
        mappable = ScalarMappable(cmap=kwargs["cmap"])
        mappable.set_array([])
        mappable.set_clim(vmin, vmax)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("bottom", size="2%", pad=0.5)
        logging.debug(f"Generating colorbar plot with {levels}")
        cbar = plt.colorbar(
            mappable,
            cax=cax,
            # extend=cmap_extend,
            orientation="horizontal",
        )
        if col_val != 0:
            cbar.set_ticks([vmin, vmin + col_val * (vmax - vmin), vmax])
            cbar.set_ticklabels([np.around(vmin, 2), 0.0, np.around(vmax, 2)])
        else:
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([np.around(vmin, 2), np.around(vmax, 2)])
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        return axes

    def tags(self, i=None):
        if i is None:
            return self.src.tags()
        else:
            return self.src.tags(i)

    def read(self, band, masked=True, **kwargs):
        return self.src.read(band, masked=masked, **kwargs)

    def dtype(self, band):
        return self.src.dtypes[band - 1]

    def nodataval(self, band):
        return self.src.nodatavals[band - 1]

    def sample(self, xy, i):
        """
        Wrapper to :func:`rasterio.Dataset.sample` method.
        """
        return self.src.sample(xy, i)

    def close(self):
        """
        Closes the raster file.
        """
        del self._src

    def add_band(self, values, **tags):
        """
        Appends array data to the raster as a band.
        """
        kwargs = self.src.meta.copy()
        band_id = kwargs["count"] + 1
        kwargs.update(count=band_id)
        tmpfile = tempfile.NamedTemporaryFile(prefix=str(self.tmpdir))
        with rasterio.open(tmpfile.name, "w", **kwargs) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
            dst.write_band(band_id, values.astype(self.src.dtypes[i - 1]))
        self._tmpfile = tmpfile
        return band_id

    def fill_nodata(self):
        """
        Wrapper to rasterio's  fillnodata function.
        """
        # A parallelized version is presented here: https://github.com/basaks/rasterio/blob/master/examples/fill_large_raster.py
        tmpfile = tempfile.NamedTemporaryFile(prefix=str(self.tmpdir))
        with rasterio.open(tmpfile.name, "w", **self.src.meta.copy()) as dst:
            for window in self.iter_windows():
                dst.write(
                    fillnodata(self.src.read(window=window, masked=True)), window=window
                )
        self._tmpfile = tmpfile

    def gaussian_filter(self, band: int, **kwargs) -> None:
        """
        Adds a gaussian filter to a band.

        :param band: Band for which to apply gaussian filter.
        :param kwargs: Passed directly as arguments to :func:`scipy.ndimage.gaussian_filter`
        """
        meta = self.src.meta.copy()
        tmpfile = tempfile.NamedTemporaryFile(prefix=str(self.tmpdir))
        with rasterio.open(tmpfile.name, "w", **meta) as dst:
            for i in range(1, self.src.count + 1):
                outband = self.src.read(i)
                if i == band:
                    outband = gaussian_filter(outband, **kwargs)
                dst.write_band(i, outband)
        self._tmpfile = tmpfile

    def mask(
        self, geometry: Union[Polygon, MultiPolygon], band: int = None, **kwargs
    ) -> None:
        """
        Applies a mask to the values within a polygon or multipolygon.

        :param geometry: Input geometry used for masking.
        :param band: Raster band over which to apply masking. If no band is given, mask is
            applied to all bands.
        """
        _kwargs = self.src.meta.copy()
        _kwargs.update(kwargs)
        out_images, out_transform = mask(self._src, geometry)
        tmpfile = tempfile.NamedTemporaryFile(prefix=str(self.tmpdir))
        with rasterio.open(tmpfile.name, "w", **_kwargs) as dst:
            if band is None:
                for j in range(1, self.src.count + 1):
                    dst.write_band(j, out_images[j - 1])
                    dst.update_tags(j, **self.src.tags(j))
            else:
                for j in range(1, self.src.count + 1):
                    if band == j:
                        dst.write_band(j, out_images[j - 1])
                        dst.update_tags(j, **self.src.tags(j))
                    else:
                        dst.write_band(j, self.src.read(j))
                        dst.update_tags(j, **self.src.tags(j))
        self._tmpfile = tmpfile

    def read_masks(self, band: int = None) -> np.ndarray:
        """
        Returns an array of mask or masks for the band(s).
        """

        if band is None:
            return np.dstack(
                [self.src.read_masks(band) for band in range(1, self.count + 1)]
            )
        else:
            return self.src.read_masks(band)

    def warp(
        self,
        dst_crs: Union[str, CRS],
        nprocs: int = -1,
        resampling_method: Resampling = None,
    ) -> None:
        """
        Wrapper to :func:`rasterio.warp.reproject`.

        Warning: Warping distorts the data, particularly for large domains.
        Using this method should be avoided.

        :param dst_crs: Destination CRS for warping raster.
        :param nprocs: Number of processors to use. If -1 is used, it will use all available cores.
        """
        nprocs = -1 if nprocs is None else nprocs
        nprocs = multiprocessing.cpu_count() if nprocs == -1 else nprocs
        dst_crs = CRS.from_user_input(dst_crs)
        transform, width, height = warp.calculate_default_transform(
            self.src.crs,
            dst_crs.srs,
            self.src.width,
            self.src.height,
            *self.src.bounds,
            dst_width=self.src.width,
            dst_height=self.src.height,
        )
        kwargs = self.src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs.srs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )
        tmpfile = tempfile.NamedTemporaryFile(prefix=str(self.tmpdir))

        with rasterio.open(tmpfile.name, "w", **kwargs) as dst:
            for i in range(1, self.src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(self._src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=self.src.transform,
                    crs=self.src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs.srs,
                    resampling=Resampling.nearest
                    if resampling_method is None
                    else resampling_method,
                    num_threads=nprocs,
                )

        self._tmpfile = tmpfile

    def resample(self, scaling_factor: int, resampling_method: Resampling = None):
        """ """

        # if resampling_method is None:
        #     resampling_method = self.resampling_method
        # else:
        #     msg = "resampling_method must be None or one of "
        #     msg += f"{self._resampling_methods.keys()}"
        #     assert resampling_method in self._resampling_methods.keys(), msg
        #     resampling_method = self._resampling_methods[resampling_method]
        resampling_method = (
            Resampling.nearest if resampling_method is None else resampling_method
        )
        if not isinstance(resampling_method, Resampling):
            raise TypeError(
                f"Argument resampling_method must be of type {Resampling}, not type {type(resampling_method)}"
            )

        tmpfile = tempfile.NamedTemporaryFile(prefix=str(self.tmpdir))
        # resample data to target shape
        width = int(self.src.width * scaling_factor)
        height = int(self.src.height * scaling_factor)
        data = self.src.read(
            out_shape=(self.src.count, height, width), resampling=resampling_method
        )
        kwargs = self.src.meta.copy()
        transform = self.src.transform * self.src.transform.scale(
            (self.src.width / data.shape[-1]), (self.src.height / data.shape[-2])
        )
        kwargs.update({"transform": transform, "width": width, "height": height})
        with rasterio.open(tmpfile.name, "w", **kwargs) as dst:
            dst.write(data)
        self._tmpfile = tmpfile
        return self

    def save(self, path):
        with rasterio.open(pathlib.Path(path), "w", **self.src.meta) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
                dst.update_tags(i, **self.src.tags(i))

    def clip(self, geom: Union[Polygon, MultiPolygon]):
        """
        Masks the raster for values outside of input polygon/multipolygon

        :param geom: Input polygon or multipolygo used for masking.
        """

        if isinstance(geom, Polygon):
            geom = MultiPolygon([geom])
        # TODO: rasterio warning: py.warnings WARNING: /home/jreniel/thesis/.conda_env/lib/python3.9/site-packages/rasterio/features.py:284: ShapelyDeprecationWarning: Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.
        # for index, item in enumerate(shapes):
        out_image, out_transform = rasterio.mask.mask(self.src, geom, crop=True)
        out_meta = self.src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        tmpfile = tempfile.NamedTemporaryFile(prefix=str(self.tmpdir))
        with rasterio.open(tmpfile.name, "w", **out_meta) as dest:
            dest.write(out_image)
        self._tmpfile = tmpfile

    def iter_windows(self, chunk_size=0, overlap=2):
        chunk_size = chunk_size if self.chunk_size is None else self.chunk_size
        overlap = overlap if self.overlap is None else overlap

        if chunk_size in [0, None, False]:
            yield rasterio.windows.Window(0, 0, self.width, self.height)
            return

        for window in get_iter_windows(self.width, self.height, chunk_size, overlap):
            yield window

    def get_window_data(self, window, masked=True, band=None):
        x0, y0, x1, y1 = self.get_window_bounds(window)
        x = np.linspace(x0, x1, window.width)
        y = np.linspace(y1, y0, window.height)
        if band is not None:
            data = self.src.read(band, masked=masked, window=window)
        else:
            data = self.src.read(masked=masked, window=window)
        return x, y, data

    def get_window_bounds(self, window):
        return array_bounds(
            window.height, window.width, self.get_window_transform(window)
        )

    def get_window_transform(self, window):
        if window is None:
            return
        return windows.transform(window, self.transform)

    @property
    def x(self):
        return self.get_x()

    @property
    def y(self):
        return self.get_y()

    @property
    def values(self):
        return self.get_values()

    @property
    def md5(self):
        hash_md5 = hashlib.md5()
        with open(self._tmpfile.resolve(), "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @property
    def count(self):
        return self.src.count

    @property
    def is_masked(self):
        for window in self.iter_windows(self.chunk_size):
            if self.src.nodata in self.src.read(window=window):
                return True
        return False

    @property
    def shape(self):
        return self.src.shape

    @property
    def height(self):
        return self.src.height

    @property
    def bbox(self) -> Bbox:
        """Returns default bbox."""
        return self.get_bbox()

    @property
    def path(self):
        return self._path

    @property
    def _path(self):
        return self.__path

    @_path.setter
    def _path(self, path: Union[str, os.PathLike]):
        if pathlib.Path(path).exists() is False:
            try:
                r = requests.get(path, stream=True)
                r.raise_for_status()
                tmpfile = tempfile.NamedTemporaryFile()
                with open(tmpfile.name, "wb") as f:
                    logger.debug(f"Downloading raster data from {path}...")
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                self._tmpfile = tmpfile
                self.__path = pathlib.Path(tmpfile.name)
            except requests.exceptions.MissingSchema:
                raise Exception(
                    f"Given path {path} is neither a valid URL nor a system file."
                )
        else:
            self.__path = pathlib.Path(path)

    @property
    def tmpfile(self) -> pathlib.Path:
        return self._tmpfile

    @property
    def _tmpfile(self):
        try:
            return pathlib.Path(self.__tmpfile.name)
        except AttributeError:
            return self.path

    @_tmpfile.setter
    def _tmpfile(self, tmpfile: tempfile._TemporaryFileWrapper):
        logger.debug(f"rasterio.open({tmpfile.name})")
        self._src = rasterio.open(tmpfile.name)
        logger.debug(f"save tmpfile {tmpfile.name}")
        self.__tmpfile = tmpfile

    @property
    def src(self) -> rasterio.io.DatasetReader:
        return self._src

    @property
    def _src(self) -> rasterio.io.DatasetReader:
        try:
            return self.__src
        except AttributeError:
            self._src = rasterio.open(self.path)
            return self.__src

    @_src.setter
    def _src(self, src: rasterio.io.DatasetReader):
        assert isinstance(src, rasterio.io.DatasetReader)
        self.__src = src

    @_src.deleter
    def _src(self):
        try:
            del self.__src
        except AttributeError:
            pass

    @property
    def tmpdir(self) -> pathlib.Path:
        """
        Directory used for copy-on-write operations.
        """
        if not hasattr(self, "_tmpdir"):
            tmpdir = os.getenv("GEOMESH_TEMPORARY_DIRECTORY")
            if tmpdir is None:
                self.__tmpdir = tempfile.TemporaryDirectory()
                tmpdir = self.__tmpdir.name
            self._tmpdir = pathlib.Path(tmpdir)
        return self._tmpdir

    @tmpdir.setter
    def tmpdir(self, tmpdir: Union[tempfile.TemporaryDirectory, str, os.PathLike]):
        if isinstance(tmpdir, tempfile.TemporaryDirectory):
            self.__tmpdir = tmpdir
            tmpdir = self.__tmpdir.name
        self._tmpdir = pathlib.Path(tmpdir)

    @property
    def width(self):
        return self.src.width

    @property
    def dx(self):
        return self.src.transform[0]

    @property
    def dy(self):
        return -self.src.transform[4]

    @property
    def crs(self) -> CRS:
        # cast rasterio.CRS to pyproj.CRS for API consistency
        return CRS.from_user_input(self.src.crs)

    @property
    def _crs(self):
        return self.src.crs

    @_crs.setter
    def _crs(self, crs: Union[None, str, CRS]):
        # check if CRS is in file
        if crs is None:
            with rasterio.open(self.path) as src:
                # Raise if CRS not in file and the user did not provide a CRS.
                # All Rasters objects must have a defined CRS.
                # Program cannot operate with an undefined CRS.
                crs = src.crs
                if crs is None:
                    raise IOError("CRS not found in raster file. Must specify CRS.")
        # CRS is specified by user rewrite raster but add CRS to meta
        else:
            if isinstance(crs, str):
                crs = CRS.from_user_input(crs)

            if not isinstance(crs, CRS):
                raise TypeError(
                    f"Argument crs must be of type {str} or {CRS},"
                    f" not type {type(crs)}."
                )
            # create a temporary copy of the original file and update meta.
            tmpfile = tempfile.NamedTemporaryFile()
            with rasterio.open(self.path) as src:
                if self.chunk_size is not None:
                    windows = get_iter_windows(
                        src.width, src.height, chunk_size=self.chunk_size
                    )
                else:
                    windows = [rasterio.windows.Window(0, 0, src.width, src.height)]
                meta = src.meta.copy()
                meta.update({"crs": crs, "driver": "GTiff"})
                with rasterio.open(
                    tmpfile,
                    "w",
                    **meta,
                ) as dst:
                    for window in windows:
                        dst.write(src.read(window=window), window=window)
            self._tmpfile = tmpfile

    @property
    def nodatavals(self):
        return self.src.nodatavals

    @property
    def transform(self):
        return self.src.transform

    @property
    def dtypes(self):
        return self.src.dtypes

    @property
    def nodata(self):
        return self.src.nodata

    @property
    def xres(self):
        return self.transform[0]

    @property
    def yres(self):
        return self.transform[4]

    @property
    def chunk_size(self) -> Union[int, None]:
        """
        Maximum allowable window size on either direction (e.g. iteration windows shall not exceed [chunk_size x chunk_size]).
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size: Union[int, None]):
        self._chunk_size = chunk_size

    @property
    def overlap(self) -> Union[int, None]:
        """
        Number of pixel overlap between two adjacent windows.
        """
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: Union[int, None]):
        self._overlap = overlap

import math
def get_iter_windows(
        width,
        height,
        chunk_size=0,
        overlap=0,
        # row_off=0,
        # col_off=0
):
    # win_h = chunk_size + overlap
    # win_w = chunk_size + overlap
    n_win_h = math.ceil(height / chunk_size)
    n_win_w = math.ceil(width / chunk_size)
    for i in range(n_win_h):
        for j in range(n_win_w):
            off_h = i * chunk_size
            off_w = j * chunk_size
            h = chunk_size + overlap
            h = h - (off_h + h) % height if off_h + h > height else h
            w = chunk_size + overlap
            w = w - (off_w + w) % width if off_w + w > width else w
            yield windows.Window(off_w, off_h, w, h)


# def get_multipolygon_from_axes(ax):
#     # extract linear_rings from plot
#     linear_ring_collection = list()
#     for path_collection in ax.collections:
#         for path in path_collection.get_paths():
#             polygons = path.to_polygons(closed_only=True)
#             for linear_ring in polygons:
#                 if linear_ring.shape[0] > 3:
#                     linear_ring_collection.append(
#                         LinearRing(linear_ring))
#     if len(linear_ring_collection) > 1:
#         # reorder linear rings from above
#         areas = [Polygon(linear_ring).area
#                  for linear_ring in linear_ring_collection]
#         idx = np.where(areas == np.max(areas))[0][0]
#         polygon_collection = list()
#         outer_ring = linear_ring_collection.pop(idx)
#         path = Path(np.asarray(outer_ring.coords), closed=True)
#         while len(linear_ring_collection) > 0:
#             inner_rings = list()
#             for i, linear_ring in reversed(
#                     list(enumerate(linear_ring_collection))):
#                 xy = np.asarray(linear_ring.coords)[0, :]
#                 if path.contains_point(xy):
#                     inner_rings.append(linear_ring_collection.pop(i))
#             polygon_collection.append(Polygon(outer_ring, inner_rings))
#             if len(linear_ring_collection) > 0:
#                 areas = [Polygon(linear_ring).area
#                          for linear_ring in linear_ring_collection]
#                 idx = np.where(areas == np.max(areas))[0][0]
#                 outer_ring = linear_ring_collection.pop(idx)
#                 path = Path(np.asarray(outer_ring.coords), closed=True)
#         multipolygon = MultiPolygon(polygon_collection)
#     else:
#         multipolygon = MultiPolygon(
#             [Polygon(linear_ring_collection.pop())])
#     return multipolygon
