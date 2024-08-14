from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Dict, Generator
from urllib.parse import urlparse
import argparse
import hashlib
import os
import tempfile
import warnings
import logging

from appdirs import user_data_dir
from matplotlib.transforms import Bbox
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from shapely.geometry import box, Polygon
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
import numpy as np
import fiona
import pyarrow
import rasterio
import wget

from geomesh.mesh import Mesh
from geomesh.raster import Raster, get_iter_windows


logger = logging.getLogger(__name__)


warnings.filterwarnings(
    'ignore',
    message="Sequential read of iterator was interrupted. Resetting iterator. This can negatively impact the performance."
    )


class RasterAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        temp_parser = argparse.ArgumentParser(add_help=False)
        temp_parser.add_argument('raster', type=Path)
        _add_raster_opts(temp_parser)
        tmp_args, _ = temp_parser.parse_known_args()

        # if np.any([tmp_args.xmin, tmp_args.xmax, tmp_args.ymin, tmp_args.ymax]) and args.window is not None:
        #     raise ValueError('Arguments [--xmin, --xmax, --ymin, --ymax] and --window are mutually exclusive.')

        # if np.any([tmp_args.xmin, tmp_args.xmax, tmp_args.ymin, tmp_args.ymax]):
        #     print(profile['transform'])
        #     exit()
        #     xmin = tmp_args.xmin if tmp_args.xmin is not None else profile['transform']
        #     ymin = tmp_args.ymin if tmp_args.ymin is not None else profile['transform']
        #     xmax = tmp_args.xmax if tmp_args.xmax is not None else profile['transform']
        #     ymax = tmp_args.ymax if tmp_args.ymax is not None else profile['transform']
        # else:
        #     bbox = None
        if tmp_args.clip is not None:
            with rasterio.open(tmp_args.raster) as src:
                crs = src.crs
            tmp_args.clip.to_crs(crs, inplace=True)
            clip = tmp_args.clip.unary_union
        else:
            clip = None

        if tmp_args.mask is not None:
            with rasterio.open(tmp_args.raster) as src:
                crs = src.crs
            tmp_args.mask.to_crs(crs, inplace=True)
            mask = tmp_args.mask.unary_union
        else:
            mask = None
        raster = Raster(
            tmp_args.raster,
            crs=tmp_args.src_crs,
            chunk_size=tmp_args.chunk_size,
            overlap=tmp_args.overlap,
            resampling_factor=tmp_args.resampling_factor,
            resampling_method=tmp_args.resampling_method,
            bbox=tmp_args.bbox,
            window=tmp_args.window,
            clip=clip,
            mask=mask,
            )

        setattr(namespace, self.dest, raster)


class WindowAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        setattr(namespace, self.dest, rasterio.windows.Window(*[float(_) for _ in values]))


class BboxAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):

        _tmp_parser = argparse.ArgumentParser(add_help=False)
        _tmp_parser.add_argument('raster')
        _tmp_parser.add_argument('--xmin', type=float)
        _tmp_parser.add_argument('--ymin', type=float)
        _tmp_parser.add_argument('--xmax', type=float)
        _tmp_parser.add_argument('--ymax', type=float)
        _tmp_parser.add_argument('--window')

        known, unknow = _tmp_parser.parse_known_args()

        if known.window is not None:
            raise ValueError('Cannot use both window and bbox at the same time.')

        if isinstance(getattr(namespace, self.dest), Bbox):
            return

        with rasterio.open(known.raster) as src:
            _full_window = rasterio.windows.Window(0, 0, src.width, src.height)
            x0, y0, x1, y1 = rasterio.transform.array_bounds(
                _full_window.height,
                _full_window.width,
                rasterio.windows.transform(_full_window, src.transform)
            )

        xmin = x0 if known.xmin is None else known.xmin
        ymin = y0 if known.ymin is None else known.ymin
        xmax = x0 if known.xmax is None else known.xmax
        ymax = y0 if known.ymax is None else known.ymax

        setattr(namespace, self.dest, Bbox.from_extents([xmin, ymin, xmax, ymax]))


def _load_geometry_file(fname):
    gdf = None
    for func in [gpd.read_file, gpd.read_feather]:
        try:
            gdf = func(fname)
        except pyarrow.lib.ArrowInvalid as err:
            if 'Not a Feather V1 or Arrow IPC file' in str(err):
                continue
            raise err
        except fiona.errors.DriverError as err:
            if 'not recognized as a supported file format.' in str(err):
                continue
            raise err
    if gdf is None and fname is not None:
        raise Exception(f'Failed to read file {fname}.')
    return gdf


def _add_raster_opts(parser):
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--crs', '--src-crs', dest='src_crs', type=CRS.from_user_input)
    parser.add_argument('--window', nargs=4, action=WindowAction)
    parser.add_argument('--xmin', type=float, action=BboxAction, dest='bbox')
    parser.add_argument('--ymin', type=float, action=BboxAction, dest='bbox')
    parser.add_argument('--xmax', type=float, action=BboxAction, dest='bbox')
    parser.add_argument('--ymax', type=float, action=BboxAction, dest='bbox')
    parser.add_argument('--resampling-factor', type=float)
    parser.add_argument('--resampling-method', type=lambda x: Resampling[x], default=Resampling.bilinear)
    parser.add_argument('--clip', type=_load_geometry_file)
    parser.add_argument('--mask', type=_load_geometry_file)


def add_raster_args(parser):
    parser.add_argument('raster', help='URI (path or url).', action=RasterAction, type=Path)
    _add_raster_opts(parser)


def append_raster_cmd_opts(cmd, opts):
    # raster_opts = {}
    if 'clip' in opts:
        # raster_opts.update({'clip': opts['clip']})
        cmd.append(f'--clip={Path(opts["clip"]).resolve()}')

    if 'chunk_size' in opts:
        # raster_opts.update({'chunk_size': opts['chunk_size']})
        cmd.append(f'--chunk-size={opts["chunk_size"]}')

    if 'overlap' in opts:
        cmd.append(f'--overlap={opts["overlap"]}')
        # raster_opts.update({'overlap': opts['overlap']})

    # if 'gaussian_filter' in opts:
    #     raster_opts.update({'gaussian_filter': opts['gaussian_filter']})

    if 'resampling_factor' in opts:
        # raster_opts.update({'resample': opts['resample']})
        cmd.append(f'--resampling-factor={opts["resampling_factor"]}')

    if 'resampling_method' in opts:
        cmd.append(f'--resampling-method={opts["resampling_method"]}')

    if 'bbox' in opts:

        xmin = opts.get('xmin', )
        if xmin is not None:
            cmd.append(f'--xmin={xmin}')

        xmax = opts.get('xmax')
        if xmax is not None:
            cmd.append(f'--xmax={xmax}')

        ymin = opts.get('ymin')
        if ymin is not None:
            cmd.append(f'--ymin={ymin}')

        ymax = opts.get('ymax')
        if ymax is not None:
            cmd.append(f'--ymax={ymax}')

    if 'window' in opts:
        raise NotImplementedError('window argument')

# if 'fill_nodata' in opts:
    #     raster_opts.update({'fill_nodata': opts['fill_nodata']})
        
    # TODO: There are additional more raster_opts to consider !!!!!!!!!!
    # fill_nodata
    # resample
    # if len(raster_opts) > 0:
        # cmd.append(f"--raster-opts='{json.dumps(raster_opts)}'")


def iter_raster_requests(self):
    for request in self.get('rasters', []):
        if 'path' in request:
            # logger.info(f'Requested raster is a local path: {request["path"]}')
            for path, request in expand_raster_request_path(request):
                yield path, request
        elif 'tile_index' in request:
            for path, request in expand_tile_index(self, request):
                yield path, request
        else:
            raise TypeError(f'Unhandled type: {request}')


def expand_tile_index(self, request: Dict) -> Generator:
    tile_index_file = request.get("tile_index") if request.get('tile-index') is None else request.get('tile-index')
    if not isinstance(tile_index_file, gpd.GeoDataFrame):
        bbox = request.get('bbox')
        if bbox is not None:
            if not isinstance(bbox, Polygon):
                has_mesh = bool(bbox.get('mesh', False))
                has_xmin = bool(bbox.get('xmin', False))
                has_xmax = bool(bbox.get('xmax', False))
                has_ymin = bool(bbox.get('ymin', False))
                has_ymax = bool(bbox.get('ymax', False))
                gdf = gpd.read_file(tile_index_file)
                if has_mesh:
                    _bbox = Mesh.open(bbox.get('mesh'), crs=bbox.get('crs')).get_bbox(crs=gdf.crs)
                    bbox = box(*_bbox.extents)
                    request.update({'bbox': {
                            'xmin': _bbox.xmin,
                            'xmax': _bbox.xmax,
                            'ymin': _bbox.ymin,
                            'ymax': _bbox.ymax,
                            'crs': gdf.crs
                        }})
                elif np.any([has_xmin, has_xmax, has_ymin, has_ymax]):
                    gdf_bounds = gdf.bounds
                    bbox = box(
                        bbox.get('xmin', np.min(gdf_bounds['minx'])),
                        bbox.get('ymin', np.min(gdf_bounds['miny'])),
                        bbox.get('xmax', np.max(gdf_bounds['maxx'])),
                        bbox.get('ymax', np.max(gdf_bounds['maxx'])),
                    )
        gdf = gpd.read_file(
            tile_index_file,
            bbox=bbox,
        )
    cache_opt = request.get("cache", True)

    # default. User wants the user_data_dir cache
    if cache_opt is True:
        cache_dir = (
            Path(user_data_dir()) / "geomesh" / "raster_cache"
            # self.path.parent / ".cache" / "raster_cache"
        )
        cache_dir.mkdir(exist_ok=True, parents=True)

    # User wants fresh download all the time (pointless?).
    elif cache_opt is None or cache_opt is False:
        self._raster_cache_tmpdir = tempfile.TemporaryDirectory()
        cache_dir = Path(self._raster_cache_tmpdir.name)
    # User wants specific directory
    else:
        cache_dir = Path(cache_opt)

    for row in gdf.itertuples():
        parsed_url = urlparse(row.URL)
        fname = cache_dir / parsed_url.netloc / parsed_url.path[1:]
        fname.parent.mkdir(exist_ok=True, parents=True)
        if not fname.is_file():
            logger.debug(f"Downloading {row.URL} to {fname}\n")
            wget.download(
                row.URL,
                out=str(fname.parent),
            )
        yield fname, request
        # logger.info(f'Yield raster {fname}')


def expand_raster_request_path(request: Dict) -> Generator:
    request = request.copy()
    requested_paths = request.pop("path")
    if isinstance(requested_paths, str):
        requested_paths = [requested_paths]
    for requested_path in list(requested_paths):
        requested_path = os.path.expandvars(requested_path)
        if '*' in requested_path:
            paths = list(glob(str(Path(requested_path).resolve())))
            if len(paths) == 0:
                raise ValueError(f'No rasters found on path {requested_path}')
        else:
            paths = [requested_path]
        for path in paths:
            if 'bbox' in request:
                request_bbox = get_bbox_from_request(path, request)
                with rasterio.open(path) as src:
                    profile = src.profile
                raster_bbox = Bbox.from_extents(*rasterio.transform.array_bounds(
                    profile['height'],
                    profile['width'],
                    profile['transform'],
                ))
                if not request_bbox.overlaps(raster_bbox):
                    continue
                if 'clip' in request and request['clip'] is not None:
                    clip = _load_geometry_file(request['clip']['path']).to_crs(profile['crs'])
                    if not np.any(clip.intersects(box(*request_bbox.get_points().flatten()))):
                        continue
                if 'mask' in request and request['mask'] is not None:
                    raise NotImplementedError('[FutureMeError]: lines are untested')
                    mask = _load_geometry_file(request['mask']['path']).to_crs(profile['crs'])
                    if not np.all(mask.intersects(box(*request_bbox.get_points().flatten()))):
                        continue
            yield Path(path).resolve(), request


def get_bbox_from_request(raster_path, request_opts):

    with rasterio.open(raster_path) as src:
        profile = src.profile
    x0, y0, x1, y1 = rasterio.transform.array_bounds(
        profile['height'],
        profile['width'],
        profile['transform'],
    )
    # print(x0, y0, x1, y1)
    bbox_request = request_opts.get('bbox') or {}
    xmin = bbox_request.get('xmin', x0)
    ymin = bbox_request.get('ymin', y0)
    xmax = bbox_request.get('xmax', x1)
    ymax = bbox_request.get('ymax', y1)

    bbox = Bbox.from_extents(xmin, ymin, xmax, ymax)

    if 'crs' in request_opts and request_opts['crs'] is not None:
        bbox_crs = CRS.from_user_input(request_opts['crs'])
    else:
        bbox_crs = CRS.from_user_input(profile['crs'])

    if not bbox_crs.equals(CRS.from_user_input(profile['crs'])):
        bbox = transform_bbox(bbox, bbox_crs, profile['crs'])

    return bbox


@lru_cache
def transform_bbox(bbox, bbox_crs, dst_crs):
    (xmin, xmax), (ymin, ymax) = Transformer.from_crs(bbox_crs, dst_crs, always_xy=True
                                                      ).transform([bbox.xmin, bbox.xmax], [bbox.ymin, bbox.ymax])
    return Bbox.from_extents(xmin, ymin, xmax, ymax)


def iter_raster_windows(raster_path, request_opts):
    with rasterio.open(raster_path) as src:
        profile = src.profile

    if 'bbox' in request_opts and request_opts['bbox'] is not None:
        x0, y0, x1, y1 = rasterio.transform.array_bounds(
            profile['height'],
            profile['width'],
            profile['transform'],
        )
        # print(x0, y0, x1, y1)
        bbox_dump = request_opts['bbox'].model_dump()
        xmin = bbox_dump.get('xmin', x0)
        ymin = bbox_dump.get('ymin', y0)
        xmax = bbox_dump.get('xmax', x1)
        ymax = bbox_dump.get('ymax', y1)

        xmin = np.max([x0, xmin])
        ymin = np.max([y0, ymin])
        xmax = np.min([x1, xmax])
        ymax = np.min([y1, ymax])

        bbox = Bbox.from_extents(xmin, ymin, xmax, ymax)

        if 'crs' in bbox_dump and bbox_dump['crs'] is not None:
            bbox_crs = CRS.from_user_input(bbox_dump['crs'])
        else:
            bbox_crs = CRS.from_user_input(profile['crs'])

        if not bbox_crs.equals(CRS.from_user_input(profile['crs'])):
            bbox = transform_bbox(bbox, bbox_crs, profile['crs'])

        window = rasterio.windows.from_bounds(
                # x0, y0, x1, y1,
                bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax,
                transform=profile['transform'],
                # height=profile['height'],
                # width=profile['width'],
                )
        row_off = int(np.floor(window.row_off))
        col_off = int(np.floor(window.col_off))
        width = min([int(np.ceil(window.width)), profile['width']])
        height = min([int(np.ceil(window.height)), profile['height']])
    else:
        row_off = 0
        col_off = 0
        width = profile['width']
        height = profile['height']

    chunk_size = request_opts.get('chunk_size')
    if chunk_size is not None:
        resampling_factor = request_opts.get('resampling_factor')
        if resampling_factor is not None:
            chunk_size /= resampling_factor
    # clip = request_opts.get('clip')
    # from geomesh.cli.mpi.lib import RasterClipStrConfig
    # if clip is not None:
    #     # what type of clip?
    #     if isinstance(clip, RasterClipStrConfig):
    #         # We assume strings to be loadable by geopandas.
    #         # There is a function later that will determine whether or not
    #         # it is geopandas loadable, and that should probably be done here
    #         # instead.
    #         request_opts.update({'clip': Path(clip)})
    #     elif isinstance(clip, Path):
    #         pass
    #     elif isinstance(clip, dict):
    #         if 'mesh' in clip:
    #             if cache_directory is None:
    #                 # TODO: fix this shortcut
    #                 cache_directory = Path('/tmp')
    #             outname = cache_directory / f'{get_digest(clip["mesh"])}.feather'
    #             if outname.exists():
    #                 request_opts.update({'clip': outname})
    #             else:
    #                 crs = clip.get('crs')
    #                 if crs is not None:
    #                     crs = CRS.from_user_input(crs)
    #                 mesh = Mesh.open(clip['mesh'], crs=crs)
    #                 clip_poly = mesh.hull.multipolygon()
    #                 gpd.GeoDataFrame([{'geometry': clip_poly}], crs=crs).to_feather(outname)
    #                 request_opts.update({'clip': outname})
    #     else:
    #         raise NotImplementedError(
    #                 f'Unhandled clip argument for {raster_path} as {clip}.'
    #             )

    for j, window in enumerate(get_iter_windows(
            width,
            height,
            chunk_size=chunk_size,
            overlap=request_opts.get('overlap', 0),
            row_off=row_off,
            col_off=col_off,
            )):
        yield j, window

def iter_raster_window_requests(config_request, cache_directory=None):

    for i, (raster_path, request_opts) in enumerate(iter_raster_requests(config_request)):
        for j, window in iter_raster_windows(raster_path, request_opts):
            yield (i, raster_path), (j, window)


def get_digest(file_path):
    h = hashlib.sha256()

    with open(file_path, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def get_raster_from_opts(
        raster_path,
        crs=None,
        clip=None,
        mask=None,
        window=None,
        chunk_size=None,
        overlap=0,
        resampling_factor=None,
        resampling_method=None,
        ):
    from geomesh.cli.mpi.lib import RasterClipStrConfig

    if crs is not None:
        crs = CRS.from_user_input(crs)
    else:
        with rasterio.open(raster_path) as src:
            src_crs = src.crs
    if clip is None:
        pass
    elif isinstance(clip, RasterClipStrConfig):
        clip = _load_geometry_file(clip.path)
        clip.to_crs(crs if crs is not None else src_crs, inplace=True)
        clip = clip.unary_union
    else:
        raise TypeError(f"Unreachable: Expected clip of type None or RasterClipStrConfig but got {type(clip)=}")

    if mask is not None:
        if isinstance(mask, BaseGeometry):
            pass
        else:
            mask = _load_geometry_file(mask['path'])
            mask.to_crs(crs if crs is not None else src_crs, inplace=True)
            mask = mask.unary_union
    return Raster(
        raster_path,
        crs=crs,
        chunk_size=chunk_size,
        overlap=overlap,
        resampling_factor=resampling_factor,
        resampling_method=resampling_method,
        # bbox=tmp_args.bbox,
        window=window,
        clip=clip,
        mask=mask,
        )
