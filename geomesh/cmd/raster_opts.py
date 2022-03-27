import argparse
from glob import glob
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Generator
from urllib.parse import urlparse

from appdirs import user_data_dir
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import wget

from geomesh.raster import Raster
from geomesh.mesh import Mesh


class RasterOptsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        raster_opts = json.loads(values)
        if raster_opts is None:
            setattr(namespace, self.dest, {})
            return
        if 'clip' in raster_opts:
            if 'bbox' in raster_opts['clip']:
                raster = self.get_raster(raster_opts.get('crs'))
                raster_opts['clip'] = box(
                    raster_opts['clip']['bbox'].get('xmin', np.min(raster.x)),
                    raster_opts['clip']['bbox'].get('ymin', np.min(raster.y)),
                    raster_opts['clip']['bbox'].get('xmax', np.max(raster.x)),
                    raster_opts['clip']['bbox'].get('ymax', np.max(raster.y))
                )
        setattr(namespace, self.dest, raster_opts)
        
    def get_raster(self, crs=None):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('raster', type=lambda x: Path(x))
        known_args, _ = parser.parse_known_args()
        return Raster(known_args.raster, crs=crs)
    
class RasterAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('raster', type=lambda x: Path(x))
        parser.add_argument('--raster-opts', '--raster_opts', action=RasterOptsAction, default={})
        args, _ = parser.parse_known_args()
        raster = Raster(
            args.raster,
            crs=args.raster_opts.get('crs'),
            chunk_size=args.raster_opts.get('chunk_size'),
            overlap=args.raster_opts.get('overlap', 2),
            )
        if 'clip' in args.raster_opts:
            raster.clip(args.raster_opts['clip'])
        if 'resample' in args.raster_opts:
            if isinstance(args.raster_opts['resample'], float):
                args.raster_opts['resample'] = {
                    'scaling_factor': args.raster_opts['resample']
                }
            if not isinstance(args.raster_opts['resample'], dict):
                raise ValueError('resample argument must be a dict with req key scaling_factor ')
            raster.resample(
                args.raster_opts['resample']['scaling_factor'],
                resampling_method=args.raster_opts['resample'].get('resampling_method')
            )
        setattr(namespace, self.dest, raster)


def add_raster_args(parser):
    parser.add_argument('raster', help='URI (path or url).', action=RasterAction)
    parser.add_argument('--raster-opts', '--raster_opts',
                        # action=RasterOptsAction
                        )
    
def append_cmd_opts(cmd, opts):
    raster_opts = {}
    if 'clip' in opts:
        raster_opts.update({'clip': opts['clip']})
        
    if 'chunk_size' in opts:
        raster_opts.update({'chunk_size': opts['chunk_size']})

    if 'overlap' in opts:
        raster_opts.update({'overlap': opts['overlap']})
        
    if 'gaussian_filter' in opts:
        raster_opts.update({'gaussian_filter': opts['gaussian_filter']})

    if 'resample' in opts:
        raster_opts.update({'resample': opts['resample']})

    if 'fill_nodata' in opts:
        raster_opts.update({'fill_nodata': opts['fill_nodata']})
        
    # TODO: There are additional more raster_opts to consider !!!!!!!!!!
    # fill_nodata
    # resample
    if len(raster_opts) > 0:
        cmd.append(f"--raster-opts='{json.dumps(raster_opts)}'")
        

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
    bbox = request.get('bbox')
    if bbox is not None:
        has_mesh = bool(bbox.get('mesh', False))
        has_xmin = bool(bbox.get('xmin', False))
        has_xmax = bool(bbox.get('xmax', False))
        has_ymin = bool(bbox.get('ymin', False))
        has_ymax = bool(bbox.get('ymax', False))
        gdf = gpd.read_file(tile_index_file)

        if has_mesh:
            bbox = box(*Mesh.open(bbox.pop('mesh'), **bbox).get_bbox(crs=gdf.crs).extents)
            
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
            for path in paths:
                yield path, request
                    
        else:
            yield requested_path, request