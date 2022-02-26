from glob import glob
import logging
import os
import pathlib
import tempfile
from typing import Callable, Dict, Generator, Union
from urllib.parse import urlparse

from appdirs import user_data_dir
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import box, Polygon
import wget

from .yamlparser import YamlParser, YamlComponentParser
from ...raster import Raster

logger = logging.getLogger(__name__)


class RasterConfig(YamlComponentParser):

    def __init__(self, parser: YamlParser, skip_checks: bool = False):

        if 'rasters' not in parser.yaml:
            return
        if skip_checks == False:
            for raster in parser.yaml['rasters']:
                self.validate_raster_request(raster)
                has_tile_index = bool(raster.get("tile-index", False))
                has_tile_index_under = bool(raster.get("tile_index", False))
                if has_tile_index or has_tile_index_under:
                    # this forces the download upon init
                    for _ in self.expand_tile_index(raster, yield_type='path'):
                        pass
                    
                
            # if key == "mesh":
            #     logger.info(f'Loading mesh from file: {pathlib.Path(feature[key])}.')
            #     mesh = feature.pop(key)
            #     feature.update({
            #         key: mesh,
            #         'object': Mesh.open(mesh, **feature)
            #     })
        super().__init__(parser)

    def from_request(self, request: Dict, yield_type='object') -> Generator:
        logger.info(f"Get raster from request: {request}")
        rtype = self.get_request_type(request)
        if rtype == "path":
            logger.info(f'Requested raster is a local path: {request["path"]}')
            for raster in self.expand_path(request, yield_type=yield_type):
                yield raster
        elif rtype == "tile-index" or rtype == "tile_index":
            logger.info(f'Requested raster is a tile index: {request[rtype]}')
            for raster in self.expand_tile_index(request, yield_type=yield_type):
                yield raster
        else:
            raise TypeError(f'Unhandled type: {rtype}')

    def get_request_type(self, request):
        self.validate_raster_request(request)
        if "path" in request:
            return "path"
        elif "tile-index" in request:
            return "tile_index"
        elif 'tile_index' in request:
            return 'tile_index'
        elif "url" in request:
            return "url"
        
    def _check_yield_type(f: Callable):
        def decorator(self, request: Dict, yield_type='object'):
            if yield_type not in ['path', 'object']:
                raise ValueError(f'Argument `yield_type` must be a string of value "path" or "object", not {yield_type}.')
            return f(self, request, yield_type)
        return decorator

    @_check_yield_type
    def expand_path(self, request: Dict, yield_type='object') -> Generator:
        requested_paths = request["path"]
        if isinstance(requested_paths, str):
            requested_paths = [requested_paths]
        for requested_path in list(requested_paths):
            requested_path = os.path.expandvars(requested_path)
            if '*' in requested_path:
                paths = list(glob(str(pathlib.Path(requested_path).resolve())))
                if len(paths) == 0:
                    raise ValueError(f'No rasters found on path {requested_path}')
                for path in paths:
                    if yield_type == "object":
                        yield self.apply_opts(Raster(path), request)
                    elif yield_type == "path":
                        yield path, request
                        
            else:
                if yield_type == "object":
                    yield self.apply_opts(Raster(requested_path), request)
                elif yield_type == "path":
                    yield requested_path, request


    def validate_raster_request(self, request: Dict):

        has_url = bool(request.get("url", False))
        has_tile_index = bool(request.get("tile-index", False))
        has_tile_index_under = bool(request.get("tile_index", False))
        has_path = bool(request.get("path", False))

        if not (has_url ^ has_tile_index ^ has_path ^ has_tile_index_under):

            class YamlRastersInputError(Exception):
                pass

            raise YamlRastersInputError(
                f"{self.key} entry in yaml file must contain at least one of `url`, `tile_index` of `path`"
            )

# 
    # def get_opts(self, request) -> Dict:
    #     print('get_opts', request)
    #     exit()
    #     rtype = self.get_request_type(request)
    #     opts = {}
    #     for raster_request in self.config.rasters:
    #         print(raster_request)
    #         exit()
    #         if rtype in raster_request:
    #             if request[rtype] == raster_request[rtype]:
    #                 opts = raster_request.copy()
    #                 opts.pop(rtype)
    #     return opts

    @staticmethod
    def apply_opts(raster: Raster, opts: Dict) -> Union[Raster, None]:
        import numpy as np
        for key, selector in opts.items():
            if key == "resample":
                raster.resample(selector["scaling_factor"])
            if key == "warp":
                raster.warp(selector)
            if key == "fill_nodata":
                raster.fill_nodata()
            if key == "clip":
                if 'bbox' in selector:
                    bbox_selector = selector['bbox']
                    if isinstance(bbox_selector, dict):
                        has_mesh = bool(bbox_selector.get('mesh', False))
                        has_xmin = bool(bbox_selector.get('xmin', False))
                        has_xmax = bool(bbox_selector.get('xmax', False))
                        has_ymin = bool(bbox_selector.get('ymin', False))
                        has_ymax = bool(bbox_selector.get('ymax', False))
                        if has_mesh:
                            mesh_bbox = selector['bbox']['object'].get_bbox(crs=raster.crs)
                            raster.clip(box(mesh_bbox.xmin, mesh_bbox.ymin, mesh_bbox.xmax, mesh_bbox.ymax))
                        elif np.any([has_xmin, has_xmax, has_ymin, has_ymax]):
                            raster.clip(box(
                                bbox_selector.get('xmin', np.min(raster.x)),
                                bbox_selector.get('ymin', np.min(raster.y)),
                                bbox_selector.get('xmax', np.max(raster.x)),
                                bbox_selector.get('ymax', np.max(raster.y))
                            ))
                        else:
                            raise NotImplementedError(f'clip option {selector} 1')

                    else:
                        raise NotImplementedError(f'invalid bbox_selector: {bbox_selector}')

                else:
                    raise NotImplementedError(f'clip option {selector} 3')
            if key == "chunk_size":
                raster.chunk_size = selector
            if key == "overlap":
                raster.overlap = selector
            # if key == 'gaussian_filter':

        return raster
    
    @_check_yield_type
    def expand_tile_index(self, request: Dict, yield_type='object') -> Generator:
        tile_index_file = request.get("tile_index") if request.get('tile-index') is None else request.get('tile-index')
        bbox = request.get('bbox')
        if bbox is not None:
            has_mesh = bool(bbox.get('mesh', False))
            has_xmin = bool(bbox.get('xmin', False))
            has_xmax = bool(bbox.get('xmax', False))
            has_ymin = bool(bbox.get('ymin', False))
            has_ymax = bool(bbox.get('ymax', False))
            gdf = gpd.read_file(tile_index_file)
            import numpy as np
            if has_mesh:
                # print(gdf.crs)
                # exit()
                bbox = bbox['object'].get_bbox(crs=gdf.crs, return_type='shapely')
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

        cache_dir = self.get_raster_cache_directory_from_request(request)

        for row in gdf.itertuples():
            parsed_url = urlparse(row.URL)
            fname = cache_dir / parsed_url.netloc / parsed_url.path[1:]
            fname.parent.mkdir(exist_ok=True, parents=True)
            if not fname.is_file():
                logger.info(f"Downloading url {row.URL} to {fname}")
                wget.download(
                    row.URL,
                    out=str(fname.parent),
                )
            if yield_type == "object":
                yield self.apply_opts(Raster(fname), request)
            elif yield_type == "path":
                yield fname, request
            # logger.info(f'Yield raster {fname}')
            


    def get_raster_cache_directory_from_request(self, request: Dict) -> pathlib.Path:
        cache_opt = request.get("cache", True)
        if cache_opt is True:
            default_cache_dir = (
                pathlib.Path(user_data_dir()) / "geomesh" / "raster_cache"
                # self.path.parent / ".cache" / "raster_cache"
            )
            default_cache_dir.mkdir(exist_ok=True, parents=True)
            return default_cache_dir

        elif cache_opt is None or cache_opt is False:
            self.tmpdir = tempfile.TemporaryDirectory()
            return pathlib.Path(self.tmpdir.name)

        else:
            return pathlib.Path(cache_opt)

    # @staticmethod
    # def get_clip_from_opts(self, opts: Dict) -> MultiPolygon:
    #     for feat_request in self.config.raw.get("features", []):
    #         for key, val in feat_request.items():
    #             if val == opts:
    #                 feat_crs = feat_request.get("crs")
    #                 if feat_crs is not None:
    #                     feat_crs = CRS.from_user_input(feat_crs)
    #     return feat_crs
    
    # def get_bbox_clip_from_opts(self, opts: Dict, raster_crs: Union[CRS, None]) -> Polygon:
    #     if 'mesh' in opts: return box(*opts['object'].get_bbox(crs=raster_crs).bounds)
    #     raise NotImplementedError(opts)
        
        # opts may be a mesh, a bbox array or a shapefile
        # print(key, opts)
        # exit()
        # for feat_request in self.config.raw.get("features", []):
        #     for key, val in feat_request.items():
        #         if val == opts:
        #             feat_crs = feat_request.get("crs")
        #             if feat_crs is not None:
        #                 feat_crs = CRS.from_user_input(feat_crs)
        # return feat_crs

    @property
    def key(self):
        return "raster"
