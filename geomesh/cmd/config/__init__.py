from abc import ABC, abstractmethod
# from collections import UserDict
from functools import cached_property
# from genericpath import exists
from glob import glob
import hashlib
import logging
import os
import pathlib
# from pprint import pformat
# from turtle import rt
from typing import Dict, Generator, List, Union
from urllib.parse import urlparse

from appdirs import user_data_dir
import geopandas as gpd
from pyproj import CRS
import requests
from shapely import ops
from shapely.geometry import MultiPolygon, Polygon, box
import tempfile
import wget
import yaml

from geomesh import db, Raster, Mesh, Geom


logger = logging.getLogger(__name__)


class YamlComponentParser(ABC):

    def __init__(self, config: "YamlParser"):
        self.config = config

    @property
    @abstractmethod
    def key(self):
        raise NotImplementedError("Attribute `key` must be implemented by subclass.")


class FeatureConfig(YamlComponentParser):

    def __init__(self, parser: "YamlParser"):

        if 'features' not in parser.yaml: return
        for feature in parser.yaml["features"].values():
            key = self.validate_feature_request(feature)
            if key == "mesh":
                logger.info(f'Loading mesh from file: {pathlib.Path(feature[key])}.')
                mesh = feature.pop(key)
                feature.update({
                    key: mesh,
                    'object': Mesh.open(mesh, **feature)
                })
        super().__init__(parser)



    # def from_request(self, request: Dict) -> Generator:
    #     rtype = self.get_request_type(request)
    #     if rtype == "mesh":
    #         opts = self.get_opts(request)
    #         mesh = request.get("obj")
    #         if mesh is None:
    #             mesh = Mesh.open(request["mesh"], crs=opts.get("crs", None))
    #             request.setdefault("obj", mesh)
    #         yield mesh
    #     else:
    #         raise NotImplementedError(f"Unhandled request type: {rtype}")
        # elif rtype == 'tile-index':
        #     for path in self.expand_tile_index(request, cache):
        #         yield Raster(path)
        # raise NotImplementedError(request, 123)

    # def get_request_type(self, request) -> str:
    #     self.validate_feature_request(request)
    #     if "mesh" in request:
    #         rtype = "mesh"
    #     elif "shape" in request:
    #         rtype = "shape"
    #     return rtype


    def validate_feature_request(self, request: Dict):
        has_mesh = bool(request.get("mesh", False))
        has_shape = bool(request.get("shape", False))
        if not (has_mesh ^ has_shape):

            class YamlFeaturesInputError(Exception):
                pass

            raise YamlFeaturesInputError(
                f"{self.key} entry in yaml file must contain at least one of `mesh` or `shape`"
            )
        if has_mesh: return "mesh"
        if has_shape: return "shape"

    # def get_opts(self, request: Dict) -> Dict:
    #     print('feature', request)
    #     ftype = self.get_request_type(request)
    #     opts = {}
    #     for feat_request in self.config.features:
    #         if ftype in feat_request:
    #             if request[ftype] == feat_request[ftype]:
    #                 opts = feat_request.copy()
    #                 opts.pop(ftype)
    #     return opts

    # def get_bbox_from_request(
    #     self, request: Dict
    # ) -> Union[Tuple[float, float, float, float], None]:
    #     opts = self.get_opts(request)
    #     logger.info(f"Get bbox from request: {request}, opts {opts}")
    #     raise NotImplementedError("continue")
    #     # print(opts)
    #     feat = opts.get("bbox", {})

    #     # _bbox = None
    #     # if isinstance(feat, dict) and len(feat) > 0:
    #     #     _bbox = (feat["xmin"], feat["ymin"], feat["xmax"], feat["ymax"])
    #     # if isinstance(feat, str):
    #     #     mesh = self.config.features.get('mesh')
    #     #     bbox = Mesh.open(feat, crs=self.get_crs(feat)).bbox
    #     #     _bbox = (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
    #     # return _bbox

    # def get_crs(self, feat: str) -> Union[CRS, None]:
    #     feat_crs = None
    #     for feat_request in self.config.raw.get("features", []):
    #         for val in feat_request.values():
    #             if val == feat:
    #                 feat_crs = feat_request.get("crs")
    #                 if feat_crs is not None:
    #                     feat_crs = CRS.from_user_input(feat_crs)
    #     return feat_crs

    @property
    def key(self):
        return "feature"


class RasterConfig(YamlComponentParser):

    def from_request(self, request: Dict) -> Generator:
        logger.info(f"Get raster from request: {request}")
        rtype = self.get_request_type(request)
        if rtype == "path":
            logger.info(f'Requested raster is a local path: {request["path"]}')
            for raster in self.expand_path(request):
                yield raster
        elif rtype == "tile-index" or rtype == "tile_index":
            logger.info(f'Requested raster is a tile index: {request[rtype]}')
            for raster in self.expand_tile_index(request):
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

    def expand_path(self, request: Dict) -> Generator:
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
                    yield self.apply_opts(Raster(path), request)               
            else:
                yield self.apply_opts(Raster(requested_path), request)


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

    def apply_opts(self, raster: Raster, opts: Dict) -> Union[Raster, None]:
        for key, opt in opts.items():
            if key == "resample":
                raster.resample(opt["scaling_factor"])
            if key == "warp":
                raster.warp(opt)
            if key == "fill_nodata":
                raster.fill_nodata()
            if key == "clip":
                raster.clip(self.get_clip_from_opts(opt))
            if key == "chunk_size":
                raster.chunk_size = opt
            if key == "overlap":
                raster.overlap = opt
            if key == "bbox":
                try:
                    raster.clip(self.get_bbox_clip_from_opts(opt, raster.crs))
                except ValueError:
                    return None
        return raster

    def expand_tile_index(self, request: Dict) -> Generator:
        tile_index_file = request.get("tile_index") if request.get('tile-index') is None else request.get('tile-index')
        bbox = request.get('bbox')
        if bbox is not None:
            gdf_crs = gpd.read_file(tile_index_file).crs
            bbox = box(*bbox['object'].get_bbox(crs=gdf_crs).bounds)
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
            logger.info(f'Yield raster {fname}')
            yield self.apply_opts(Raster(fname), request)


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

    # def get_clip_from_opts(self, opts: Dict) -> MultiPolygon:
    #     for feat_request in self.config.raw.get("features", []):
    #         for key, val in feat_request.items():
    #             if val == opts:
    #                 feat_crs = feat_request.get("crs")
    #                 if feat_crs is not None:
    #                     feat_crs = CRS.from_user_input(feat_crs)
    #     return feat_crs

    def get_bbox_clip_from_opts(self, opts: Dict, raster_crs: Union[CRS, None]) -> Polygon:
        if 'mesh' in opts: return box(*opts['object'].get_bbox(crs=raster_crs).bounds)
        raise NotImplementedError(opts)
        
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


class GeomConfig(YamlComponentParser):

    def __init__(self, parser: "YamlParser"):

        super().__init__(parser)

        if 'geom' not in parser.yaml:
            self.config = None
            return
        
        if 'rasters' not in parser.yaml["geom"]:
            geom_raster_config = None

        geom_raster_config = parser.yaml["geom"].get("raster")
        if not isinstance(geom_raster_config, list):
            raise ValueError(f"geom.raster entry must be of type list, not {type(geom_raster_config)}.")
        
        for item in geom_raster_config:
            if not isinstance(item, dict):
                raise ValueError("geom.raster entries must be a mapping.")
            
            if not (
                bool(item.get("path", False)) ^
                bool(item.get("tile-index", False)) ^
                bool(item.get("tile_index", False))):
                raise ValueError("geom.raster entries must contain only one of 'path' or 'tile_index' keys.")


        self.geom_raster_config = geom_raster_config
        
        if 'features' not in parser.yaml["geom"]:
            geom_feature_config = None
            
        geom_feature_config = parser.yaml["geom"].get("features", [])
        if not isinstance(geom_feature_config, list):
            raise ValueError(f"geom.features entry must be of type list, not {type(geom_feature_config)}.")
        
        for item in geom_feature_config:
            if not isinstance(item, dict):
                raise ValueError("geom.features entries must be a mapping.")
            
            if not (
                bool(item.get("mesh", False)) ^
                bool(item.get("geometry", False))):
                raise ValueError("geom.features entries must contain only one of 'mesh' or 'geometry' keys.")
        
        self.geom_features_config = geom_feature_config

    def __call__(self):

        if self.config is None:
            return
        if self.config.cache is None:
            raise NotImplementedError("Cache is disabled, what to do? -- Easy build an in-memory spatialite db")
        build_id = self.get_build_id()
        # print(build_id)
        if build_id is None:
            return
        res = self.config.cache.geom.get(build_id, db.orm.Geom)
        if res is None:
            res = self.build_geom()
            self.config.cache.geom.add(build_id, db.orm.Geom)
        return res

    #     # geom_build_id = self.get_build_id()
    #     # geom = cache.geom.fetch_by_build_id(geom_build_id)

    @property
    def rasters(self):
        """
        Returns a generator of (Raster, geom_opts) tuple.
        """
        for geom_raster_request in self.geom_raster_config:
            for raster in self.config.rasters.from_request(geom_raster_request):
                if raster is not None:
                    yield raster, {
                        'zmin': geom_raster_request.get('zmin'),
                        'zmax': geom_raster_request.get('zmax'),
                    }

    @property
    def features(self):
        for geom_feature_request in self.geom_features_config:
            for feature in self.config.features.from_request(geom_feature_request):
                if feature is not None:
                    yield feature, {
                        # 'zmin': geom_feature_request.get('zmin'),
                        # 'zmax': geom_feature_request.get('zmax'),
                    }

        # config = self.config.yaml.get("feature", [])
        # if isinstance(config, dict):
        #     config = [config]
        # for request in config:
        #     for feature in self.config.features.from_request(request):
        #         yield feature, request

    def get_build_id(self) -> Union[str, None]:
        """
        Will generate an ID for the unique combination of user requests.
        """
        f: List[str] = []
        # breakpoint()
        for raster, geom_opts in self.rasters:
            # print(raster, geom_opts)
            zmin = geom_opts.get("zmin")
            zmax = geom_opts.get("zmax")
            zmin = "" if zmin is None else f"{zmin:G}"
            zmax = "" if zmax is None else f"{zmax:G}"
            f.append(f"{raster.md5}{zmin}{zmax}")
        for feat, geom_opts in self.features:
            f.append(f"{feat.md5}")
        if len(f) == 0:
            return None
        return hashlib.md5("".join(f).encode("utf-8")).hexdigest()

    def build_geom(self):
        rasters_geoms = self.build_rasters_geoms()
        features_geoms = self.build_features_geoms()
        res = ops.unary_union(
            [
                *[rg.multipolygon for rg in rasters_geoms],
                *[fg.multipolygon for fg in features_geoms],
            ]
        )
        return Geom(res)

    def build_rasters_geoms(self):
        rasters = list(self.rasters)
        geoms = len(rasters) * [None]
        for i, feat_build_id in enumerate(self.iter_raster_build_ids()):
            res = self.config.cache.geom.get(feat_build_id, db.orm.GeomCollection)
            if res is not None:
                geoms[i] = res
        jobs = [rasters[i] for i, val in enumerate(geoms) if val is None]
        if len(jobs) > 0:
            breakpoint()
        return []

    def build_features_geoms(self):
        return [
            Geom(feature.multipolygon, crs=feature.crs) for feature, _ in self.features
        ]

    # def iter_raster_build_ids(self) -> Generator:
    #     for raster, request in self.rasters:
    #         zmin = request.get("zmin")
    #         zmax = request.get("zmax")
    #         zmin = "" if zmin is None else f"{zmin:G}"
    #         zmax = "" if zmax is None else f"{zmax:G}"
    #         yield hashlib.md5(f"{raster.md5}{zmin}{zmax}".encode("utf-8")).hexdigest()

    # def iter_features_build_ids(self):
    #     for feat, request in self.features:
    #         yield feat.md5

    @property
    def key(self):
        return "geom"


class YamlParser:

    VERSION = 0

    def __init__(self, path, cache: db.Cache = None):
        self._path = pathlib.Path(path)
        self._cache = cache
        self._features = FeatureConfig(self)
        self._rasters = RasterConfig(self)
        self._geom = GeomConfig(self)
        # self._hfun = HfunConfig(self)


    @property
    def geom(self) -> GeomConfig:
        return self._geom

    # @property
    # def hfun(self) -> HfunConfig:
    #     return self._hfun

    @property
    def rasters(self) -> RasterConfig:
        return self._rasters

    @property
    def features(self) -> FeatureConfig:
        return self._features

    @property
    def cache(self) -> db.Cache:
        return self._cache

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @cached_property
    def yaml(self):
        with open(self.path) as fh:
            logger.info(f"Loading configuration file: {self.path}")
            return yaml.load(fh, Loader=yaml.SafeLoader)

    @property
    def yml(self):
        return self.yaml