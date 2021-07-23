from abc import ABC, abstractmethod
from collections import UserDict
from glob import glob
import hashlib
import logging
import os
import pathlib
from pprint import pformat
from typing import Dict, Generator, List, Tuple, Union

from appdirs import user_data_dir
import geopandas as gpd
from pyproj import CRS
import requests
from shapely import ops
from shapely.geometry import MultiPolygon, Polygon
import tempfile
import yaml

from geomesh import db, Raster, Mesh, Geom


logger = logging.getLogger(__name__)


class YamlComponentParser(ABC):
    def __init__(self, parser: "YamlParser"):
        self.yaml = parser

    @property
    def config(self):
        return self.yaml.raw.get(self.key, {})

    def validate_raster_request(self, request: Dict):
        has_url = bool(request.get("url", False))
        has_tile_index = bool(request.get("tile-index", False))
        has_path = bool(request.get("path", False))
        if not (has_url ^ has_tile_index ^ has_path):

            class YamlRastersInputError(Exception):
                pass

            raise YamlRastersInputError(
                f"{self.key} entry in yaml file must contain at least one of `url`, `tile-index` of `path`"
            )

    def validate_feature_request(self, request: Dict):
        has_mesh = bool(request.get("mesh", False))
        has_shape = bool(request.get("shape", False))
        if not (has_mesh ^ has_shape):

            class YamlFeaturesInputError(Exception):
                pass

            raise YamlFeaturesInputError(
                f"{self.key} entry in yaml file must contain at least one of `mesh` or `shape`"
            )

    @property
    def path(self):
        return self._path

    @property
    @abstractmethod
    def key(self):
        raise NotImplementedError("Attribute `key` must be implemented by subclass.")


class FeaturesYamlParser(YamlComponentParser, UserDict):
    def from_request(self, request: Dict) -> Generator:
        rtype = self.get_request_type(request)
        if rtype == "mesh":
            opts = self.get_opts(request)
            mesh = request.get("obj")
            if mesh is None:
                mesh = Mesh.open(request["mesh"], crs=opts.get("crs", None))
                request.setdefault("obj", mesh)
            yield mesh
        else:
            raise NotImplementedError(f"Unhandled request type: {rtype}")
        # elif rtype == 'tile-index':
        #     for path in self.expand_tile_index(request, cache):
        #         yield Raster(path)
        # raise NotImplementedError(request, 123)

    def get_request_type(self, request) -> str:
        self.validate_feature_request(request)
        if "mesh" in request:
            rtype = "mesh"
        elif "shape" in request:
            rtype = "shape"
        return rtype

    def get_opts(self, request: Dict) -> Dict:
        ftype = self.get_request_type(request)
        opts = {}
        for feat_request in self.config:
            if ftype in feat_request:
                if request[ftype] == feat_request[ftype]:
                    opts = feat_request.copy()
                    opts.pop(ftype)
        return opts

    def get_bbox_from_request(
        self, request: Dict
    ) -> Union[Tuple[float, float, float, float], None]:
        opts = self.get_opts(request)
        logger.info(f"Get bbox from request: {request}, opts {opts}")
        raise NotImplementedError("continue")
        # print(opts)
        feat = opts.get("bbox", {})

        # _bbox = None
        # if isinstance(feat, dict) and len(feat) > 0:
        #     _bbox = (feat["xmin"], feat["ymin"], feat["xmax"], feat["ymax"])
        # if isinstance(feat, str):
        #     mesh = self.yaml.features.get('mesh')
        #     bbox = Mesh.open(feat, crs=self.get_crs(feat)).bbox
        #     _bbox = (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        # return _bbox

    def get_crs(self, feat: str) -> Union[CRS, None]:
        feat_crs = None
        for feat_request in self.yaml.raw.get("features", []):
            for val in feat_request.values():
                if val == feat:
                    feat_crs = feat_request.get("crs")
                    if feat_crs is not None:
                        feat_crs = CRS.from_user_input(feat_crs)
        return feat_crs

    @property
    def key(self):
        return "features"


class RastersYamlParser(YamlComponentParser, UserDict):
    def from_request(self, request: Dict) -> Generator:
        logger.info(f"Get raster from request: {request}")
        rtype = self.get_request_type(request)
        if rtype == "path":
            logger.info(f'Requested raster is a local path: {request["path"]}')
            for raster in self.expand_path(request):
                yield raster
        elif rtype == "tile-index":
            logger.info(f'Requested raster is a tile index: {request["tile-index"]}')
            for raster in self.expand_tile_index(request):
                yield raster

    def get_request_type(self, request):
        self.validate_raster_request(request)
        if "path" in request:
            return "path"
        elif "tile-index" in request:
            return "tile-index"
        elif "url" in request:
            return "url"

    def expand_path(self, request: Dict) -> Generator:
        for path in glob(os.path.expandvars(request["path"])):
            opts = self.get_opts(request)
            raster = Raster(path)
            self.apply_opts(raster, opts)
            yield raster

    def get_opts(self, request) -> Dict:
        rtype = self.get_request_type(request)
        opts = {}
        for raster_request in self.config:
            if rtype in raster_request:
                if request[rtype] == raster_request[rtype]:
                    opts = raster_request.copy()
                    opts.pop(rtype)
        return opts

    def apply_opts(self, raster: Raster, opts: Dict) -> None:
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

    def expand_tile_index(self, request: Dict) -> Generator:
        opts = self.get_opts(request)
        if len(opts) > 0:
            logger.info(f"Options for requested tile index are: {opts}")
        gdf = gpd.read_file(
            request["tile-index"],
            bbox=self.yaml.features.get_bbox_from_request(request),
        )
        cache = self.get_cache_from_opts(opts)
        for row in gdf.itertuples():
            fname = cache / row.URL.split("/")[-1]
            if not fname.is_file():
                logger.info(f"Downloading file {fname} to {cache}")
                with open(cache / fname, "wb") as fh:
                    fh.write(requests.get(row.URL, allow_redirects=True).content)
            raster = Raster(cache / fname)
            self.apply_opts(raster, opts)
            yield raster

    def get_cache_from_opts(self, opts: Dict) -> pathlib.Path:
        cache_opt = opts.get("cache", True)
        if cache_opt is True:
            default_cache_dir = (
                pathlib.Path(user_data_dir()) / "geomesh" / "raster_cache"
            )
            default_cache_dir.mkdir(exist_ok=True)
            return default_cache_dir

        elif cache_opt is None or cache_opt is False:
            self.tmpdir = tempfile.TemporaryDirectory()
            return pathlib.Path(self.tmpdir.name)

        else:
            return pathlib.Path(cache_opt)

    def get_clip_from_opts(self, opts: Dict) -> MultiPolygon:
        for feat_request in self.yaml.raw.get("features", []):
            for key, val in feat_request.items():
                if val == opts:
                    feat_crs = feat_request.get("crs")
                    if feat_crs is not None:
                        feat_crs = CRS.from_user_input(feat_crs)
        return feat_crs

    @property
    def key(self):
        return "rasters"


class GeomYamlParser(YamlComponentParser, UserDict):
    def __call__(self):
        if self.yaml.cache is None:
            raise NotImplementedError("Cache is disabled, what to do?")
        build_id = self.get_build_id()
        res = self.yaml.cache.geom.get(build_id, db.orm.Geom)
        if res is None:
            res = self.build_geom()
            print(res)
            self.yaml.cache.geom.add(build_id, db.orm.Geom)
        return res

        # geom_build_id = self.get_build_id()
        # geom = cache.geom.fetch_by_build_id(geom_build_id)

    @property
    def rasters(self):
        config = self.config.get("rasters", [])
        if isinstance(config, dict):
            config = [config]
        for request in config:
            for raster in self.yaml.rasters.from_request(request):
                yield raster, request

    @property
    def features(self):
        config = self.config.get("features", [])
        if isinstance(config, dict):
            config = [config]
        for request in config:
            for feature in self.yaml.features.from_request(request):
                yield feature, request

    @property
    def key(self):
        return "geom"

    def get_build_id(self) -> str:
        """
        Will generate an ID for the unique combination of user requests.
        """
        f: List[str] = []
        for raster, geom_opts in self.rasters:
            zmin = geom_opts.get("zmin")
            zmax = geom_opts.get("zmax")
            zmin = "" if zmin is None else f"{zmin:G}"
            zmax = "" if zmax is None else f"{zmax:G}"
            f.append(f"{raster.md5}{zmin}{zmax}")
        for feat, geom_opts in self.features:
            f.append(f"{feat.md5}")
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
            res = self.yaml.cache.geom.get(feat_build_id, db.orm.GeomCollection)
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

    def iter_raster_build_ids(self) -> Generator:
        for raster, request in self.rasters:
            zmin = request.get("zmin")
            zmax = request.get("zmax")
            zmin = "" if zmin is None else f"{zmin:G}"
            zmax = "" if zmax is None else f"{zmax:G}"
            yield hashlib.md5(f"{raster.md5}{zmin}{zmax}".encode("utf-8")).hexdigest()

    # def iter_features_build_ids(self):
    #     for feat, request in self.features:
    #         yield feat.md5


class YamlParser:

    VERSION = 0

    def __init__(self, path, cache: db.Cache = None):
        self._path = pathlib.Path(path)
        self.cache = cache

    @property
    def geom(self) -> GeomYamlParser:
        return GeomYamlParser(self)

    @property
    def rasters(self) -> RastersYamlParser:
        return RastersYamlParser(self)

    @property
    def features(self) -> FeaturesYamlParser:
        return FeaturesYamlParser(self)

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def raw(self):
        if not hasattr(self, "_raw"):
            with open(self.path) as fh:
                logger.info(f"Loading configuration file: {self.path}")
                self._raw = yaml.load(fh, Loader=yaml.SafeLoader)
        return self._raw
