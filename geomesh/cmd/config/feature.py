import logging
import pathlib
from typing import Dict, Generator

from .yamlparser import YamlParser, YamlComponentParser
from ...mesh import Mesh

logger = logging.getLogger(__name__)

class FeatureConfig(YamlComponentParser):

    def __init__(self, parser: YamlParser):

        if 'features' not in parser.yaml: return
        for feature in parser.yaml["features"]:
            key = self.validate_feature_request(feature)
            if key == "mesh":
                logger.info(f'Loading mesh from file: {pathlib.Path(feature[key])}.')
                mesh = feature.pop(key)
                feature.update({
                    key: mesh,
                    'object': Mesh.open(mesh, **feature)
                })
        super().__init__(parser)



    def from_request(self, request: Dict) -> Generator:
        rtype = self.get_request_type(request)
        if rtype == "mesh":
            opts = self.get_opts(request)
            mesh = request.get("obj")
            if mesh is None:
                mesh = Mesh.open(request["mesh"], crs=opts.get("crs", None))
                request.setdefault("obj", mesh)
            yield mesh
        # elif rtype == 'geometry':
            
        else:
            raise NotImplementedError(f"Unhandled request type: {rtype}")

    def get_request_type(self, request) -> str:
        return self.validate_feature_request(request)

    def validate_feature_request(self, request: Dict):
        has_mesh = bool(request.get("mesh", False))
        has_shape = bool(request.get("geometry", False))
        if not (has_mesh ^ has_shape):

            class YamlFeaturesInputError(Exception):
                pass

            raise YamlFeaturesInputError(
                f"{self.key} entry in yaml file must contain at least one of `mesh` or `geometry`"
            )
        if has_mesh: return "mesh"
        if has_shape: return "geometry"

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

