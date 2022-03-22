from pathlib import Path
import tempfile
from typing import Dict, Generator
from urllib.parse import urlparse

from appdirs import user_data_dir
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import wget

from geomesh.mesh import Mesh

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