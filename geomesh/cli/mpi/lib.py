from datetime import datetime
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import hashlib
import inspect
import json
import logging
import tempfile
import warnings

from colored_traceback.colored_traceback import Colorizer
from dask_geopandas.hilbert_distance import _hilbert_distance
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from psutil import cpu_count
from pydantic import BaseModel
from pydantic import model_validator
from pyproj import CRS
from scipy.optimize import curve_fit
from shapely import LineString
from shapely import MultiPolygon
from shapely import Polygon
from shapely import ops
import dask_geopandas as dgpd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from geomesh.cli.raster_opts import expand_raster_request_path
from geomesh.cli.raster_opts import expand_tile_index
from geomesh.cli.raster_opts import get_raster_from_opts
from geomesh.cli.raster_opts import iter_raster_windows
from geomesh.geom.raster import RasterGeom

logger = logging.getLogger(__name__)


def hardware_info(comm=None):
    comm = MPI.COMM_WORLD if comm is None else comm
    logical_proc_counts = comm.gather(cpu_count(logical=True), root=0)
    hw_proc_counts = comm.gather(cpu_count(logical=False), root=0)
    hwinfo = None
    processor_names = comm.gather(MPI.Get_processor_name(), root=0)
    if comm.Get_rank() == 0:
        hwinfo = pd.DataFrame([{
                'node_name': _name,
                'hw_proc_count': _hpc,
                'thread_count': _lpc,
                'color': None,
                # 'executor': None,
            } for _name, _hpc, _lpc in zip(processor_names, hw_proc_counts, logical_proc_counts)]
            )
        hwinfo.index.name = 'rank'
        nodes = hwinfo.drop_duplicates(ignore_index=True)
        # assign colors
        for row in nodes.itertuples():
            hwinfo.loc[hwinfo['node_name'] == row.node_name, 'color'] = row.Index
    return comm.bcast(hwinfo, root=0)


def get_split_array_weighted(comm, array, weights, size=None):
    rank = comm.Get_rank()
    size = size or comm.Get_size()
    # items per rank
    if rank == 0:
        array = [(global_k, row) for global_k, row in enumerate(array)]
        lst = list(zip(array, weights))
        buckets = evenly_distribute(lst, size)
        array = [[val for val, weights in buckets[_rank]] for _rank in range(size)]
    return comm.bcast(array, root=0)


def get_split_array(comm, array, weights=None):

    if weights is not None:
        return get_split_array_weighted(comm, array, weights)

    if comm.Get_rank() == 0:
        array = [(global_k, row) for global_k, row in enumerate(array)]
        array = np.array(array, dtype=object)
        array = np.array_split(array, comm.Get_size())
    else:
        array = None
    return comm.bcast(array, root=0)


def evenly_distribute(lst, n):
    """Distribute items in lst (tuples of (val, weight)) into n buckets"""
    buckets = [[] for i in range(n)]
    weights = [[0, i] for i in range(n)]
    for item in sorted(lst, key=lambda x: x[1], reverse=True):
        idx = weights[0][1]
        buckets[idx].append(item)
        weights[0][0] += item[1]
        weights = sorted(weights)
    return buckets


def get_normalized_raster_request_opts(raster_hash, request_opts, window):
    clip = request_opts.get('clip')
    if clip is not None:
        clip = str(clip)
    mask = request_opts.get('mask')
    if mask is not None:
        mask = str(mask)
    return {
        'raster_hash': raster_hash,
        'window': str(window),
        'resampling_factor': request_opts.get('resampling_factor'),
        'resampling_method': request_opts.get('resampling_method'),
        'clip': clip,
        'mask': mask,
        }


def split_into_batches(comm, local_jobs):
    # not currently used but kept for reference
    max_size = comm.Get_size()
    local_jobs = sorted([(row, np.min([max_size, nprocs])) for row, nprocs in local_jobs], key=lambda x: x[1])
    sublists = []
    while local_jobs:
        current_sum = 0
        current_sublist = []
        for i, (row, nprocs) in reversed(list(enumerate(local_jobs))):
            if len(current_sublist) == 0:
                current_sublist.append((row, nprocs))
                local_jobs.pop(i)
                current_sum += nprocs
                if current_sum == max_size:
                    break
                else:
                    continue
            if current_sum + nprocs > max_size:
                continue
            if current_sum + nprocs < max_size:
                current_sublist.append((row, nprocs))
                local_jobs.pop(i)
                current_sum += nprocs
                continue
            if current_sum + nprocs == max_size:
                current_sublist.append((row, nprocs))
                local_jobs.pop(i)
                current_sum += nprocs
                continue
            if current_sum == max_size:
                break
        sublists.append(current_sublist)
    if len(current_sublist) > 0:
        sublists.append(current_sublist)
    return sublists


def _interpolate_list(lst):
    prev_val = None
    interpolated_lst = []
    for val in lst:
        if val is not None:
            prev_val = val
            interpolated_lst.append(val)
        else:
            interpolated_lst.append(prev_val)
    return interpolated_lst


def get_raster_md5(comm, raster_window_requests):
    raster_window_requests = get_split_array(comm, raster_window_requests)[comm.Get_rank()]
    raster_hashes = []
    for global_k, ((i, raster_path, request_opts), (j, window)) in raster_window_requests:
        if j == 0:
            hash_md5 = hashlib.md5()
            with open(raster_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            raster_hashes.append(hash_md5.hexdigest())
        else:
            raster_hashes.append(None)
    return _interpolate_list([item for sublist in comm.allgather(raster_hashes) for item in sublist])


def get_bbox_gdf(comm, raster_window_request, cache_directory, raster_hashes):
    cache_directory /= 'bbox_gdf'
    cache_directory.mkdir(exist_ok=True)
    epsg4326 = CRS.from_epsg(4326)
    raster_window_request = get_split_array(comm, raster_window_request)[comm.Get_rank()]
    bbox_gdf = []
    for global_k, ((i, raster_path, request_opts), (j, window)) in raster_window_request:
        normalized_raster_request = get_normalized_raster_request_opts(raster_hashes[global_k], request_opts, window)
        normalized_raster_request.pop('resampling_factor')
        normalized_raster_request.pop('resampling_method')
        tmpfile = cache_directory / (hashlib.sha256(json.dumps(normalized_raster_request).encode()).hexdigest() + '.feather')
        if tmpfile.exists():
            logger.debug(f"Loading bbox for {Path(raster_path).name} {window=} from {tmpfile}")
            bbox_gdf.append(gpd.read_feather(tmpfile))
            continue
        logger.debug(f"Building bbox for {Path(raster_path).name} {window=} to {tmpfile}")
        raster = get_raster_from_opts(raster_path, request_opts, window)
        bbox_mp = Geom(raster).get_multipolygon(dst_crs=epsg4326)
        # if raster.clip is not None:
        #     geom_bbox = geom_mp
        # if geom_mp is None:
        #     geom_mp = MultiPolygon([])
        bbox_gdf.append(gpd.GeoDataFrame([{
            'index': global_k,
            'geometry': bbox_mp,
            'raster_hash': raster_hashes[global_k],
            }], crs=epsg4326).set_index('index'))
        # bbox_gdf[-1].to_feather(tmpfile)
    bbox_gdf = pd.concat([item for sublist in comm.allgather(bbox_gdf) for item in sublist]).sort_index()
    # verify
    # if comm.Get_rank() == 0:
    #     logger.info('plotting')
    #     import matplotlib.pyplot as plt
    #     bbox_gdf.plot(facecolor='none', cmap='jet')
    #     plt.show(block=True)
    return bbox_gdf





def mpiabort_excepthook(type, value, traceback):
    Colorizer('default').colorize_traceback(type, value, traceback)
    MPI.COMM_WORLD.Abort(-1)


class BboxConfig(BaseModel):
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    crs: Optional[str] = None



# from enum import Enum

# class ClipType(Enum):
#     PATH = 'path'
#     DICT = 'dict'


# class PathClipConfig(BaseModel):
#     type: ClipType = ClipType.PATH
#     path: str

# class DictClipConfig(BaseModel):
#     type: ClipType = ClipType.DICT
#     config: Dict[str, float]

from pydantic.class_validators import validator


# RasterClipConfigTypes = Union[dict, str]

class RasterClipStrConfig(BaseModel):
    path: Path

RasterClipConfigTypes = Union[RasterClipStrConfig]

class RasterConfig(BaseModel):
    path: Optional[str] = None
    tile_index: Optional[Path] = None
    resampling_factor: Optional[float] = None
    clip: Optional[RasterClipConfigTypes] = None
    overlap: Optional[int] = 0
    chunk_size: Optional[int] = None
    bbox: Optional[BboxConfig] = None
    crs: Optional[str] = None
    _normalization_keys = [
            "resampling_factor",
            "clip",
            "bbox",
            ]


    @model_validator(mode='before')
    @classmethod
    def precheck(cls, data: Any) -> Any:
        path = data.get('path')
        tile_index = data.get('tile_index')
        if path and tile_index:
            raise ValueError("path and tile_index are mutually exclusive.")
        if not path and not tile_index:
            raise ValueError("One of path or tile_index is required.")
        clip = data.get('clip')
        if clip is not None:
            if isinstance(clip, str):
                data['clip'] = RasterClipStrConfig(path=Path(clip).resolve())
            elif isinstance(clip, dict):
                pass
            else:
                raise ValueError("`clip` key must be a string")
        crs = data.get("crs")
        if crs is not None:
            data['crs'] = CRS.from_user_input(data["crs"])
        # print(data.get('bbox'))
        return data


    # @model_validator(mode='after')
    # def postcheck(self) -> "Self":
    #     if self.bbox is not None:

    #     pass


    def iter_raster_requests(self):
        global_kwargs = self.model_dump()
        global_kwargs.pop("path")
        global_kwargs.pop("tile_index")
        if self.path:
            for fname, raster_request in expand_raster_request_path({"path": self.path, **global_kwargs}):
                yield fname, global_kwargs
        elif self.tile_index:
            for fname, raster_request in expand_tile_index(self, {"tile_index": self.tile_index, **global_kwargs}):
                yield fname, global_kwargs
        else:
            raise NotImplementedError("Unreachable: neither self.path nor self.tile_index")

    def iter_normalized_raster_requests(self):
        for raster_path, raster_request in self.iter_raster_requests():
            yield raster_path, {key: raster_request[key] for key in self._normalization_keys}

    def iter_raster_window_requests(self):
        for i, (raster_path, request_opts) in enumerate(list(self.iter_raster_requests())):
            for j, window in iter_raster_windows(raster_path, request_opts):
                yield (i, raster_path, request_opts), (j, window)

    def iter_normalized_raster_window_requests(self):
        for i, (raster_path, request_opts) in enumerate(list(self.iter_normalized_raster_requests())):
            for j, window in iter_raster_windows(raster_path, {**request_opts, 'chunk_size': self.chunk_size, 'overlap': self.overlap}):
                yield (i, raster_path, request_opts), (j, window)





class IterableRasterWindowTrait:

    def iter_raster_window_requests(self):
        for raster_config in self.rasters:
            for this_raster_config in raster_config.iter_raster_window_requests():
                yield this_raster_config

    def iter_raster_requests(self):
        for raster_config in self.rasters:
            for this_raster_config in raster_config.iter_raster_requests():
                yield this_raster_config

    def iter_normalized_raster_requests(self):
        for raster_config in self.rasters:
            for this_raster_config in raster_config.iter_normalized_raster_requests():
                yield this_raster_config

    def iter_normalized_raster_window_requests(self):
        for raster_config in self.rasters:
            for this_raster_config in raster_config.iter_normalized_raster_window_requests():
                yield this_raster_config

    def df(self):
        df = pd.DataFrame([{'path': raster_path, 'window': window, **request_opts} for (_, raster_path, request_opts), (_, window) in self.iter_raster_window_requests()])
        if df.empty:
            global_kwargs = self.model_dump()
            global_kwargs.pop("path")
            global_kwargs.pop("tile_index")
            return pd.DataFrame(columns=['path', 'window', *list(global_kwargs)])
        return df






# class MeshgenConfig(BaseModel):

#     geom: Optional[GeomConfig]
#     hfun: Optional[HfunConfig]

#     @classmethod
#     def try_from_yaml_path(cls, path: Path) -> "MeshgenConfig":
#         import yaml
#         with open(path, 'r') as file:
#             data = yaml.safe_load(file)
#         return cls.try_from_dict(data)

#     @classmethod
#     def try_from_dict(cls, data: dict) -> "MeshgenConfig":
#         if 'hgrid' in data:
#             data = data['hgrid']
#         geom_config = GeomConfig(**data['geom']) if 'geom' in data else GeomConfig(rasters=[])
#         hfun_config = HfunConfig(**data['hfun']) if 'hfun' in data else HfunConfig(rasters=[])
#         return cls(
#                 geom=geom_config,
#                 hfun=hfun_config,
#             )







