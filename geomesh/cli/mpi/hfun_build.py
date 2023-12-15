#!/usr/bin/env python
from functools import lru_cache
from functools import partial
from pathlib import Path
from time import time
from typing import Any
from typing import List
from typing import Optional
import argparse
import hashlib
import json
import logging
import os
import pickle
import queue
import sys
import tempfile
import warnings

from colored_traceback.colored_traceback import Colorizer
from jigsawpy import jigsaw_msh_t, savemsh, loadmsh
from matplotlib.transforms import Bbox
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
# from mpi4py.futures import MPIPoolExecutor
from pydantic import BaseModel
from pydantic import model_validator
from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
import geopandas as gpd
import fasteners
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from geomesh import Geom, Hfun
from geomesh import utils
from geomesh.cli.build import BuildCli
from geomesh.cli.mpi import lib
from geomesh.cli.mpi.geom_build import GeomConfig
from geomesh.geom.shapely_geom import ShapelyGeom
from geomesh.cli.mpi.geom_build import GeomRasterConfig
from geomesh.cli.raster_opts import iter_raster_window_requests, get_raster_from_opts
from geomesh.cli.schedulers.local import LocalCluster
from geomesh.geom.shapely_geom import MultiPolygonGeom
from geomesh.hfun.raster import RasterHfun

warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
# warnings.filterwarnings("error")

logger = logging.getLogger(f'[rank: {MPI.COMM_WORLD.Get_rank()}]: {__name__}')


class HfunRasterContourRequest(BaseModel):
    level: float | List[float]
    expansion_rate: float
    target_size: Optional[float] = None
    # nprocs: Optional[int] = None

class HfunRasterConstantValueRequest(BaseModel):
    value: float
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None


class HfunRasterGradientDelimiterRequest(BaseModel):
    multiplier: float = 1./3.
    hmin: Optional[float] = None
    hmax: Optional[float] = None
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None


class HfunGeomConfig(BaseModel):
    zmin: Optional[float] = None
    zmax: Optional[float] = None


class HfunRequestConfig(BaseModel):
    contour: Optional[HfunRasterContourRequest | List[HfunRasterContourRequest]] = None
    gradient_delimiter: Optional[HfunRasterGradientDelimiterRequest | List[HfunRasterGradientDelimiterRequest]] = None
    constant_value: Optional[HfunRasterConstantValueRequest | List[HfunRasterConstantValueRequest]] = None
    geom: Optional[HfunGeomConfig] = None


class HfunRasterConfig(lib.RasterConfig, lib.IterableRasterWindowTrait, HfunRequestConfig):
    hmax: Optional[float] = None
    hmin: Optional[float] = None
    _normalization_keys = [
            "resampling_factor",
            "clip",
            "bbox",
            "hmin",
            "hmax",
            ]

    def iter_normalized_raster_requests(self):
        for normalized_path, normalized_raster_hfun_request in super().iter_normalized_raster_requests():
            for normalization_key in self._normalization_keys:
                normalized_raster_hfun_request[normalization_key] = getattr(self, normalization_key)
            yield normalized_path, normalized_raster_hfun_request

    def iter_normalized_raster_window_requests(self):
        for (i, normalized_path, normalized_raster_hfun_request), (j, window) in super().iter_normalized_raster_window_requests():
            for normalization_key in self._normalization_keys:
                normalized_raster_hfun_request[normalization_key] = getattr(self, normalization_key)
            yield (i, normalized_path, normalized_raster_hfun_request), (j, window)


class HfunConfig(BaseModel):
    geom: Optional[GeomConfig] = None
    hmax: Optional[float] = None
    hmin: Optional[float] = None
    marche: Optional[bool] = False
    rasters: List[HfunRasterConfig]
    verbosity: Optional[int] = 0
    cpus_per_task: Optional[int] = None
    max_cpus_per_task: Optional[int] = None
    min_cpus_per_task: Optional[int] = None
    _constraints_keys = [
            'contour',
            'gradient_delimiter',
            'constant_value',
            ]

    _normalization_keys = [ "geom", "hmax", "hmin", "marche", ]
    # min_cpus_per_task: 8

    @model_validator(mode='before')
    @classmethod
    def precheck(cls, data: Any) -> Any:
        cpus_per_task = data.get('cpus_per_task')
        max_cpus_per_task = data.get('max_cpus_per_task')
        min_cpus_per_task = data.get('min_cpus_per_task')
        if np.all([bool(cpus_per_task), np.any([bool(max_cpus_per_task), bool(min_cpus_per_task)])]):
            raise ValueError('Arguments cpus_per_task and max_cpus_per_task are mutually exclusive.')
        if cpus_per_task is not None and (not isinstance(cpus_per_task, int) or cpus_per_task < 1):
            raise ValueError(f'Argument cpus_per_task must be an int > 0 or None (for auto dist) but got {cpus_per_task=}.')
        for i, raster_config in enumerate(data.get("rasters", [])):
            constant_value_request = raster_config.get("constant_value")
            if isinstance(constant_value_request, float):
                data["rasters"][i]["constant_value"] = HfunRasterConstantValueRequest(value=constant_value_request)
        return data

    @classmethod
    def try_from_yaml_path(cls, path: Path) -> "HfunConfig":
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return cls.try_from_dict(data)

    @classmethod
    def try_from_dict(cls, data: dict) -> "HfunConfig":
        return cls(**data['hfun'])

    def iter_raster_window_requests(self):
        for raster_config in self.rasters:
            for (i, normalized_path, raster_hfun_request), (j, window) in raster_config.iter_raster_window_requests():
                for key in self._normalization_keys:
                    raster_hfun_request[key] = getattr(self, key)
                yield (i, normalized_path, raster_hfun_request), (j, window)

    def iter_normalized_raster_requests(self):
        for raster_config in self.rasters:
            for normalized_path, normalized_raster_hfun_request in raster_config.iter_normalized_raster_requests():
                for normalization_key in self._normalization_keys:
                    normalized_raster_hfun_request[normalization_key] = getattr(self, normalization_key)
                yield normalized_path, normalized_raster_hfun_request

    def iter_normalized_raster_window_requests(self):
        k = 0
        for g, raster_config in enumerate(self.rasters):
            for (i, raster_path, normalized_raster_hfun_request), (j, window) in raster_config.iter_normalized_raster_window_requests():
                for normalization_key in self._normalization_keys:
                    normalized_raster_hfun_request[normalization_key] = getattr(self, normalization_key)
                raster_request_opts = dict(
                        # raster_path=normalized_path,
                        crs=self.rasters[g].crs,
                        clip=normalized_raster_hfun_request.pop("clip"),
                        # bbox=self.rasters[k].bbox,
                        window=window,
                        chunk_size=self.rasters[g].chunk_size,
                        overlap=self.rasters[g].overlap,
                        resampling_factor=normalized_raster_hfun_request.pop("resampling_factor"),
                        resampling_method=normalized_raster_hfun_request.pop("resampling_method", None),  # Not implemented in config file.
                        )
                hfun_request_opts = dict(
                        hmin=normalized_raster_hfun_request.pop("hmin"),
                        hmax=normalized_raster_hfun_request.pop("hmax"),
                        verbosity=self.verbosity,
                        marche=normalized_raster_hfun_request.pop("marche"),
                        nprocs=1,
                        geom=normalized_raster_hfun_request.pop("geom"),
                        constraints={key: getattr(self.rasters[g], key) for key in self._constraints_keys},
                        )

                # hfun_raster_constraints = self._get_raster_window_constraints(k, j)
                yield k, ((i, raster_path, raster_request_opts, hfun_request_opts), (j, window))
                k += 1


    def _build_base_geoms_gdf(self, comm, output_rank=None, cache_directory=None):
        output_rank = 0 if output_rank is None else output_rank
        data = {
                "rasters": [],
                # "sieve": False,
                # "partition_size": 2,
                # "grid_size": None,
                }
        for raster_config in self.rasters:
            raster_geom_data = dict(
                    path=raster_config.path,
                    tile_index=raster_config.tile_index,
                    resampling_factor=raster_config.resampling_factor,
                    clip=raster_config.clip,
                    overlap=raster_config.overlap,
                    chunk_size=raster_config.chunk_size,
                    bbox=raster_config.bbox,
                    crs=raster_config.crs,
                    )
            if raster_config.geom is not None:
                raster_geom_data['zmin'] = raster_config.geom.zmin
                raster_geom_data['zmax'] = raster_config.geom.zmax
            data["rasters"].append(GeomRasterConfig(**raster_geom_data))
        geom_config = GeomConfig(**data)
        geom_cache_directory = cache_directory.parent / "geom_build" if cache_directory is not None else None
        base_geoms_gdf = geom_config.build_clipped_raster_geom_gdf_mpi(comm, output_rank=output_rank, cache_directory=geom_cache_directory)
        # if comm.Get_rank() == output_rank:
        #     base_geoms_gdf.plot(ax=plt.gca(), facecolor='none', color='b')
        #     plt.show(block=True)
        # comm.barrier()
        return base_geoms_gdf

    def _get_hfun_raster_window_geoms_gdf(self, comm, output_rank=None, cache_directory=None):
        base_hfun_window_geoms_gdf = self._build_base_geoms_gdf(comm, output_rank=output_rank, cache_directory=cache_directory)
        if self.geom is not None:
            warnings.warn("hfun.geom is experimental")
            base_geom_gdf = self.geom.build_combined_geoms_gdf_mpi(
                        comm,
                        output_rank=output_rank,
                        cache_directory=cache_directory.parent / "geom_build"
                        )
            if base_geom_gdf is not None:
                base_geom_mp = MultiPolygon(list(base_geom_gdf.geometry))
                del base_geom_gdf
            else:
                base_geom_mp = None

        # if self.geom is not None and base_geom_mp is not None:
        #     base_hfun_window_geoms_gdf.geometry = base_hfun_window_geoms_gdf.geometry.difference(base_geom_mp)

        return base_hfun_window_geoms_gdf

    @staticmethod
    def _build_contours_gdf_from_raster(raster_path, raster_request_opts, contour_request_config):
        # g, (k, ((i, raster_path, raster_request_opts, hfun_request_opts), (j, window))) = indices
        def extract_contours_from_arrays(xvals, yvals, zvals, level):
            # plt.ioff()
            logger.debug(f"Processing contour {level=} for {raster_path=}")
            import matplotlib as mpl
            _old_backend = mpl.get_backend()
            plt.switch_backend('agg')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                ax = plt.contour(xvals, yvals, zvals, levels=[level])
            plt.close(plt.gcf())
            plt.switch_backend(_old_backend)
            for i, (_level, path_collection) in enumerate(zip(ax.levels, ax.collections)):
                if _level != level:
                    continue
                linestrings = []
                for path in path_collection.get_paths():
                    try:
                        linestrings.append(LineString(path.vertices))
                    except ValueError as e:
                        if "LineStrings must have at least 2 coordinate tuples" in str(e):
                            continue
                        raise e
            return MultiLineString(linestrings)
        # lock = fasteners.InterProcessLock(cache_directory / f"{raster_path.name}.lock")  # for processes
        data = []
        this_raster_window_data = []
        contour_request = contour_request_config.model_dump()
        request_levels = [contour_request.pop('level')] if not isinstance(contour_request['level'], list) else contour_request.pop('level')

        def get_window_data_from_raster():
            raster = get_raster_from_opts(raster_path=raster_path, **raster_request_opts)
            return get_window_data(
                        raster.src,
                        window=raster.window,
                        # band=band,
                        masked=True,
                        resampling_method=raster.resampling_method,
                        resampling_factor=raster.resampling_factor,
                        clip=raster.clip,
                        mask=raster.mask,
                        )

        raster = get_raster_from_opts(raster_path=raster_path, **raster_request_opts)
        crs = raster.crs
        from geomesh.raster.raster import get_window_data
        x, y, values = get_window_data_from_raster()
        for requested_level in request_levels:
            geometry = extract_contours_from_arrays(x, y, values, requested_level)
            this_raster_window_data.append({
                'geometry': geometry,
                'level': requested_level,
                'raster_path': raster_path,
                # **contour_request
                })
        return gpd.GeoDataFrame(this_raster_window_data, crs=crs)

    @staticmethod
    def _get_contours_gdf_from_raster_feather_path(raster_path, raster_request_opts, contours_request_config, cache_directory):
        raster_request_opts = raster_request_opts.copy()
        raster_request_opts.pop("chunk_size")
        raster_request_opts.pop("overlap")
        contours_request_config = contours_request_config.model_dump()
        contours_request_config.pop("target_size")
        contours_request_config.pop("expansion_rate")
        normalized_requests = [raster_path, raster_request_opts, contours_request_config]  # global requets
        serialized_requests = json.dumps(normalized_requests, default=str)
        return cache_directory / (hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".feather")

    @classmethod
    def _get_contours_gdf_from_raster(cls, indices, raster_path, raster_request_opts, contour_request_config, dst_crs, cache_directory):
        k, i, j = indices
        filepath = cls._get_contours_gdf_from_raster_feather_path(raster_path, raster_request_opts, contour_request_config, cache_directory)
        if not filepath.is_file():
            contours_gdf = cls._build_contours_gdf_from_raster(raster_path, raster_request_opts, contour_request_config)
        else:
            contours_gdf = gpd.read_feather(filepath).to_crs(dst_crs)
        contours_gdf["global_index"] = k
        contours_gdf["target_size"] = contour_request_config.target_size
        contours_gdf["expansion_rate"] = contour_request_config.expansion_rate
        return contours_gdf.to_crs(dst_crs)


    # @staticmethod
    # def _collapse_by_required_columns(gdf):
    #     required_columns = ['level', 'target_size', 'expansion_rate', 'global_id', 'raster_id', 'window_id']
    #     data = []
    #     column_data = []
    #     for (level, target_size, expansion_rate, global_id, raster_id, window_id), group in gdf.groupby(required_columns):
    #         # Filter out empty geometries and flatten MultiLineStrings
    #         geometries = [geom for row in group.itertuples() if not row.geometry.is_empty
    #                       for geom in (row.geometry.geoms if isinstance(row.geometry, MultiLineString) else [row.geometry])
    #                       if isinstance(geom, LineString) and not geom.is_empty]
    #         # row_data = raster_geom_gdf.iloc[].drop('geometry')
    #         # column_data.append(row_data)
    #         # if geometries:
    #         data.append({
    #             'geometry': MultiLineString(geometries),
    #             'level': level,
    #             'target_size': target_size,
    #             'expansion_rate': expansion_rate,
    #             'global_id': global_id,
    #             'raster_id': raster_id,
    #             'window_id': window_id,
    #         })
    #     return gpd.GeoDataFrame(data, crs=gdf.crs)


    @staticmethod
    def _get_raster_window_bbox(indices, raster_path, request_opts):
        k, i, j = indices
        from shapely.geometry import box
        raster = get_raster_from_opts(
                raster_path=raster_path,
                window=request_opts.pop("window"),
                **request_opts)
        bbox = raster.get_bbox()
        shapely_box = box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        return gpd.GeoDataFrame(
                [{'geometry': shapely_box, 'global_id': k, 'raster_id': i, 'window_id': j}],
                crs=raster.crs
                ).to_crs("epsg:4326")

    def _compute_hfun_nprocs(self, max_threads, local_contours_gdf):
        if self.cpus_per_task is not None:
            return self.cpus_per_task
        def count_points(geometry):
            if hasattr(geometry, 'geoms'):
                return sum(len(geom.coords) for geom in geometry.geoms)
            else:
                return len(geometry.coords)
        try:
            total_points = local_contours_gdf.geometry.apply(count_points).sum()
        except:
            return 1
        if total_points == 0:
            return 1
        recommended_processes = np.max([1, np.min([total_points % 30000 // 1000, max_threads])])
        if self.max_cpus_per_task is not None and recommended_processes > self.max_cpus_per_task:
            recommended_processes = self.max_cpus_per_task
        if self.min_cpus_per_task is not None and recommended_processes < self.min_cpus_per_task:
            recommended_processes = self.min_cpus_per_task
        return recommended_processes

    def _get_hfun_raster_window_contours_gdf(self, comm, output_rank=None, cache_directory=None):
        root_rank = 0 if output_rank is None else output_rank
        if cache_directory is not None:
            raster_window_cache_dir = cache_directory / "raster_window_contours"
        else:
            if comm.Get_rank() == root_rank:
                _tmpdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".tmp-hfun_build-raster_window_contours")
                raster_window_cache_dir = Path(_tmpdir.name)
            else:
                raster_window_cache_dir = None
            raster_window_cache_dir = comm.bcast(raster_window_cache_dir, root=root_rank)
            # raster_window_cache_dir.parent.mkdir()
            # raster_window_cache_dir.mkdir(exist_ok=True)
        dst_crs = "epsg:4326"
        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                hfun_contours_gdf = []
                raster_window_bbox = []
                for k, ((i, raster_path, raster_request_opts, hfun_request_opts), (j, window)) in self.iter_normalized_raster_window_requests():
                    raster_window_bbox.append((
                        (k, i, j),
                        raster_path,
                        raster_request_opts,
                        ))
                    for constraint_name, constraint_configs in hfun_request_opts['constraints'].items():
                        if not isinstance(constraint_configs, list):
                            constraint_configs = [constraint_configs]
                        for constraint_config in constraint_configs:
                            if isinstance(constraint_config, HfunRasterContourRequest):
                                hfun_contours_gdf.append((
                                    (k, i, j),
                                    raster_path,
                                    raster_request_opts,
                                    constraint_config,
                                    dst_crs,
                                    raster_window_cache_dir
                                    ))
                                # self._get_contours_gdf_from_raster(*hfun_contours_gdf[-1])
                                # hfun_contours_gdf[-1] = executor.submit(
                                #         self._get_contours_gdf_from_raster,
                                #         *hfun_contours_gdf[-1]
                                #         )
                                # import psutil
                                # # Get the current process
                                # process = psutil.Process()
                                # # Get the list of open files
                                # open_files = process.open_files()
                                # # Count the number of open files
                                # num_open_files = len(open_files)
                                # print(f"Number of open files: {num_open_files}", flush=True)
                if len(hfun_contours_gdf) > 0:
                    # from concurrent.futures import as_completed
                    # _coll = []
                    # for hfun_contour_gdf in as_completed(hfun_contours_gdf):
                    #     import psutil

                    #     # Get the current process
                    #     process = psutil.Process()

                    #     # Get the list of open files
                    #     open_files = process.open_files()

                    #     # Count the number of open files
                    #     num_open_files = len(open_files)

                    #     print(f"Number of open files: {num_open_files}", flush=True)
                    #     _coll.append(hfun_contour_gdf)
                    # hfun_contours_gdf = pd.concat(_coll, ignore_index=True)
                    # del _coll

                    # hfun_contours_gdf = pd.concat([future.result() for future in hfun_contours_gdf], ignore_index=True)
                    hfun_contours_gdf = pd.concat(list(executor.starmap(self._get_contours_gdf_from_raster, hfun_contours_gdf, chunksize=2)), ignore_index=True)
                    raster_window_bbox = pd.concat(list(executor.starmap(self._get_raster_window_bbox, raster_window_bbox)), ignore_index=True)
                    raster_window_bbox = raster_window_bbox.loc[hfun_contours_gdf.global_index]
                    joined = gpd.sjoin(
                            hfun_contours_gdf,
                            raster_window_bbox,
                            how='left',
                            predicate='intersects',
                            )
                    joined = joined[joined.index != joined.index_right]
                    joined.index.name = 'index_left'
                    groups = joined.groupby("index_left")
                    logger.debug("making differences")
                    tasks = []
                    valid_rows = joined[joined['index_right'].notna()]
                    lower_indices = valid_rows['index_right'] < valid_rows.global_index
                    valid_rows = valid_rows[lower_indices]
                    for index_left, group in groups:
                        base_geom = hfun_contours_gdf.iloc[[index_left]].geometry.squeeze()
                        filtered_indices = valid_rows.loc[valid_rows.index == index_left, 'index_right']
                        envelope_geometries = raster_window_bbox.loc[filtered_indices].geometry.tolist() if not filtered_indices.empty else []
                        tasks.append((base_geom, envelope_geometries))
                    hfun_contours_gdf.geometry = list(executor.starmap(self._process_geom_difference, tasks))
                    hfun_contours_gdf = hfun_contours_gdf[~hfun_contours_gdf.geometry.is_empty]
                    logger.debug("making differences done")
                else:
                    hfun_contours_gdf = None

                # hfun_contours_gdf.plot(ax=plt.gca(), cmap='jet')

            else:
                hfun_contours_gdf = None
                raster_window_bbox = None
        # plt.show(block=False)
        # if comm.Get_rank() == output_rank:
        #     breakpoint()
        comm.barrier()
        # raise
        if output_rank is None:
            return comm.bcast(hfun_contours_gdf, root=root_rank), comm.bcast(raster_window_bbox, root=root_rank)
        else:
            return hfun_contours_gdf, raster_window_bbox

    @staticmethod
    def _process_geom_difference(base_contours, envelope_geometries):
        if envelope_geometries:
            combined_geom = ops.unary_union(envelope_geometries)
            base_contours = base_contours.difference(combined_geom.buffer(-np.finfo(np.float16).eps) or MultiPolygon([]))
        base_contours = ops.unary_union(base_contours)
        return base_contours
    
    @staticmethod
    def _loadmsh_wrapper(filepath: Path) -> jigsaw_msh_t:
        # msh_t = jigsaw_msh_t()
        # loadmsh(str(filepath.resolve()), msh_t)
        # return msh_t
        return pickle.load(open(filepath, 'rb'))


    def build_uncombined_hfuns_mpi(self, comm, output_rank=None, cache_directory=None) -> List[jigsaw_msh_t] | None:
        root_rank = 0 if output_rank is None else output_rank
        hfun_raster_window_geoms_gdf = self._get_hfun_raster_window_geoms_gdf(comm, output_rank=root_rank, cache_directory=cache_directory)
        hfun_raster_window_contours_gdf, raster_window_bbox = self._get_hfun_raster_window_contours_gdf(
                comm,
                output_rank=root_rank,
                cache_directory=cache_directory
                )
        # verify
        # if comm.Get_rank() == 0:
        #     hfun_raster_window_contours_gdf.plot(cmap="jet", ax=plt.gca())
        #     raster_window_bbox.plot(ax=plt.gca(), facecolor='none', edgecolor='k')
        #     plt.show()
        # comm.barrier()
        max_threads = lib.hardware_info(comm)['thread_count'].min()
        if cache_directory is not None:
            hfun_cache_directory = cache_directory / "raster_window"
        else:
            if comm.Get_rank() == root_rank:
                _tmpdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.tmp-hfun_build-uncombined-hfuns')
                hfun_cache_directory = Path(_tmpdir.name)
            else:
                hfun_cache_directory = None
            hfun_cache_directory = comm.bcast(hfun_cache_directory, root=root_rank)

        def iter_raster_hfun_msh_t_args():

            if hfun_raster_window_contours_gdf is not None:
                joined = gpd.sjoin(
                        raster_window_bbox,
                        raster_window_bbox,
                        how='left',
                        predicate='intersects',
                        )
            for k, ((i, raster_path, raster_request_opts, hfun_request_opts), (j, window)) in self.iter_normalized_raster_window_requests():
                if k not in hfun_raster_window_geoms_gdf.index:
                    continue
                hfun_request_opts["geom"] = ShapelyGeom(
                    hfun_raster_window_geoms_gdf.loc[k].geometry,
                    crs=hfun_raster_window_geoms_gdf.crs
                )
                hfun_constraints = hfun_request_opts.pop("constraints")
                if hfun_raster_window_contours_gdf is not None:
                    local_contours_gdf = hfun_raster_window_contours_gdf[hfun_raster_window_contours_gdf['global_index'].isin(list(joined.loc[[k]].index_right))]
                    hfun_constraints["contour"] = local_contours_gdf
                    hfun_request_opts["nprocs"] = self._compute_hfun_nprocs(max_threads, local_contours_gdf)
                raster_request_opts["raster_path"] = raster_path
                yield raster_request_opts, hfun_request_opts, hfun_constraints, hfun_cache_directory


        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                hfun_cache_directory.mkdir(exist_ok=True, parents=True)
                # serial (debug)
                # uncombined_hfuns = []
                # for args in iter_raster_hfun_msh_t_args():
                #     uncombined_hfuns.append(self._build_raster_hfun_msh_t(*args))
                # parallel
                logger.debug("Begin building hfuns in parallel")
                start = time()
                uncombined_hfuns: List[Path] = list(executor.starmap(
                    self._build_raster_hfun_msh_t,
                    iter_raster_hfun_msh_t_args(),
                    ))
                logger.debug(f"done building uncombined hfuns in parallel, took {time()-start}")
                logger.debug("Begin loading hfuns in parallel")
                start = time()
                uncombined_hfuns: List[jigsaw_msh_t] = list(executor.map(self._loadmsh_wrapper, uncombined_hfuns))
                logger.debug(f"done building uncombined hfuns in parallel, took {time()-start}")
                # import pickle
                # pickle.dump(
                #         uncombined_hfuns,
                #         open("uchfns.pkl", "wb")
                #         )
                # print(uncombined_hfuns, flush=True)
                # verify:
                # for hfun_msh_t in uncombined_hfuns:
                #     # print(hfun_msh_t.tria3["index"], flush=True)
                #     if len(hfun_msh_t.tria3["index"]) > 0:
                #         utils.triplot(hfun_msh_t, ax=plt.gca())
                #     plt.gca().axis("scaled")
                #     plt.show()
                # uncombined_hfun.plot(ax=plt.gca(), cmap='jet')
            else:
                uncombined_hfuns = None
        comm.barrier()
        if output_rank is None:
            return comm.bcast(uncombined_hfuns, root=0)
        return uncombined_hfuns

    # def _custom_job_scheduler(jobs, comm, output_rank):

    #     rank = comm.Get_rank()
    #     hwinfo = lib.hardware_info(comm)
    #     unique_colors = hwinfo['color'].unique()
    #     local_color = hwinfo.iloc[rank]['color']
    #     color_sizes = hwinfo['color'].value_counts()
    #     local_comm = comm.Split(local_color, rank)
    #     local_jobs = [job for _, job in lib.get_split_array_weighted(
    #             comm,
    #             jobs,
    #             [job[2] for job in jobs],
    #             len(unique_colors),
    #             )[local_color]]

    #     # Create a queue for jobs
    #     job_queue = queue.PriorityQueue()

    #     # Add jobs to queue, prioritized by number of processors needed
    #     for job, args, procs_needed in sorted(local_jobs, key=lambda x: x[2]):
    #         args = pickle.dumps(args)
    #         job_queue.put((-procs_needed, (job, args, procs_needed)))  # Negate procs_needed to sort in descending order

    #     # Create a list to hold futures
    #     futures = []
    #     results = []
    #     with MPICommExecutor(local_comm) as executor:
    #         if executor is not None:
    #             free_procs = color_sizes[local_color]
    #             while not job_queue.empty() or futures:
    #                 # Remove finished futures and update the number of free processors
    #                 for future in futures:
    #                     if future.done():
    #                         output_path, fprocs = future.result()
    #                         free_procs += fprocs  # Assumes the job returns the number of processors it used
    #                         futures.remove(future)
    #                         if output_path is not None:
    #                             results.append(output_path)
    #                 # ------- Option 1
    #                 # Keep track of jobs we can't fit right now
    #                 deferred_jobs = []

    #                 # Try to fill all available processors
    #                 while not job_queue.empty() and free_procs > 0:
    #                     _, (job, args, procs_needed) = job_queue.get()
    #                     if procs_needed <= free_procs:
    #                         args = pickle.loads(args)
    #                         future = executor.submit(job, *args)
    #                         futures.append(future)
    #                         free_procs -= procs_needed
    #                         from time import sleep
    #                         sleep(5)
    #                     else:
    #                         # If a job can't be scheduled, defer it
    #                         deferred_jobs.append((-procs_needed, (job, args, procs_needed)))

    #                 # Return deferred jobs back into the queue
    #                 for job in deferred_jobs:
    #                     job_queue.put(job)
        # return [item for sublist in comm.allgather(results) for item in sublist]

    @staticmethod
    def _get_cached_hfun_filename(raster_request_opts, hfun_request_opts, hfun_constraints, cache_directory):
        normalized_requests = [raster_request_opts, hfun_request_opts, hfun_constraints]  # global requets
        serialized_requests = json.dumps(normalized_requests, default=str)
        return cache_directory / (hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".msh")

    @classmethod
    def _build_raster_hfun_msh_t(
            cls,
            raster_request_opts,
            hfun_request_opts,
            hfun_constraints,
            cache_directory,
            ) -> jigsaw_msh_t:
        raster_hfun_filename = cls._get_cached_hfun_filename(
                raster_request_opts,
                hfun_request_opts,
                hfun_constraints,
                cache_directory
                )
        if raster_hfun_filename.is_file():
            logger.debug(f"Hfun for raster_path={raster_request_opts['raster_path']} window={raster_request_opts['window']} is already cached at {raster_hfun_filename}")
            return raster_hfun_filename
        logger.debug(f"Begin building hfun for raster_path={raster_request_opts['raster_path']} window={raster_request_opts['window']}")
        start = time()
        hfun = RasterHfun(
                raster=get_raster_from_opts(
                    **raster_request_opts
                    ),
                **hfun_request_opts
                )
        for constraint, constraint_requests in hfun_constraints.items():
            if constraint == "contour":
                if constraint_requests is not None:
                    for row in constraint_requests.itertuples():
                        hfun.add_feature(
                                row.geometry,
                                expansion_rate=row.expansion_rate,
                                target_size=row.target_size
                                )
                    # print(row, flush=True)
            else:
                if not isinstance(constraint_requests, list):
                    constraint_requests = [constraint_requests]
                for constraint_request_config in constraint_requests:
                    if constraint_request_config is not None:
                        try:
                            getattr(hfun, f"add_{constraint}")(**constraint_request_config.model_dump())
                        except Exception as e:
                            raise Exception(f"failed at {constraint=} {constraint_request_config=} {e=}")
        msh_t = hfun.msh_t()
        logger.debug(f"Building hfun for raster_path={raster_request_opts['raster_path']} window={raster_request_opts['window']} took {time()-start}")
        # savemsh(str(raster_hfun_filename.resolve()), msh_t)
        pickle.dump(msh_t, open(raster_hfun_filename, 'wb'))
        return raster_hfun_filename



        # the rest of the keys must be constraints

        # print(request_opts.keys(), flush=True)
        # return constraints.apply(hfun)

                
        # if local_contours_path is not None:
        #     local_contours = gpd.read_feather(local_contours_path)
        #     if len(local_contours) > 0:
        #         for k, row in local_contours.iterrows():
        #             row = row.copy()
        #             row.pop('level')
        #             row['feature'] = row.pop('geometry')
        #             # add as features bc theire precomputed + use additionals
        #             # logging.getLogger("geomesh").setLevel(logging.DEBUG)
        #             start = time()
        #             logger.info('start adding contour constraint ..')
        #             hfun.add_feature(**row)
        #             # logging.getLogger("geomesh").setLevel(logging.WARNING)
        #             logger.debug(f'adding contour constraints took {time()-start} on {nprocs=}')
        #     del local_contours
        # for request_type, request_values in request_opts.items():
        #     # special handle for contours/features
        #     if request_type == 'contours':
        #         pass
        #     elif request_type == 'features':
        #         pass
                # raise NotImplementedError


        # rank = comm.Get_rank()
        # for item in self.iter_raster_window_requests():
        #     print(item)
        # raise
        # hfun_raster_window_requests = list(self.iter_normalized_raster_window_requests())
        # cache_directory = hfun_config['cache_directory']
        # Now that we have an expanded list of raster window requests, we nee to compute
        # a system-independent hash for each of the requests. The fisrt step is to compute the md5
        # of each of the rasteres in raw form. We then use this md5 as a salt for adding other requests to the
        # raster request. This should theoretically result in the same hash on any platform, becaue it based on
        # the file content instead of path.
        # if rank == 0:
        #     logger.info('will get raster md5')
        # raster_hashes = lib.get_raster_md5(comm, hfun_raster_window_requests)
        # if rank == 0:
        #     logger.info('will get bbox gdf')
        # bbox_gdf = lib.get_bbox_gdf(comm, hfun_raster_window_requests, cache_directory, raster_hashes)
        # if rank == 0:
        #     logger.info('will get local contours paths')
        # local_contours_paths = get_local_contours_paths(comm, hfun_raster_window_requests, bbox_gdf, cache_directory, raster_hashes)
        # if rank == 0:
        #     logger.info('will get cpus_per_task')
        # cpus_per_task = get_cpus_per_task(comm, hfun_raster_window_requests, hfun_config, local_contours_paths, bbox_gdf)
        # if rank == 0:
        #     logger.info('will build msh_ts')
        # out_msh_t_tempfiles = get_out_msh_t(
        #             hfun_raster_window_requests,
        #             cpus_per_task,
        #             comm,
        #             hfun_config,
        #             local_contours_paths,
        #             bbox_gdf,
        #             cache_directory,
        #             raster_hashes,
        #             )


    @staticmethod
    def combine_msh_t_list(msh_t_list):
        msh_t = jigsaw_msh_t()
        msh_t.mshID = "euclidean-mesh"
        msh_t.ndims = +2
        msh_t.crs = CRS.from_epsg(4326)
        for window_mesh in msh_t_list:
            window_mesh.tria3["index"] += len(msh_t.vert2)
            msh_t.tria3 = np.append(
                    msh_t.tria3,
                    window_mesh.tria3,
                    axis=0
                    )
            if not window_mesh.crs.equals(msh_t.crs):
                utils.reproject(window_mesh, msh_t.crs)
            msh_t.vert2 = np.append(
                msh_t.vert2,
                window_mesh.vert2,
                axis=0
            )
            msh_t.value = np.append(msh_t.value, window_mesh.value)
        return msh_t


    def build_combined_hfun_msh_t_mpi(self, comm, output_rank=None, cache_directory=None):
        if cache_directory:
            cached_filepath = self._get_final_hfun_msh_path(comm, cache_directory)
            if cached_filepath.is_file() and (output_rank is None or comm.Get_rank() == output_rank):
                # hfun_msh_t = jigsaw_msh_t()
                # loadmsh(str(cached_filepath.resolve()), hfun_msh_t)
                # return hfun_msh_t
                return pickle.load(open(cached_filepath, 'rb'))
        final_hfun_msh_t = self._build_final_raster_hfun_msh_t_mpi(comm, output_rank=output_rank, cache_directory=cache_directory)
        should_write_cache = cache_directory and (output_rank is None or comm.Get_rank() == output_rank)
        if should_write_cache:
            savemsh(str(cached_filepath.resolve()), final_hfun_msh_t)
        if output_rank is None:
            return comm.bcast(final_hfun_msh_t, root=output_rank)
        return final_hfun_msh_t

    def _get_final_hfun_msh_path(self, comm, cache_directory):
        if comm.Get_rank() == 0:
            normalized_requests = list(self.iter_normalized_raster_window_requests())
            normalized_requests.append([self.hmin, self.hmax, self.marche])  # global requets
            serialized_requests = json.dumps(normalized_requests, default=str)
            # cached_filename = hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".msh"
            cached_filename = hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".pkl"
            combined_hfun_cached_directory = cache_directory / "combined"
            combined_hfun_cached_directory.mkdir(parents=True, exist_ok=True)
            combined_filepath = combined_hfun_cached_directory / cached_filename
        else:
            combined_filepath = None
        return comm.bcast(combined_filepath, root=0)

    def _build_final_raster_hfun_msh_t_mpi(self, comm, output_rank=None, cache_directory=None):
        root_rank = 0 if output_rank is None else output_rank
        hfun_msh_t_list = self.build_uncombined_hfuns_mpi(comm, output_rank=output_rank, cache_directory=cache_directory)

        def partition_list(input_list, N):
            """Partition `input_list` into sublists with roughly `N` items each."""
            array = np.array(input_list, dtype=object)  # Create a NumPy array of objects
            num_partitions = -(-len(input_list) // N)  # Ceiling division to get the number of partitions
            return np.array_split(array, num_partitions)  # Split the array into subarrays
        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                logger.debug("Begin partial combine of hfun windows")
                while len(hfun_msh_t_list) > 1:
                    hfun_msh_t_list = list(executor.map(
                        self.combine_msh_t_list,
                        partition_list(hfun_msh_t_list, 2)
                        ))
                    logger.debug(f"{len(hfun_msh_t_list)=}")
                hfun_msh_t = hfun_msh_t_list.pop()
            else:
                hfun_msh_t = None


        if output_rank is None:
            return comm.bcast(hfun_msh_t, root=root_rank)
        return hfun_msh_t









def submit(
        executor,
        config_path,
        to_msh=None,
        to_pickle=None,
        dst_crs=None,
        show=False,
        cache_directory=None,
        log_level: str = None,
        **kwargs
        ):
    if isinstance(executor, LocalCluster):
        raise TypeError('LocalCluster is not supported, use MPICluster instead.')
    delete_pickle = False if to_pickle is not None else True
    if delete_pickle is True and cache_directory is None:
        cache_directory = cache_directory or Path(os.getenv('GEOMESH_TEMPDIR', Path.cwd() / f'.tmp-geomesh/{Path(__file__).stem}'))
        cache_directory.mkdir(parents=True, exist_ok=True)
    if delete_pickle is True:
        to_pickle = Path(tempfile.NamedTemporaryFile(dir=cache_directory, suffix='.pkl').name)
    build_cmd = get_cmd(
            config_path,
            to_msh=to_msh,
            to_pickle=to_pickle,
            dst_crs=dst_crs,
            show=show,
            cache_directory=cache_directory,
            log_level=log_level,
            )

    async def callback():
        await executor.submit(build_cmd, **kwargs)
        hfun = pickle.load(open(to_pickle, 'rb'))
        if delete_pickle:
            to_pickle.unlink()
        return hfun

    return callback()


def get_cmd(
        config_path,
        to_msh=None,
        to_pickle=None,
        dst_crs=None,
        show=False,
        cache_directory=None,
        log_level: str = None,
        ):
    build_cmd = [
            sys.executable,
            f'{Path(__file__).resolve()}',
            f'{Path(config_path).resolve()}',
            ]
    if to_msh is not None:
        build_cmd.append(f'--to-file={to_msh}')
    if to_pickle is not None:
        build_cmd.append(f'--to-pickle={to_pickle}')
    if show:
        build_cmd.append('--show')
    if cache_directory is not None:
        build_cmd.append(f'--cache-directory={cache_directory}')
    if dst_crs is not None:
        build_cmd.append(f'--dst-crs={dst_crs}')
    if log_level is not None:
        build_cmd.append(f'--log-level={log_level}')
    logger.debug(' '.join(build_cmd))
    return build_cmd


def init_logger(log_level: str):
    logging.basicConfig(
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        force=True,
        # datefmt="%Y-%m-%d %H:%M:%S "
    )
    log_level = {
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "critical": logging.CRITICAL,
            "notset": logging.NOTSET,
        }[str(log_level).lower()]
    logger.setLevel(log_level)
    # if int(log_level) < 40:
    #     logging.getLogger("geomesh").setLevel(log_level)
    # logging.Formatter.converter = lambda *args: datetime.now(tz=pytz.timezone("UTC")).timetuple()
    logging.captureWarnings(True)


def get_gdf_from_axes(axes, contour_opts, crs):

    # path_collection = axes.Collection[0]
    level = contour_opts['level']
    level_data = {}
    for i, (_level, path_collection) in enumerate(zip(axes.levels, axes.collections)):
        if _level != level:
            continue
        linestrings = []
        for path in path_collection.get_paths():
            try:
                linestrings.append(LineString(path.vertices))
            except ValueError as e:
                if "LineStrings must have at least 2 coordinate tuples" in str(e):
                    continue
                raise e
        level_data.update({level: MultiLineString(linestrings)})
        # break
        # print(i, level)
        # if level == args.level:
        #     level_data.update({level: MultiLineString(linestrings)})
    if len(level_data) == 0:
        return None

    return gpd.GeoDataFrame([{
        'geometry': MultiLineString(linestrings),
        'level': level,
        'expansion_rate': contour_opts.get('expansion_rate'),
        'target_size': contour_opts.get('target_size'),
        # 'color': rgb2hex(axes.collections[i].get_facecolor())
        } for i, (level, data) in enumerate(level_data.items())],
        crs=crs
        ).to_crs(CRS.from_epsg(4326))


def get_contours_gdf_from_raster(raster_path, request_opts, window, contour_request, cache_directory):
    cache_directory /= 'raster_window_contours'
    cache_directory.mkdir(exist_ok=True)
    # normalized_contour_request = {
    #     'level': contour_request['level'],
    #     'expansion_rate': contour_request.get('expansion_rate'),
    #     'target_size': contour_request.get('target_size'),
    #     }
    # normalized_contour_request.update(lib.get_normalized_raster_request_opts(raster_hash, request_opts, window))
    tmpfile = cache_directory / (hashlib.sha256(json.dumps(normalized_contour_request).encode()).hexdigest() + '.feather')
    if tmpfile.is_file():
        logger.debug(f"Load contours for {normalized_contour_request=} from {tmpfile=}")
        return gpd.read_feather(tmpfile)

    logger.debug(f"Building contours for {normalized_contour_request=} to {tmpfile=}")
    raster = get_raster_from_opts(raster_path, request_opts, window)
    gdf = []
    for xvals, yvals, zvals in raster:
        try:
            # plt.ioff()
            import matplotlib as mpl
            _old_backend = mpl.get_backend()
            plt.switch_backend('agg')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                ax = plt.contour(xvals, yvals, zvals[0, :], levels=[contour_request['level']])
            plt.close(plt.gcf())
            plt.switch_backend(_old_backend)
            _gdf = get_gdf_from_axes(ax, contour_request, raster.crs)
            # _gdf.to_crs(CRS.from_epsg(4326), inplace=True)
            if _gdf is not None:
                gdf.append(_gdf)
        except TypeError as err:
            if 'Input z must be at least a (2, 2) shaped array, but has shape' in str(err):
                plt.switch_backend(_old_backend)
                continue
            raise
    if len(gdf) == 0:
        return None
    gdf = pd.concat(gdf, ignore_index=True)
    # gdf = gdf.loc[gdf['geometry'].is_valid, :]
    # gdf = gdf.loc[~gdf['geometry'].is_empty, :]
    gdf.to_feather(tmpfile)
    return gdf


def get_hfun_contours_gdf(comm, hfun_raster_window_requests, bbox_gdf, cache_directory):
    hfun_contours_requests = []
    if comm.Get_rank() == 0:
        for global_k, ((i, raster_path, request_opts), (j, window)) in enumerate(hfun_raster_window_requests):
            if 'contours' in request_opts:
                for contour_request in request_opts['contours']:
                    hfun_contours_requests.append(((global_k, ((i, raster_path, request_opts), (j, window))), contour_request))
        hfun_contours_requests = np.array(hfun_contours_requests, dtype=object)
        hfun_contours_requests = np.array_split(hfun_contours_requests, comm.Get_size())
    hfun_contours_requests = comm.bcast(hfun_contours_requests, root=0)[comm.Get_rank()]
    contours_gdf = []
    for (global_k, ((i, raster_path, request_opts), (j, window))), contour_request in hfun_contours_requests:
        local_gdf = get_contours_gdf_from_raster(
                raster_path,
                request_opts,
                window,
                raster_hashes[global_k],
                contour_request,
                cache_directory
                )
        if local_gdf is None:
            continue
        idxs = bbox_gdf.iloc[:global_k].sindex.query(local_gdf.unary_union)
        if len(idxs) > 0:
            local_gdf = local_gdf.difference(bbox_gdf.iloc[idxs].unary_union)
            local_gdf = gpd.GeoDataFrame([{'geometry': local_gdf.unary_union, **contour_request}], crs=bbox_gdf.crs)
        contours_gdf.append(local_gdf.to_crs('epsg:4326'))
    contours_gdf = pd.concat([item for sublist in comm.allgather(contours_gdf) for item in sublist], ignore_index=True)
    # verify
    # if comm.Get_rank() == 0:
    #     contours_gdf.plot(cmap='jet')
    #     plt.show(block=True)
    return contours_gdf


# def collapse_by_required_columns(gdf):
#     required_columns = set(['level', 'target_size', 'expansion_rate'])
#     data = []
#     for indexes in gdf.groupby(list(required_columns)).groups.values():
#         # if len(indexes) == 0:
#         #     continue
#         rows = gdf.iloc[indexes]
#         geometries = []
#         for row in rows.itertuples():
#             if row.geometry.is_empty:
#                 continue
#             elif isinstance(row.geometry, LineString):
#                 geometries.append(row.geometry)
#             elif isinstance(row.geometry, MultiLineString):
#                 for ls in row.geometry.geoms:
#                     if not ls.is_empty:
#                         geometries.append(ls)
#         data.append({
#             'geometry': MultiLineString(geometries),
#             'level': gdf.iloc[indexes[0]].level,
#             'target_size': gdf.iloc[indexes[0]].target_size,
#             'expansion_rate': gdf.iloc[indexes[0]].expansion_rate,
#             })
#     return gpd.GeoDataFrame(data, crs=gdf.crs)



def get_cpus_per_task(comm, hfun_raster_window_requests, hfun_config, local_contours_paths, bbox_gdf):

    cpus_per_task = hfun_config.get('cpus_per_task')
    max_cpus_per_task = hfun_config.get('max_cpus_per_task')
    min_cpus_per_task = hfun_config.get('min_cpus_per_task')

    if np.all([bool(cpus_per_task), np.any([bool(max_cpus_per_task), bool(min_cpus_per_task)])]):
        raise ValueError('Arguments cpus_per_task and max_cpus_per_task are mutually exclusive.')

    if cpus_per_task is not None:
        if not isinstance(cpus_per_task, int) or cpus_per_task < 1:
            raise ValueError(f'Argument cpus_per_task must be an int > 0 or None (for auto dist) but got {cpus_per_task=}.')

    if cpus_per_task is not None:
        return [cpus_per_task]*len(hfun_raster_window_requests)

    if all(local_contours_path is None for local_contours_path in local_contours_paths):
        if cpus_per_task is None:
            cpus_per_task = 1
        return [cpus_per_task]*len(hfun_raster_window_requests)

    nprocs_list = []
    local_contours_paths = np.array_split(local_contours_paths, comm.Get_size())[comm.Get_rank()]
    max_threads = lib.hardware_info(comm)['thread_count'].min()
    for local_contours_path in local_contours_paths:
        if local_contours_path is None:
            continue
        total_points = 0
        for row in gpd.read_feather(local_contours_path).itertuples():
            if hasattr(row.geometry, 'geoms'):
                for geom in row.geometry.geoms:
                    total_points += len(geom.coords)
            else:
                total_points += len(row.geometry.coords)

        nprocs_list.append(np.max([1, np.min([total_points % 30000 // 1000, max_threads])]))
        if max_cpus_per_task is not None and nprocs_list[-1] > max_cpus_per_task:
            nprocs_list[-1] = max_cpus_per_task
        if min_cpus_per_task is not None and nprocs_list[-1] < min_cpus_per_task:
            nprocs_list[-1] = min_cpus_per_task

    return [item for sublist in comm.allgather(nprocs_list) for item in sublist if item is not None]


def pickle_dump_callback(filepath, obj):
    pickle.dump(obj,  open(filepath, 'wb'))
    return filepath


def job_scheduler(jobs, comm):

    rank = comm.Get_rank()
    hwinfo = lib.hardware_info(comm)
    unique_colors = hwinfo['color'].unique()
    local_color = hwinfo.iloc[rank]['color']
    color_sizes = hwinfo['color'].value_counts()
    local_comm = comm.Split(local_color, rank)
    local_jobs = [job for _, job in lib.get_split_array_weighted(
            comm,
            jobs,
            [job[2] for job in jobs],
            len(unique_colors),
            )[local_color]]

    # Create a queue for jobs
    job_queue = queue.PriorityQueue()

    # Add jobs to queue, prioritized by number of processors needed
    for job, args, procs_needed in sorted(local_jobs, key=lambda x: x[2]):
        args = pickle.dumps(args)
        job_queue.put((-procs_needed, (job, args, procs_needed)))  # Negate procs_needed to sort in descending order

    # Create a list to hold futures
    futures = []
    results = []
    with MPICommExecutor(local_comm) as executor:
        if executor is not None:
            free_procs = color_sizes[local_color]
            while not job_queue.empty() or futures:
                # Remove finished futures and update the number of free processors
                for future in futures:
                    if future.done():
                        output_path, fprocs = future.result()
                        free_procs += fprocs  # Assumes the job returns the number of processors it used
                        futures.remove(future)
                        if output_path is not None:
                            results.append(output_path)
                # ------- Option 1
                # Keep track of jobs we can't fit right now
                deferred_jobs = []

                # Try to fill all available processors
                while not job_queue.empty() and free_procs > 0:
                    _, (job, args, procs_needed) = job_queue.get()
                    if procs_needed <= free_procs:
                        args = pickle.loads(args)
                        future = executor.submit(job, *args)
                        futures.append(future)
                        free_procs -= procs_needed
                        from time import sleep
                        sleep(5)
                    else:
                        # If a job can't be scheduled, defer it
                        deferred_jobs.append((-procs_needed, (job, args, procs_needed)))

                # Return deferred jobs back into the queue
                for job in deferred_jobs:
                    job_queue.put(job)
    return [item for sublist in comm.allgather(results) for item in sublist]

# def job_scheduler(jobs, comm):
#     # Create a queue for jobs
#     job_queue = queue.PriorityQueue()
#     # Add jobs to queue, prioritized by number of processors needed
#     for job, args, procs_needed in sorted(jobs, key=lambda x: x[2]):
#         args = pickle.dumps(args)
#         job_queue.put((-procs_needed, (job, args, procs_needed)))  # Negate procs_needed to sort in descending order
#     # Create a list to hold futures
#     futures = []
#     results = []
#     with MPICommExecutor(comm, root=0) as executor:
#         if executor is not None:
#             free_procs = comm.Get_size()
#             while not job_queue.empty() or futures:
#                 # Remove finished futures and update the number of free processors
#                 for future in futures:
#                     if future.done():
#                         output_path, fprocs = future.result()
#                         free_procs += fprocs  # Assumes the job returns the number of processors it used
#                         futures.remove(future)
#                         if output_path is not None:
#                             results.append(output_path)
#                 # ------- Option 1
#                 # # Keep track of jobs we can't fit right now
#                 # deferred_jobs = []
#                 # # Try to fill all available processors
#                 # while not job_queue.empty() and free_procs > 0:
#                 #     _, (job, args, procs_needed) = job_queue.get()
#                 #     if procs_needed <= free_procs:
#                 #         args = pickle.loads(args)
#                 #         future = executor.submit(job, *args)
#                 #         futures.append(future)
#                 #         free_procs -= procs_needed
#                 #     else:
#                 #         # If a job can't be scheduled, defer it
#                 #         deferred_jobs.append((-procs_needed, (job, args, procs_needed)))

#                 # # Return deferred jobs back into the queue
#                 # for job in deferred_jobs:
#                 #     job_queue.put(job)

#                 # Submit new jobs if there are enough free processors
#                 while not job_queue.empty():
#                     procs_needed = -job_queue.queue[0][0]  # Peek at the number of processors needed by the next job
#                     if procs_needed <= free_procs:
#                         _, (job, args, procs_needed) = job_queue.get()
#                         args = pickle.loads(args)
#                         future = executor.submit(job, *args)
#                         futures.append(future)
#                         free_procs -= procs_needed
#                     else:
#                         break
#     # Collect results
#     if comm.Get_rank() == 0:
#         results.extend([res for res in [f.result()[0] for f in futures] if res is not None])
#     return comm.bcast(results, root=0)


def get_out_msh_t(
        hfun_raster_window_requests,
        cpus_per_task,
        comm,
        hfun_config,
        local_contours_paths,
        bbox_gdf,
        cache_directory,
        raster_hashes,
        ):
    results = []
    jobs = []
    # cache_directory /= 'window_msh_t'
    # cache_directory.mkdir(exist_ok=True)

    hfun_constraints = [
            'constant_value',
            'contour',
            'feature',
            'gradient_delimiter',
            'narrow_channel_anti_aliasing',
            'gaussian_filter',
            ]
    for global_k, ((i, raster_path, request_opts), (j, window)) in enumerate(hfun_raster_window_requests):
        normalized_raster_opts = lib.get_normalized_raster_request_opts(raster_hashes[global_k], request_opts, window)
        normalized_hfun_request = {
                    'hmin': hfun_config.get('hmin'),
                    'hmax': hfun_config.get('hmax'),
                    'marche': hfun_config.get('marche'),
                    'normalized_raster_opts': normalized_raster_opts,
                }
        if local_contours_paths[global_k] is not None:
            normalized_hfun_request.update({
                        'local_contours_sha256': local_contours_paths[global_k].stem,
                    })
        # for requests in request_opts:
        for request_type, request_values in request_opts.items():
            if request_type in list(hfun_constraints):
                normalized_hfun_request.update({
                    request_type: request_values,
                    })
        tmpfile = cache_directory / (hashlib.sha256(
            json.dumps(
                normalized_hfun_request,
                ).encode()).hexdigest() + '.pkl')
        if tmpfile.exists():
            results.append(tmpfile)
        else:
            args = (
                    raster_path,
                    request_opts,
                    hfun_config,
                    window,
                    cpus_per_task[global_k],
                    local_contours_paths[global_k],
                    bbox_gdf,
                    i,
                    global_k,
                    partial(pickle_dump_callback, tmpfile),
                    )
            jobs.append((
                    self._get_local_msh_t_tmpfile,
                    args,
                    cpus_per_task[global_k]
                    ))
    results.extend(job_scheduler(jobs, comm))
    return results


def get_local_contours_paths(comm, hfun_raster_window_requests, bbox_gdf, cache_directory):

    if comm.Get_rank() == 0:
        logger.info('Begin build raster contour requests...')

    contours_gdf = get_hfun_contours_gdf(comm, hfun_raster_window_requests, bbox_gdf, cache_directory)

    # verify:
    if contours_gdf is None:
        return [None]*len(hfun_raster_window_requests)

    contours_hash = hashlib.sha256(str(contours_gdf).encode('utf-8')).hexdigest()

    cache_directory /= 'local_contours'
    cache_directory.mkdir(exist_ok=True)
    # hfun_raster_window_requests = np.array_split(
    #         np.array(hfun_raster_window_requests, dtype=object),
    #         comm.Get_size()
    #         )[comm.Get_rank()]
    hfun_raster_window_requests = lib.get_split_array(comm, hfun_raster_window_requests)[comm.Get_rank()]

    local_contours_paths = []
    for global_k, ((i, raster_path, raster_opts), (j, window)) in hfun_raster_window_requests:
        # normalized_raster_request = lib.get_normalized_raster_request_opts(
        #         raster_hashes[global_k], raster_opts, window)
        # normalized_raster_request['contours_hash'] = contours_hash
        feather = cache_directory / (hashlib.sha256(json.dumps(normalized_raster_request).encode()).hexdigest() + '.feather')
        if feather.exists():
            local_contours_paths.append(feather)
            continue
        base_mp = bbox_gdf.iloc[[global_k]]
        local_proj = CRS.from_user_input(
                ' '.join([
                        "+proj=aeqd",
                        "+R=6371000",
                        "+units=m",
                        f"+lat_0={base_mp.unary_union.centroid.y}",
                        f"+lon_0={base_mp.unary_union.centroid.x}",
                    ])
                )
        base_mp = base_mp.to_crs(local_proj)
        bbox_centroid = base_mp.unary_union.centroid
        bbox_bounds = Bbox.from_extents(*base_mp.unary_union.bounds)
        bbox_centroid = base_mp.unary_union.centroid
        bbox_radius = np.sqrt((bbox_bounds.xmax - bbox_centroid.x)**2+(bbox_bounds.ymax - bbox_centroid.y)**2)
        # NOTE: hardcoded to 2 times the radius
        clip_poly = bbox_centroid.buffer(2*bbox_radius)
        clip_poly = ops.transform(
                Transformer.from_crs(local_proj, contours_gdf.crs, always_xy=True).transform,
                clip_poly)
        local_contours = gpd.clip(contours_gdf, clip_poly).reset_index()
        if len(local_contours) > 0:
            local_contours = collapse_by_required_columns(local_contours)
        local_contours.to_feather(feather)
        local_contours_paths.append(feather)
    return [item for sublist in comm.allgather(local_contours_paths) for item in sublist]


# def get_hfun_raster_window_requests(comm, hfun_config):
#     if comm.Get_rank() == 0:
#         logger.info('Loading hfun raster window requests...')
#         hfun_raster_window_requests = list(iter_raster_window_requests(hfun_config))
#         logger.debug('Done loading hfun raster window requests.')
#     else:
#         hfun_raster_window_requests = None
#     hfun_raster_window_requests = comm.bcast(hfun_raster_window_requests, root=0)
#     return hfun_raster_window_requests


# def get_uncombined_hfuns(comm, hfun_config):
#     rank = comm.Get_rank()
#     hfun_raster_window_requests = get_hfun_raster_window_requests(comm, hfun_config)
#     cache_directory = hfun_config['cache_directory']
#     # Now that we have an expanded list of raster window requests, we nee to compute
#     # a system-independent hash for each of the requests. The fisrt step is to compute the md5
#     # of each of the rasteres in raw form. We then use this md5 as a salt for adding other requests to the
#     # raster request. This should theoretically result in the same hash on any platform, becaue it based on
#     # the file content instead of path.
#     if rank == 0:
#         logger.info('will get raster md5')
#     raster_hashes = lib.get_raster_md5(comm, hfun_raster_window_requests)
#     if rank == 0:
#         logger.info('will get bbox gdf')
#     bbox_gdf = lib.get_bbox_gdf(comm, hfun_raster_window_requests, cache_directory, raster_hashes)
#     if rank == 0:
#         logger.info('will get local contours paths')
#     local_contours_paths = get_local_contours_paths(comm, hfun_raster_window_requests, bbox_gdf, cache_directory, raster_hashes)
#     if rank == 0:
#         logger.info('will get cpus_per_task')
#     cpus_per_task = get_cpus_per_task(comm, hfun_raster_window_requests, hfun_config, local_contours_paths, bbox_gdf)
#     if rank == 0:
#         logger.info('will build msh_ts')
#     out_msh_t_tempfiles = get_out_msh_t(
#                 hfun_raster_window_requests,
#                 cpus_per_task,
#                 comm,
#                 hfun_config,
#                 local_contours_paths,
#                 bbox_gdf,
#                 cache_directory,
#                 raster_hashes,
#                 )
#     if rank == 0:
#         logger.info(f'built {len(out_msh_t_tempfiles)} msh_ts')
#     return out_msh_t_tempfiles


def combine_msh_t_list(msh_t_list):
    msh_t = jigsaw_msh_t()
    msh_t.mshID = "euclidean-mesh"
    msh_t.ndims = +2
    msh_t.crs = CRS.from_epsg(4326)
    for window_mesh in msh_t_list:
        window_mesh.tria3["index"] += len(msh_t.vert2)
        msh_t.tria3 = np.append(
                msh_t.tria3,
                window_mesh.tria3,
                axis=0
                )
        if not window_mesh.crs.equals(msh_t.crs):
            utils.reproject(window_mesh, msh_t.crs)
        msh_t.vert2 = np.append(
            msh_t.vert2,
            window_mesh.vert2,
            axis=0
        )
        msh_t.value = np.append(msh_t.value, window_mesh.value)
    return msh_t


def combine_msh_t_partition(msh_t_list, cache_directory):
    msh_t_names = [msh_t_path.name for msh_t_path in msh_t_list]
    tmpfile = cache_directory / (hashlib.sha256(json.dumps(msh_t_names).encode()).hexdigest() + ".pkl")
    msh_t_list = np.array(msh_t_list, dtype=object).flatten()
    msh_ts = []
    for msh_t_path in msh_t_list:
        # if msh_t_path.exists():
        try:
            with open(msh_t_path, 'rb') as fh:
                msh_ts.append(pickle.load(fh))
        except EOFError:
            pass
    msh_t = combine_msh_t_list(msh_ts)
    with open(tmpfile, 'wb') as fh:
        pickle.dump(msh_t, fh)
    return tmpfile


def distribute_buckets(buckets, size):
    """Distribute buckets into 'size' groups as evenly as possible."""
    result = [[] for _ in range(size)]
    index = 0
    for bucket in buckets:
        result[index].append(bucket)
        index = (index + 1) % size
    return result


def combine_hfuns(comm, out_msh_t_tempfiles, hfun_config):

    cache_directory = hfun_config["cache_directory"]
    cache_directory /= "msh_t_combined"
    cache_directory.mkdir(exist_ok=True)

    rank = comm.Get_rank()
    size = comm.Get_size()

    divisions = [len(out_msh_t_tempfiles)]
    while divisions[-1] > 1:
        divisions.append(int(np.floor(divisions[-1]/2)))

    for i, npartitions in enumerate(divisions[1:]):
        if rank == 0:
            if npartitions <= size:
                out_msh_t_tmpfiles = np.array(out_msh_t_tempfiles, dtype=object)
                buckets = [out_msh_t_tmpfiles[bucket] for bucket in np.array_split(
                    range(len(out_msh_t_tempfiles)), npartitions)]
                buckets = np.array(buckets + [[None]]*(size - len(buckets)), dtype=object)
            else:
                buckets = np.array_split(out_msh_t_tempfiles, npartitions)
                buckets = distribute_buckets(buckets, size)
        else:
            buckets = None

        bucket = comm.bcast(buckets, root=0)[rank]
        local_list = []
        for partition in bucket:
            if partition is None:
                continue
            if isinstance(partition, Path):
                local_list.append(partition)
            else:
                combined_msh_t_tempfile = combine_msh_t_partition(partition, cache_directory)
                local_list.append(combined_msh_t_tempfile)

        out_msh_t_tempfiles = [item for sublist in comm.allgather(local_list) for item in sublist]

    if rank == 0:
        combined_msh_t_tempfile = combine_msh_t_partition(out_msh_t_tempfiles, cache_directory)
    else:
        combined_msh_t_tempfile = None

    return comm.bcast(combined_msh_t_tempfile, root=0)


class ConfigPathAction(argparse.Action):

    def __call__(self, parser, namespace, config_path, option_string=None):
        if not config_path.is_file():
            raise FileNotFoundError(f'{config_path=} is not an existing file.')
        setattr(namespace, self.dest, config_path)


def get_argument_parser():

    def cache_directory_bootstrap(path_str):
        path = Path(path_str)
        if not path.name == "hfun_build":
            path /= "hfun_build"
        return path

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, action=ConfigPathAction)
    # parser.add_argument('--max-cpus-per-task', type=int)
    parser.add_argument('--to-msh', type=Path)
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--dst-crs', '--dst_crs', type=CRS.from_user_input, default=CRS.from_epsg(4326))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=cache_directory_bootstrap)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


def make_plot(hfun_msh_t):
    utils.tricontourf(hfun_msh_t, axes=plt.gca(), cmap='jet')
    utils.triplot(hfun_msh_t, axes=plt.gca())
    plt.gca().axis('scaled')
    plt.show(block=True)
    return


def to_msh(args, hfun_msh_t_tmpfile):
    logger.info('Write msh_t...')
    with open(hfun_msh_t_tmpfile, 'rb') as fh:
        msh_t = pickle.load(fh)
    savemsh(f'{args.to_msh.resolve()}', Hfun(msh_t).msh_t(dst_crs=args.dst_crs))
    return


def to_pickle(args, hfun_msh_t_tmpfile):
    logger.info('Write pickle...')
    with open(hfun_msh_t_tmpfile, 'rb') as fh:
        msh_t = pickle.load(fh)
    with open(args.to_pickle, 'wb') as fh:
        pickle.dump(Hfun(msh_t), fh)
    logger.info('Done writing pickle...')
    return


# def get_hfun_config_from_args(comm, args):
#     rank = comm.Get_rank()
#     # We begin by validating the user's request. The validation is done by the BuildCi class,
#     # it does not coerce or change the data of the hfun_config dictionary.
#     # The yaml configuration data needs to be expanded for parallelization,
#     # and the expansion is done by the iter_raster_window_requests method.
#     # TODO: Optimize (parallelize) the initial data loading.
#     if rank == 0:
#         logger.info('Validating user raster hfun requests...')
#         hfun_config = BuildCli(args).config.hfun.hfun_config
#         if hfun_config is None:
#             raise RuntimeError(f'No hfun to process in {args.config}')
#         tmpdir = hfun_config.get('cache_directory', Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp-geomesh')))
#         hfun_config.setdefault("cache_directory", tmpdir)
#         if args.cache_directory is not None:
#             hfun_config['cache_directory'] = args.cache_directory
#         hfun_config["cache_directory"].mkdir(exist_ok=True, parents=True)
#     else:
#         hfun_config = None

#     return comm.bcast(hfun_config, root=0)


def main(args: argparse.Namespace, comm=None):
    """This program uses MPI and memoization. The memoization directory can be provided
    as en evironment variable GEOMESH_TEMPDIR, as the key 'cache_directory' in the yaml
    configuration file, or as the command line argument --cache-directory, and they
    superseed each other in this same order. The default is `os.getcwd() + '/.tmp'`.
    """

    comm = MPI.COMM_WORLD if comm is None else comm
    hfun_config = HfunConfig.try_from_yaml_path(args.config)
    # hfun_config = get_hfun_config_from_args(comm, args)
    hfun_msh_t = hfun_config.build_combined_hfun_msh_t_mpi(
            comm,
            output_rank=0,
            cache_directory=args.cache_directory,
            )
    finalization_tasks = []

    if args.show:
        finalization_tasks.append(make_plot)

    if args.to_msh:
        finalization_tasks.append(partial(to_msh, args))

    if args.to_pickle:
        finalization_tasks.append(partial(to_pickle, args))

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            futures = [executor.submit(task, hfun_msh_t) for task in finalization_tasks]
            for future in futures:
                future.result()
            # executor.shutdown(wait=True)


def entrypoint():
    sys.excepthook = lib.mpiabort_excepthook
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        try:
            args = get_argument_parser().parse_args()
        except:
            comm.Abort(-1)
    else:
        args = None
    args = comm.bcast(args, root=0)
    init_logger(args.log_level)
    from geomesh.cli.mpi import geom_build
    geom_build.init_logger(args.log_level)
    main(args)


if __name__ == "__main__":
    entrypoint()
