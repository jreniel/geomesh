#!/usr/bin/env python
from functools import partial
from pathlib import Path
from time import time
from typing import List
from typing import Optional
from typing import Union
import argparse
import hashlib
import inspect
import json
import logging
import os
import pickle
import sys
import tempfile
import warnings

from dask_geopandas.hilbert_distance import _hilbert_distance
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pydantic import BaseModel
from shapely import ops
from shapely import unary_union
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
import dask_geopandas as dgpd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from geomesh import Geom
from geomesh.geom.raster import RasterGeom
from geomesh.cli.mpi import lib
from geomesh.cli.raster_opts import get_raster_from_opts

logger = logging.getLogger(f'[rank: {MPI.COMM_WORLD.Get_rank()}]: {__name__}')

warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

warnings.filterwarnings(
        'ignore',
        message='All-NaN slice encountered'
        )

warnings.filterwarnings(
        'ignore',
        message="`unary_union` returned None due to all-None GeoSeries. "
                "In future, `unary_union` will return 'GEOMETRYCOLLECTION EMPTY' instead."
        )


class GeomRasterConfig(lib.RasterConfig, lib.IterableRasterWindowTrait):
    zmax: Optional[float] = None
    zmin: Optional[float] = None
    _normalization_keys = [
            "resampling_factor",
            "clip",
            "bbox",
            "zmin",
            "zmax"
            ]

    def iter_normalized_raster_requests(self):
        for normalized_path, normalized_raster_geom_request in super().iter_normalized_raster_requests():
            for normalization_key in self._normalization_keys:
                normalized_raster_geom_request[normalization_key] = getattr(self, normalization_key)
            yield normalized_path, normalized_raster_geom_request

    def iter_normalized_raster_window_requests(self):
        for (i, normalized_path, normalized_raster_geom_request), (j, window) in super().iter_normalized_raster_window_requests():
            for normalization_key in self._normalization_keys:
                normalized_raster_geom_request[normalization_key] = getattr(self, normalization_key)
            yield (i, normalized_path, normalized_raster_geom_request), (j, window)




class GeomConfig(BaseModel):
    rasters: List[GeomRasterConfig]
    sieve: Optional[Union[bool, float]] = None
    partition_size: Optional[int] = 2
    grid_size: Optional[float] = None
    _normalization_keys = ["grid_size"]

    @classmethod
    def try_from_yaml_path(cls, path: Path) -> "GeomConfig":
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return cls.try_from_dict(data)

    @classmethod
    def try_from_dict(cls, data: dict) -> "GeomConfig":
        return cls(**data['geom']) if 'geom' in data else cls(rasters=[])

    def build_raster_geom_gdf_mpi(self, comm, output_rank=None, cache_directory=None):
        with MPICommExecutor(
                comm,
                root=0 if output_rank is None else output_rank
                ) as executor:
            if executor is not None:
                # raster_geom_gdf = []
                # for raster_request_opts, geom_request_opts, geom_raster_constraints in self.iter_normalized_raster_window_requests():
                #     raster_geom_gdf.append(self._build_raster_geom_gdf_wrapper(raster_request_opts, geom_request_opts, geom_raster_constraints, cache_directory))
                # raster_geom_gdf = pd.concat(raster_geom_gdf, ignore_index=True)
                raster_geom_gdf = pd.concat(list(executor.starmap(
                            self._build_raster_geom_gdf_wrapper,
                            [(*args, cache_directory) for args in self.iter_normalized_raster_window_requests()],
                            )), ignore_index=True)
                # verify:
                # raster_geom_gdf.plot(ax=plt.gca(), cmap='jet')
                # plt.show(block=True)
                # breakpoint()
            else:
                raster_geom_gdf = None
        if output_rank is None:
            return comm.bcast(raster_geom_gdf, root=0)
        else:
            return raster_geom_gdf

    @staticmethod
    def _process_geom_difference(base_geom, envelope_geometries):
        if envelope_geometries:
            combined_geom = ops.unary_union(envelope_geometries)
            base_geom = base_geom.difference(combined_geom.buffer(-np.finfo(np.float16).eps) or MultiPolygon([]))
        base_geom = ops.unary_union(base_geom)
        return base_geom

    def build_clipped_raster_geom_gdf_mpi(self, comm, output_rank=None, cache_directory=None):
        root_rank = 0 if output_rank is None else output_rank
        raster_geom_gdf = self.build_raster_geom_gdf_mpi(comm, output_rank=output_rank, cache_directory=cache_directory)
        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                raster_geom_envelopes = gpd.GeoDataFrame(
                        geometry=raster_geom_gdf.geometry.envelope,
                        crs=raster_geom_gdf.crs
                        )
                joined = gpd.sjoin(
                        raster_geom_gdf,
                        raster_geom_envelopes,
                        how='left',
                        predicate='intersects',
                        )
                # joined = joined[joined.index != joined.index_right]
                joined.index.name = 'index_left'
                groups = joined.groupby("index_left")
                logger.debug("making differences")
                tasks = []
                valid_rows = joined[joined['index_right'].notna()]
                different_paths = raster_geom_gdf.loc[valid_rows['index_right'], 'raster_path'].values != \
                                  raster_geom_gdf.loc[valid_rows.index, 'raster_path'].values
                lower_indices = valid_rows['index_right'] < valid_rows.index
                valid_rows = valid_rows[different_paths & lower_indices]
                for index_left, group in groups:
                    base_geom = raster_geom_gdf.iloc[[index_left]].geometry.squeeze()
                    filtered_indices = valid_rows.loc[valid_rows.index == index_left, 'index_right']
                    envelope_geometries = raster_geom_envelopes.loc[filtered_indices].geometry.tolist() if not filtered_indices.empty else []
                    tasks.append((base_geom, envelope_geometries))
                # raster_geom_gdf = gpd.GeoDataFrame(geometry=geometries, crs=raster_geom_gdf.crs)
                raster_geom_gdf.geometry = list(executor.starmap(self._process_geom_difference, tasks))
                raster_geom_gdf = raster_geom_gdf[~raster_geom_gdf.geometry.is_empty]
                logger.debug("making differences done")
                # raster_geom_gdf.plot(facecolor='none', cmap='jet')
                # plt.title("after")
                # plt.show(block=True)
                # raise
                # raster_geom_gdf = raster_geom_gdf[raster_geom_gdf.geometry.is_valid]
                # verify
        if output_rank is None:
            return comm.bcast(raster_geom_gdf, root=root_rank)
        return raster_geom_gdf

    def _apply_sieve_to_gdf(self, geom_gdf):
        if geom_gdf is not None:
            if self.sieve is True:
                geom_gdf["area"] = geom_gdf.to_crs("epsg:6933").geometry.area
                geom_gdf = geom_gdf.loc[geom_gdf["area"] == geom_gdf["area"].max()].reset_index(drop=True)
                geom_gdf.drop(columns=['area'], inplace=True)
            elif isinstance(self.sieve, float):
                geom_gdf["area"] = geom_gdf.to_crs("epsg:6933").geometry.area
                geom_gdf = geom_gdf.loc[geom_gdf["area"] < self.sieve].reset_index(drop=True)
                geom_gdf.drop(columns=['area'], inplace=True)
        return geom_gdf

    def build_combined_geoms_gdf_mpi(self, comm, output_rank=None, cache_directory=None):
        """
        Builds combined geoms_gdf using MPI comm.
        """
        root_rank = 0 if output_rank is None else output_rank

        if cache_directory is not None:
            cached_filepath = self._get_final_geom_gdf_feather_path(comm, cache_directory)
            if cached_filepath.is_file():
                if comm.Get_rank() == root_rank:
                    logger.debug("Loading geom_gdf from cache: %s", str(cached_filepath))
                    geom_gdf = gpd.read_feather(cached_filepath)
                else:
                    geom_gdf = None
            else:
                geom_gdf = self._build_final_raster_geom_gdf_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
                if comm.Get_rank() == root_rank and geom_gdf is not None:
                    geom_gdf.to_feather(cached_filepath)
        else:
            geom_gdf = self._build_final_raster_geom_gdf_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)

        if geom_gdf is not None:
            geom_gdf = self._apply_sieve_to_gdf(geom_gdf)

        if output_rank is None:
            return comm.bcast(geom_gdf, root=root_rank)
        return geom_gdf



        # geom_gdf = None
        # if cache_directory is not None:
        #     cached_filepath = self._get_final_geom_gdf_feather_path(comm, cache_directory)
        #     if cached_filepath.is_file():
        #         if comm.Get_rank() == root_rank:
        #             geom_gdf = gpd.read_feather(cached_filepath)
        #         else:
        #             geom_gdf = None

        # final_geom_gdf = self._build_final_raster_geom_gdf_mpi(comm, output_rank=output_rank, cache_directory=cache_directory)
        # should_write_cache = cache_directory and (output_rank is None or comm.Get_rank() == output_rank)
        # if should_write_cache:
        #     final_geom_gdf.to_feather(cached_filepath)




                # if output_rank is None comm.Get_rank() == output_rank:
                #     # Only one rank returns the cached file, others continue with subsequent code
                #     return self._apply_sieve_to_gdf(gpd.read_feather(cached_filepath))
                # else:
                #     return None

        # final_geom_gdf = self._build_final_raster_geom_gdf_mpi(comm, output_rank=output_rank, cache_directory=cache_directory)
        # should_write_cache = cache_directory and (output_rank is None or comm.Get_rank() == output_rank)
        # if should_write_cache:
        #     final_geom_gdf.to_feather(cached_filepath)

        # if final_geom_gdf is not None:
        #     final_geom_gdf = self._apply_sieve_to_gdf(final_geom_gdf)

        # return final_geom_gdf if output_rank is None or comm.Get_rank() == output_rank else None

    @staticmethod
    def _build_raster_geom_gdf(
            raster_opts,
            raster_geom_kwargs,
            get_multipolygon_kwargs,
            ):
        logger.debug(f"Building raster geom for {raster_opts=} {raster_geom_kwargs=}")
        geom_mp = RasterGeom(
                raster=get_raster_from_opts(**raster_opts),
                **raster_geom_kwargs
                ).get_multipolygon(
                **get_multipolygon_kwargs,
                )
        if geom_mp is None:
            geom_mp = MultiPolygon([])
        gdf = gpd.GeoDataFrame([{
            'geometry': geom_mp,
            "raster_path": str(raster_opts.pop("raster_path").resolve()),
            "window": str(raster_opts.pop("window")),
            "clip": json.dumps(raster_opts.pop("clip").model_dump(), default=str) if raster_opts["clip"] is not None else raster_opts.pop("clip"),
            **raster_opts,
            **raster_geom_kwargs,
            }],
            crs=get_multipolygon_kwargs["dst_crs"]
            )
        return gdf


    @staticmethod
    def _get_raster_geom_gdf_feather_path(raster_opts, raster_geom_kwargs, get_multipolygon_kwargs, cache_directory):
        normalized_geom_window_request = {
                "dst_crs": get_multipolygon_kwargs["dst_crs"],
                **raster_opts,
                **raster_geom_kwargs,
                }
        normalized_geom_window_request.pop("crs")
        normalized_geom_window_request.pop("chunk_size")
        raster_geom_cache_dir = Path(cache_directory) / 'raster_geom'
        raster_geom_cache_dir.mkdir(parents=True, exist_ok=True)
        cached_filename = hashlib.sha256(json.dumps(normalized_geom_window_request, default=str).encode('utf-8')).hexdigest() + '.feather'
        return raster_geom_cache_dir / cached_filename

    @classmethod
    def _build_raster_geom_gdf_wrapper(
            cls,
            raster_opts,
            raster_geom_kwargs,
            get_multipolygon_kwargs,
            cache_directory
            ):
        if cache_directory is None:
            return cls._build_raster_geom_gdf(raster_opts, raster_geom_kwargs, get_multipolygon_kwargs)
        cached_filepath = cls._get_raster_geom_gdf_feather_path(raster_opts, raster_geom_kwargs, get_multipolygon_kwargs, cache_directory)
        if cached_filepath.is_file():
            logger.debug(f"loading from cache {cached_filepath}")
            return gpd.read_feather(cached_filepath)
        raster_geom_gdf = cls._build_raster_geom_gdf(raster_opts, raster_geom_kwargs, get_multipolygon_kwargs)
        raster_geom_gdf.to_feather(cached_filepath)
        return raster_geom_gdf

    # @staticmethod
    # def _dask_mpi_core_init_wrapper(
    #         comm,
    #         interface=None,
    #         protocol=None,
    #         dashboard=True,
    #         dashboard_address=None,
    #         nthreads=None,
    #         memory_limit="auto",
    #         local_directory=None,
    #         worker_class="distributed.Worker",
    #         worker_options=None,
    #         exit=False,
    #         ):
    #     logging.getLogger("distributed").setLevel(logging.ERROR)
    #     return dask_mpi.core.initialize(
    #                 interface=interface,
    #                 nthreads=nthreads or 1,
    #                 local_directory=local_directory,
    #                 memory_limit=memory_limit,
    #                 dashboard=dashboard,
    #                 dashboard_address=dashboard_address or ":8787",
    #                 protocol=protocol,
    #                 worker_class=worker_class,
    #                 worker_options=worker_options,
    #                 comm=comm,
    #                 exit=exit,
    #             )

    @staticmethod
    def _polygon_to_buffered_edges(polygon, buffer_distance=1.):
        envelope = polygon.envelope
        exterior = envelope.exterior
        line_segments_buffered = []
        for i in range(len(exterior.coords) - 1):
            line_segment = LineString([exterior.coords[i], exterior.coords[i + 1], exterior.coords[(i + 2) % len(exterior.coords)]])
            buffered_segment = line_segment.buffer(buffer_distance)
            line_segments_buffered.append(buffered_segment)
        return MultiPolygon([line_segment_buffered for line_segment_buffered in line_segments_buffered if line_segment_buffered.is_valid])

    def _build_final_raster_geom_gdf_mpi(self, comm, output_rank=None, cache_directory=None):

        root_rank = 0 if output_rank is None else output_rank
        def split_gdf(gdf):
            exterior_rings_gdf = gpd.GeoDataFrame(
                    geometry=list(gdf.to_crs("epsg:6933").geometry.apply(self._polygon_to_buffered_edges).to_crs(gdf.crs)),
                    crs=gdf.crs)
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
            gdf = gdf[gdf.geometry.is_valid]
            exterior_rings_gdf = exterior_rings_gdf.explode(index_parts=False).reset_index(drop=True)
            exterior_rings_gdf = exterior_rings_gdf[exterior_rings_gdf.geometry.is_valid]
            joined = gpd.sjoin(
                    gdf,
                    exterior_rings_gdf,
                    how='left',
                    predicate='intersects'
                    )
            excluded = gdf.loc[joined[joined.index_right.isna()].index.unique()]
            gdf = gdf.loc[joined[~joined.index_right.isna()].index.unique()]
            return gdf, excluded

        def split_gdf2(gdf):
        #     exterior_rings_gdf = gpd.GeoDataFrame(
        #             geometry=list(gdf.to_crs("epsg:6933").geometry.apply(self._polygon_to_buffered_edges).to_crs(gdf.crs)),
        #             crs=gdf.crs)
        #     gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        #     gdf = gdf[gdf.geometry.is_valid]
        #     exterior_rings_gdf = exterior_rings_gdf.explode(index_parts=False).reset_index(drop=True)
        #     exterior_rings_gdf = exterior_rings_gdf[exterior_rings_gdf.geometry.is_valid]
            gdf = gdf.explode(index_parts=False)
            joined = gpd.sjoin(
                    gdf,
                    gdf,
                    how='left',
                    predicate='intersects'
                    )
            excluded = gdf.loc[joined[joined.index_right.isna()].index.unique()]
            gdf = gdf.loc[joined[~joined.index_right.isna()].index.unique()]
            return gdf, excluded

        gdf = self.build_clipped_raster_geom_gdf_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
        if gdf is not None:
            gdf = gdf.to_crs("epsg:6933")
        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                excluded = []
                start = time()
                logger.info(f"Begin split gdf with {len(gdf)=}")
                gdf, excluded_from_computation = split_gdf(gdf)
                excluded.append(excluded_from_computation)
                del excluded_from_computation
                logger.debug(f"Split took {time()-start}")
                excluded = []
                cnt = 1
                while len(gdf) > 1:
                    gdf = gdf.set_index(_hilbert_distance(gdf, gdf.total_bounds, level=10)).sort_index()
                    npartitions = int(len(gdf) // self.partition_size)
                    logger.info(f"Begin iteration {cnt} with {len(gdf)=} and using {npartitions=}")
                    iteration_start = time()  # Start timing the iteration
                    gdf = gpd.GeoDataFrame(
                            geometry=list(executor.map(
                                partial(unary_union, grid_size=self.grid_size),
                                [list(partition.geometry) for partition in dgpd.from_geopandas(gdf, npartitions=npartitions).partitions],
                                )),
                                crs=gdf.crs
                                )
                    iteration_time = time() - iteration_start
                    logger.info(f"Iteration took {iteration_time}")
                    # logger.info(f"Begin split gdf with {len(gdf)=}")
                    # gdf, excluded_from_computation = split_gdf2(gdf)
                    # excluded.append(excluded_from_computation)
                    # del excluded_from_computation

                    # logger.info(f"Begin split gdf with {len(gdf)=}")
                    # logger.info("Beging final unary_union operation")
                    # start = time()
                    # gdf, excluded_from_computation = split_gdf(gdf)
                    # excluded_from_computation = pd.concat([excluded_from_computation, newly_excluded], ignore_index=True)
                    # all_excluded.append(excluded_from_computation)
                    # del excluded_from_computation
                    # logger.debug(f"Split took {time()-start}")
                    # # Check for exponential growth
                    # is_exponential, eta, next_iter_time = check_exponential_growth(iter_times)
                    # if is_exponential:
                    #     logger.info(f"The time is growing exponentially. Estimated time for next iteration: {timedelta(seconds=next_iter_time)}.")
                    #     logger.info(f"Estimated total time remaining: {timedelta(seconds=eta)}.")
                    #     logger.info(f"Should finish by {datetime.now() + timedelta(seconds=eta)}.")
                    #     break  # Optional: break the loop if exponential growth is detected
                    # else:
                    #     logger.info(f"Iteration {cnt} took {timedelta(seconds=iteration_time)} seconds.")



                    gdf = gdf[~gdf.geometry.is_empty]
                    cnt += 1
                gdf = gdf.explode(index_parts=False)
                    # gdf = pd.concat([gdf.explode(index_parts=False), gpd.read_feather(_tmp_excluded.name)], ignore_index=True)
                # gdf = gdf.set_index(_hilbert_distance(gdf, gdf.total_bounds, level=10)).sort_index()

                # logger.info("Beging final unary_union operation")
                # start = time()
                # gdf, newly_excluded = split_gdf(gdf)
                # excluded_from_computation = pd.concat([excluded_from_computation, newly_excluded], ignore_index=True)
                # del newly_excluded
                # gdf = gdf.set_index(_hilbert_distance(gdf, gdf.total_bounds, level=10)).sort_index()
                # gdf = gpd.GeoDataFrame([{'geometry': [unary_union(list(gdf.geometry), grid_size=grid_size)]}], crs=gdf.crs)
                # # gdf = pd.concat([gdf.explode(index_parts=False), gpd.read_feather(_tmp_excluded.name)], ignore_index=True)
                # gdf = pd.concat([gdf.explode(index_parts=False), pd.concat(excluded, ignore_index=True)], ignore_index=True)
                # gdf = gdf.set_index(_hilbert_distance(gdf, gdf.total_bounds, level=10)).sort_index()
                # gdf = gdf.explode(index_parts=False)
                gdf.reset_index(drop=True, inplace=True)
                gdf.index.name = None
                # gdf = pd.concat([gdf.explode(index_parts=False), pd.concat(excluded, ignore_index=True).explode(index_parts=False)], ignore_index=True)
                gdf = gdf.to_crs("epsg:4326")
                # logger.info(f'Final iteration took: {time()-start}')

        comm.barrier()
        if output_rank is None:
            return comm.bcast(gdf, root=root_rank)
        return gdf

    def _get_final_geom_gdf_feather_path(self, comm, cache_directory):
        if comm.Get_rank() == 0:
            serialized_requests = json.dumps(list(self.iter_normalized_raster_requests()), default=str)
            cached_filename = hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".feather"
            combined_geom_cached_directory = cache_directory / "combined"
            combined_geom_cached_directory.mkdir(parents=True, exist_ok=True)
            combined_filepath = combined_geom_cached_directory / cached_filename
        else:
            combined_filepath = None
        return comm.bcast(combined_filepath, root=0)

    def iter_raster_window_requests(self):
        for raster_config in self.rasters:
            for (i, normalized_path, raster_geom_request), (j, window) in raster_config.iter_raster_window_requests():
                yield (i, normalized_path, raster_geom_request), (j, window)

    def iter_normalized_raster_requests(self):
        for raster_config in self.rasters:
            for normalized_path, normalized_raster_geom_request in raster_config.iter_normalized_raster_requests():
                for normalization_key in self._normalization_keys:
                    normalized_raster_geom_request[normalization_key] = getattr(self, normalization_key)
                yield normalized_path, normalized_raster_geom_request

    def iter_normalized_raster_window_requests(self):
        for k, raster_config in enumerate(self.rasters):
            for (i, normalized_path, normalized_raster_geom_request), (j, window) in raster_config.iter_normalized_raster_window_requests():
                raster_request_opts = dict(
                        raster_path=normalized_path,
                        crs=self.rasters[k].crs,
                        clip=normalized_raster_geom_request.pop("clip"),
                        window=window,
                        chunk_size=self.rasters[k].chunk_size,
                        overlap=self.rasters[k].overlap,
                        resampling_factor=normalized_raster_geom_request.pop("resampling_factor"),
                        resampling_method=normalized_raster_geom_request.pop("resampling_method", None),  # Not implemented in config file.
                        )
                sig = inspect.signature(RasterGeom.__init__)
                param_names = [param.name for param in sig.parameters.values() if param.name != 'self']
                raster_geom_kwargs = {name: normalized_raster_geom_request[name] for name in param_names if name in normalized_raster_geom_request}
                get_multipolygon_kwargs = {
                        'dst_crs': "epsg:4326",
                        'nprocs': 1,
                        }
                yield raster_request_opts, raster_geom_kwargs, get_multipolygon_kwargs


def submit(
        executor,
        config_path,
        to_file=None,
        to_feather=None,
        to_pickle=None,
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
            to_file=to_file,
            to_feather=to_feather,
            to_pickle=to_pickle,
            show=show,
            cache_directory=cache_directory,
            log_level=log_level,
            )

    async def callback():
        await executor.submit(build_cmd, **kwargs)
        geom = pickle.load(open(to_pickle, 'rb'))
        if delete_pickle:
            to_pickle.unlink()
        return geom

    return callback()


def get_cmd(
        config_path,
        to_file=None,
        to_feather=None,
        to_pickle=None,
        show=False,
        cache_directory=None,
        log_level: str = None,
        ):
    build_cmd = [
            sys.executable,
            f'{Path(__file__).resolve()}',
            f'{Path(config_path).resolve()}',
            ]
    if to_file is not None:
        build_cmd.append(f'--to-file={to_file}')
    if to_feather is not None:
        build_cmd.append(f'--to-feather={to_feather}')
    if to_pickle is not None:
        build_cmd.append(f'--to-pickle={to_pickle}')
    if show:
        build_cmd.append('--show')
    if cache_directory is not None:
        build_cmd.append(f'--cache-directory={cache_directory}')
    if log_level is not None:
        build_cmd.append(f'--log-level={log_level}')
    return build_cmd

def plot_geom(gdf):
    from geomesh.geom.shapely_geom import MultiPolygonGeom
    logger.info('Generating plot')
    geom = MultiPolygonGeom(MultiPolygon(list(gdf.geometry)), crs=gdf.crs)
    geom.make_plot()
    plt.gca().axis('scaled')
    plt.show(block=True)
    return


def write_feather(path, gdf):
    logger.info(f'Writting feather to {path}.')
    gdf.to_feather(path)
    logger.info(f'Done file to {path}.')


def write_file(path, gdf):
    logger.info(f'Writting file to {path}.')
    gdf.to_file(path)

def write_pickle(path, gdf):
    logger.info(f'Writting file to {path}.')
    pickle.dump(Geom(gdf.iloc[0].geometry, crs=gdf.crs), open(path, 'wb'))
    logger.info(f'Done file to {path}.')

# def write_msh(gdf, path):
#     logger.info(f'Writting feather to {path}.')
#     gdf.to_feather(path)

# def write_vtk(gdf):
#     pass


def entrypoint(args: argparse.Namespace, comm=None):

    comm = comm or MPI.COMM_WORLD

    # geom_config = get_geom_config_from_args(comm, args)
    geom_config = GeomConfig.try_from_yaml_path(args.config)

    gdf = geom_config.build_combined_geoms_gdf_mpi(
            comm,
            output_rank=0,
            cache_directory=args.cache_directory,
            )

    finalization_tasks = []

    if args.to_feather:
        finalization_tasks.append(partial(write_feather, args.to_feather))

    if args.to_file:
        finalization_tasks.append(partial(write_file, args.to_file))

    if args.to_pickle:
        finalization_tasks.append(partial(write_pickle, args.to_pickle))

    # if args.to_msh:
    #     finalization_tasks.append(partial(write_msh, args.to_msh))

    # if args.to_vtk:
    #     finalization_tasks.append(partial(write_vtk, args.to_vtk))

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            # for task in finalization_tasks:
            #     task(gdf)
            results = [future.result() for future in [executor.submit(task, gdf) for task in finalization_tasks]]
            if args.show:
                plot_geom(gdf)

        # else:
            # results = None

def get_argument_parser():

    def cache_directory_bootstrap(path_str):
        path = Path(path_str)
        if not path.name == "geom_build":
            path /= "geom_build"
        return path

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--to-file', type=Path)
    parser.add_argument('--to-feather', type=Path)
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=cache_directory_bootstrap)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    # sieve_options = parser.add_argument_group('sieve_options').add_mutually_exclusive_group()
    # sieve_options.add_argument('--sieve')
    # sieve_options.add_argument_group('--sieve')
    return parser


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


def main():
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
    entrypoint(args, comm=comm)


if __name__ == "__main__":
    main()


def test_main():
    from unittest.mock import patch
    yaml_str = """
mesh_bbox: &mesh_bbox
    xmin: -98.00556
    ymin: 8.534422
    xmax: -60.040005
    ymax: 45.831431
    crs: 'epsg:4326'


rasters:
    - &GEBCO_2021_sub_ice_topo_1
      path: '../../data/GEBCO_2021_sub_ice_topo/gebco_2021_sub_ice_topo_n90.0_s0.0_w-180.0_e-90.0.tif'
      bbox: *mesh_bbox
      chunk_size: 1622
    - &GEBCO_2021_sub_ice_topo_2
      path: '../../data/GEBCO_2021_sub_ice_topo/gebco_2021_sub_ice_topo_n90.0_s0.0_w-90.0_e0.0.tif'
      bbox: *mesh_bbox
      chunk_size: 1622
    - &PRUSVI2019
      path: '../../data/prusvi_19_topobathy_2019/*.tif'
    - &NCEITileIndex
      tile_index: '../../data/tileindex_NCEI_ninth_Topobathy_2014.zip'
      bbox: *mesh_bbox

geom:
    zmax: &zmax 10.
    rasters:
      - <<: *PRUSVI2019
        zmax: *zmax
        resampling_factor: 0.2 # 50 meters
      - <<: *NCEITileIndex
        zmax: *zmax
        resampling_factor: 0.2 # 50 meters
      - <<: *GEBCO_2021_sub_ice_topo_1
        clip: '../../data/floodplain_patch.gpkg'
        zmin: 0.
        zmax: *zmax
        overlap: 5
      - <<: *GEBCO_2021_sub_ice_topo_2
        clip: '../../data/floodplain_patch.gpkg'
        zmin: 0.
        zmax: *zmax
        overlap: 5
      - <<: *GEBCO_2021_sub_ice_topo_1
        zmax: 0.
        overlap: 5
      - <<: *GEBCO_2021_sub_ice_topo_2
        zmax: 0.
        overlap: 5
    sieve: True
"""
    import tempfile
    tmp_yaml = tempfile.NamedTemporaryFile()
    open(tmp_yaml.name, 'w').write(yaml_str)
    with patch.object(
            sys,
            'argv',
            [
                __file__,
                tmp_yaml.name,
                '--cache-directory=./.cache/geom_build',
                '--show',
                # '--log-level=debug',
                # '/sciclone/pscr/jrcalzada/thesis/runs/tropicald-validations/hindcasts/Harvey2017/.cache/geom_build/1b075beedc11b6aa3c45a246dc3d8db345ce827484d9fd9d451f67689d8157b8/config.yaml'
            ]
            ):
        main()

