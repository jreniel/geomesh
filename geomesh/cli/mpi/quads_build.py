#!/usr/bin/env python
from pathlib import Path
from time import time
from time import sleep
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import argparse
import hashlib
import inspect
import json
import logging
import os
import sys
import tempfile
from geomesh.mesh.mesh import EuclideanMesh2D

from matplotlib.tri import Triangulation
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator
from pyproj import CRS
from shapely.geometry import Polygon, LinearRing
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from geomesh.cli.mpi import lib
from geomesh.cli.raster_opts import iter_raster_window_requests, get_raster_from_opts
from geomesh.geom.quadgen import Quads, logger as quadgen_logger
from geomesh.geom.quadgen import generate_quad_gdf_from_mp, check_conforming
from geomesh.mesh import Mesh


logger = logging.getLogger(__name__)


def get_cmd(
        config_path,
        # to_msh=None,
        # to_pickle=None,
        to_feather=None,
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
    if to_feather is not None:
        build_cmd.append(f'--to-feather={to_feather}')
    # if to_msh is not None:
    #     build_cmd.append(f'--to-file={to_msh}')
    # if to_pickle is not None:
    #     build_cmd.append(f'--to-pickle={to_pickle}')
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
    quadgen_logger.setLevel(log_level)
    # if int(log_level) < 40:
    #     logging.getLogger("geomesh").setLevel(log_level)
    # logging.Formatter.converter = lambda *args: datetime.now(tz=pytz.timezone("UTC")).timetuple()
    logging.captureWarnings(True)


def get_argument_parser():

    def cache_directory_bootstrap(path_str):
        path = Path(path_str)
        if not path.name == "quads_build":
            path /= "quads_build"
        return path


    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--to-feather', type=Path)
    parser.add_argument('--dst-crs', '--dst_crs', type=CRS.from_user_input, default=CRS.from_epsg(4326))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=cache_directory_bootstrap)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


# def get_quads_config_from_args(comm, args):
#     rank = comm.Get_rank()
#     if rank == 0:
#         logger.info('Validating user raster quad requests...')
#         quads_config = BuildCli(args).config.quads.quads_config
#         if quads_config is None:
#             raise RuntimeError(f'No quads to process in {args.config}')
#     else:
#         quads_config = None

#     return comm.bcast(quads_config, root=0)


def get_quads_raster_window_requests(comm, quads_config):
    if comm.Get_rank() == 0:
        logger.info('Loading quads raster window requests...')
        quads_raster_window_requests = list(iter_raster_window_requests(quads_config))
        logger.debug('Done loading quads raster window requests.')
    else:
        quads_raster_window_requests = None
    quads_raster_window_requests = comm.bcast(quads_raster_window_requests, root=0)
    return quads_raster_window_requests


# import threading


# class TimeoutException(Exception):
#     pass


# def function_to_run_with_timeout(raster, normalized_quads_opts, tmpfile):
#     Quads.from_raster(raster, **normalized_quads_opts).quads_gdf.to_feather(tmpfile)
#     print("Function completed", flush=True)


# def run_with_timeout(func, timeout, *args, **kwargs):
#     thread = threading.Thread(target=func, args=args, kwargs=kwargs)
#     thread.start()
#     thread.join(timeout)
#     if thread.is_alive():
#         thread.join()  # Optional: you might want to let the thread finish in the background
#         raise TimeoutException("Function exceeded the timeout limit")


def get_quad_feather(args):
    # normalized_quads_window_request, cache_directory = args
    raster_path, quads_opts, window, raster_hash, cache_directory = args
    # normalized_raster_opts = lib.get_normalized_raster_request_opts(raster_hash, quads_opts, window)


    def get_default_kwargs_from_signature(sig):
        return {key: param.default for key, param in sig.parameters.items() if param.default != inspect.Parameter.empty}

    # Get default kwargs from both functions
    sig1_kwargs = get_default_kwargs_from_signature(inspect.signature(generate_quad_gdf_from_mp))
    sig2_kwargs = get_default_kwargs_from_signature(inspect.signature(Quads.from_raster))

    # Merge both default kwargs
    from_raster_kwargs = {**sig1_kwargs, **sig2_kwargs}
    # technically max_quad_length is required for the function call, but we make it optional
    # for the user, using a deafult of 500 units (hopefully meters).
    from_raster_kwargs.setdefault('max_quad_length', 500.)
    # adjust default nprocs to play better with MPI
    from multiprocessing import cpu_count
    from_raster_kwargs['nprocs'] = cpu_count()
    # Override merged defaults with quads_opts values where they exist
    for key, value in quads_opts.items():
        if key in from_raster_kwargs:
            from_raster_kwargs[key] = value

    tmpfile = cache_directory / (hashlib.sha256(
        json.dumps(
            {
            **lib.get_normalized_raster_request_opts(raster_hash, quads_opts, window),
            **from_raster_kwargs,
            },
            ).encode()).hexdigest() + '.feather')

    if not tmpfile.exists():

        from time import time
        logger.info(f'rank={MPI.COMM_WORLD.Get_rank()} start generating quads for {raster_path=}...')
        start = time()
        raster = get_raster_from_opts(raster_path, quads_opts, window)
        # try:
        #     run_with_timeout(function_to_run_with_timeout, 300, raster, normalized_quads_opts, tmpfile)
        # except TimeoutException:
        #     print(f'{raster_path=} timeout', flush=True)
        #     raise
        quads = Quads.from_raster(
                    raster,
                    **from_raster_kwargs,
                    )
        logger.debug(f'rank={MPI.COMM_WORLD.Get_rank()} took {time()-start} to generate quads')
        quads.quads_gdf.to_feather(tmpfile)
    else:
        logger.debug(f'{tmpfile=} exists...')
    return tmpfile


def get_uncombined_quads_from_raster_requests(comm, quads_config, cache_directory):
    problematic_rasters = [
        'FL/ncei19_n25x50_w080x50_2016v1',
        'FL/ncei19_n25x50_w081x00_2022v1',
        'FL/ncei19_n25x25_w081x00_2022v2',
        'LA_MS/ncei19_n29x75_w090x00_2020v1',
        'LA_MS/ncei19_n29x75_w090x25_2020v1',
    ]
    results = []
    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:
            logger.info('will build quads from rasters')
            cache_directory /= 'window_quads'
            cache_directory.mkdir(exist_ok=True)
            quads_raster_window_requests = get_quads_raster_window_requests(comm, quads_config)
            raster_hashes = []
            for (i, raster_path, request_opts), (j, window) in quads_raster_window_requests:
                raster_hashes.append(f'{Path(raster_path).resolve()}')
            job_args = []
            for raster_hash, ((i, raster_path, quads_opts), (j, window)) in zip(raster_hashes, quads_raster_window_requests):
                # TODO: Hardcoding problematic rasters to skip:
                if any(problematic_raster in str(raster_path) for problematic_raster in problematic_rasters):
                    continue
                job_args.append((raster_path, quads_opts, window, raster_hash, cache_directory))
            # Dumping job_args to JSON file
            results = list(executor.map(get_quad_feather, job_args))
    results = comm.bcast(results, root=0)
    if comm.Get_rank() == 0:
        logger.info(f'built {len(results)} feathers')
    return results


def get_uncombined_quads_from_triplets(comm, quads_config):
    triplets_requests = quads_config['triplets']
    out_feathers = []
    for i, quad_opts in enumerate(triplets_requests):
        quad_opts = quad_opts.copy()
        fname = quad_opts.pop('path')
        out_feathers.append(Quads.from_triplets_mpi(comm, fname, **quad_opts))
    return out_feathers


def get_uncombined_quads_from_banks(comm, quads_config, cache_directory):
    banks_requests = quads_config['banks']
    out_feathers = []
    for i, quad_opts in enumerate(banks_requests):
        quad_opts = quad_opts.copy()
        fname = Path(quad_opts.pop('path'))
        quads = Quads.from_banks_file_mpi(
                    comm,
                    fname,
                    rank=0,  # all other ranks return None
                    cache_directory=cache_directory,
                    **quad_opts
                    )
        fname_str = str(fname.resolve()).replace('/', '_').replace('.', '_')  # Make it filename-safe
        # Convert kwargs to a string

        cache_directory = cache_directory / 'uncombined_quads_from_banks_requests'
        cache_directory.mkdir(exist_ok=True, parents=True)
        kwargs_str = '_'.join(f'{key}={value}' for key, value in sorted(quad_opts.items()))
        # Create explicit, unique filename
        unique_filename = hashlib.sha256(f"{fname_str}__{kwargs_str}".encode()).hexdigest()
        out_feather = cache_directory / f'{unique_filename}.feather'
        if comm.Get_rank() == 0:
            quads.quads_gdf.to_feather(out_feather)
        out_feathers.append(out_feather)
    return out_feathers

def get_uncombined_quads_from_hgrid(comm, quads_config, cache_directory):
    pass
    raise


def get_uncombined_quads(comm, quads_config, cache_directory):

    quads_gen_funcs = {
        'rasters': get_uncombined_quads_from_raster_requests,
        'triplets': get_uncombined_quads_from_triplets,
        'banks': get_uncombined_quads_from_banks,
        'from_hgrid': get_uncombined_quads_from_hgrid,
        }
    uncombined_quads_feathers = []
    for key in quads_config:
        if key not in quads_gen_funcs:
            continue
        uncombined_quads_feathers.extend(quads_gen_funcs[key](comm, quads_config, cache_directory))

    return uncombined_quads_feathers

def concat_multiple_quads_gdf(gdfs):
    # Start with an offset of 0
    # offset = 0
    # concatenated_gdfs = []
    for file_index, gdf in enumerate(gdfs):
        if len(gdf) == 0:
            continue
        # force local sequentialism
        # gdf['quad_group_id'] = gdf.quad_group_id.astype(int)
        # unique_ids = sorted(gdf['quad_group_id'].unique())
        # id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
        # gdf['quad_group_id'] = gdf['quad_group_id'].map(id_mapping)
        gdf['file_index'] = file_index
        # gdf['quad_group_id'] += offset
        # print(gdf['quad_group_id'])
        # Update the offset for the next GeoDataFrame
        # print(offset, flush=True)
        # offset += gdf['quad_group_id'].max() + 1
        # print(offset, flush=True)
        # Add the updated gdf to the list
        # concatenated_gdfs.append(gdf)

    # Concatenate all updated gdfs
    return pd.concat(gdfs, ignore_index=True)

def load_the_quads_gdf_from_cache(comm, files_in_window_cache_dir):
    gdf = None
    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:
            gdf_list = list(executor.map(gpd.read_feather, files_in_window_cache_dir))
            gdf = concat_multiple_quads_gdf(gdf_list)
    return gdf

def check_conforming_wrapper(args):
    lr_left, lr_right = args
    return check_conforming(Polygon(lr_left.coords), Polygon(lr_right.coords))


# def get_overlaps(index, tmpfeather):
#     # Checking for overlaps with the rest of the dataset
#     gdf = gpd.read_feather(tmpfeather)
#     sindex = quads_gdf.sindex
#     polygon = gdf.at[index, 'geometry']
#     potential_overlaps = gdf[gdf.geometry.overlaps(polygon)]
#     # Return the pairs where the index is not equal to the other index
#     return potential_overlaps[potential_overlaps.index != index].index.tolist()

def get_overlaps(index, feather_path):
    # Load the GeoDataFrame and its spatial index
    gdf = gpd.read_feather(feather_path)
    polygon = gdf.loc[index, 'geometry']
    # Use the spatial index to narrow down the candidates
    possible_matches_index = list(gdf.sindex.intersection(polygon.bounds))
    possible_matches = gdf.iloc[possible_matches_index]

    # Check for actual overlaps
    actual_overlaps = possible_matches[possible_matches.geometry.overlaps(polygon)]
    overlaps = actual_overlaps.index.tolist()

    # Remove self index if present
    if index in overlaps:
        overlaps.remove(index)

    return index, overlaps


def cleanup_overlapping_pairs_mut_parallel(quads_gdf, executor):
    quads_gdf['geometry'] = quads_gdf['geometry'].apply(lambda x: Polygon(x))
    sindex = quads_gdf.sindex
    import tempfile
    # tmpdir = tempfile.TemporaryDirectory()
    tmpfeather = Path.cwd() / 'tmp_quads_gdf.feather'
    quads_gdf.to_feather(tmpfeather)
    # tmpsindex = tmpfeather.parent / 'sindex.pickle'
    # pickle.dump(sindex, open(tmpsindex, 'wb'))
    tasks = [(index, tmpfeather) for index in quads_gdf.index]
    logger.info("start the parallel part...")
    results = list(executor.starmap(get_overlaps, tasks))
    logger.info("Done!")

    # Create overlap_dict from results
    logger.info("creating the dict")
    overlap_dict = {index: set(overlaps) for index, overlaps in results if overlaps}
    logger.info("done!")

    logger.info("now the serial part...")
    # Sequentially process the overlap_dict
    while overlap_dict:
        max_area_idx = max(overlap_dict.keys(), key=lambda x: quads_gdf.at[x, 'geometry'].area)
        quads_gdf.drop(max_area_idx, inplace=True)
        for idx in list(overlap_dict[max_area_idx]):
            overlap_dict[idx].discard(max_area_idx)
            if not overlap_dict[idx]:
                del overlap_dict[idx]
        del overlap_dict[max_area_idx]

    logger.info("now the serial part is done!", flush=True)
    quads_gdf['geometry'] = quads_gdf['geometry'].apply(lambda x: LinearRing(x.exterior.coords))

    return quads_gdf

def cleanup_quads_gdf_mut_local(quads_gdf, executor):
    from time import time
    start = time()
    logger.info("Start cleanup of overlapping pairs.")
    cleanup_overlapping_pairs_mut_parallel(quads_gdf, executor)
    logger.debug(f"cleanup overlapping pairs took: {time()-start}")
    logger.info("Start cleanup of slightly touching pairs.")
    start = time()
    from geomesh.geom.quadgen import cleanup_touches_with_eps_tolerance_mut
    cleanup_touches_with_eps_tolerance_mut(quads_gdf)
    logger.debug(f"cleanup of slightly touching pairs took: {time()-start}")


def get_combined_quads_gdf(comm, uncombined_quads_paths):
    quads_gdf = load_the_quads_gdf_from_cache(comm, uncombined_quads_paths)
    # with MPICommExecutor(comm) as executor:
    #     if executor is not None:
    #         uncombined_quads_gdf.plot(ax=plt.gca())
    #         plt.show(block=True)
    with MPICommExecutor(comm) as executor:
        if executor is not None:
            print("do first sjoin", flush=True)
            quads_gdf['geometry'] = quads_gdf['geometry'].apply(Polygon)
            joined = gpd.sjoin(
                    quads_gdf,
                    quads_gdf,
                    how='inner',
                    predicate='contains'
                    )
            joined = joined[joined.index != joined["index_right"]]
            quads_gdf.drop(index=joined.index.unique(), inplace=True)
            quads_gdf.reset_index(inplace=True, drop=True)

            print("do second sjoin", flush=True)
            quads_gdf['geometry'] = quads_gdf['geometry'].apply(lambda x: LinearRing(x.exterior.coords))
            joined = gpd.sjoin(
                    quads_gdf,
                    quads_gdf,
                    how='inner',
                    predicate='intersects'
                    )
            joined = joined[joined.index != joined["index_right"]]
            joined['sorted_index_pair'] = joined.apply(lambda row: tuple(sorted([row.name, row['index_right']])), axis=1)
            joined = joined.drop_duplicates(subset='sorted_index_pair')
            joined = joined.drop(columns=['sorted_index_pair'])
            def make_args_list(row):
                return row.geometry, quads_gdf.loc[row.index_right].geometry
            joined["is_conforming"] = list(executor.map(check_conforming_wrapper, joined.apply(make_args_list, axis=1)))


            # quads_gdf = gpd.read_feather("this_quads_gdf.feather")
            # joined = gpd.read_feather("this_joined.feather")
            non_conforming_indexes = joined[joined['is_conforming'] == False].index.unique().union(joined[joined['is_conforming'] == False].index_right.unique())
            quads_non_conforming = quads_gdf.loc[non_conforming_indexes]
            joined2 = gpd.sjoin(
                    quads_non_conforming,
                    quads_non_conforming,
                    how='inner',
                    predicate='intersects'
                    )
            joined2 = joined2[joined2.index != joined2.index_right]
            # Create a graph
            import networkx as nx
            G = nx.Graph()

            # Add edges based on sjoin results. Nodes will be automatically created.
            for index, row in joined2.iterrows():
                G.add_edge(index, row.index_right)

            # Find connected components - each component is a set of indices that are connected through intersections
            components = list(nx.connected_components(G))

            # Create a mapping from index to group
            index_to_group = {}
            for group_id, component in enumerate(components):
                for index in component:
                    index_to_group[index] = group_id

            # Map the group ID to each polygon in the original GeoDataFrame
            quads_non_conforming['group'] = quads_non_conforming.index.map(index_to_group)
            groups = quads_non_conforming.groupby("group")
            from multiprocessing import Pool, cpu_count
            with Pool(cpu_count()) as pool:
                results = pool.map(process_group, groups)
            indexes_to_drop = []
            for _, idxs in results:
                indexes_to_drop.extend(idxs)
            quads_gdf = quads_gdf.drop(index=list(set(indexes_to_drop)))
    return quads_gdf


def process_group(group_data):
    name, quads_gdf = group_data
    indices_to_drop = get_indices_to_drop(quads_gdf)
    return name, indices_to_drop

def get_indices_to_drop(quads_gdf):
    quads_gdf = quads_gdf.copy()
    quads_gdf['geometry'] = quads_gdf['geometry'].apply(Polygon)
    def get_overlapping_pairs(gdf):
        overlaps = gpd.sjoin(gdf, gdf, how='inner', predicate='overlaps')
        return overlaps[overlaps.index != overlaps.index_right]

    overlapping_pairs = get_overlapping_pairs(quads_gdf)

    overlap_dict = {}
    for idx, row in overlapping_pairs.iterrows():
        overlap_dict.setdefault(idx, set()).add(row['index_right'])
        overlap_dict.setdefault(row['index_right'], set()).add(idx)

    indices_to_drop = []
    while overlap_dict:
        max_area_idx = max(overlap_dict.keys(), key=lambda x: quads_gdf.at[x, 'geometry'].area)
        indices_to_drop.append(max_area_idx)
        for idx in overlap_dict[max_area_idx]:
            overlap_dict[idx].discard(max_area_idx)
            if not overlap_dict[idx]:
                del overlap_dict[idx]
        del overlap_dict[max_area_idx]
    return indices_to_drop


class QuadsRasterConfig(lib.RasterConfig, lib.IterableRasterWindowTrait):
    max_quad_length: float
    zmin: Optional[float] = None
    zmax: Optional[float] = None
    min_cross_section_node_count: Optional[int] = 4
    min_quad_length: Optional[float] = None
    min_quad_width: Optional[float] = None
    max_quad_width: Optional[float] = None
    shrinkage_factor: Optional[float] = 0.9
    cross_distance_factor: Optional[float] = 0.9
    min_branch_length: Optional[float] = None
    threshold_size: Optional[float] = None
    resample_distance: Optional[float] = None
    simplify_tolerance: Optional[float] = None
    interpolation_distance: Optional[float] = None
    min_area_to_length_ratio: Optional[float] = 0.1
    min_area: Optional[float] = float(np.finfo(np.float32).eps)
    min_quads_per_group: Optional[int] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    # nprocs=cpu_count(),
    _normalization_keys = [
            "resampling_factor",
            "clip",
            "bbox",
            ]

    @field_validator('max_quad_length')
    @classmethod
    def max_quad_length_validator(cls, v: float) -> float:
        if v is not None:
            assert v > 0., "max_quad_length must be > 0."
        return v

    @field_validator('min_cross_section_node_count')
    @classmethod
    def min_cross_section_node_count_validator(cls, v: int) -> int:
        if v is not None:
            assert v > 1, "min_cross_section_node_count must be > 1"
        return v

    @field_validator('min_quads_per_group')
    @classmethod
    def min_quads_per_group_validator(cls, v: int) -> int:
        if v is not None:
            assert v >= 0, "min_quads_per_group must be >= 0"
        return v

    def iter_normalized_raster_requests(self):
        # raster_quads_requests = self.model_dump()
        for normalized_path, normalized_raster_quads_request in super().iter_normalized_raster_requests():
            for normalization_key in self._normalization_keys:
                normalized_raster_quads_request[normalization_key] = getattr(self, normalization_key)
            yield normalized_path, normalized_raster_quads_request

    def iter_normalized_raster_window_requests(self):
        for (i, normalized_path, normalized_raster_quads_request), (j, window) in super().iter_normalized_raster_window_requests():
            for normalization_key in self._normalization_keys:
                normalized_raster_quads_request[normalization_key] = getattr(self, normalization_key)
            yield (i, normalized_path, normalized_raster_quads_request), (j, window)

class QuadsFromHgridConfig(BaseModel):
    path: Path
    use_split_quads: Optional[bool] = True
    use_quads: Optional[bool] = True

    def gdf(self) -> gpd.GeoDataFrame:
        if not hasattr(self, "_gdf"):
            hgrid = Mesh.open(self.path)
            geometry = self._extract_quads(hgrid)
            self._gdf = gpd.GeoDataFrame(geometry=geometry, crs=hgrid.crs)
        return self._gdf

    def _extract_quads(self, mesh: EuclideanMesh2D) -> List[LinearRing]:
        geometry = []
        if self.use_quads:
            vert2 = mesh.msh_t.vert2["coord"]
            quad4 = mesh.msh_t.quad4["index"]
            geometry.extend([Polygon(quad) for quad in vert2[quad4]])
        if self.use_split_quads:
            geometry.extend(self._compute_split_quads(mesh))
        return geometry

    @staticmethod
    def _is_quad(angles1, angles2, threshold_angle_deg) -> bool:
        # Check if all angles are close to 90 degrees within the threshold
        return np.all(np.abs(angles1 - 90) <= np.abs(threshold_angle_deg)) and \
               np.all(np.abs(angles2 - 90) <= np.abs(threshold_angle_deg))

    def _compute_split_quads(self, mesh: EuclideanMesh2D) -> List[LinearRing]:
        vert2 = mesh.msh_t.vert2["coord"]
        tria3 = mesh.msh_t.tria3["index"]
        # interior_angles = np.rad2deg(self._compute_interior_angles(vert2, tria3))
        # abs_diff = np.abs(interior_angles - 90)
        # threshold = 20.
        # bool_list = np.any(abs_diff <= threshold, axis=1)
        gdf = gpd.GeoDataFrame(geometry=[Polygon(tria) for tria in vert2[tria3]], crs=mesh.crs)

        # def angle_between(v1, v2):
        #     from numpy.linalg import norm
        #     dot_product = np.dot(v1, v2)
        #     magnitude_product = norm(v1) * norm(v2)
        #     # Check if magnitudes are zero to avoid division by zero error
        #     if magnitude_product == 0:
        #         return 0
        #     else:
        #         # Clip the value to avoid RuntimeWarning in arccos
        #         clipped_value = np.clip(dot_product / magnitude_product, -1.0, 1.0)
        #         angle = np.arccos(clipped_value)
        #         return np.degrees(angle)

        # def compute_interior_angles(linear_ring):

        #     coords = list(linear_ring.coords[:-1])
        #     n = len(coords)
        #     angles = []
        #     for i in range(n):
        #         prev_point = np.array(coords[(i-2) % n])
        #         curr_point = np.array(coords[(i-1) % n])
        #         next_point = np.array(coords[i])
        #         prev_vector = curr_point - prev_point
        #         next_vector = next_point - curr_point
        #         angle = angle_between(prev_vector, next_vector)
        #         angles.append(angle)
        #     return angles

        # def has_angle_near_90_degrees(angles, threshold=5):
        #     for angle in angles:
        #         if 90 - threshold <= angle <= 90 + threshold:
        #             return True
        #     return False


        from shapely.geometry import LineString
        gdf = gpd.GeoDataFrame(geometry=[LinearRing(tria) for tria in vert2[tria3]], crs=mesh.crs)
        def guess_if_its_a_half_quad(linestring: LineString, threshold) -> bool:
            segments = list(linestring.coords)
            if len(segments) != 4:  # A LineString with 3 segments will have 4 coordinates
                return False

            lengths = [LineString(segments[i:i+2]).length for i in range(len(segments)-1)]
            lengths.sort()
            # a >= threshold * b

            # return lengths[2] > lengths[1] > lengths[0]
            return lengths[2] >= threshold*lengths[0]

        # gdf["possibly_a_half_quad"] = gdf.geometry.apply(guess_if_its_a_half_quad)

        # gdf["angles"] = gdf.geometry.apply(compute_interior_angles)
        # gdf["lengths

        # Define the threshold
        threshold = 5

        # Apply the filter condition
        # gdf["possibly_a_half_quad"] = gdf["angles"].apply(lambda angles: has_angle_near_90_degrees(angles, threshold))
        # gdf["angle_sums"] = gdf["angles"].apply(lambda angles: np.sum(angles))
        # filtered_gdf = gdf[gdf["possibly_a_half_quad"]]
        # gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='k', linewidth=0.3)
        filtered_gdf = gdf[gdf.geometry.apply(guess_if_its_a_half_quad, threshold=6.)]
        # filtered_gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='r')
        # plt.show(block=False)

        # joined = gpd.sjoin(gdf, gdf, how="inner", predicate="intersects")
        # joined[""]

        # breakpoint()

        # neighbors = Triangulation(vert2[:, 0], vert2[:, 1], tria3).neighbors
        # from collections import defaultdict
        # Dictionary to store edges and corresponding triangles
        # edge_to_tri = defaultdict(list)

        # Populate the dictionary with edges and their corresponding triangles
        # for i, tris in enumerate(tria3):
        #     for j in range(3):
        #         edge = tuple(sorted([tris[j], tris[(j + 1) % 3]]))
        #         edge_to_tri[edge].append(i)

        # quads = []
        # # Iterate through each triangle and its neighbors
        # for i, tris in enumerate(tria3):
        #     for neighbor in neighbors[i]:
        #         if neighbor != -1 and neighbor > i:  # Ensure each pair is considered once
        #             common_edge = np.intersect1d(tris, tria3[neighbor])
        #             if len(common_edge) == 2:
        #                 other_vertices = [np.setdiff1d(tris, common_edge)[0], np.setdiff1d(tria3[neighbor], common_edge)[0]]
        #                 angles1 = interior_angles[i][np.isin(tris, common_edge)]
        #                 angles2 = interior_angles[neighbor][np.isin(tria3[neighbor], common_edge)]

        #                 if self._is_quad(angles1, angles2, threshold_angle_deg=20.):
        #                     quad = vert2[np.concatenate((common_edge, other_vertices))]
        #                     quads.append(LinearRing(quad))


        return [Polygon(geom) for geom in filtered_gdf.geometry]




        # geometry = [Polygon(tria) for tria in tria3[vert2]]
        # gdf = gpd.GeoDataFrame(geometry=geometry, crs=mesh.crs)

    @staticmethod
    def _compute_interior_angles(vertices, triangles):
        # vertices: array of vertex coordinates, shape (n_vertices, 2)
        # triangles: array of triangle indices, shape (n_triangles, 3)

        def angle_between_vectors(v1, v2):
            # Compute the angle between two vectors in radians
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clip for numerical stability
            return np.arccos(cos_theta)

        interior_angles = np.zeros((triangles.shape[0], 3))

        for i, tri in enumerate(triangles):
            # Get the coordinates of the vertices of the triangle
            p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

            # Compute the edge vectors
            v0 = p1 - p0
            v1 = p2 - p1
            v2 = p0 - p2

            # Calculate the angles
            angle0 = angle_between_vectors(-v2, v0)
            angle1 = angle_between_vectors(-v0, v1)
            angle2 = angle_between_vectors(-v1, v2)

            # Store the angles
            interior_angles[i] = [angle0, angle1, angle2]

        return interior_angles



class QuadsConfig(BaseModel):
    rasters: Optional[List[QuadsRasterConfig]] = None
    from_hgrid: Optional[Union[Path, QuadsFromHgridConfig]] = None
    # geometry: Optional[Path] = None
    min_quads_per_group: Optional[int] = None
    _cached_raster_windows = []

    # @model_validator(mode="before")
    # def _validate_before(cls, data: Any) -> Any:

    #     # if isinstance(data, dict):
    #     #     assert (
    #     #         'card_number' not in data
    #     #     ), 'card_number should not be included'
    #     return data

    @model_validator(mode="after")
    def _validate_model(self):
        requirement_count = sum([
            self.rasters is not None, # has_rasters
            self.from_hgrid is not None,
            ])
        if requirement_count > 1:
            raise ValueError("rasters and geometry are mutually exclusive")
        if requirement_count == 0:
            raise ValueError("at least one of rasters or geometry is required")

        if isinstance(self.from_hgrid, Path):
            self.from_hgrid = QuadsFromHgridConfig(path=self.from_hgrid, use_split_quads=True, use_quads=True)
        return self

    @field_validator('min_quads_per_group')
    @classmethod
    def min_quads_per_group_validator(cls, v: int) -> int:
        if v is not None:
            assert v >= 0, "min_quads_per_group must be >= 0"
        return v

    @classmethod
    def try_from_yaml_path(cls, path: Path) -> "QuadsConfig":
        with open(path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return cls.try_from_dict(data)

    @classmethod
    def try_from_dict(cls, data: dict) -> "QuadsConfig":
        return cls(**data['quads'])


    def iter_normalized_raster_window_requests(self):
        k = 0
        if self.rasters is not None:
            for g, raster_config in enumerate(self.rasters):
                for (i, raster_path, normalized_raster_quads_request), (j, window) in raster_config.iter_normalized_raster_window_requests():
                    # for normalization_key in self._normalization_keys:
                    #     normalized_raster_quads_request[normalization_key] = getattr(self, normalization_key)
                    # print(normalized_raster_quads_request, flush=True)
                    raster_request_opts = dict(
                            # raster_path=normalized_path,
                            crs=self.rasters[g].crs,
                            clip=normalized_raster_quads_request.pop("clip"),
                            # bbox=self.rasters[k].bbox,
                            # window=window,
                            chunk_size=self.rasters[g].chunk_size,
                            overlap=self.rasters[g].overlap,
                            resampling_factor=normalized_raster_quads_request.pop("resampling_factor"),
                            resampling_method=normalized_raster_quads_request.pop("resampling_method", None),  # Not implemented in config file.
                            )
                    # print(raster_config.model_dump(), flush=True)
                    quads_request_opts = dict(
                        max_quad_length=self.rasters[g].max_quad_length,
                        min_cross_section_node_count=self.rasters[g].min_cross_section_node_count,
                        min_quad_length=self.rasters[g].min_quad_length,
                        min_quad_width=self.rasters[g].min_quad_width,
                        shrinkage_factor=self.rasters[g].shrinkage_factor,
                        cross_distance_factor=self.rasters[g].cross_distance_factor,
                        min_branch_length=self.rasters[g].min_branch_length,
                        threshold_size=self.rasters[g].threshold_size,
                        resample_distance=self.rasters[g].resample_distance,
                        simplify_tolerance=self.rasters[g].simplify_tolerance,
                        interpolation_distance=self.rasters[g].interpolation_distance,
                        min_area_to_length_ratio=self.rasters[g].min_area_to_length_ratio,
                        min_area = self.rasters[g].min_area,
                        min_quads_per_group = self.rasters[g].min_quads_per_group,
                        zmin = self.rasters[g].zmin,
                        zmax = self.rasters[g].zmax,
                        lower_bound = self.rasters[g].lower_bound,
                        upper_bound = self.rasters[g].upper_bound,
                        nprocs=1,
                        )

                    # quads_raster_constraints = self._get_raster_window_constraints(k, j)
                    yield k, ((i, raster_path, raster_request_opts, quads_request_opts), (j, window))
                    k += 1
        else:
            raise ValueError("Unreachable: self.rasters is None")

    def _get_final_quads_gdf_feather_path(self, comm, cache_directory):
        if comm.Get_rank() == 0:
            serialized_requests = []
            if self.rasters is not None:
                normalized_requests = list(self.iter_normalized_raster_window_requests())
                serialized_requests.append(json.dumps(normalized_requests, default=str))
            # if self.geometry is not None:
            #     gdf = gpd.read_file(self.geometry)
            #     geoms = gdf.geometry.apply(lambda x: x.wkb)
            #     hashes = geoms.apply(lambda x: hashlib.md5(x).hexdigest())
            #     serialized_requests = json.dumps(hashes, default=str)
            if isinstance(self.from_hgrid, QuadsFromHgridConfig):
                # gdf = gpd.read_file(self.geometry)
                gdf = self.from_hgrid.gdf()
                # DEBUG!
                # gdf.plot(facecolor='none', edgecolor='r')
                # plt.show()
                geoms = gdf.geometry.apply(lambda x: x.wkb)
                hashes = geoms.apply(lambda x: hashlib.md5(x).hexdigest())
                serialized_requests.append(json.dumps(hashes, default=str))
            if len(serialized_requests) == 0:
                raise ValueError("Unreachable: unexpected condition: serialized_requests is empty")
            cached_filename = hashlib.sha256(json.dumps(serialized_requests, default=str).encode('utf-8')).hexdigest() + ".feather"
            combined_quads_cached_directory = cache_directory / "combined"
            combined_quads_cached_directory.mkdir(parents=True, exist_ok=True)
            combined_filepath = combined_quads_cached_directory / cached_filename
        else:
            combined_filepath = None
        return comm.bcast(combined_filepath, root=0)

    @staticmethod
    def _build_raster_quads_gdf(raster_path, raster_request_opts, quads_request_opts, window, quads_cache_directory):
        # TODO: Verify sanity of the normalization.
        _qro = quads_request_opts.copy()
        _qro.pop("nprocs")
        _rro = raster_request_opts.copy()
        _rro.pop("chunk_size")
        normalized_requests = [raster_path, _rro, _qro, window]
        # print(quads_q
        serialized_requests = json.dumps(normalized_requests, default=str)
        # cached_filename = hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".msh"
        cached_filename = hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".feather"
        quads_filepath = quads_cache_directory / cached_filename
        if not quads_filepath.is_file():
            logger.debug(f"Processing {raster_path=} {raster_request_opts=} {quads_request_opts=} {window=}")
            start = time()
            quads_gdf = Quads.from_raster(
                    raster=raster_path,
                    window=window,
                    raster_opts=raster_request_opts,
                    **quads_request_opts,
                    ).quads_gdf
            logger.debug(f"Processing {raster_path=} {raster_request_opts=} {quads_request_opts=} {window=} took {time()-start}")
            quads_gdf.to_feather(quads_filepath)
        else:
            logger.debug("Loading %s from cache", str(quads_filepath))
            start = time()
            quads_gdf = gpd.read_feather(quads_filepath)
            if time() - start < 2.:
                sleep(1)
        return quads_gdf

    def build_uncombined_quads_gdf_mpi(self, comm, output_rank=None, cache_directory=None) -> Union[List[gpd.GeoDataFrame], None]:
        root_rank = 0 if output_rank is None else output_rank
        if cache_directory is not None:
            quads_cache_directory = cache_directory / "raster_window"
        else:
            if comm.Get_rank() == root_rank:
                _tmpdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.tmp-quads_build-uncombined-quads')
                quads_cache_directory = Path(_tmpdir.name)
            else:
                quads_cache_directory = None
            quads_cache_directory = comm.bcast(quads_cache_directory, root=root_rank)
        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                quads_cache_directory.mkdir(exist_ok=True, parents=True)
                # serial (debug)
                # uncombined_quads = []
                # for args in self.iter_normalized_raster_window_requests():
                #     uncombined_quads.append(self._build_raster_quads_gdf(*args))
                # parallel
                logger.debug("Begin building quads in parallel")
                start = time()
                uncombined_quads: List[gpd.GeoDataFrame] = list(executor.starmap(
                    self._build_raster_quads_gdf,
                    [(raster_path, raster_request_opts, quads_request_opts, window, quads_cache_directory)
                        for _, ((_, raster_path, raster_request_opts, quads_request_opts), (_, window))
                        in self.iter_normalized_raster_window_requests()],
                    ))
                logger.debug("done building uncombined quads in parallel, took %s", time()-start)
                # print(uncombined_quads, flush=True)

                # logger.debug("Begin loading quads in parallel")
                # start = time()
                # uncombined_quads: List[jigsaw_msh_t] = list(executor.map(self._loadmsh_wrapper, uncombined_quads))
                # logger.debug(f"done building uncombined quads in parallel, took {time()-start}")
                # import pickle
                # pickle.dump(
                #         uncombined_quads,
                #         open("uchfns.pkl", "wb")
                #         )
                # print(uncombined_quads, flush=True)
                # verify:
                # logger.debug("plotting")
                # for quads_gdf in uncombined_quads:
                #     if len(quads_gdf) > 0:
                #         quads_gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='r')
                # plt.gca().axis("scaled")
                # plt.show(block=True)
            else:
                uncombined_quads = None
        comm.barrier()
        if output_rank is None:
            return comm.bcast(uncombined_quads, root=root_rank)
        return uncombined_quads

    @staticmethod
    def _concat_quads_gdf_list(quads_gdf_list: List[gpd.GeoDataFrame]):
        return pd.concat(quads_gdf_list, ignore_index=True)

    def build_combined_quads_gdf_mpi(self, comm, output_rank=None, cache_directory=None):

        root_rank = 0 if output_rank is None else output_rank
        if cache_directory is not None:
            cached_filepath = self._get_final_quads_gdf_feather_path(comm, cache_directory)
            cached_filepath.unlink(missing_ok=True)
            if cached_filepath.is_file():
                if comm.Get_rank() == root_rank:
                    logger.debug("Loading quads_gdf from cache: %s", str(cached_filepath))
                    quads_gdf = gpd.read_feather(cached_filepath)
                    # with open(cached_filepath, "rb") as fh:
                    #     quads_gdf = pickle.load(fh)
                else:
                    quads_gdf = None
            else:
                quads_gdf = self._build_final_quads_gdf_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
                if comm.Get_rank() == root_rank and quads_gdf is not None:
                    quads_gdf.to_feather(cached_filepath)
                    # with open(cached_filepath, "wb") as fh:
                    #     pickle.dump(quads_gdf, fh)
        else:
            quads_gdf = self._build_final_quads_gdf_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
        logger.debug(f"{comm.Get_rank()=} loaded {quads_gdf=}")
        if quads_gdf is not None and self.min_quads_per_group is not None:
            quads_gdf = self._apply_sieve_to_gdf(quads_gdf)

        if output_rank is None:
            return comm.bcast(quads_gdf, root=root_rank)
        return quads_gdf

    def _apply_sieve_to_gdf(self, quads_gdf):
        counts = quads_gdf.groupby('quad_group_id').size()
        quads_gdf = quads_gdf[quads_gdf['quad_group_id'].isin(counts[counts >= self.min_quads_per_group].index)]
        # renumber the id's so that they are continuous
        # Get the unique remaining group ids
        unique_ids = quads_gdf['quad_group_id'].unique()

        # Sort them and generate a new consecutive range
        unique_ids.sort()
        new_ids = range(1, len(unique_ids)+1)

        # Map old ids to new ids in a dictionary
        id_map = dict(zip(unique_ids, new_ids))

        # Replace old ids with new ids
        quads_gdf.loc[:, 'quad_group_id'] = quads_gdf['quad_group_id'].map(id_map)
        return quads_gdf

    def _build_final_quads_gdf_mpi_from_hgrid(self, comm, output_rank=None, cache_directory=None):
        assert isinstance(self.from_hgrid, QuadsFromHgridConfig)
        root_rank = 0 if output_rank is None else output_rank
        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                logger.debug("Should be building the quad_group_id")
                quads_gdf = self.from_hgrid.gdf()
                gdf_uu = gpd.GeoDataFrame(geometry=[quads_gdf.unary_union], crs=quads_gdf.crs).explode(index_parts=False).reset_index(drop=True)
                gdf_centroids = gpd.GeoDataFrame(geometry=quads_gdf.to_crs("epsg:6933").geometry.centroid, crs="epsg:6933").to_crs(quads_gdf.crs)
                joined = gpd.sjoin(gdf_centroids, gdf_uu, predicate="within", how="left")
                quads_gdf["quad_group_id"] = joined.index_right
            else:
                quads_gdf = None
        if output_rank is None:
            return comm.bcast(quads_gdf, root=root_rank)
        return quads_gdf

    def _build_final_quads_gdf_mpi(self, comm, output_rank=None, cache_directory=None):
        if isinstance(self.from_hgrid, QuadsFromHgridConfig):
            return self._build_final_quads_gdf_mpi_from_hgrid(comm, output_rank=output_rank, cache_directory=cache_directory)
        elif isinstance(self.rasters, list):
            return self._build_final_quads_gdf_mpi_raster(comm, output_rank=output_rank, cache_directory=cache_directory)
        raise NotImplementedError("Unreachable: Should've done one of the above.")

    def _build_final_quads_gdf_mpi_raster(self, comm, output_rank=None, cache_directory=None):
        # logger.debug("Begin build_quads_gdf_mpi")
        root_rank = 0 if output_rank is None else output_rank
        # if cache_directory is not None:
        #     quads_cache_directory = cache_directory / "raster_window"
        # else:
        #     if comm.Get_rank() == root_rank:
        #         _tmpdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.tmp-quads_build-uncombined-quads')
        #         quads_cache_directory = Path(_tmpdir.name)
        #     else:
        #         quads_cache_directory = None
        #     quads_cache_directory = comm.bcast(quads_cache_directory, root=root_rank)
        # logger.debug(f"{comm.Get_rank()=} will build uncombined quads_gdf")
        print("will build uncombined quads", flush=True)
        quads_gdf = self.build_uncombined_quads_gdf_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
        print("DONE with will build uncombined quads", flush=True)


        def partition_list(input_list, items_per_bucket):
            """Partition `input_list` into sublists with roughly `items_per_bucket` items each."""
            array = np.array(input_list, dtype=object)
            num_partitions = -(-len(input_list) // items_per_bucket)
            return np.array_split(array, num_partitions)

        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                logger.debug("Doing concat")
                # print(quads_gdf, flush=True)
                while len(quads_gdf) > 1:
                    quads_gdf = list(executor.map(
                        self._concat_quads_gdf_list,
                        partition_list(quads_gdf, 2)
                        ))
                    logger.debug("%s", len(quads_gdf))
                quads_gdf = quads_gdf.pop()
                # quads_gdf = pd.concat(quads_gdf, ignore_index=True)
                logger.debug("do first sjoin")
                quads_gdf['geometry'] = quads_gdf['geometry'].apply(Polygon)
                joined = gpd.sjoin(
                        quads_gdf,
                        quads_gdf,
                        how='inner',
                        predicate='contains'
                        )
                joined = joined[joined.index != joined["index_right"]]
                quads_gdf.drop(index=joined.index.unique(), inplace=True)
                quads_gdf.reset_index(inplace=True, drop=True)

                logger.debug("do second sjoin")
                quads_gdf['geometry'] = quads_gdf['geometry'].apply(lambda x: LinearRing(x.exterior.coords))
                joined = gpd.sjoin(
                        quads_gdf,
                        quads_gdf,
                        how='inner',
                        predicate='intersects'
                        )
                joined = joined[joined.index != joined["index_right"]]
                joined['sorted_index_pair'] = joined.apply(lambda row: tuple(sorted([row.name, row['index_right']])), axis=1)
                joined = joined.drop_duplicates(subset='sorted_index_pair')
                joined = joined.drop(columns=['sorted_index_pair'])
                def make_args_list(row):
                    return row.geometry, quads_gdf.loc[row.index_right].geometry
                joined["is_conforming"] = list(executor.map(check_conforming_wrapper, joined.apply(make_args_list, axis=1)))


                # quads_gdf = gpd.read_feather("this_quads_gdf.feather")
                # joined = gpd.read_feather("this_joined.feather")
                non_conforming_indexes = joined[joined['is_conforming'] == False].index.unique().union(joined[joined['is_conforming'] == False].index_right.unique())
                quads_non_conforming = quads_gdf.loc[non_conforming_indexes]
                joined2 = gpd.sjoin(
                        quads_non_conforming,
                        quads_non_conforming,
                        how='inner',
                        predicate='intersects'
                        )
                joined2 = joined2[joined2.index != joined2.index_right]
                # Create a graph
                import networkx as nx
                G = nx.Graph()

                # Add edges based on sjoin results. Nodes will be automatically created.
                for index, row in joined2.iterrows():
                    G.add_edge(index, row.index_right)

                # Find connected components - each component is a set of indices that are connected through intersections
                components = list(nx.connected_components(G))

                # Create a mapping from index to group
                index_to_group = {}
                for group_id, component in enumerate(components):
                    for index in component:
                        index_to_group[index] = group_id

                # Map the group ID to each polygon in the original GeoDataFrame
                quads_non_conforming['group'] = quads_non_conforming.index.map(index_to_group)
                groups = quads_non_conforming.groupby("group")
                from multiprocessing import Pool, cpu_count
                with Pool(cpu_count()) as pool:
                    results = pool.map(process_group, groups)
                indexes_to_drop = []
                for _, idxs in results:
                    indexes_to_drop.extend(idxs)
                quads_gdf = quads_gdf.drop(index=list(set(indexes_to_drop)))

        if output_rank is None:
            return comm.bcast(quads_gdf, root=root_rank)

        return quads_gdf

    # def _build_final_raster_quads_msh_t_mpi(self, comm, output_rank=None, cache_directory=None):

    #     quads_gdf = self.build_combined_quads_gdf_mpi(comm, output_rank=output_rank, cache_directory=cache_directory)
    #     comm.barrier()
    #     root_rank = 0 if output_rank is None else output_rank




# #         def partition_list(input_list, items_per_bucket):
# #             """Partition `input_list` into sublists with roughly `items_per_bucket` items each."""
# #             array = np.array(input_list, dtype=object)
# #             num_partitions = -(-len(input_list) // items_per_bucket)
# #             return np.array_split(array, num_partitions)

    #     with MPICommExecutor(comm, root=root_rank) as executor:
    #         if executor is not None:
    #             # verify
    #             quads_gdf.plot(ax=plt.gca(), cmap='jet')
    #             plt.show(block=True)

    #     comm.barrier()
    #     raise
# #                 logger.debug("Begin partial combine of quads windows")
# #                 while len(quads_msh_t_list) > 1:
# #                     quads_msh_t_list = list(executor.map(
# #                         self._combine_quads_gdf_list,
# #                         partition_list(quads_msh_t_list, 2)
# #                         ))
# #                     logger.debug("%s", len(quads_msh_t_list))
# #                 quads_msh_t = quads_msh_t_list.pop()
# #             else:
# #                 quads_msh_t = None


#         if output_rank is None:
#             return comm.bcast(quads_msh_t, root=root_rank)
        return quads_gdf

    # def build_combined_quads_msh_t_mpi(self, comm, output_rank=None, cache_directory=None):
    #     # if cache_directory:
    #     #     cached_filepath = self._get_final_quads_pkl_path(comm, cache_directory)
    #     #     logger.debug(f"{cached_filepath.is_file()=}")
    #         # if cached_filepath.is_file() and (output_rank is None or comm.Get_rank() == output_rank):
    #         #     # quads_msh_t = jigsaw_msh_t()
    #         #     # loadmsh(str(cached_filepath.resolve()), quads_msh_t)
    #         #     # return quads_msh_t
    #         #     return pickle.load(open(cached_filepath, 'rb'))
    #     # comm.barrier()
    #     # raise
    #     final_quads_msh_t = self._build_final_raster_quads_msh_t_mpi(comm, output_rank=output_rank, cache_directory=cache_directory)
    #     # should_write_cache = cache_directory and (output_rank is None or comm.Get_rank() == output_rank)
    #     # if should_write_cache:
    #     #     pickle.dump(final_quads_msh_t, open(cached_filepath, "wb"))
    #     if output_rank is None:
    #         return comm.bcast(final_quads_msh_t, root=output_rank)
    #     return final_quads_msh_t
    #     # uncombined_quads_msh_t_list = self.build_uncombined_quads_msh_t(comm, output_rank=output_rank, cache_directory=cache_directory)

    def build_spliced_msh_t_mpi(self, comm, input_msh_t, output_rank=None, cache_directory=None):
        root_rank = 0 if output_rank is None else output_rank
        quads_gdf = self.build_combined_quads_gdf_mpi(
                comm,
                output_rank=root_rank,
                cache_directory=cache_directory
                )
        if comm.Get_rank() == root_rank:
            logger.debug("Begin splicing quads into base mesh")
            spliced_msh_t = Quads(quads_gdf)(input_msh_t)
        else:
            spliced_msh_t = None
        if output_rank is None:
            return comm.bcast(spliced_msh_t, root=root_rank)
        return spliced_msh_t


def entrypoint(args, comm=None):
    comm = MPI.COMM_WORLD if comm is None else comm
    cache_directory = args.cache_directory or Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp-geomesh'))
    cache_directory.mkdir(exist_ok=True, parents=True)
    quads_config = QuadsConfig.try_from_yaml_path(args.config)
    quads_gdf = quads_config.build_combined_quads_gdf_mpi(
            comm,
            output_rank=0,
            cache_directory=cache_directory
            )
    # print(quads_config, flush=True)
    # exit90
    # uncombined_quads_paths = get_uncombined_quads(comm, quads_config, cache_directory=cache_directory)
    # combined_quads_gdf = get_combined_quads_gdf(comm, uncombined_quads_paths)
    # if comm.Get_rank() == 0:
    #     combined_quads_gdf.to_feather(args.to_feather)


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
    entrypoint(args)


if __name__ == "__main__":
    main()


def test_quads_build_on_USVIPR():
    import yaml
    import tempfile
    raster_path_glob_string = '/sciclone/pscr/jrcalzada/thesis/runs/tropicald-validations/data/prusvi_19_topobathy_2019/*.tif'
    opts = {
        'quads': {
            'rasters': [
                {
                    'max_quad_length': 500.,
                    'max_quad_width': 500.,
                    'path': raster_path_glob_string,
                    # 'resampling_factor': 0.2,
                    'resample_distance': 100.,
                    'threshold_size': 1500.,
                    # 'zmin': -30.,
                    'zmax': 0.,
                },
                {
                    'max_quad_length': 500.,
                    'max_quad_width': 500.,
                    'path': raster_path_glob_string,
                    'threshold_size': 1500.,
                    'resample_distance': 100.,
                    'zmin': 0.,
                    'zmax': 10.,
                },
                ]
            }
        }
    args = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        tmpfile_config = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        tmpfile_feather = tempfile.NamedTemporaryFile(delete=False, suffix=".feather")
        cache_tmpdir = Path(tempfile.TemporaryDirectory().name)
        cache_tmpdir.mkdir()
        yaml.dump(opts, open(tmpfile_config.name, 'w'))
        cmd_opts = [
                tmpfile_config.name,
                f'--to-feather={tmpfile_feather.name}', # required
                f'--cache-directory={cache_tmpdir.name}'
                ]
        args = get_argument_parser().parse_args(cmd_opts)
        print(f"{tmpfile_config=}")
        print(f"{tmpfile_feather=}")
        print(f"{' '.join(cmd_opts)}")
        print(f"{args}")
    args = MPI.COMM_WORLD.bcast(args, root=0)
    # init_logger(args.log_level)

    entrypoint(args)
    # print(args)

def test_quads_build_on_chesapeake_bay():
    import yaml
    import tempfile
    # nwatl_bbox = {
    #         'crs': 'epsg:4326',
    #         'xmax': -60.040005,
    #         'xmin': -98.00556,
    #         'ymax': 45.831431,
    #         'ymin': 8.534422
    #         }
    xmin, ymin, xmax, ymax = -77.964478,36.681636,-72.504272,40.400948
    chesepeake_bay_bbox = {
                        'crs': 'epsg:4326',
                        'xmax': xmax,
                        'xmin': xmin,
                        'ymax': ymax,
                        'ymin': ymin
                    }

    ncei_bbox = chesepeake_bay_bbox

    opts = {
        'quads': {
            'rasters': [
                # {
                #     'max_quad_length': 500.0,
                #     'max_quad_width': 500.0,
                #     'path': '/sciclone/pscr/jrcalzada/thesis/runs/tropicald-validations/data/prusvi_19_topobathy_2019/*.tif',
                #     # 'resampling_factor': 0.2,
                #     'resample_distance': 100.0,
                #     'threshold_size': 1500.,
                #     # 'zmin': -30.0,
                #     'zmax': 0.0
                # },
                # {
                #     'max_quad_length': 500.0,
                #     'max_quad_width': 500.0,
                #     'path': '/sciclone/pscr/jrcalzada/thesis/runs/tropicald-validations/data/prusvi_19_topobathy_2019/*.tif',
                #     'threshold_size': 1500.,
                #     # 'resampling_factor': 0.2,
                #     'resample_distance': 100.0,
                #     'zmin': 0.0,
                #     'zmax': 10.0
                # },
                {
                    'bbox': ncei_bbox,
                    'max_quad_length': 500.0,
                    'max_quad_width': 500.0,
                    'resample_distance': 100.0,
                    'tile_index': '/sciclone/pscr/jrcalzada/thesis/runs/tropicald-validations/data/tileindex_NCEI_ninth_Topobathy_2014.zip',
                    # 'resampling_factor': 0.2,
                    # 'zmin': -30.0,
                    'threshold_size': 1500.,
                    'zmax': 0.0
                },
                {
                    'bbox': ncei_bbox,
                    'max_quad_length': 500.0,
                    'max_quad_width': 500.0,
                    'resample_distance': 100.0,
                    'tile_index': '/sciclone/pscr/jrcalzada/thesis/runs/tropicald-validations/data/tileindex_NCEI_ninth_Topobathy_2014.zip',
                    'resampling_factor': 0.2,
                    'zmin': 0.0,
                    'zmax': 10.0
                }
            ]
        }
    }
    args = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        tmpfile_config = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        tmpfile_feather = tempfile.NamedTemporaryFile(delete=False, suffix=".feather")
        cache_tmpdir = Path(tempfile.TemporaryDirectory().name)
        cache_tmpdir.mkdir()
        yaml.dump(opts, open(tmpfile_config.name, 'w'))
        cmd_opts = [
                tmpfile_config.name,
                f'--to-feather={tmpfile_feather.name}', # required
                f'--cache-directory={cache_tmpdir.name}'
                ]
        args = get_argument_parser().parse_args(cmd_opts)
        print(f"{tmpfile_config=}")
        print(f"{tmpfile_feather=}")
        print(f"{' '.join(cmd_opts)}")
        print(f"{args}")
    args = MPI.COMM_WORLD.bcast(args, root=0)
    # init_logger(args.log_level)

    entrypoint(args)
    # print(args)


