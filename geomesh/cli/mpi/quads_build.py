#!/usr/bin/env python
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import logging
import os
import sys

from colored_traceback.colored_traceback import Colorizer
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor, MPIPoolExecutor
from pyproj import CRS
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LinearRing

from geomesh.cli.build import BuildCli
from geomesh.geom.quadgen import generate_quad_gdf_from_mp, check_conforming
from geomesh.geom.quadgen import Quads, logger as quadgen_logger
from geomesh.cli.mpi import lib
from geomesh.cli.raster_opts import iter_raster_window_requests, get_raster_from_opts


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



def mpiabort_excepthook(type, value, traceback):
    Colorizer('default').colorize_traceback(type, value, traceback)
    MPI.COMM_WORLD.Abort(-1)


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


class ConfigPathAction(argparse.Action):

    def __call__(self, parser, namespace, config_path, option_string=None):
        if not config_path.is_file():
            raise FileNotFoundError(f'{config_path=} is not an existing file.')
        setattr(namespace, self.dest, config_path)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, action=ConfigPathAction)
    parser.add_argument('--to-feather', type=Path, required=True)
    parser.add_argument('--dst-crs', '--dst_crs', type=CRS.from_user_input, default=CRS.from_epsg(4326))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=Path)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


def get_quads_config_from_args(comm, args):
    rank = comm.Get_rank()
    if rank == 0:
        logger.info('Validating user raster quad requests...')
        quads_config = BuildCli(args).config.quads.quads_config
        if quads_config is None:
            raise RuntimeError(f'No quads to process in {args.config}')
    else:
        quads_config = None

    return comm.bcast(quads_config, root=0)


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


def get_uncombined_quads(comm, quads_config, cache_directory):

    quads_gen_funcs = {
        'rasters': get_uncombined_quads_from_raster_requests,
        'triplets': get_uncombined_quads_from_triplets,
        'banks': get_uncombined_quads_from_banks,
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


            quads_gdf = gpd.read_feather("this_quads_gdf.feather")
            joined = gpd.read_feather("this_joined.feather")
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


def entrypoint(args, comm=None):
    comm = MPI.COMM_WORLD if comm is None else comm
    quads_config = get_quads_config_from_args(comm, args)
    cache_directory = args.cache_directory or Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp-geomesh'))
    cache_directory.mkdir(exist_ok=True, parents=True)
    uncombined_quads_paths = get_uncombined_quads(comm, quads_config, cache_directory=cache_directory)
    combined_quads_gdf = get_combined_quads_gdf(comm, uncombined_quads_paths)
    if comm.Get_rank() == 0:
        combined_quads_gdf.to_feather(args.to_feather)


def main():
    sys.excepthook = mpiabort_excepthook
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


