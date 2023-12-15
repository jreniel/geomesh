#!/usr/bin/env python
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from time import time
import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import tempfile
import warnings

from colored_traceback.colored_traceback import Colorizer
from dask_geopandas.hilbert_distance import _hilbert_distance
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from scipy.optimize import curve_fit
from shapely import ops
from shapely.geometry import Polygon, MultiPolygon
import dask_geopandas as dgpd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geomesh import Geom
from geomesh.cli.build import BuildCli
from geomesh.cli.mpi import lib
from geomesh.cli.raster_opts import iter_raster_window_requests, get_raster_from_opts
from geomesh.cli.schedulers.local import LocalCluster

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


def mpiabort_excepthook(type, value, traceback):
    Colorizer('default').colorize_traceback(type, value, traceback)
    MPI.COMM_WORLD.Abort(-1)


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


def init_logger(log_level: str):
    logging.basicConfig(
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        force=True,
        # datefmt="%Y-%m-%d %H:%M:%S "
    )
    logging.getLogger("geomesh").setLevel({
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "critical": logging.CRITICAL,
            "notset": logging.NOTSET,
        }["info".lower()])
    logger.setLevel({
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "critical": logging.CRITICAL,
            "notset": logging.NOTSET,
        }[str(log_level).lower()])
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
    parser.add_argument('--to-file', type=Path)
    parser.add_argument('--to-feather', type=Path)
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=Path)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    # sieve_options = parser.add_argument_group('sieve_options').add_mutually_exclusive_group()
    # sieve_options.add_argument('--sieve')
    # sieve_options.add_argument_group('--sieve')
    return parser


def get_geom_config_from_args(comm, args):
    rank = comm.Get_rank()
    # We begin by validating the user's request. The validation is done by the BuildCi class,
    # it does not coerce or change the data of the geom_config dictionary.
    # The yaml configuration data needs to be expanded for parallelization,
    # and the expansion is done by the iter_raster_window_requests method.
    # TODO: Optimize (parallelize) the initial data loading.
    if rank == 0:
        logger.info('Validating user raster geom requests...')
        geom_config = BuildCli(args).config.geom.geom_config
        if geom_config is None:
            raise RuntimeError(f'No geom to process in {args.config}')
        tmpdir = geom_config.get('cache_directory', Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp')))
        tmpdir /= 'geom_build'
        geom_config.setdefault("cache_directory", tmpdir)
        if args.cache_directory is not None:
            geom_config['cache_directory'] = args.cache_directory
        geom_config["cache_directory"].mkdir(exist_ok=True, parents=True)
    else:
        geom_config = None

    return comm.bcast(geom_config, root=0)


def get_uncombined_geoms(comm, geom_config) -> gpd.GeoDataFrame:

    rank = comm.Get_rank()

    # TODO: Optimize initial data loading.
    if rank == 0:
        start = time()
        logger.info('Loading geom raster window requests...')
        geom_raster_window_requests = list(iter_raster_window_requests(geom_config))
        logger.info(f'Took {time()-start} to load geom raster window requests.')
    else:
        geom_raster_window_requests = None

    # get raster hashes
    if rank == 0:
        logger.info(f"[{MPI.Get_processor_name()}] computing raster hashes...")
    raster_hashes = lib.get_raster_md5(comm, geom_raster_window_requests)
    cache_directory = geom_config['cache_directory']

    if rank == 0:
        logger.info(f"[{MPI.Get_processor_name()}] get bboxes gdf...")
    bboxes_gdf = lib.get_bbox_gdf(comm, geom_raster_window_requests, cache_directory, raster_hashes)

    geom_raster_window_requests = comm.bcast(geom_raster_window_requests, root=0)
    geom_raster_window_requests = lib.get_split_array(
            comm,
            geom_raster_window_requests,
            weights=list(range(len(geom_raster_window_requests)))
            )
    # build both geom and bbox gdfs
    mps_gdf = []
    for global_k, ((i, raster_path, geom_opts), (j, window)) in geom_raster_window_requests[comm.Get_rank()]:

        logger.info(f"[{MPI.Get_processor_name()}: rank: {comm.Get_rank()}] computing geom...")
        normalized_geom_window_request = {
                'zmin': geom_opts.get('zmin'),
                'zmax': geom_opts.get('zmax'),
                'sieve': geom_opts.get('sieve'),
                **lib.get_normalized_raster_request_opts(raster_hashes[global_k], geom_opts, window),
                }
        tmpfile = cache_directory / (hashlib.sha256(
            json.dumps(normalized_geom_window_request).encode('utf-8')).hexdigest() + '.feather')
        if tmpfile.exists():
            mps_gdf.append(gpd.read_feather(tmpfile))
            continue
        raster = get_raster_from_opts(raster_path, geom_opts, window)
        geom = Geom(
                raster,
                zmin=geom_opts.get('zmin'),
                zmax=geom_opts.get('zmax')
                )
        # If you have additional geom criteria, call it now
        # geom.generate_quads()
        geom_mp = geom.get_multipolygon(dst_crs='epsg:4326', nprocs=1)
        if geom_mp is None:
            geom_mp = MultiPolygon([])
        gdf = gpd.GeoDataFrame([{
            'geometry': geom_mp,
            'raster_hash': raster_hashes[global_k],
            'index': global_k,
            }],
            crs='epsg:4326'
            ).set_index('index')
        # gdf.to_feather(tmpfile)
        mps_gdf.append(gdf)

    mps_gdf = pd.concat([item for sublist in comm.allgather(mps_gdf) for item in sublist]).sort_index()

    # clip geoms based on "priority"
    if rank == 0:
        logger.info(f"[{MPI.Get_processor_name()}] Begin clipping geoms...")

    gdf = []
    for global_k, ((i, raster_path, geom_opts), (j, window)) in geom_raster_window_requests[comm.Get_rank()]:
        logger.debug(f"[{MPI.Get_processor_name()}: rank: {comm.Get_rank()}] clipping geom...")
        base_geom = mps_gdf.iloc[[global_k]]
        other_bboxes = bboxes_gdf.iloc[:global_k]
        # other_geoms = mps_gdf.iloc[:global_k]
        rtree = other_bboxes.sindex
        idxs = rtree.query(base_geom.unary_union)
        if len(idxs) > 0:
            _mps_gdf = other_bboxes.iloc[idxs]
            _mps_gdf = _mps_gdf.loc[_mps_gdf['raster_hash'] != raster_hashes[global_k]]
            if not _mps_gdf.empty:
                base_geom = base_geom.difference(_mps_gdf.unary_union or MultiPolygon([])).unary_union
                from shapely import wkb
                geom_id = hashlib.sha256(wkb.dumps(base_geom or MultiPolygon([]))).hexdigest()
                base_geom = gpd.GeoDataFrame([{
                    'geometry': base_geom,
                    'geom_id': geom_id,
                    }], crs='epsg:4326')
        gdf.append(base_geom)

    gdf = pd.concat([item for sublist in comm.allgather(gdf) for item in sublist]).reset_index()
    gdf = gdf.loc[gdf['geometry'].is_valid, :]
    gdf = gdf.loc[~gdf['geometry'].is_empty, :]

    if rank == 0:
        logger.info('Done building window geoms!')
    # verify
    # if rank == 0:
    #     gdf.plot(facecolor='none', cmap='jet', legend=True)
    #     plt.show(block=True)
    #     raise NotImplementedError
    # else:
    #     from time import sleep
    #     sleep(10000)
    #     raise
    return gdf


def get_combined_geoms(comm, gdf: gpd.GeoDataFrame, geom_config: dict):
    geom_outfile = None
    if comm.Get_rank() == 0:
        # generate unique id for this geom
        cache_directory = geom_config['cache_directory']
        geom_id_str = ''.join(sorted(gdf['geom_id'].dropna().tolist()))
        geom_hash = hashlib.sha256(json.dumps(geom_id_str).encode('utf-8')).hexdigest()
        geom_outfile = cache_directory / (geom_hash + '.feather')
    geom_outfile = comm.bcast(geom_outfile, root=0)
    if not geom_outfile.exists():
        gdf = combine_geoms(comm, gdf)
        if comm.Get_rank() == 0:
            gdf.to_feather(geom_outfile)
    else:
        logger.info(f"Geom {geom_outfile} already exists. Loading...")
        if comm.Get_rank() == 0:
            gdf = gpd.read_feather(geom_outfile)
        gdf = comm.bcast(gdf, root=0)
    return gdf


def point_slope(x, m, b):
    return m*x + b


def exponential(x, a, b):
    return a * np.exp(b * x)


def combine_geoms(comm, gdf):

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        divisions = [len(gdf)]
        iter_times = []
        while divisions[-1] > 1:
            divisions.append(int(np.floor(divisions[-1]/2)))

    else:
        divisions = None

    divisions = comm.bcast(divisions, root=0)
    if rank == 0:
        logger.info(f'Begin combining geoms (up to {len(divisions[1:])} iterations required) {divisions=}.\n')
    for i, npartitions in enumerate(divisions[1:]):
        if rank == 0:
            start = time()
            gdf = gdf.set_index(_hilbert_distance(gdf, gdf.total_bounds, level=10)).sort_index()
            if npartitions <= size:
                buckets = [[gdf.iloc[bucket].geometry] for bucket in np.array_split(range(len(gdf)), npartitions)]
                buckets = np.array(buckets + [[None]]*(size - len(buckets)), dtype=object)
            else:
                dgdf = dgpd.from_geopandas(gdf, npartitions=npartitions)
                partitions = np.array([part for part in dgdf.partitions], dtype=object)
                buckets = np.array_split(partitions, size)
        else:
            gdf = None
            buckets = None

        gdf = comm.bcast(gdf, root=0)
        buckets = comm.bcast(buckets, root=0)

        local_list = []
        for partition in buckets[rank]:

            # _local_time = time()
            if partition is None:
                continue
            elif isinstance(partition, np.ndarray):
                partition = [poly for poly in partition.flatten() if isinstance(poly, (Polygon, MultiPolygon))]
                gdf = gpd.GeoDataFrame([{'geometry': geom} for geom in partition], crs=gdf.crs)
                local_list.append(ops.unary_union(partition))
            else:
                local_list.append(gdf.loc[np.array(partition.index)].unary_union)
            # _local_time = time() - _local_time
            # plt.ioff()
            # gpd.GeoDataFrame([{'geometry': local_list[-1]}], crs=gdf.crs).plot(cmap='jet')
            # plt.title(f'{rank=} {_local_time=}')
            # savedir = Path(f'testout/{i+1:02d}')
            # savedir.mkdir(exist_ok=True, parents=True)
            # plt.savefig(savedir / f'rank_{rank}.png', dpi=200)
            # plt.close(plt.gcf())
            # plt.ion()

        global_list = [item for sublist in comm.allgather(local_list) for item in sublist]
        del local_list
        if rank == 0:
            gdf = gpd.GeoDataFrame([{'geometry': mp} for mp in global_list], crs=gdf.crs)
            gdf = gdf.set_index(_hilbert_distance(gdf, gdf.total_bounds, level=10)).sort_index()
        del global_list

        gdf = comm.bcast(gdf, root=0)
        is_growing_exponentially = False
        if rank == 0:
            iter_times.append(time() - start)
            # if old_iter_time is not None:
            #     iter_time = old_iter_time -
            # old_iter_time = iter_time
            eta = 0
            if len(iter_times) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params, params_covariance = curve_fit(exponential, range(i+1), iter_times)
                for j, _ in enumerate(divisions):
                    if j >= i:
                        eta += exponential(j, *params)
            if len(iter_times) > 1:
                if abs(params[1] - 1) < 0.5:
                    logger.info("The time is growing exponentially")
                    formula = exponential
                    is_growing_exponentially = True
                else:
                    logger.info("The time is not growing exponentially")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        formula = point_slope
                        params, params_covariance = curve_fit(formula, range(i+1), iter_times)

                logger.info(
                        f'\n\titeration {i+1} took {timedelta(seconds=iter_times[-1])} seconds.\n'
                        f'\tExpected wait time for next iteration: {timedelta(seconds=formula(i+1, *params))}.\n'
                        f'\tEstimated total time remaining: {timedelta(seconds=eta)}.\n'
                        f'\tShould finish by {datetime.now() + timedelta(seconds=eta)}.\n'
                        f'\tThere are {len(gdf)} items remaining.\n'
                        )
            else:
                logger.info(
                        f'\n\titeration {i+1} took {timedelta(seconds=iter_times[-1])} seconds.\n'
                        f'\tThere are {len(gdf)} items remaining.\n'
                        )
        is_growing_exponentially = comm.bcast(is_growing_exponentially, root=0)

        if is_growing_exponentially:
            break

    if rank == 0:
        start = time()
        gdf = gdf.set_index(_hilbert_distance(gdf, gdf.total_bounds, level=10)).sort_index()
        gdf = gpd.GeoDataFrame([{'geometry': gdf.unary_union}], crs=gdf.crs)
        logger.info(f'Final iteration took: {time()-start}')
    else:
        gdf = None

    return comm.bcast(gdf, root=0)


def plot_geom(gdf):
    logger.info('Generating plot')
    geom = Geom(gdf.unary_union, crs=gdf.crs)
    geom.make_plot()
    plt.gca().axis('scaled')
    plt.show(block=True)


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


def main(args: argparse.Namespace):

    comm = MPI.COMM_WORLD

    init_logger('info')

    geom_config = get_geom_config_from_args(comm, args)

    gdf = get_combined_geoms(comm, get_uncombined_geoms(comm, geom_config), geom_config)

    finalization_tasks = []

    if args.show:
        finalization_tasks.append(plot_geom)

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
            for task in finalization_tasks:
                executor.submit(task, gdf).result()
            executor.shutdown(wait=False)


def entrypoint():
    sys.excepthook = mpiabort_excepthook
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        args = get_argument_parser().parse_args()
    else:
        args = None
    args = comm.bcast(args, root=0)
    main(args)


if __name__ == "__main__":
    entrypoint()
