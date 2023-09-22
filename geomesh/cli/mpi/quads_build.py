#!/usr/bin/env python
from pathlib import Path
import argparse
import hashlib
import json
import logging
import os
import sys

from colored_traceback.colored_traceback import Colorizer
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pyproj import CRS
import numpy as np

from geomesh.cli.build import BuildCli
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
    normalized_raster_opts = lib.get_normalized_raster_request_opts(raster_hash, quads_opts, window)
    normalized_quads_opts = {
            'zmin': quads_opts.get('zmin'),
            'zmax': quads_opts.get('zmax'),
            'pad_width': quads_opts.get('pad_width'),
            'max_quad_length': quads_opts.get('max_quad_length', 500.),
            'shrinkage_factor': quads_opts.get('shrinkage_factor', 0.9),
            'cross_distance_factor': quads_opts.get('cross_distance_factor', 0.95),
            'min_branch_length': quads_opts.get('min_branch_length', None),
            'threshold_size': quads_opts.get('threshold_size', None),
            'resample_distance': quads_opts.get('resample_distance', None),
            'simplify_tolerance': quads_opts.get('simplify_tolerance', None),
            'interpolation_distance': quads_opts.get('interpolation_distance', None),
            'min_ratio': quads_opts.get('min_ratio', 0.1),
            'min_area': quads_opts.get('min_area', np.finfo(np.float64).min),
            'min_cross_section_node_count': quads_opts.get('min_cross_section_node_count', 4),
            'min_quad_length': quads_opts.get('min_quad_length', None),
            'min_quad_width': quads_opts.get('min_quad_width', None),
            'max_quad_width': quads_opts.get('max_quad_width', None),
            }
    normalized_quads_window_request = {
            **normalized_quads_opts,
            **normalized_raster_opts,
            }
    tmpfile = cache_directory / (hashlib.sha256(
        json.dumps(
            normalized_quads_window_request,
            ).encode()).hexdigest() + '.feather')

    if not tmpfile.exists():

        from time import time
        logger.debug(f'rank={MPI.COMM_WORLD.Get_rank()} start generating quads...')
        start = time()
        raster = get_raster_from_opts(raster_path, quads_opts, window)
        # try:
        #     run_with_timeout(function_to_run_with_timeout, 300, raster, normalized_quads_opts, tmpfile)
        # except TimeoutException:
        #     print(f'{raster_path=} timeout', flush=True)
        #     raise
        Quads.from_raster(
                    raster,
                    **normalized_quads_opts
                    ).quads_gdf.to_feather(tmpfile)
        logger.debug(f'rank={MPI.COMM_WORLD.Get_rank()} took {time()-start} to generate quads')
    else:
        logger.debug(f'{tmpfile=} exists...')
    return tmpfile


def get_uncombined_quads_from_raster_requests(comm, quads_config, cache_directory):
    rank = comm.Get_rank()

    quads_raster_window_requests = get_quads_raster_window_requests(comm, quads_config)

    # cache_directory = quads_config['cache_directory']

    # Now that we have an expanded list of raster window requests, we nee to compute
    # a system-independent hash for each of the requests. The fisrt step is to compute the md5
    # of each of the rasteres in raw form. We then use this md5 as a salt for adding other requests to the
    # raster request. This should theoretically result in the same hash on any platform, becaue it based on
    # the file content instead of path.
    if rank == 0:
        logger.info('will get raster md5')
    raster_hashes = lib.get_raster_md5(comm, quads_raster_window_requests)

#     raise NotImplementedError
#     if rank == 0:
#         logger.info('will get local contours paths')
#     local_contours_paths = get_local_contours_paths(comm, quads_raster_window_requests, bbox_gdf, cache_directory, raster_hashes)

#     if rank == 0:
#         logger.info('will get cpus_per_task')
#     cpus_per_task = get_cpus_per_task(comm, quads_raster_window_requests, quads_config, local_contours_paths, bbox_gdf)

    if rank == 0:
        logger.info('will build quads from rasters')

    cache_directory /= 'window_quads'
    cache_directory.mkdir(exist_ok=True)
    results = []
    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:
            job_args = []
            for raster_hash, ((i, raster_path, quads_opts), (j, window)) in zip(raster_hashes, quads_raster_window_requests):
                job_args.append((raster_path, quads_opts, window, raster_hash, cache_directory))
            results = list(executor.map(get_quad_feather, job_args))
    out_feathers = comm.bcast(results, root=0)

    if rank == 0:
        logger.info(f'built {len(out_feathers)} feathers')
    return out_feathers


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


def main(args, comm=None):
    comm = MPI.COMM_WORLD if comm is None else comm
    quads_config = get_quads_config_from_args(comm, args)
    cache_directory = args.cache_directory or Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp-geomesh'))
    cache_directory.mkdir(exist_ok=True, parents=True)
    uncombined_quads_paths = get_uncombined_quads(comm, quads_config, cache_directory=cache_directory)

    print(quads_config)


def entrypoint():
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
    main(args)


if __name__ == "__main__":
    entrypoint()
