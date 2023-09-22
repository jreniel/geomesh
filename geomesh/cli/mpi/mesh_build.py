#!/usr/bin/env python
from functools import partial
from pathlib import Path
import argparse
import hashlib
import logging
import os
import pickle
import sys

from colored_traceback.colored_traceback import Colorizer
from jigsawpy import savevtk, savemsh
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pyproj import CRS
import matplotlib.pyplot as plt
import numpy as np


from geomesh.cmd.mpi import hfun_build, geom_build
from geomesh.cmd.mpi import lib
from geomesh.driver import JigsawDriver
from geomesh.geom import Geom
from geomesh import Mesh, utils


logger = logging.getLogger(__name__)


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
    geom_build.logger.setLevel(log_level)
    hfun_build.logger.setLevel(log_level)
    lib.logger.setLevel(log_level)
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
    # parser.add_argument('--max-cpus-per-task', type=int)
    parser.add_argument('--to-msh', type=Path)
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--dst-crs', '--dst_crs', type=CRS.from_user_input, default=CRS.from_epsg(4326))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=Path)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


def get_geom_tempfile_from_args(comm, args):
    geom_config = geom_build.get_geom_config_from_args(comm, args)
    gdf = geom_build.combine_geoms(comm, geom_build.get_uncombined_geoms(comm, geom_config))
    if comm.Get_rank() == 0:
        geom = Geom(gdf.unary_union, crs=gdf.crs)
        cache_directory = geom_config["cache_directory"]
        cache_directory /= 'geom'
        cache_directory.mkdir(parents=True, exist_ok=True)
        geom_tmpfile = cache_directory / hashlib.sha256(str(gdf).encode('utf-8')).hexdigest()
        pickle.dump(geom, open(geom_tmpfile, 'wb'))
    else:
        geom_tmpfile = None
        gdf = None
    return comm.bcast(geom_tmpfile, root=0)


def get_hfun_tempfile_from_args(comm, args):
    hfun_config = hfun_build.get_hfun_config_from_args(comm, args)
    uncombined_hfun_paths = hfun_build.get_uncombined_hfuns(comm, hfun_config)
    return hfun_build.combine_hfuns(comm, uncombined_hfun_paths, hfun_config)


def get_output_msh_t_tempfile_from_args(comm, args):

    rank = comm.Get_rank()
    # we need 1 color for the geom and the remainder of the colors for the hfun
    hwinfo = lib.hardware_info()

    unique_colors = hwinfo['color'].unique()

    local_color = np.min([hwinfo.iloc[rank]['color'], 1])
    local_comm = comm.Split(local_color, rank)

    # asynchronous (implicit)
    if len(unique_colors) > 1:
        geom_tempfile = None
        hfun_tempfile = None
        if local_color == 0:
            geom_tempfile = get_geom_tempfile_from_args(local_comm, args)
        else:
            hfun_tempfile = get_hfun_tempfile_from_args(local_comm, args)

        local_comm.Barrier()
        comm.Barrier()

        # Broadcast from local_ranks to COMM_WORLD
        if local_comm.Get_rank() == 0 and local_color == 0:
            comm.bcast(geom_tempfile, root=0)
        else:
            geom_tempfile = comm.bcast(geom_tempfile, root=0)

        hfun_root = np.min(hwinfo[hwinfo['color'] != 0].index)
        if rank == hfun_root:
            comm.bcast(hfun_tempfile, root=hfun_root)
        else:
            hfun_tempfile = comm.bcast(hfun_tempfile, root=hfun_root)

    # synchronous
    else:
        geom_tempfile = get_geom_tempfile_from_args(local_comm, args)
        hfun_tempfile = get_hfun_tempfile_from_args(local_comm, args)

    if args.cache_directory is not None:
        cache_directory = args.cache_directory / 'msh_t'
    else:
        cache_directory = Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp')) / 'msh_t'
    cache_directory.mkdir(exist_ok=True, parents=True)

    msh_t_outfile = cache_directory / (hashlib.sha256(
        f"{geom_tempfile}{hfun_tempfile}".encode('utf-8')).hexdigest() + ".pkl")
    # logger.info(f'{msh_t_outfile=}')
    if not msh_t_outfile.is_file():
        if comm.Get_rank() == 0:
            geom = pickle.load(open(geom_tempfile, 'rb'))
            hfun = pickle.load(open(hfun_tempfile, 'rb'))
            driver = JigsawDriver(
                    geom,
                    hfun,
                    verbosity=1,
                    # nprocs=local_comm.Get_size()
                    )
            driver.opts.numthread = local_comm.Get_size()
            mesh = driver.output_mesh
            pickle.dump(mesh.msh_t, open(msh_t_outfile, 'wb'))
    return msh_t_outfile


def make_plot(hfun_msh_t_tmpfile):
    logger.info('Drawing plot...')
    with open(hfun_msh_t_tmpfile, 'rb') as fh:
        msh_t = pickle.load(fh)
    utils.triplot(msh_t, axes=plt.gca())
    plt.gca().axis('scaled')
    plt.show(block=True)
    return


def to_msh(args, hfun_msh_t_tmpfile):
    logger.info('Write msh_t...')
    with open(hfun_msh_t_tmpfile, 'rb') as fh:
        msh_t = pickle.load(fh)
    savemsh(f'{args.to_msh.resolve()}', msh_t)
    return


def to_pickle(args, hfun_msh_t_tmpfile):
    logger.info('Write pickle...')
    with open(hfun_msh_t_tmpfile, 'rb') as fh:
        msh_t = pickle.load(fh)
    with open(args.to_pickle, 'wb') as fh:
        pickle.dump(Mesh(msh_t), fh)
    logger.info('Done writing pickle...')
    return


def main(args):

    comm = MPI.COMM_WORLD

    out_msh_t_tempfile = get_output_msh_t_tempfile_from_args(comm, args)

    finalization_tasks = []

    if args.show:
        finalization_tasks.append(make_plot)

    if args.to_msh:
        finalization_tasks.append(partial(to_msh, args))

    if args.to_pickle:
        finalization_tasks.append(partial(to_pickle, args))

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            futures = [executor.submit(task, out_msh_t_tempfile) for task in finalization_tasks]
            for future in futures:
                future.result()

    # if args.to_pickle is not None:
    #     mesh.to_pickle(args.to_pickle)
    # if args.to_msh is not None:
    #     mesh.to_msh(args.to_msh)
    # if args.show:
    #     mesh.show()


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
