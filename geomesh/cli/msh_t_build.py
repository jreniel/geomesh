#!/usr/bin/env python
from multiprocessing import cpu_count
from pathlib import Path
import argparse
import logging
import os
import pickle
import sys
import tempfile

from pyproj import CRS
from jigsawpy import savemsh
import matplotlib.pyplot as plt

from geomesh import Mesh
from geomesh import utils
from geomesh.cli.schedulers.mpi import MPICluster
from geomesh.driver import JigsawDriver

logger = logging.getLogger(__name__)


def submit(
        executor,
        # config_path,
        geom_pickle_path,
        hfun_pickle_path,
        initial_mesh_pickle_path=None,
        numthread=cpu_count(),
        verbosity=0,
        finalize=True,
        dst_crs=None,
        to_msh=None,
        to_pickle=None,
        sieve=True,
        sieve_area=None,
        show=False,
        cache_directory=None,
        log_level: str = None,
        **kwargs
        ):
    if isinstance(executor, MPICluster):
        raise TypeError('MPICluster is not supported, use LocalCluster instead.')
    cache_directory = cache_directory or Path(os.getenv('GEOMESH_TEMPDIR', Path.cwd() / '.tmp-geomesh'))
    delete_pickle = False if to_pickle is not None else True
    to_pickle = to_pickle or Path(tempfile.NamedTemporaryFile(dir=cache_directory, suffix='.pkl').name)
    build_cmd = get_cmd(
            # config_path,
            geom_pickle_path,
            hfun_pickle_path,
            initial_mesh_pickle_path=initial_mesh_pickle_path,
            numthread=numthread,
            verbosity=verbosity,
            finalize=finalize,
            dst_crs=dst_crs,
            to_msh=to_msh,
            to_pickle=to_pickle,
            sieve=sieve,
            sieve_area=sieve_area,
            show=show,
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
        # config_path,
        geom_pickle_path,
        hfun_pickle_path,
        initial_mesh_pickle_path=None,
        numthread=cpu_count(),
        verbosity=0,
        geom_feat: bool = None,
        finalize=True,
        dst_crs=None,
        to_msh=None,
        to_pickle=None,
        sieve=True,
        sieve_area=None,
        show=False,
        log_level: str = None,
        ):
    cwd = Path(os.getenv('GEOMESH_TEMPDIR', Path.cwd() / '.tmp-geomesh'))
    cache_directory = cwd / f'{Path(__file__).stem}'
    cache_directory.mkdir(parents=True, exist_ok=True)
    build_cmd = [
            sys.executable,
            f'{Path(__file__).resolve()}',
            # f'{Path(config_path).resolve()}',
            ]
    build_cmd.append(f'--geom-pickle={geom_pickle_path}')
    build_cmd.append(f'--hfun-pickle={hfun_pickle_path}')
    if initial_mesh_pickle_path is not None:
        build_cmd.append(f'--initial-mesh={initial_mesh_pickle_path}')
    if finalize is False:
        build_cmd.append('--no-finalize')
    if to_msh is not None:
        build_cmd.append(f'--to-file={to_msh}')
    if to_pickle is not None:
        build_cmd.append(f'--to-pickle={to_pickle}')
    if show:
        build_cmd.append('--show')
    if dst_crs is not None:
        build_cmd.append(f'--dst-crs={dst_crs}')
    if log_level is not None:
        build_cmd.append(f'--log-level={log_level}')
    if sieve:
        build_cmd.append('--sieve')
    if sieve_area is not None:
        build_cmd.append(f'--sieve-area={sieve_area}')
    if verbosity > 0:
        build_cmd.append(f'--verbosity={verbosity}')
    if numthread is not None:
        build_cmd.append(f'--numthread={numthread}')
    if geom_feat is True:
        build_cmd.append('--enable-geom-feat')
    logger.debug(' '.join(build_cmd))
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


def get_output_msh_t_from_args(args):
    logger.debug("Building mesh")
    driver = JigsawDriver(
            geom=args.geom,
            hfun=args.hfun,
            initial_mesh=args.initial_mesh,
            verbosity=args.verbosity,
            dst_crs=args.dst_crs,
            sieve_area=args.sieve,
            finalize=args.finalize,
            geom_feat=args.geom_feat,
            )

    # msh_t_config = BuildCli(args).config.msh_t.msh_t_config
    # driver.opts = msh_t_config['opt'].get('numthread', cpu_count())
    # for key, value in msh_t_config['opt'].items():
    #     this_attr = getattr(driver.opts, key)
    #     setattr(driver.opts, key, value)

    logger.debug("Building msh_t()")
    return driver.msh_t()


def make_plot(msh_t):
    logger.info('Drawing plot...')
    utils.tricontourf(msh_t, axes=plt.gca(), cmap='jet')
    utils.triplot(msh_t, axes=plt.gca())
    plt.gca().axis('scaled')
    plt.show(block=True)


def to_msh(args, msh_t):
    logger.info('Write msh_t...')
    savemsh(f'{args.to_msh.resolve()}', msh_t)


def to_pickle(args, msh_t):
    logger.info('Write pickle...')
    with open(args.to_pickle, 'wb') as fh:
        pickle.dump(Mesh(msh_t), fh)
    logger.info('Done writing pickle...')


def main(args):

    init_logger(args.log_level)
    out_msh_t = get_output_msh_t_from_args(args)
    if args.to_pickle is not None:
        to_pickle(args, out_msh_t)
    if args.to_msh is not None:
        to_msh(args, out_msh_t)
    # if args.show:
    #     mesh.show()


class PickleLoadAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, pickle.load(values.open('rb')))


def get_argument_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('config', type=Path)
    parser.add_argument('--geom-pickle', type=Path, required=True, dest='geom', action=PickleLoadAction)
    parser.add_argument('--hfun-pickle', type=Path, required=True, dest='hfun', action=PickleLoadAction)
    parser.add_argument('--initial-mesh', type=Path, dest='initial_mesh', action=PickleLoadAction)
    parser.add_argument('--numthread', type=int, default=cpu_count())
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--no-finalize', action='store_false', dest='finalize', default=True)
    parser.add_argument('--dst-crs', type=CRS.from_user_input, default=CRS.from_epsg(4326))
    parser.add_argument('--to-msh', type=Path)
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--enable-geom-feat', action="store_true", dest="geom_feat")
    sieve_opts = parser.add_argument_group('sieve options').add_mutually_exclusive_group()
    sieve_opts.add_argument('--sieve', action='store_true')
    sieve_opts.add_argument('--sieve-area', type=float, dest="sieve")
    sieve_opts.set_defaults(sieve=True)
    parser.add_argument(
            "--log-level",
            choices=["warning", "info", "debug"],
            default="warning",
        )
    return parser


if __name__ == "__main__":
    main(get_argument_parser().parse_args())
