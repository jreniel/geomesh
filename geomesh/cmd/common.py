import argparse
from multiprocessing import cpu_count
import pathlib

from appdirs import user_data_dir

from geomesh import db


def add_log_level_to_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )


def add_cache_to_parser(parser: argparse.ArgumentParser):

    class SpatialiteCacheAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, db.Cache(db.orm.spatialite_session(pathlib.Path(values))()))

    cache_group = parser.add_argument_group('Geomesh cache options')
    cache_parser = cache_group.add_mutually_exclusive_group()
    cache_parser.add_argument(
        '--spatialite',
        action=SpatialiteCacheAction,
        dest='cache',
        default=db.Cache(db.orm.spatialite_session(pathlib.Path(user_data_dir('geomesh')) / 'cache.sqlite')())
    )
    cache_group.add_argument('--disable-cache', dest='cache', action='store_const', const=None)


def add_nprocs_to_parser(parser: argparse.ArgumentParser):

    # class NprocsAction(argparse.Action):
    #     def __call__(self, parser, namespace, values, option_string=None):
    #         if 
    parser.add_argument(
        '--nprocs',
        default=1,
        type=int,
        help='Number of processors for processing.'
    )
