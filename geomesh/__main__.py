#! /usr/bin/env python
import argparse
from datetime import datetime
import logging
import sys

from pytz import timezone

from .cmd.build import BuildCli
# from .cmd.raster_geom_build import GeomCli

# pygeos==0.10.2 && shapely==1.8.0


def init_logger():
    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    tmp_args, _ = tmp_parser.parse_known_args()
    if tmp_args.log_level is not None:
        logging.basicConfig(
            format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            force=True,
        )

        logging.Formatter.converter = lambda *args: datetime.now(
            tz=timezone("UTC")
        ).timetuple()

        logging.captureWarnings(True)
        logging.getLogger('geomesh').setLevel({
                "warning": logging.WARNING,
                "info": logging.INFO,
                "debug": logging.DEBUG,
                "critical": logging.CRITICAL,
                "notset": logging.NOTSET,
            }[tmp_args.log_level])

def main():
    init_logger()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="clitype")
    _add_build_cli_actions(subparsers)
    # _add_geom_cli_actions(subparsers)
    args = parser.parse_args()
    if args.clitype == 'build':
        BuildCli(args.config).main()
    # elif args.clitype == 'geom':
    #     GeomCli(args).main()

def _add_build_cli_actions(subparsers):
    parser = subparsers.add_parser('build')
    parser.add_argument('config')

# def _add_geom_cli_actions(subparsers):
#     parser = subparsers.add_parser('geom')
#     actions= parser.add_subparsers(dest='geom_actions')
#     build_parser = actions.add_parser('build')
#     build_parser.add_argument('--raster', '-r', dest='rasters', action='append', metavar='uri')
#     build_parser.add_argument('feature', nargs='?')


if __name__ == "__main__":
    main()
