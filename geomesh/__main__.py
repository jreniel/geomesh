#! /usr/bin/env python
import argparse
from datetime import datetime
import logging
import sys

from pytz import timezone

from geomesh.cmd.build import BuildCli
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
    clitypes = {
        'build': BuildCli,
        }
    for name, clitype in clitypes.items():
        clitype.add_parser_arguments(subparsers.add_parser(name))
    args = parser.parse_args()
    clitypes[args.clitype](args).main()


if __name__ == "__main__":
    main()
