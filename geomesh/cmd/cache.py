import argparse
import pathlib

from appdirs import user_data_dir

from geomesh.cmd import CliComponent


class CacheCli(CliComponent):

    def __init__(self, args: argparse.Namespace):
        if args.path is None:
            args.path = pathlib.Path(user_data_dir('geomesh'))
            args.path.mkdir(exist_ok=True)
        print(args.path)

    @staticmethod
    def add_subparser_action(subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser("cache")
        parser.add_argument('--path')
