import argparse

from geomesh.cmd import CliComponent


class BuildCli(CliComponent):
    def __init__(self, args):
        self.args = args
        

    @staticmethod
    def add_subparser_action(subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser("build")
        add_geom_to_parser(parser)
        add_hfun_to_parser(parser)


def add_geom_to_parser(parser: argparse.ArgumentParser):
    parser.add_argument("geom")


def add_hfun_to_parser(parser: argparse.ArgumentParser):
    parser.add_argument("hfun")
