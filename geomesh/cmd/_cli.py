# import argparse

# from geomesh.cmd import CliComponent

# from .slurmbuild.__main__ import MeshBuild

# class MeshBuildCli(CliComponent):

#     def __init__(self, args):
#         print('??')
#         self.args = args
#         MeshBuild(args.config).main()

#     @staticmethod
#     def add_subparser_action(subparsers: argparse._SubParsersAction):
#        add_common_parser_arguments(subparsers.add_parser("build"))

# # pygeos==0.10.2 && shapely==1.8.0


# def add_common_parser_arguments(parser):
#     parser.add_argument("config")
