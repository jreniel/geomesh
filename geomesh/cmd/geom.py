import argparse
from functools import cached_property
import logging
from typing import List

import yaml

from geomesh import Geom, db
from geomesh.cmd import CliComponent, common
from geomesh.cmd.config import YamlParser
# from geomesh.geom.collector import GeomCollector


logger = logging.getLogger(__name__)


class GeomCli(CliComponent):

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # print(self.geom)
        if self.geom is None:
            return
        # print(self.geom)
        raise NotImplementedError


    @cached_property
    def geom(self):
        # geom_build_id = self.args.config.geom.get_build_id()
        # geom = self.cache.geom().get_by_build_id(geom_build_id)
        logger.info('Begin loading requested Geom.')
        # return self.args.config.geom()
        # if geom is None:

        # print(geom_build_id)
        # exit()

        # logger.info(f'Searching for pre-built geom on cache database {self.args.cache}.')

        # geom = db.orm.Geom.fetch( self.cache)
        # geom = self.args.config.geom.fetch(self.cache)
        # TODO: Try to predict final polygon and fetch from DB, otherwise create and populate
        # print(self.args.config.geom.rasters)
        # exit()
        # for request in self.args.config.geom.rasters:
            # for raster in self.get_raster_from_uri(request['uri']):
            #     geom = self.cache.geom(
            #         raster,
            #         zmin=request.get('zmin'),
            #         zmax=request.get('zmax')
            #     )
            # self.args.config.geom.rasters
            # print(request)

        # geom_list: List[Geom] = []
        # for geom_source in self.args.geom_sources:
        #     geom_list.append(self.cache.geom(geom_source))
        # self._geom = GeomCombine(geom_list)
        return self.args.config.geom()


    @staticmethod
    def add_subparser_action(subparsers: argparse._SubParsersAction) -> None:
        add_geom_to_parser(subparsers.add_parser("geom"))

    @property
    def cache(self):
        if not hasattr(self, '_cache'):
            self._cache = db.Cache(self.args.cache)
        return self._cache


def add_geom_to_parser(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="action")
    add_build_to_parser(subparsers.add_parser('build'))


def add_build_to_parser(parser: argparse.ArgumentParser):

    class YamlParserAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            tmp_parser = argparse.ArgumentParser(add_help=False)
            common.add_cache_to_parser(tmp_parser)
            tmp_args = tmp_parser.parse_known_args()[0]
            setattr(namespace, self.dest, YamlParser(values, tmp_args.cache))

    parser.add_argument(
        'config',
        action=YamlParserAction,
        metavar='yaml_file'
    )
    common.add_log_level_to_parser(parser)
    common.add_cache_to_parser(parser)
    common.add_nprocs_to_parser(parser)
