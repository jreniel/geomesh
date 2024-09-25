#!/usr/bin/env python
import argparse
import hashlib
import json
import logging
import pickle
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from inpoly import inpoly2
from jigsawpy import jigsaw_msh_t, savemsh
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pydantic import BaseModel
from pyproj import CRS
from shapely.geometry import MultiPolygon

from geomesh import Mesh, utils
from geomesh.cli.mpi import geom_build, hfun_build, lib, quads_build
from geomesh.cli.mpi.geom_build import GeomConfig
from geomesh.cli.mpi.hfun_build import HfunConfig
from geomesh.cli.mpi.quads_build import QuadsConfig
from geomesh.driver import JigsawDriver
from geomesh.geom import quadgen
from geomesh.geom.base import multipolygon_to_jigsaw_msh_t
from geomesh.geom.shapely_geom import MultiPolygonGeom

logger = logging.getLogger(__name__)


def init_logger(log_level: str):
    logging.basicConfig(
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        force=True,
        # datefmt="%Y-%m-%d %H:%M:%S "
    )
    _log_level = getattr(logging, str(log_level).upper())
    logger.setLevel(_log_level)
    geom_build.logger.setLevel(_log_level)
    hfun_build.logger.setLevel(_log_level)
    quads_build.logger.setLevel(_log_level)
    quadgen.logger.setLevel(_log_level)
    lib.logger.setLevel(_log_level)
    # if int(log_level) < 40:
    #     logging.getLogger("geomesh").setLevel(log_level)
    # logging.Formatter.converter = lambda *args: datetime.now(tz=pytz.timezone("UTC")).timetuple()
    logging.captureWarnings(True)


def get_argument_parser():

    def cache_directory_bootstrap(path_str):
        path = Path(path_str)
        if path.name != "mesh_build":
            path /= "mesh_build"
        path.mkdir(exist_ok=True, parents=True)
        return path

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    # parser.add_argument('--max-cpus-per-task', type=int)
    parser.add_argument('--to-msh', type=Path)
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--dst-crs', '--dst_crs', type=CRS.from_user_input, default=CRS.from_epsg(4326))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=cache_directory_bootstrap)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


# def get_geom_tempfile_from_args(comm, args):
#     geom_config = geom_build.get_geom_config_from_args(comm, args)
#     gdf = geom_build.combine_geoms(comm, geom_build.get_uncombined_geoms(comm, geom_config))
#     if comm.Get_rank() == 0:
#         geom = Geom(gdf.unary_union, crs=gdf.crs)
#         cache_directory = geom_config["cache_directory"]
#         cache_directory /= 'geom'
#         cache_directory.mkdir(parents=True, exist_ok=True)
#         geom_tmpfile = cache_directory / hashlib.sha256(str(gdf).encode('utf-8')).hexdigest()
#         pickle.dump(geom, open(geom_tmpfile, 'wb'))
#     else:
#         geom_tmpfile = None
#         gdf = None
#     return comm.bcast(geom_tmpfile, root=0)


# def get_hfun_tempfile_from_args(comm, args):
#     hfun_config = hfun_build.get_hfun_config_from_args(comm, args)
#     uncombined_hfun_paths = hfun_build.get_uncombined_hfuns(comm, hfun_config)
#     return hfun_build.combine_hfuns(comm, uncombined_hfun_paths, hfun_config)


# def get_output_msh_t_tempfile_from_args(comm, args):

#     rank = comm.Get_rank()
#     # we need 1 color for the geom and the remainder of the colors for the hfun
#     hwinfo = lib.hardware_info()

#     unique_colors = hwinfo['color'].unique()

#     local_color = np.min([hwinfo.iloc[rank]['color'], 1])
#     local_comm = comm.Split(local_color, rank)

#     # asynchronous (implicit)
#     if len(unique_colors) > 1:
#         geom_tempfile = None
#         hfun_tempfile = None
#         if local_color == 0:
#             geom_tempfile = get_geom_tempfile_from_args(local_comm, args)
#         else:
#             hfun_tempfile = get_hfun_tempfile_from_args(local_comm, args)

#         local_comm.Barrier()
#         comm.Barrier()

#         # Broadcast from local_ranks to COMM_WORLD
#         if local_comm.Get_rank() == 0 and local_color == 0:
#             comm.bcast(geom_tempfile, root=0)
#         else:
#             geom_tempfile = comm.bcast(geom_tempfile, root=0)

#         hfun_root = np.min(hwinfo[hwinfo['color'] != 0].index)
#         if rank == hfun_root:
#             comm.bcast(hfun_tempfile, root=hfun_root)
#         else:
#             hfun_tempfile = comm.bcast(hfun_tempfile, root=hfun_root)

#     # synchronous
#     else:
#         geom_tempfile = get_geom_tempfile_from_args(local_comm, args)
#         hfun_tempfile = get_hfun_tempfile_from_args(local_comm, args)

#     if args.cache_directory is not None:
#         cache_directory = args.cache_directory / 'msh_t'
#     else:
#         cache_directory = Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp')) / 'msh_t'
#     cache_directory.mkdir(exist_ok=True, parents=True)

#     msh_t_outfile = cache_directory / (hashlib.sha256(
#         f"{geom_tempfile}{hfun_tempfile}".encode('utf-8')).hexdigest() + ".pkl")
#     # logger.info(f'{msh_t_outfile=}')
#     if not msh_t_outfile.is_file():
#         if comm.Get_rank() == 0:
#             geom = pickle.load(open(geom_tempfile, 'rb'))
#             hfun = pickle.load(open(hfun_tempfile, 'rb'))
#             driver = JigsawDriver(
#                     geom,
#                     hfun,
#                     verbosity=1,
#                     # nprocs=local_comm.Get_size()
#                     )
#             driver.opts.numthread = local_comm.Get_size()
#             mesh = driver.output_mesh
#             pickle.dump(mesh.msh_t, open(msh_t_outfile, 'wb'))
#     return msh_t_outfile


class MeshConfig(BaseModel):

    geom: Optional[GeomConfig] = None
    hfun: Optional[HfunConfig] = None
    quads: Optional[QuadsConfig] = None
    opts: Optional[lib.OptsConfig] = None
    finalize: Optional[bool] = True
    verbosity: Optional[int] = 0
    sieve: Optional[bool | float] = None

    @classmethod
    def try_from_yaml_path(cls, path: Path) -> "MeshConfig":
        with open(path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return cls.try_from_dict(data)

    @classmethod
    def try_from_dict(cls, data: dict) -> "MeshConfig":
        return cls(**data['mesh'])

    def _get_base_mesh_path(self, comm, cache_directory):
        # serialized_requests = json.dumps(requests, default=str)
        # return hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".pkl"
        normalized_requests = []
        if self.geom is not None:
            normalized_requests.append(self.geom._get_final_geom_gdf_feather_path(comm, cache_directory.parent / "geom_build").stem)
        if self.hfun is not None:
            normalized_requests.append(self.hfun._get_final_hfun_msh_path(comm, cache_directory.parent / "hfun_build").stem)
        if comm.Get_rank() == 0:
            normalized_requests.append(self.finalize)
            normalized_requests.append(self.sieve)
            serialized_requests = json.dumps(normalized_requests, default=str)
            cached_filename = hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".pkl"
            combined_msh_t_cache_directory = cache_directory / "combined_msh_t"
            combined_msh_t_cache_directory.mkdir(parents=True, exist_ok=True)
            combined_filepath = combined_msh_t_cache_directory / cached_filename
        else:
            combined_filepath = None
        # comm.barrier()
        # logger.debug(f"{comm.Get_rank()=} {combined_filepath=}")
        return comm.bcast(combined_filepath, root=0)

    def _get_quads_combined_msh_t_path(self, comm, cache_directory):
        jigsaw_msh_t_path = self._get_base_mesh_path(comm, cache_directory)
        if self.quads is None:
            return jigsaw_msh_t_path
        normalized_requests = [jigsaw_msh_t_path.stem, self.quads._get_final_quads_gdf_feather_path(comm, cache_directory.parent / "quads_build").stem]
        if comm.Get_rank() == 0:
            serialized_requests = json.dumps(normalized_requests, default=str)
            cached_filename = hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".pkl"
            combined_hfun_cached_directory = cache_directory / "quads_combined_msh_t"
            combined_hfun_cached_directory.mkdir(parents=True, exist_ok=True)
            combined_filepath = combined_hfun_cached_directory / cached_filename
        else:
            combined_filepath = None
        return comm.bcast(combined_filepath, root=0)

    def _build_base_mesh_mpi(self, comm, output_rank, cache_directory):
        root_rank = 0 if output_rank is None else output_rank
        hwinfo = lib.hardware_info(comm)
        geom = self.get_geom_msh_t(comm, output_rank=root_rank, cache_directory=cache_directory)
        hfun = self.get_hfun_msh_t(comm, output_rank=root_rank, cache_directory=cache_directory)
        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                # numthread = np.min([
                #         hwinfo[hwinfo.color == hwinfo.loc[root_rank].color].index.nunique(),
                #         16])
                logger.debug("init JigsawDriver")
                # utils.tricontourf(hfun, show=True)
                driver = JigsawDriver(
                    geom=geom,
                    hfun=hfun,
                    verbosity=self.verbosity,
                    # dst_crs=self.dst_crs,
                    sieve=self.sieve,
                    finalize=self.finalize,
                    # nprocs=n_local_colors,
                    )
                if self.opts is not None:
                    for key, item in self.opts.model_dump().items():
                        setattr(driver.opts, key, item)

                if driver.opts.numthread is None:
                    driver.opts.numthread = hwinfo[hwinfo.color == hwinfo.loc[root_rank].color].index.nunique()
                logger.debug("Call msh_t()")
                output_msh_t = driver.msh_t()
            else:
                output_msh_t = None
        if output_rank is None:
            return comm.bcast(output_msh_t, root=root_rank)
        return output_msh_t

    def get_base_mesh_mpi(self, comm, output_rank=None, cache_directory=None):
        root_rank = 0 if output_rank is None else output_rank
        if cache_directory is not None:
            cached_filepath = self._get_base_mesh_path(comm, cache_directory)
            if cached_filepath.is_file():
                if comm.Get_rank() == root_rank:
                    logger.debug("Loading base_msh_t from cache: %s", str(cached_filepath))
                    with open(cached_filepath, "rb") as fh:
                        base_msh_t = pickle.load(fh)
                else:
                    base_msh_t = None
            else:
                logger.debug(f"{cached_filepath=} doesn't exists, building mesh")
                base_msh_t = self._build_base_mesh_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
                if comm.Get_rank() == root_rank and base_msh_t is not None:
                    with open(cached_filepath, "wb") as fh:
                        pickle.dump(base_msh_t, fh)
        else:
            base_msh_t = self._build_base_mesh_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)

        # if base_msh_t is not None:
        #     base_msh_t = self._apply_sieve_to_gdf(base_msh_t)

        if output_rank is None:
            return comm.bcast(base_msh_t, root=root_rank)
        return base_msh_t

    def _apply_sieve_to_output_msh_t(self, output_msh_t):

        mp = utils.geom_to_multipolygon(output_msh_t)

        areas = [poly.area for poly in mp.geoms]
        if self.sieve is True:
            polygon = mp.geoms[areas.index(max(areas))]
            mp = MultiPolygon([polygon])
        elif self.sieve is None or self.sieve is False:
            pass
        else:
            # TODO:
            raise NotImplementedError(f'Unhandled sieve: expected None or bool but got {self.sieve=}')

        _msh_t = multipolygon_to_jigsaw_msh_t(mp)

        in_poly = inpoly2(output_msh_t.vert2['coord'], _msh_t.vert2['coord'], _msh_t.edge2['index'])[0]

        tria3_mask = np.any(~in_poly[output_msh_t.tria3['index']], axis=1)

        # tria3_idxs_take = np.where(~tria3_mask)[0]
        tria3_index = output_msh_t.tria3['index'][~tria3_mask, :]
        # tria3_index = output_msh_t.tria3['index'].take(tria3_idxs_take)
        used_indexes, inverse = np.unique(tria3_index, return_inverse=True)
        node_indexes = np.arange(output_msh_t.vert2['coord'].shape[0])
        isin = np.isin(node_indexes, used_indexes)
        # tria3_idxs = np.where(~isin)[0]
        vert2_idxs = np.where(isin)[0]

        df = pd.DataFrame(index=node_indexes).iloc[vert2_idxs].reset_index()
        mapping = {v: k for k, v in df.to_dict()['index'].items()}
        tria3_index = np.array([mapping[x] for x in used_indexes])[inverse].reshape(tria3_index.shape)
        output_msh_t.vert2 = output_msh_t.vert2.take(vert2_idxs, axis=0)

        tria3_IDtag = output_msh_t.tria3['IDtag'][~tria3_mask]

        # update value
        if len(output_msh_t.value) > 0:
            output_msh_t.value = output_msh_t.value.take(vert2_idxs)

        output_msh_t.tria3 = np.array(
                [(tuple(indices), tria3_IDtag[i]) for i, indices in enumerate(tria3_index)],
                dtype=jigsaw_msh_t.TRIA3_t)

        return output_msh_t


    def get_output_msh_t_mpi(self, comm, output_rank=None, cache_directory=None):

        if cache_directory is not None and Path(cache_directory).name != 'mesh_build':
            cache_directory = Path(cache_directory) / "mesh_build"
            cache_directory.mkdir(exist_ok=True, parents=True)

        root_rank = 0 if output_rank is None else output_rank
        if self.quads is None:
            logger.debug("quads is None")
            return self.get_base_mesh_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
        if cache_directory is not None:
            cached_filepath = self._get_quads_combined_msh_t_path(comm, cache_directory)
            if cached_filepath.is_file():
                logger.debug("The final output_msh_t is in cache")
                if comm.Get_rank() == root_rank:
                    logger.debug("Loading output_msh_t from cache: %s", str(cached_filepath))
                    with open(cached_filepath, "rb") as fh:
                        output_msh_t = pickle.load(fh)
                else:
                    output_msh_t = None

            else:
                logger.debug("The final output_msh_t is not in cache")
                # if comm.Get_rank() == root_rank:
                #     logger.debug(f"{cached_filepath=} {cache_directory=}")
                #     comm.Abort()
                logger.debug("getting the base mesh")
                output_msh_t = self.get_base_mesh_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
                logger.debug("quad-splicing the base mesh")
                output_msh_t = self.quads.build_spliced_msh_t_mpi(comm, output_msh_t, output_rank=root_rank, cache_directory=cache_directory.parent / "quads_build")

                if comm.Get_rank() == root_rank and output_msh_t is not None:
                    with open(cached_filepath, "wb") as fh:
                        pickle.dump(output_msh_t, fh)
        else:
            output_msh_t = self.get_base_mesh_mpi(comm, output_rank=root_rank)
            output_msh_t = self.quads.build_spliced_msh_t_mpi(comm, output_msh_t, output_rank=root_rank)

        if output_msh_t is not None:
            output_msh_t = self._apply_sieve_to_output_msh_t(output_msh_t)

        if output_rank is None:
            return comm.bcast(output_msh_t, root=root_rank)
        return output_msh_t


        # spliced_output_msh_t_path = self._get_final_output_msh_t_path(comm, cache_directory)
        # # logger.debug(f"{comm.Get_rank()=} will build uncombined quads_gdf")
        # output_msh_t = self.build_spliced_msh_t_mpi(comm, output_rank=root_rank, cache_directory=spliced_output_msh_t_path.parent)
        # # return self.quads.build_spliced_msh_t_mpi(comm, output_msh_t, output_rank=root_rank, cache_directory=spliced_quads_directory)
        # return output_msh_t

    def get_geom_msh_t(self, comm, output_rank=None, cache_directory=None):
        root_rank = 0 if output_rank is None else output_rank
        if self.geom is None:
            return
        if cache_directory is not None:
            geom_cache_directory = cache_directory.parent / "geom_build"
        else:
            if comm.Get_rank() == root_rank:
                _tmpdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.tmp-geom_build')
                geom_cache_directory = Path(_tmpdir.name)
            else:
                geom_cache_directory = None
            geom_cache_directory = comm.bcast(geom_cache_directory, root=root_rank)

        geom = self.geom.build_combined_geoms_gdf_mpi(comm, output_rank=root_rank, cache_directory=geom_cache_directory)
        if geom is not None:
            geom = MultiPolygonGeom(MultiPolygon(list(geom.explode(index_parts=False).geometry)), crs=geom.crs)
        if comm.Get_rank() == root_rank:
            logger.debug("Done bulding geom!")
        if output_rank is None:
            return comm.bcast(geom, root=root_rank)
        return geom

    def get_hfun_msh_t(self, comm, output_rank=None, cache_directory=None):
        root_rank = 0 if output_rank is None else output_rank
        if self.hfun is None:
            return
        if cache_directory is not None:
            hfun_cache_directory = cache_directory.parent / "hfun_build"
        else:
            if comm.Get_rank() == root_rank:
                _tmpdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.tmp-hfun_build')
                hfun_cache_directory = Path(_tmpdir.name)
            else:
                hfun_cache_directory = None
            hfun_cache_directory = comm.bcast(hfun_cache_directory, root=root_rank)
        hfun = self.hfun.build_combined_hfun_msh_t_mpi(comm, output_rank=root_rank, cache_directory=hfun_cache_directory)
        if output_rank is None:
            return comm.bcast(hfun, root=0)
        return hfun

def make_plot(output_msh_t):
    logger.info('Drawing plot...')
    utils.triplot(output_msh_t, axes=plt.gca())
    plt.title(f"{len(output_msh_t.vert2)=}")
    plt.gca().axis('scaled')
    plt.show(block=True)


def to_msh(args, output_msh_t):
    logger.info('Write msh_t...')
    savemsh(f'{args.to_msh.resolve()}', output_msh_t)


def to_pickle(args, output_msh_t):
    logger.info('Write pickle...')
    with open(args.to_pickle, 'wb') as fh:
        pickle.dump(Mesh(output_msh_t), fh)
    logger.info('Done writing pickle...')
    return

def entrypoint(args):

    comm = MPI.COMM_WORLD

    # out_msh_t_tempfile = get_output_msh_t_tempfile_from_args(comm, args)
    mesh_config = MeshConfig.try_from_yaml_path(args.config)

    output_msh_t = mesh_config.get_output_msh_t_mpi(comm, output_rank=0, cache_directory=args.cache_directory)

    finalization_tasks = []

    if args.show:
        finalization_tasks.append(make_plot)

    if args.to_msh:
        finalization_tasks.append(partial(to_msh, args))

    if args.to_pickle:
        finalization_tasks.append(partial(to_pickle, args))

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            futures = [executor.submit(task, output_msh_t) for task in finalization_tasks]
            for future in futures:
                future.result()

    # if args.to_pickle is not None:
    #     mesh.to_pickle(args.to_pickle)
    # if args.to_msh is not None:
    #     mesh.to_msh(args.to_msh)
    # if args.show:
    #     mesh.show()


def main():
    sys.excepthook = lib.mpiabort_excepthook
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
    entrypoint(args)


if __name__ == "__main__":
    main()
