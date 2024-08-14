#!/usr/bin/env python
"""
mesh_build2 is a test for checking the mesh combine
by using PythonCDT instead of JIGSAW
"""
from pathlib import Path
from typing import Optional
import argparse
import logging
import sys
import tempfile

from jigsawpy import jigsaw_msh_t
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pydantic import BaseModel
from pyproj import CRS
import PythonCDT as cdt
import numpy as np
import yaml

from geomesh.cli.mpi import hfun_build, geom_build, quads_build
from geomesh.cli.mpi import lib
from geomesh.cli.mpi.geom_build import GeomConfig
from geomesh.cli.mpi.hfun_build import HfunConfig
from geomesh.cli.mpi.quads_build import QuadsConfig


logger = logging.getLogger(__name__)


class MeshConfig(BaseModel):
    geom: Optional[GeomConfig] = None
    hfun: Optional[HfunConfig] = None
    quads: Optional[QuadsConfig] = None

    @classmethod
    def try_from_yaml_path(cls, path: Path) -> "MeshConfig":
        with open(path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return cls.try_from_dict(data)

    @classmethod
    def try_from_dict(cls, data: dict) -> "MeshConfig":
        return cls(**data['mesh'])

    def output_msh_t_mpi(self, comm, output_rank=None, cache_directory=None) -> jigsaw_msh_t:
        root_rank = 0 if output_rank is None else output_rank
        # geom = self.get_geom_msh_t(comm, output_rank=root_rank, cache_directory=cache_directory)
        hfun_msh_t = self.get_hfun_msh_t(comm, output_rank=root_rank, cache_directory=cache_directory)
        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                t = cdt.Triangulation(
                        cdt.VertexInsertionOrder.AS_PROVIDED,
                        cdt.IntersectingConstraintEdges.RESOLVE,
                        0.
                        )
                t.insert_vertices([cdt.V2d(*coord) for coord in hfun_msh_t.vert2["coord"]])

                print("get constrained triangulation vertices", flush=True)
                vertices = np.array([(v.x, v.y) for v in t.vertices])
                print("get constrained triangulation elements", flush=True)
                elements = np.array([tria.vertices for tria in t.triangles])
                print("getting constrained triangulation elements done!", flush=True)
                # verify
                # from multiprocessing import Pool, cpu_count
                # import geopandas as gpd
                # from shapely.geometry import Polygon
                import matplotlib.pyplot as plt
                # nprocs = cpu_count()
                # chunksize = len(vertices) // nprocs
                # with Pool(nprocs) as pool:
                #     original_mesh_gdf = gpd.GeoDataFrame(
                #         geometry=list(pool.map(
                #             Polygon,
                #             vertices[elements, :].tolist(),
                #             chunksize
                #             )),
                #         crs=hfun_msh_t.crs
                #     )
                # original_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
                plt.triplot(vertices[:, 0], vertices[:, 1], elements)
                plt.title("this is contrained mesh")
                plt.show()
                # breakpoint()
        comm.barrier()

        if output_rank is None:
            return comm.bcast(output_msh_t, root=root_rank)
        return output_msh_t


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
    quads_build.logger.setLevel(log_level)
    lib.logger.setLevel(log_level)
    logging.captureWarnings(True)



def get_argument_parser():

    def cache_directory_bootstrap(path_str):
        path = Path(path_str)
        if path.name != "mesh_build2":
            path /= "mesh_build2"
        path.mkdir(exist_ok=True, parents=True)
        return path

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=lambda x: MeshConfig.try_from_yaml_path(Path(x)))
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


def entrypoint(args, comm=None):
    comm = comm or MPI.COMM_WORLD
    output_msh_t = args.config.output_msh_t_mpi(comm, output_rank=0, cache_directory=args.cache_directory)


def main():
    sys.excepthook = lib.mpiabort_excepthook
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        try:
            args = get_argument_parser().parse_args()
        except Exception as err:
            print(f"ERROR: {err}", flush=True)
            comm.Abort(-1)
    else:
        args = None
    args = comm.bcast(args, root=0)
    init_logger(args.log_level)
    entrypoint(args, comm=comm)

if __name__ == "__main__":

    main()
