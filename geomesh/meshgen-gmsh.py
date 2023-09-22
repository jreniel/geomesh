#!/usr/bin/env python
import argparse
import logging
import sys

from colored_traceback.colored_traceback import Colorizer
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np

from geomesh import Geom, Raster


logger = logging.getLogger(f'[rank: {MPI.COMM_WORLD.Get_rank()}]: {__name__}')


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


def mpiabort_excepthook(type, value, traceback):
    Colorizer('default').colorize_traceback(type, value, traceback)
    MPI.COMM_WORLD.Abort(-1)


# def add_multipolygon_to_gmsh_model(model_name, multipolygon):
#     gmsh.initialize()
#     gmsh.model.add(model_name)

#     for i, polygon in enumerate(multipolygon.geoms):
#         print(f"begin adding polygon {i+1}")
#         exterior_coords = list(polygon.exterior.coords)
#         interior_coords = [list(ring.coords) for ring in polygon.interiors]

#         print('add exteriors')
#         # Add exterior
#         exterior_points = [gmsh.model.geo.addPoint(*coord, 0) for coord in exterior_coords]
#         exterior_lines = [gmsh.model.geo.addLine(exterior_points[i - 1], exterior_points[i]) for i in range(1, len(exterior_points))] 
#         exterior_lines.append(gmsh.model.geo.addLine(exterior_points[-1], exterior_points[0]))  # Connect the last point to the first
#         exterior_curve_loop = gmsh.model.geo.addCurveLoop(exterior_lines)

#         print('add interiors')
#         # Add interiors
#         interior_curve_loops = []
#         for coords in interior_coords:
#             interior_points = [gmsh.model.geo.addPoint(*coord, 0) for coord in coords]
#             interior_lines = [gmsh.model.geo.addLine(interior_points[i - 1], interior_points[i]) for i in range(1, len(interior_points))]
#             interior_lines.append(gmsh.model.geo.addLine(interior_points[-1], interior_points[0]))  # Connect the last point to the first
#             interior_curve_loop = gmsh.model.geo.addCurveLoop(interior_lines)
#             interior_curve_loops.append(interior_curve_loop)

#         # Create surface
#         gmsh.model.geo.addPlaneSurface([exterior_curve_loop] + interior_curve_loops, i+1)
#     print('calling synchronize')
#     gmsh.model.geo.synchronize()
#     # gmsh.option.setNumber("Mesh.Algorithm", 2)
#     current_algorithm = gmsh.option.getNumber("Mesh.Algorithm")
#     print(f"{current_algorithm=}")
#     print('calling mesh generate')
#     gmsh.model.mesh.generate(2)

#     _, node_coords, _ = gmsh.model.mesh.getNodes()
#     element_types, element_tags, element_connectivity = gmsh.model.mesh.getElements()
#     trias = np.where(element_types == 2)[0]
#     trias = element_connectivity[trias[0]]
#     trias = trias.reshape((-1, 3)) - 1
#     # Cleanup
#     gmsh.finalize()
#     # Reshape the node coordinates and element connectivity to form a more convenient data structure
#     node_coords = node_coords.reshape((-1, 3))[:, :2]
#     print(f'element_connectivity:\n{element_connectivity}')
#     print(f'element_types:\n{element_types}')
#     plt.triplot(node_coords[:, 0], node_coords[:, 1], trias)
#     plt.show(block=True)
#     return node_coords, element_connectivity


def main(args, comm=None):
    init_logger(args.log_level)
    comm = comm or MPI.COMM_WORLD
    import geopandas as gpd
    import pygmsh
    from pyproj import CRS
    # Let's hardcode a set of rasters to process for now
    from appdirs import user_data_dir
    import numpy as np
    rootdir = user_data_dir('geomesh')
    # Harlem River tile
    raster = Raster(
            f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/ncei19_n41x00_w074x00_2015v1.tif',
            # resampling_factor=0.2,
            )
    geom = Geom(
            raster,
            zmax=20.
            )
    mp = geom.get_multipolygon()
    centroid = np.array(mp.centroid.coords).flatten()
    local_azimuthal_projection = CRS.from_user_input(
        f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
        )
    gdf = gpd.GeoDataFrame([{'geometry': mp}], crs=raster.crs).to_crs(local_azimuthal_projection).explode()

    with pygmsh.geo.Geometry() as geom:
        for row in gdf.itertuples():
            polygon = row.geometry
                    # Get the exterior coordinates
            exterior_coords = list(polygon.exterior.coords)
            
            # Get the interior coordinates (holes), reversed
            interior_coords = [list(ring.coords)[::-1] for ring in polygon.interiors]

            # Add polygon to the geometry
            geom.add_polygon(
                [exterior_coords] + interior_coords,
                mesh_size=100.
            )

            mesh = geom.generate_mesh()
    print(mesh)




    # coords, elements = add_multipolygon_to_gmsh_model('harlem_river', gdf.geometry[0])
    # print(coords, elements)





def get_argument_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=Path, required=True)
    # parser.add_argument('--msh_t-pickle', type=Path, required=True)
    # parser.add_argument('--to-pickle', type=Path)
    # parser.add_argument('--to-gr3', type=Path)
    # parser.add_argument('--show', action='store_true')
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


def entrypoint():
    sys.excepthook = mpiabort_excepthook
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        try:
            args = get_argument_parser().parse_args()
        except SystemExit:
            comm.Abort(-1)
    else:
        args = None
    args = comm.bcast(args, root=0)
    main(args, comm)


if __name__ == "__main__":
    entrypoint()
