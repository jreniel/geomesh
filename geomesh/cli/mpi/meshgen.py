#!/usr/bin/env python
from pathlib import Path
import argparse
import logging
import sys

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

from geomesh.cli.mpi import lib


logger = logging.getLogger(__name__)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


def entrypoint(config_path: Path, comm=None):
    comm = comm or MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        # Let the root rank do the validation
        user_config = lib.MeshgenConfig.try_from_yaml_path(config_path)
    else:
        user_config = None
    user_config = comm.bcast(user_config, root=0)
    raster_geom_gdf = user_config.geom.get_raster_geom_gdf(comm, output_rank=0)


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
    lib.init_logger(args.log_level)
    entrypoint(
            args.config,
            comm=comm,
            )


if __name__ == "__main__":
    main()

def test_main():
    from unittest.mock import patch
    yaml_str = """
femto.sciclone.wm.edu: &femto
  slurm: &femto_slurm
    walltime: '3-00:00:00'
    job_name: ' '

frontera.tacc.utexas.edu: &frontera
  slurm: &frontera_slurm
    partition: 'flex'
    walltime: '3-00:00:00'
    job_name: ' '

log_level: "info"
scheduler:
    slurm:
      geom:
        <<: *femto_slurm
        ntasks: 32
      hfun:
        <<: *femto_slurm
        ntasks: 352
        # min_ntasks: 32
        # max_ntasks: 320
      quads:
        <<: *femto_slurm
        cpus_per_task: 2
        # ntasks: 176
        min_ntasks: 16
        max_ntasks: 176
      msh_t:
        <<: *femto_slurm
      hgrid:
        <<: *femto_slurm
        min_ntasks: 32
        max_ntasks: 320

    mesh_bbox: &mesh_bbox
        xmin: -98.00556
        ymin: 8.534422
        xmax: -60.040005
        ymax: 45.831431
        crs: 'epsg:4326'

rasters:
    - &GEBCO_2021_sub_ice_topo_1
      path: '../../data/GEBCO_2021_sub_ice_topo/gebco_2021_sub_ice_topo_n90.0_s0.0_w-180.0_e-90.0.tif'
      bbox: *mesh_bbox
      chunk_size: 1622
    - &GEBCO_2021_sub_ice_topo_2
      path: '../../data/GEBCO_2021_sub_ice_topo/gebco_2021_sub_ice_topo_n90.0_s0.0_w-90.0_e0.0.tif'
      bbox: *mesh_bbox
      chunk_size: 1622
    - &PRUSVI2019
      path: '../../data/prusvi_19_topobathy_2019/*.tif'
    - &NCEITileIndex
      tile_index: '../../data/tileindex_NCEI_ninth_Topobathy_2014.zip'
      bbox: *mesh_bbox

geom:
    zmax: &zmax 10.
    rasters:
      - <<: *PRUSVI2019
        zmax: *zmax
        resampling_factor: 0.2 # 50 meters
      - <<: *NCEITileIndex
        zmax: *zmax
        resampling_factor: 0.2 # 50 meters
      - <<: *GEBCO_2021_sub_ice_topo_1
        clip: '../../data/floodplain_patch.gpkg'
        zmin: 0.
        zmax: *zmax
        overlap: 5
      - <<: *GEBCO_2021_sub_ice_topo_2
        clip: '../../data/floodplain_patch.gpkg'
        zmin: 0.
        zmax: *zmax
        overlap: 5
      - <<: *GEBCO_2021_sub_ice_topo_1
        zmax: 0.
        overlap: 5
      - <<: *GEBCO_2021_sub_ice_topo_2
        zmax: 0.
        overlap: 5
    sieve: True
hfun:
    hmax: 15000.
    # verbosity: 1
    marche: false
    # min_cpus_per_task: 8
    rasters:
      - <<: *PRUSVI2019
        resampling_factor: 0.2  # 15 m
        contours:
          - level: 0
            target_size: 100.
            expansion_rate: 0.007
          - level: 10.
            target_size: 100.
            expansion_rate: 0.007
        constant_value:
          - lower_bound: 0.
            value: 500.
        gradient_delimiter:
          - upper_bound: 0.
            hmin: 100.
      - <<: *NCEITileIndex
        resampling_factor: 0.2  # 15 m
        constant_value:
          - lower_bound: 0.
            value: 500.
        contours:
          - level: 0
            target_size: 100.
            expansion_rate: 0.007
          - level: 10.
            target_size: 100.
            expansion_rate: 0.007
        gradient_delimiter:
          - upper_bound: 0.
            hmin: 100.
      - <<: *GEBCO_2021_sub_ice_topo_1
        gradient_delimiter:
          upper_bound: 0.
          hmin: 500.
        overlap: 0
        contours:
          - level: 0
            target_size: 500.
            expansion_rate: 0.001
      - <<: *GEBCO_2021_sub_ice_topo_2
        gradient_delimiter:
          upper_bound: 0.
          hmin: 500.
        overlap: 0
        contours:
          - level: 0
            target_size: 500.
            expansion_rate: 0.001
# msh_t:
#   opt:
#     geom_feat: true
#     numthread: *geom_ntasks
# quads:
#   # banks:
#   #   - path: './full_set_of_bank_pairs/bank_pairs.gpkg'
#   #     max_quad_width: 500.
#   #     min_quad_width: 10.
#   #     min_cross_section_node_count: 4
#   #     bbox: *mesh_bbox
#   #     # shrinkage_factor: .1
#   rasters:
#     - <<: *PRUSVI2019
#       # resampling_factor: 0.2
#       # zmin: -30.
#       zmax: 0.
#       resample_distance: 100.
#       threshold_size: 1500.
#       # threshold_size: 1500.
#       # min_quad_length: 50.
#       max_quad_length: 500.
#       max_quad_width: 500.
#       # min_quad_width: 50.

#     - <<: *PRUSVI2019
#       resampling_factor: 0.2
#       zmin: 0.
#       zmax: 10.
#       resample_distance: 100.
#       threshold_size: 1500.
#       # min_quad_length: 50.
#       max_quad_length: 500.
#       max_quad_width: 500.
#       # min_quad_width: 50.

#     - <<: *NCEITileIndex
#       # resampling_factor: 0.2
#       # zmin: -30.
#       zmax: 0.
#       resample_distance: 100.
#       threshold_size: 1500.
#       # threshold_size: 1500.
#       # min_quad_length: 50.
#       max_quad_length: 500.
#       max_quad_width: 500.
#       # min_quad_width: 50.

#     - <<: *NCEITileIndex
#       resampling_factor: 0.2
#       zmin: 0.
#       zmax: 10.
#       resample_distance: 100.
#       threshold_size: 1500.
#       # min_quad_length: 50.
#       max_quad_length: 500.
#       max_quad_width: 500.
#       # min_quad_width: 50.

interpolate:
rasters:
  - <<: *GEBCO_2021_sub_ice_topo_1
  - <<: *GEBCO_2021_sub_ice_topo_2
  - <<: *NCEITileIndex
  - <<: *PRUSVI2019

boundaries:
threshold: -30
min_open_bound_length: 100

outputs:
hgrid:
  - path: 'hgrid.gr3'
    overwrite: True
2dm:
  - path: 'hgrid.2dm'
    overwrite: True

"""
    import tempfile
    tmp_yaml = tempfile.NamedTemporaryFile()
    open(tmp_yaml.name, 'w').write(yaml_str)
    with patch.object(
            sys,
            'argv',
            [
                __file__,
                tmp_yaml.name,
                # '/sciclone/pscr/jrcalzada/thesis/runs/tropicald-validations/hindcasts/Harvey2017/.cache/geom_build/1b075beedc11b6aa3c45a246dc3d8db345ce827484d9fd9d451f67689d8157b8/config.yaml'
            ]
            ):
        main()


