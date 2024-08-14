from functools import partial
from pathlib import Path
from typing import List
from typing import Optional
import asyncio
import argparse
import logging
import pickle
import sys
import tempfile
# from

from jigsawpy import jigsaw_msh_t
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pydantic import BaseModel
from pyproj import Transformer
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import yaml

from geomesh import Geom
from geomesh.cli.mpi import lib
from geomesh.cli.mpi import mesh_build
from geomesh.cli.mpi.mesh_build import MeshConfig
from geomesh.cli.mpi.mesh_build import MeshConfig
from geomesh.cli.raster_opts import iter_raster_window_requests, get_raster_from_opts
from geomesh.cli.schedulers.local import LocalCluster
from geomesh.mesh import Mesh
from geomesh.raster.raster import get_window_data


logger = logging.getLogger(__name__)


def init_logger(log_level: str):
    mesh_build.init_logger(log_level)
    logger.setLevel(getattr(logging, log_level.upper()))


def submit(
        executor,
        config_path,
        cache_directory=None,
        to_pickle=None,
        to_gr3=None,
        log_level: str = None,
        **kwargs
        ):

    if isinstance(executor, LocalCluster):
        raise TypeError('LocalCluster is not supported, use MPICluster instead.')

    delete_pickle = False if to_pickle is not None else True
    if cache_directory is not None:
        cache_directory = Path(cache_directory)
        cache_directory.mkdir(parents=True, exist_ok=True)
    to_pickle = to_pickle or Path(tempfile.NamedTemporaryFile(dir=cache_directory, suffix='.pkl').name)
    build_cmd = get_command(
            config_path,
            cache_directory=cache_directory,
            to_gr3=to_gr3,
            to_pickle=to_pickle,
            log_level=log_level,
            )
    logger.debug(f"{' '.join(build_cmd)=}")
    if asyncio.get_event_loop().is_running():
        async def callback():
            await executor.submit(build_cmd, **kwargs)
            with open(to_pickle, 'rb') as f:
                hgrid = pickle.load(f)
            if delete_pickle:
                to_pickle.unlink()
            return hgrid
        return callback()
    executor.submit(build_cmd, **kwargs)
    with open(to_pickle, 'rb') as f:
        hgrid = pickle.load(f)
    if delete_pickle:
        to_pickle.unlink()
    return hgrid


def get_command(
        config_path,
        to_pickle=None,
        to_gr3=None,
        cache_directory=None,
        log_level: str = None,
        ):
    build_cmd = [
            sys.executable,
            f'{Path(__file__).resolve()}',
            f'{Path(config_path).resolve()}',
            ]
    if to_pickle is not None:
        build_cmd.append(f'--to-pickle={to_pickle.resolve()}')
    if to_gr3 is not None:
        build_cmd.append(f'--to-gr3={to_gr3.resolve()}')
    if log_level is not None:
        build_cmd.append(f'--log-level={log_level.lower()}')
    if cache_directory is not None:
        build_cmd.append(f'--cache-directory={cache_directory.resolve()}')

    return build_cmd


def get_argument_parser():

    def cache_directory_bootstrap(path_str):
        path = Path(path_str)
        if path.name != "hgrid_build":
            path /= "hgrid_build"
        path.mkdir(exist_ok=True, parents=True)
        return path

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--to-gr3', type=Path)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=cache_directory_bootstrap)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


def make_plot(mesh):
    logger.info('Drawing plot...')
    mesh.boundaries.open.plot(ax=plt.gca(), color='b')
    # Add text labels to the centroid of each LineString
    for idx, row in mesh.boundaries.open.iterrows():
        centroid = row['geometry'].centroid
        plt.gca().text(centroid.x, centroid.y, f"{idx+1}", color='red')
    mesh.boundaries.land.plot(ax=plt.gca(), color='g')
    mesh.boundaries.interior.plot(ax=plt.gca(), color='r')
    ax = plt.gca()
    mesh.make_plot(ax=ax, elements=True)
    # quads.quads_gdf.plot(ax=ax, color='magenta')
    # mesh.wireframe(ax=plt.gca())
    logger.debug('begin making mesh triplot')
    plt.title(f'node count: {len(mesh.msh_t.vert2["coord"])}')
    plt.gca().axis('scaled')
    plt.show(block=True)


# def to_msh(args, msh_t):
#     logger.info('Write msh_t...')
#     savemsh(f'{args.to_msh.resolve()}', msh_t)


def to_pickle(args, mesh_tempfile):
    logger.info('Write pickle...')
    pickle.dump(pickle.load(open(mesh_tempfile, 'rb')), open(args.to_pickle, 'wb'))
    logger.info('Done writing pickle...')


def to_gr3(args, mesh):
    logger.info('write gr3 file...')
    mesh.write(args.to_gr3, overwrite=True, format="grd")
    logger.info(f'Done writting gr3 file: {args.to_gr3}...')



class InterpolateRasterConfig(lib.RasterConfig, lib.IterableRasterWindowTrait):
    _normalization_keys = [
            "resampling_factor",
            "clip",
            "bbox",
            ]
    def iter_normalized_raster_requests(self):
        for normalized_path, normalized_raster_geom_request in super().iter_normalized_raster_requests():
            for normalization_key in self._normalization_keys:
                normalized_raster_geom_request[normalization_key] = getattr(self, normalization_key)
            yield normalized_path, normalized_raster_geom_request

    def iter_normalized_raster_window_requests(self):
        for (i, normalized_path, normalized_raster_geom_request), (j, window) in super().iter_normalized_raster_window_requests():
            for normalization_key in self._normalization_keys:
                normalized_raster_geom_request[normalization_key] = getattr(self, normalization_key)
            yield (i, normalized_path, normalized_raster_geom_request), (j, window)


class InterpolateConfig(BaseModel):
    rasters: List[InterpolateRasterConfig]

    def iter_normalized_raster_window_requests(self):
        k = 0
        for g, raster_config in enumerate(self.rasters):
            for (i, raster_path, normalized_raster_hfun_request), (j, window) in raster_config.iter_normalized_raster_window_requests():
                raster_request_opts = dict(
                        # raster_path=normalized_path,
                        crs=self.rasters[g].crs,
                        clip=normalized_raster_hfun_request.pop("clip"),
                        # bbox=self.rasters[k].bbox,
                        window=window,
                        chunk_size=self.rasters[g].chunk_size,
                        overlap=self.rasters[g].overlap,
                        resampling_factor=normalized_raster_hfun_request.pop("resampling_factor"),
                        resampling_method=normalized_raster_hfun_request.pop("resampling_method", None),  # Not implemented in config file.
                        )
                interpolate_kwargs = {}
                yield k, ((i, raster_path, raster_request_opts, interpolate_kwargs), (j, window))
                k += 1


class BoundariesConfig(BaseModel):
    threshold: Optional[float] = 0.
    min_open_bound_length: Optional[float] = 0


class HgridConfig(BaseModel):

    mesh: MeshConfig
    interpolate: InterpolateConfig
    boundaries: BoundariesConfig

    @classmethod
    def try_from_yaml_path(cls, path: Path) -> "HgridConfig":
        with open(path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return cls.try_from_dict(data)

    @classmethod
    def try_from_dict(cls, data: dict) -> "HgridConfig":
        return cls(**data['hgrid'])


    def _build_output_mesh_mpi(self, comm, output_rank=None, cache_directory=None):
        root_rank = 0 if output_rank is None else output_rank
        if cache_directory is not None:
            output_msh_t_cache_directory = cache_directory.parent / "mesh_build"
        else:
            if comm.Get_rank() == root_rank:
                _tmpdir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.tmp-output_msh_t_build')
                output_msh_t_cache_directory = Path(_tmpdir.name)
            else:
                output_msh_t_cache_directory = None
            output_msh_t_cache_directory = comm.bcast(output_msh_t_cache_directory, root=root_rank)
        if comm.Get_rank() == root_rank:
            logger.debug(f"Begin call to self.mesh.get_output_msh_t_mpi()")
        output_msh_t = self.mesh.get_output_msh_t_mpi(comm, output_rank=root_rank, cache_directory=output_msh_t_cache_directory)
        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                raster_requests = list(self.interpolate.iter_normalized_raster_window_requests())
                job_args = []
                for k, ((i, raster_path, raster_request_opts, interpolate_kwargs), (j, window)) in raster_requests:
                    job_args.append((raster_path, raster_request_opts, output_msh_t.crs))
                raster_window_bboxes = list(executor.starmap(self._get_raster_window_bbox, job_args))

        def iter_interpolate_raster_to_msh_t_job_args():
            for k, ((i, raster_path, raster_request_opts, interpolate_kwargs), (j, window)) in raster_requests:
                raster_request_opts = raster_request_opts.copy()
                raster_request_opts["raster_path"] = raster_path
                # raster_request_opts.pop("bbox")
                # raster = get_raster_from_opts(raster_path=raster_path, **raster_request_opts)
                # xi = raster.x
                # yi = raster.y
                # (xi, yi, _), crs = self._get_window_data_from_raster(raster_path=raster_path, window=window, **raster_request_opts)
                bbox = raster_window_bboxes[k]
                vert2_idxs = np.where(
                    np.logical_and(
                        np.logical_and(bbox.xmin <= output_msh_t.vert2["coord"][:, 0], bbox.xmax >= output_msh_t.vert2["coord"][:, 0]),
                        np.logical_and(bbox.ymin <= output_msh_t.vert2["coord"][:, 1], bbox.ymax >= output_msh_t.vert2["coord"][:, 1]),
                    )
                )[0]
                msh_t_coords = output_msh_t.vert2["coord"][vert2_idxs, :]
                yield msh_t_coords, output_msh_t.crs, raster_request_opts, interpolate_kwargs, vert2_idxs

        with MPICommExecutor(comm, root=root_rank) as executor:
            if executor is not None:
                logger.debug("Begin interpolate_raster_to_msh_t()")
                results = list(executor.starmap(
                            self._interpolate_raster_to_msh_t,
                            # [(*args, cache_directory) for args in self.iter_normalized_raster_window_requests()],
                            iter_interpolate_raster_to_msh_t_job_args(),
                            ))
                output_msh_t.value = np.full(
                        (output_msh_t.vert2["coord"].shape[0], 1),
                        np.nan,
                        dtype=jigsaw_msh_t.REALS_t
                        )
                for idxs, values in results:
                    output_msh_t.value[idxs, 0] = values



                if np.any(np.isnan(output_msh_t.value)):
                    value = output_msh_t.value.flatten()
                    non_nan_idxs = np.where(~np.isnan(value))[0]
                    nan_idxs = np.where(np.isnan(value))[0]
                    value[nan_idxs] = griddata(
                            output_msh_t.vert2['coord'][non_nan_idxs, :],
                            value[non_nan_idxs],
                            output_msh_t.vert2['coord'][nan_idxs, :],
                            method='nearest'
                            )
                    output_msh_t.value = value.reshape((value.size, 1)).astype(jigsaw_msh_t.REALS_t)
                    del value


                # verify:
                # from geomesh import utils
                # import matplotlib.pyplot as plt
                # utils.tricontourf(output_msh_t, ax=plt.gca())
                mesh = Mesh(output_msh_t)
                mesh.boundaries.auto_generate(**self.boundaries.model_dump())
                # mesh.make_plot(ax=plt.gca(), vmax=10.,
                #                # elements=True
                #                )
                # plt.title(f"{len(output_msh_t.vert2['coord'])=}")
                # plt.show(block=True)

                # breakpoint()
            else:
                mesh = None
        if output_rank is None:
            return comm.bcast(mesh, root=root_rank)
        return mesh

    @staticmethod
    def _get_window_data_from_raster(raster_request_opts):
        raster = get_raster_from_opts(**raster_request_opts)
        return get_window_data(
                    raster.src,
                    window=raster.window,
                    band=1,
                    masked=True,
                    resampling_method=raster.resampling_method,
                    resampling_factor=raster.resampling_factor,
                    clip=raster.clip,
                    mask=raster.mask,
                    ), raster.crs

    @classmethod
    def _interpolate_raster_to_msh_t(
            cls,
            coords,
            coords_crs,
            raster_request_opts,
            interpolate_kwargs,
            idxs,
            ):
        # logger.debug(f"Interpolating {raster_opts=} {interpolate_kwargs=}")
        logger.debug(f"Interpolating {raster_request_opts=}")
        (xi, yi, zi), crs = cls._get_window_data_from_raster(raster_request_opts)
        if not crs.equals(coords_crs):
            transformer = Transformer.from_crs(coords_crs, crs, always_xy=True)
            coords[:, 0], coords[:, 1] = transformer.transform(coords[:, 0], coords[:, 1])
        _values = RegularGridInterpolator(
                (xi, yi),
                zi.T.astype(np.float64),
                'linear',
                bounds_error=False,
                fill_value=np.nan
                )(coords)
        nan_idxs = np.where(np.isnan(_values))
        non_nan_idxs = np.where(~np.isnan(_values))
        # start = time()
        _values[nan_idxs] = NearestNDInterpolator(
                # xyzo[non_nan_idxs],
                coords[non_nan_idxs],
                _values[non_nan_idxs],
                )(coords[nan_idxs, :])
        return idxs, _values

    @staticmethod
    def _get_raster_window_bbox(raster_path, request_opts, dst_crs):
        # k, i, j = indices
        # from shapely.geometry import box
        raster = get_raster_from_opts(
                raster_path=raster_path,
                window=request_opts.pop("window"),
                **request_opts)
        return raster.get_bbox(dst_crs=dst_crs)
        # shapely_box = box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        # return gpd.GeoDataFrame(
        #         [{'geometry': shapely_box, 'global_id': k, 'raster_id': i, 'window_id': j}],
        #         crs=raster.crs
        #         )

    # def _get_interpolated_output_mesh_mpi(self, comm, output_rank=None, cache_directory=None):

    # def _get_output_mesh_pickle_path(comm, cache_directory):
    #     if comm.Get_rank() == 0:
    #         normalized_requests.append(self.finalize)
    #         normalized_requests.append(self.sieve)
    #         serialized_requests = json.dumps(normalized_requests, default=str)
    #         cached_filename = hashlib.sha256(serialized_requests.encode('utf-8')).hexdigest() + ".pkl"
    #         combined_hfun_cached_directory = cache_directory / "combined_msh_t"
    #         combined_hfun_cached_directory.mkdir(parents=True, exist_ok=True)
    #         combined_filepath = combined_hfun_cached_directory / cached_filename
    #     else:
    #         combined_filepath = None
    #     # comm.barrier()
    #     # logger.debug(f"{comm.Get_rank()=} {combined_filepath=}")
    #     return comm.bcast(combined_filepath, root=0)



    def get_output_mesh_mpi(self, comm, output_rank=None, cache_directory=None):
        root_rank = 0 if output_rank is None else output_rank
        # if self.quads is None:
        #     return self.get_base_mesh_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
        # if cache_directory is not None:
        #     cached_filepath = self._get_output_mesh_pickle_path(comm, cache_directory)
        #     if cached_filepath.is_file():
        #         if comm.Get_rank() == root_rank:
        #             logger.debug("Loading output_mesh from cache: %s", str(cached_filepath))
        #             with open(cached_filepath, "rb") as fh:
        #                 output_mesh = pickle.load(fh)
        #         else:
        #             output_mesh = None

        #     else:
        #         output_mesh = self._build_output_mesh_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
        #         if comm.Get_rank() == root_rank and output_mesh is not None:
        #             with open(cached_filepath, "wb") as fh:
        #                 pickle.dump(output_mesh, fh)
        # else:
        #     output_mesh = self._build_output_mesh_mpi(comm, output_rank=root_rank)

        output_mesh = self._build_output_mesh_mpi(comm, output_rank=root_rank, cache_directory=cache_directory)
        if output_rank is None:
            return comm.bcast(output_mesh, root=root_rank)
        return output_mesh


def entrypoint(args):
    """
    This program takes a pickle with a jigsaw_msh_t and interpolates the bathymetry
    as defined by the config file. It will also build the boundaries.
    The output can be directly saved as a model-ready "gr3" file.
    """
    comm = MPI.COMM_WORLD
    hgrid_config = HgridConfig.try_from_yaml_path(args.config)
    mesh = hgrid_config.get_output_mesh_mpi(comm, output_rank=0, cache_directory=args.cache_directory)

    finalization_tasks = []

    # if args.show:
    #     finalization_tasks.append(make_plot)

    # if args.to_msh:
    #     finalization_tasks.append(partial(to_msh, args))

    if args.to_pickle:
        finalization_tasks.append(partial(to_pickle, args))

    if args.to_gr3:
        finalization_tasks.append(partial(to_gr3, args))
    # futures = []
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            futures =  [executor.submit(task, mesh) for task in finalization_tasks]
            if args.show:
                make_plot(mesh)
            [future.result() for future in futures]


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





# rank = comm.Get_rank()
# interpolate_config = None
# boundaries_config = None
# if rank == 0:
#     from geomesh.cli.build import BuildCli
#     cli_config = BuildCli(args).config
#     interpolate_config = cli_config.interpolate.interpolate_config
# interpolate_config = comm.bcast(interpolate_config, root=0)
# mesh = pickle.load(open(args.msh_t_pickle, 'rb'))

# interp_raster_window_requests = get_interp_raster_window_requests(comm, interpolate_config)
# mesh.msh_t.value = interpolate_topobathy(comm, interp_raster_window_requests, mesh)

# use_aa = bool(interpolate_config.get('use_aa', False))
# if use_aa:
#     do_use_aa(comm, mesh, interp_raster_window_requests, interpolate_config)

# if rank != 0:
#     mesh = None

# if rank == 0:
#     boundaries_config = cli_config.boundaries.boundaries_config
#     if boundaries_config is not None:
#         mesh.boundaries.auto_generate(**boundaries_config)

# if rank == 0:
#     cache_directory = Path(os.getenv("GEOMESH_TEMPDIR", Path.cwd() / '.tmp')) / 'build_hgrid'
#     cache_directory.mkdir(parents=True, exist_ok=True)
#     _mesh_tempfile = tempfile.NamedTemporaryFile(dir=cache_directory, suffix='.pkl')
#     mesh_tempfile = Path(_mesh_tempfile.name)
#     with open(mesh_tempfile, 'wb') as fh:
#         pickle.dump(mesh, fh)
#     mesh = None
# else:
#     mesh_tempfile = None

# mesh_tempfile = comm.bcast(mesh_tempfile, root=0)

