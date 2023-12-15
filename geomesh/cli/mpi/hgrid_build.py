from functools import partial
from pathlib import Path
import argparse
import logging
import os
import pickle
import sys
import tempfile

from colored_traceback.colored_traceback import Colorizer
from jigsawpy import jigsaw_msh_t
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pyproj import CRS, Transformer
from rasterio.mask import mask as riomask
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from shapely.geometry import mapping, Polygon, MultiPolygon, GeometryCollection
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from geomesh.cli.mpi import lib
from geomesh import Geom
from geomesh.cli.schedulers.local import LocalCluster
from geomesh.cli.raster_opts import iter_raster_window_requests, get_raster_from_opts


logger = logging.getLogger(__name__)


def mpiabort_excepthook(type, value, traceback):
    Colorizer('default').colorize_traceback(type, value, traceback)
    MPI.COMM_WORLD.Abort(-1)


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
    lib.logger.setLevel(log_level)
    # if int(log_level) < 40:
    #     logging.getLogger("geomesh").setLevel(log_level)
    # logging.Formatter.converter = lambda *args: datetime.now(tz=pytz.timezone("UTC")).timetuple()
    logging.captureWarnings(True)


def submit(
        executor,
        config_path,
        msh_t_pickle_path,
        cache_directory=None,
        to_pickle=None,
        cwd=None,
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
    build_cmd = get_cmd(
            config_path,
            msh_t_pickle_path,
            to_pickle=to_pickle,
            log_level=log_level,
            )

    async def callback():
        await executor.submit(build_cmd, **kwargs)
        hgrid = pickle.load(open(to_pickle, 'rb'))
        if delete_pickle:
            to_pickle.unlink()
        return hgrid

    return callback()


def get_cmd(
        config_path,
        msh_t_pickle_path,
        to_pickle=None,
        log_level: str = None,
        ):
    build_cmd = [
            sys.executable,
            f'{Path(__file__).resolve()}',
            f'--config={Path(config_path)}',
            f'--msh_t-pickle={msh_t_pickle_path}',
            ]
    if to_pickle is not None:
        build_cmd.append(f'--to-pickle={to_pickle}')
    if log_level is not None:
        build_cmd.append(f'--log-level={log_level}')

    return build_cmd


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--msh_t-pickle', type=Path, required=True)
    parser.add_argument('--quads-feather-path', type=Path)
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--to-gr3', type=Path)
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


def get_interp_raster_window_requests(comm, interp_config):
    if comm.Get_rank() == 0:
        logger.info('Loading hfun raster window requests...')
        interp_raster_window_requests = list(iter_raster_window_requests(interp_config))
        logger.debug('Done loading hfun raster window requests.')
    else:
        interp_raster_window_requests = None
    interp_raster_window_requests = comm.bcast(interp_raster_window_requests, root=0)
    return interp_raster_window_requests


def interpolate_raster_to_mesh(
        msh_t,
        raster,
        # use_aa: bool = False,
        # min_ratio=0.1,
        # threshold_size=None,
        # min_area=np.finfo(np.float32).eps,
        # cross_section_node_count=4,
        # zmin=None,
        # zmax=None,
        # simplify_tolerance=None,
        # resample_distance=None,
        ):
    coords = np.array(msh_t.vert2['coord'])
    coords_crs = msh_t.crs
    idxs = []
    values = []
    for xi, yi, zi in raster:
        zi = zi[0, :]
        vert2_idxs = np.where(
            np.logical_and(
                np.logical_and(np.min(xi) <= coords[:, 0], np.max(xi) >= coords[:, 0]),
                np.logical_and(np.min(yi) <= coords[:, 1], np.max(yi) >= coords[:, 1]),
            )
        )[0]
        _coords = coords[vert2_idxs, :]
        if not raster.crs.equals(coords_crs):
            transformer = Transformer.from_crs(coords_crs, raster.crs, always_xy=True)
            _coords[:, 0], _coords[:, 1] = transformer.transform(_coords[:, 0], _coords[:, 1])
        _values = RegularGridInterpolator(
                (xi, yi),
                zi.T.astype(np.float64),
                'linear',
                bounds_error=False,
                fill_value=np.nan
                )(_coords)
        nan_idxs = np.where(np.isnan(_values))
        non_nan_idxs = np.where(~np.isnan(_values))
        # start = time()
        _values[nan_idxs] = NearestNDInterpolator(
                # xyzo[non_nan_idxs],
                coords[non_nan_idxs],
                _values[non_nan_idxs],
                )(coords[nan_idxs, :])
        idxs.append(vert2_idxs)
        values.append(_values)
    if len(idxs) == 0:
        return np.array([]), np.array([])
    values = np.hstack(values)
    return np.hstack(idxs), values


def find_narrow_channels(
        raster_path,
        request_opts,
        window,
        target_crs,
        min_ratio=0.1,
        threshold_size=None,
        min_area=np.finfo(np.float32).eps,
        cross_section_node_count=4,
        zmin=None,
        zmax=None,
        simplify_tolerance=None,
        resample_distance=None,
        ):
    request_opts = request_opts.copy()
    raster = get_raster_from_opts(raster_path, request_opts, window)
    raster.resampling_factor = 1
    if zmax is None:
        zmax = 0 if zmin is None else np.finfo(np.float64).max
    mp = Geom(raster, zmin=zmin, zmax=zmax).get_multipolygon()
    if mp is None:
        return gpd.GeoDataFrame(columns=['geometry'], crs=target_crs)
    centroid = np.array(mp.centroid.coords).flatten()
    local_azimuthal_projection = CRS.from_user_input(
        f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
        )
    gdf = gpd.GeoDataFrame([{'geometry': mp}], crs=raster.crs)
    original_gdf = gdf.to_crs(local_azimuthal_projection)

    if simplify_tolerance:
        original_gdf = original_gdf.simplify(tolerance=simplify_tolerance, preserve_topology=True)
    threshold_size = threshold_size or np.finfo(np.float32).max
    buffer_size = threshold_size * cross_section_node_count
    buffered_geom = original_gdf.unary_union.buffer(-buffer_size).buffer(buffer_size)
    if not buffered_geom.is_empty:
        final_patches = original_gdf.unary_union.difference(
            gpd.GeoDataFrame([{'geometry': buffered_geom}], crs=original_gdf.crs).unary_union
                )
    else:
        final_patches = original_gdf.unary_union
    if isinstance(final_patches, Polygon):
        final_patches = [final_patches]
    elif isinstance(final_patches, MultiPolygon):
        final_patches = [patch for patch in final_patches.geoms]
    elif isinstance(final_patches, GeometryCollection):
        return gpd.GeoDataFrame(columns=['geometry'], crs=target_crs)
    final_patches = [
            patch for patch in final_patches
            if patch.is_valid and not patch.is_empty
            and patch.area > min_area
            and (patch.length / patch.area) < min_ratio
        ]

    if simplify_tolerance is not None:
        final_patches = [patch.simplify(
            tolerance=float(simplify_tolerance),
            preserve_topology=True
            ) for patch in final_patches]

    from geomesh.geom.raster import resample_polygon

    if resample_distance is not None:
        final_patches = [resample_polygon(patch, resample_distance) for patch in final_patches]

    return gpd.GeoDataFrame(final_patches, columns=['geometry'], crs=original_gdf.crs).to_crs(target_crs)


def interpolate_topobathy(comm, interp_raster_window_requests, msh_t):
    interp_raster_window_requests = lib.get_split_array(comm, interp_raster_window_requests)[comm.Get_rank()]
    interp_data = []
    for global_k, ((i, raster_path, request_opts), (j, window)) in interp_raster_window_requests:
        logger.debug(f'Interpolating {raster_path=}  {window=}')
        raster = get_raster_from_opts(raster_path, request_opts, window)
        interp_data.append(interpolate_raster_to_mesh(
            msh_t,
            raster,
            ))
    # interp_data = [item for sublist in comm.allgather(interp_data) for item in sublist]
    logger.debug(f"Gathering interpolation data on rank {comm.Get_rank()}...")
    interp_data = comm.gather(interp_data, root=0)
    if comm.Get_rank() != 0:
        del interp_data
    # do final interp:
    value = None
    if comm.Get_rank() == 0:
        value = np.full(
                (msh_t.vert2['coord'].shape[0], 1),
                np.nan,
                dtype=jigsaw_msh_t.REALS_t
                )
        for rank_data in interp_data:
            for indexes, values in rank_data:
                if len(indexes) == 0:
                    continue
                value[indexes, :] = values.reshape((values.size, 1))
        logger.debug('Will do NaN padding now (only on rank 0)...')
        if np.any(np.isnan(value)):
            value = value.flatten()
            non_nan_idxs = np.where(~np.isnan(value))[0]
            nan_idxs = np.where(np.isnan(value))[0]
            value[nan_idxs] = griddata(
                    msh_t.vert2['coord'][non_nan_idxs, :],
                    value[non_nan_idxs],
                    msh_t.vert2['coord'][nan_idxs, :],
                    method='nearest'
                    )
            value = value.reshape((value.size, 1)).astype(jigsaw_msh_t.REALS_t)
    return comm.bcast(value, root=0)


def make_plot(mesh_tempfile):
    logger.info('Drawing plot...')
    pickle.load(open(mesh_tempfile, 'rb')).make_plot()
    plt.show(block=True)


# def to_msh(args, msh_t):
#     logger.info('Write msh_t...')
#     savemsh(f'{args.to_msh.resolve()}', msh_t)


def to_pickle(args, mesh_tempfile):
    logger.info('Write pickle...')
    pickle.dump(pickle.load(open(mesh_tempfile, 'rb')), open(args.to_pickle, 'wb'))
    logger.info('Done writing pickle...')


def to_gr3(args, mesh_tempfile):
    logger.info('write gr3 file...')
    pickle.load(open(mesh_tempfile, 'rb')).write(args.to_gr3, format='gr3')
    logger.info(f'Done writting gr3 file: {args.to_gr3}...')


def get_raster_window_data_for_aa(msh_t, raster_path, request_opts, window, interp_config):
    raster = get_raster_from_opts(raster_path, request_opts, window)
    min_ratio = interp_config.get('min_ratio', 0.1)
    threshold_size = interp_config.get('threshold_size', None)
    min_area = interp_config.get('min_area', np.finfo(np.float32).eps)
    cross_section_node_count = interp_config.get('cross_section_node_count', 4)
    zmin = interp_config.get('zmin', None)
    zmax = interp_config.get('zmax', None)
    simplify_tolerance = interp_config.get('simplify_tolerance', None)
    resample_distance = interp_config.get('resample_distance', None)
    channels_gdf = find_narrow_channels(
        raster_path,
        request_opts,
        window,
        msh_t.crs,
        min_ratio=min_ratio,
        threshold_size=threshold_size,
        min_area=min_area,
        cross_section_node_count=cross_section_node_count,
        zmin=zmin,
        zmax=zmax,
        simplify_tolerance=simplify_tolerance,
        resample_distance=resample_distance,
    )
    channels_sindex = channels_gdf.sindex
    raster_chunk_data = []
    for xi, yi, zi in raster:
        zi = zi[0, :]
        min_xi = xi.min()
        max_xi = xi.max()
        min_yi = yi.min()
        max_yi = yi.max()
        # transform to mesh crs if needed:
        if not raster.crs.equals(msh_t.crs):
            transformer = Transformer.from_crs(raster.crs, msh_t.crs, always_xy=True)
            min_xi, min_yi = transformer.transform(min_xi, min_yi)
            max_xi, max_yi = transformer.transform(max_xi, max_yi)
        vert2_idxs = np.where(
            np.logical_and(
                np.logical_and(min_xi <= msh_t.vert2['coord'][:, 0], max_xi >= msh_t.vert2['coord'][:, 0]),
                np.logical_and(min_yi <= msh_t.vert2['coord'][:, 1], max_yi >= msh_t.vert2['coord'][:, 1]),
            )
        )[0]
        # _coords = msh_t.vert2['coord'][vert2_idxs, :]
        # if not raster.crs.equals(msh_t.crs):
        #     transformer = Transformer.from_crs(msh_t.crs, raster.crs, always_xy=True)
        #     _coords[:, 0], _coords[:, 1] = transformer.transform(_coords[:, 0], _coords[:, 1])
        mask = np.isin(msh_t.tria3['index'], vert2_idxs).any(axis=1)
        triangles_list = [Polygon([msh_t.vert2['coord'][i] for i in triangle_idx])
                          for triangle_idx in msh_t.tria3['index'][mask]]
        # msh_t.crs matches channels_gdf.crs at this point
        triangles_gdf = gpd.GeoDataFrame(geometry=triangles_list, crs=msh_t.crs)

        # we will store indexes of the triangles only partially within polygons here
        partial_overlap_idx = []
        intersections = []
        logger.debug('Finding intersections...')
        for idx, triangle in triangles_gdf.iterrows():
            # find approximate matches with r-tree, then precise matches from those approximate ones
            possible_matches_index = list(channels_sindex.intersection(triangle.geometry.bounds))
            possible_matches = channels_gdf.iloc[possible_matches_index]
            possible_matches = possible_matches[possible_matches.geometry.is_valid]
            precise_matches = possible_matches[possible_matches.intersects(triangle.geometry)]
            # check if triangle is only partially within any polygon and has overlapping area >= 50% of the triangle area
            for _, possible_match in precise_matches.iterrows():
                # calculate intersection
                intersection = triangle.geometry.intersection(possible_match.geometry)
                # calculate intersection area
                intersection_area = intersection.area
                # check if triangle is not completely within the polygon and
                # if the intersection area is >= 50% of the triangle area or
                # if the intersection is entirely within the triangle
                if (not triangle.geometry.within(possible_match.geometry)
                        and intersection_area >= (0.5 * triangle.geometry.area))\
                        or triangle.geometry.contains(intersection):
                    partial_overlap_idx.append(idx)
                    intersections.append(intersection)
                    break

        # select triangles that are partially within the polygons
        selected_triangles = triangles_gdf.loc[partial_overlap_idx]
        intersections = gpd.GeoDataFrame(geometry=intersections, crs=selected_triangles.crs).to_crs(raster.crs)
        # Compute mean values of intersections from the raster data
        logger.debug('computing mean values...')
        with rasterio.open(raster._tmpfile) as src:
            mean_values = []
            for geometry in intersections.geometry:
                geoms = [mapping(geometry)]
                out_image, out_transform = riomask(src, geoms, crop=True)
                valid_data = out_image[out_image != src.nodata]
                if valid_data.size > 0:  # Check if there's any valid data
                    mean_value = valid_data.mean()
                else:
                    mean_value = np.nan  # Use np.nan for cases with no valid data
                mean_values.append(mean_value)

        selected_triangles['mean_values'] = mean_values

        from collections import defaultdict
        # Initialize the dictionaries to store the sum and count for each node
        # Initialize the dictionaries to store the sum and count for each node
        node_sums = defaultdict(float)
        node_counts = defaultdict(int)

        # Iterate over the selected triangles and their corresponding mean values
        for triangle, mean_val in zip(selected_triangles.index, mean_values):
            # Get the nodes for this triangle from the connectivity table
            triangle_nodes = msh_t.tria3['index'][triangle]

            # Update the sum and count for each node
            for node in triangle_nodes:
                if np.isfinite(mean_val):  # Ignore mean values that are not finite (NaN or Inf)
                    node_sums[node] += mean_val
                    node_counts[node] += 1

        # Calculate the mean value for each node
        node_means = np.full((msh_t.vert2['coord'].shape[0],), np.nan)  # Initialize with NaN values
        non_zero_counts = [node for node, count in node_counts.items() if count > 0]  # Nodes with non-zero counts

        for node in non_zero_counts:
            node_means[node] = node_sums[node] / node_counts[node]

        # Replace values that are above the threshold only if the corresponding node mean value is not NaN
        threshold = -0.1
        local_values = np.array(msh_t.value[vert2_idxs])
        threshold_indexes = np.where((local_values > threshold) & np.isfinite(node_means[vert2_idxs]))[0]
        local_values[threshold_indexes] = node_means[vert2_idxs][threshold_indexes]
        raster_chunk_data.append((vert2_idxs, local_values))
    # concatenate the raster_chunks
    # Create an empty dictionary
    index_dict = {}
    # Iterate over each chunk
    for chunk in raster_chunk_data:
        vert2_idxs, local_values = chunk
        # Iterate over each index and corresponding value
        for idx, value in zip(vert2_idxs, local_values):
            # Check if index is already in dictionary
            if idx in index_dict:
                # Update sum and count
                sum_value, count = index_dict[idx]
                index_dict[idx] = (sum_value + value, count + 1)
            else:
                # Add new entry
                index_dict[idx] = (value, 1)

    # Now, calculate the averages
    average_dict = {}
    for idx, (sum_value, count) in index_dict.items():
        average_dict[idx] = sum_value / count

    # Create list of tuples (index, value)
    average_list = [(idx, value) for idx, value in average_dict.items()]
    # Sort list by index
    average_list.sort(key=lambda x: x[0])
    # Create arrays with indexes and values
    local_raster_idxs = [idx for idx, value in average_list]
    local_raster_values = [value for idx, value in average_list]

    return local_raster_idxs, local_raster_values


def split_into_groups(arr, n_groups):
    # calculate the size of each group
    size = len(arr) // n_groups
    # if the array cannot be evenly divided, add one more element to the last group
    remainder = len(arr) % n_groups

    groups = []
    start = 0
    for i in range(n_groups):
        # the last group gets the remainder
        end = start + size + (i < remainder)
        groups.append(arr[start:end])
        start = end

    return groups


def do_use_aa(comm, mesh,  interp_raster_window_requests, interp_config):

    # Algorith description:
    # 1. Find all narrow channels in the raster
    # 2. Find all elements crossing the narrow channels.
    # 3. Find all raster points that are inside both the narrow channels and element.
    # 4. Assign the mean value of the raster points found in #3 to all element nodes.

    msh_t = mesh.msh_t
    interp_config = interp_config.copy()
    interp_config.pop('rasters', None)
    # 1. Find all narrow channels in the raster
    # min_ratio = interp_config.get('min_ratio', 0.1)
    # threshold_size = interp_config.get('threshold_size', None)
    # min_area = interp_config.get('min_area', np.finfo(np.float32).eps)
    # cross_section_node_count = interp_config.get('cross_section_node_count', 4)
    # zmin = interp_config.get('zmin', None)
    # zmax = interp_config.get('zmax', None)
    # simplify_tolerance = interp_config.get('simplify_tolerance', None)
    # resample_distance = interp_config.get('resample_distance', None)

    # channels_gdf = []
    # for global_k, ((i, raster_path, request_opts), (j, window)) in lib.get_split_array(comm, interp_raster_window_requests)[comm.Get_rank()]:
    #     channels_gdf.append(
    #         find_narrow_channels(
    #             raster_path,
    #             request_opts,
    #             window,
    #             msh_t.crs,
    #             min_ratio=min_ratio,
    #             threshold_size=threshold_size,
    #             min_area=min_area,
    #             cross_section_node_count=cross_section_node_count,
    #             zmin=zmin,
    #             zmax=zmax,
    #             simplify_tolerance=simplify_tolerance,
    #             resample_distance=resample_distance,
    #         ))
    
    # logger.debug('Gathering channels_gdf...')
    # # channels_gdf is guaranteed to be at msh_t.crs at the concat point.
    # channels_gdf = pd.concat([item for sublist in comm.allgather(channels_gdf) for item in sublist])

    # logger.debug('got channels_gdf')

    # if len(channels_gdf) == 0:
    #     return
    # logger.debug('will build channels_sindex')
    # channels_sindex = channels_gdf.sindex
    rank = comm.Get_rank()
    # subdivide interp_raster_window_requests into local_colors groups
    hwinfo = lib.hardware_info(comm)
    unique_colors = hwinfo['color'].unique()
    local_color = hwinfo.iloc[rank]['color']
    # color_sizes = hwinfo['color'].value_counts()
    local_comm = comm.Split(local_color, rank)
    local_jobs = split_into_groups(
            [(global_k, row) for global_k, row in enumerate(interp_raster_window_requests)],
            len(unique_colors)
            )[local_color]
    local_results = None
    with MPICommExecutor(local_comm) as executor:
        if executor is not None:
            local_results = []
            for global_k, ((i, raster_path, request_opts), (j, window)) in local_jobs:
                local_results.append(executor.submit(
                    get_raster_window_data_for_aa,
                    # channels_gdf,
                    # channels_sindex,
                    msh_t,
                    raster_path,
                    request_opts,
                    window,
                    interp_config,
                    ).result())
    all_results = comm.gather(local_results, root=0)
    print(f"{rank=} finished with results {all_results=}")
    comm.barrier()
    exit()


def main(args):
    """
    This program takes a pickle with a jigsaw_msh_t and interpolates the bathymetry
    as defined by the config file. It will also build the boundaries.
    The output can be directly saved as a model-ready "gr3" file.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    interpolate_config = None
    boundaries_config = None
    if rank == 0:
        from geomesh.cli.build import BuildCli
        cli_config = BuildCli(args).config
        interpolate_config = cli_config.interpolate.interpolate_config
    interpolate_config = comm.bcast(interpolate_config, root=0)
    mesh = pickle.load(open(args.msh_t_pickle, 'rb'))

    interp_raster_window_requests = get_interp_raster_window_requests(comm, interpolate_config)
    mesh.msh_t.value = interpolate_topobathy(comm, interp_raster_window_requests, mesh)

    use_aa = bool(interpolate_config.get('use_aa', False))
    if use_aa:
        do_use_aa(comm, mesh, interp_raster_window_requests, interpolate_config)

    if rank != 0:
        mesh = None

    if rank == 0:
        boundaries_config = cli_config.boundaries.boundaries_config
        if boundaries_config is not None:
            mesh.boundaries.auto_generate(**boundaries_config)

    if rank == 0:
        cache_directory = Path(os.getenv("GEOMESH_TEMPDIR", Path.cwd() / '.tmp')) / 'build_hgrid'
        cache_directory.mkdir(parents=True, exist_ok=True)
        _mesh_tempfile = tempfile.NamedTemporaryFile(dir=cache_directory, suffix='.pkl')
        mesh_tempfile = Path(_mesh_tempfile.name)
        with open(mesh_tempfile, 'wb') as fh:
            pickle.dump(mesh, fh)
        mesh = None
    else:
        mesh_tempfile = None

    mesh_tempfile = comm.bcast(mesh_tempfile, root=0)

    finalization_tasks = []

    if args.show:
        finalization_tasks.append(make_plot)

    # if args.to_msh:
    #     finalization_tasks.append(partial(to_msh, args))

    if args.to_pickle:
        finalization_tasks.append(partial(to_pickle, args))

    if args.to_gr3:
        finalization_tasks.append(partial(to_gr3, args))
    futures = []
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            for task in finalization_tasks:
                futures.append(executor.submit(task, mesh_tempfile))
            [future.result() for future in futures]


def entrypoint():
    sys.excepthook = mpiabort_excepthook
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        args = get_argument_parser().parse_args()
    else:
        args = None
    print("bcast args", flush=True)
    args = comm.bcast(args, root=0)
    init_logger(args.log_level)
    main(args)


if __name__ == "__main__":
    entrypoint()
