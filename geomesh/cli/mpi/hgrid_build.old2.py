from functools import partial
from pathlib import Path
import argparse
import gc
import logging
import os
import pickle
import queue
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
from shapely.geometry import mapping, Polygon, MultiPolygon
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from geomesh.cli.mpi import lib
from geomesh import Geom
from geomesh.cli.schedulers.local import LocalCluster
from geomesh.cli.raster_opts import iter_raster_window_requests, get_raster_from_opts


logger = logging.getLogger(__name__)


def mpiabort_excepthook(type, value, traceback):
    Colorizer('default').colorize_traceback(type, value, traceback)
    MPI.COMM_WORLD.Abort(-1)


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
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--to-gr3', type=Path)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=Path)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


def interpolate_coords(xi, yi, zi, coords):
    values = RegularGridInterpolator(
            (xi, yi),
            zi.T.astype(np.float64),
            'linear',
            bounds_error=False,
            fill_value=np.nan
            )(coords)
    nan_idxs = np.where(np.isnan(values))
    non_nan_idxs = np.where(~np.isnan(values))
    values[nan_idxs] = NearestNDInterpolator(
            coords[non_nan_idxs],
            values[non_nan_idxs],
            )(coords[nan_idxs, :])
    return values


def interpolate_raster_to_mesh(
        msh_t_pickle_path,
        coords,
        raster_path,
        request_opts,
        window,
        output_path,
        cpus_needed,
        use_aa: bool = False,
        min_ratio=0.1,
        threshold_size=None,
        min_area=np.finfo(np.float32).eps,
        cross_section_node_count=4,
        zmin=None,
        zmax=None,
        simplify_tolerance=None,
        resample_distance=None,
        ):
    msh_t = pickle.load(open(msh_t_pickle_path, 'rb'))
    coords = np.array(msh_t.vert2['coord'])
    coords_crs = msh_t.crs
    del msh_t
    gc.collect()
    idxs = []
    raster = get_raster_from_opts(raster_path, request_opts, window)
    if not raster.crs.equals(coords_crs):
        transformer = Transformer.from_crs(coords_crs, raster.crs, always_xy=True)
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
            _coords[:, 0], _coords[:, 1] = transformer.transform(_coords[:, 0], _coords[:, 1])
        idxs.append(vert2_idxs)
        values.append(interpolate_coords(xi, yi, zi, _coords))
        del xi, yi, zi, _coords
        gc.collect()
    if len(idxs) == 0:
        return (np.array([]), np.array([]))
    values = np.hstack(values)
    idxs, values = np.hstack(idxs), values
    if use_aa:
        force_channel_open(
                msh_t_pickle_path,
                raster,
                idxs,
                values,
                min_ratio=min_ratio,
                threshold_size=threshold_size,
                min_area=min_area,
                cross_section_node_count=cross_section_node_count,
                zmin=zmin,
                zmax=zmax,
                simplify_tolerance=simplify_tolerance,
                resample_distance=resample_distance,
                )
    values = values.reshape((values.size, 1))
    pickle.dump((idxs, values), open(output_path, 'wb'))
    return output_path, cpus_needed


def force_channel_open(
        msh_t_pickle_path,
        raster,
        idxs,
        values,
        min_ratio=0.1,
        threshold_size=None,
        min_area=np.finfo(np.float32).eps,
        cross_section_node_count=4,
        zmin=None,
        zmax=None,
        simplify_tolerance=None,
        resample_distance=None,
        ):
    # Algorith description:
    # 1. Find all narrow channels in the raster
    # 2. Find all elements crossing the narrow channels.
    # 3. Find all raster points that are inside both the narrow channels and element.
    # 4. Assign the mean value of the raster points found in #3 to all element nodes.

    # 1. Find all narrow channels in the raster

    channels_gdf = find_narrow_channels(
            raster,
            min_ratio=min_ratio,
            threshold_size=threshold_size,
            min_area=min_area,
            cross_section_node_count=cross_section_node_count,
            zmin=zmin,
            zmax=zmax,
            simplify_tolerance=simplify_tolerance,
            resample_distance=resample_distance,
            )
    if len(channels_gdf) > 0:

        # 2. Find all elements crossing the narrow channels.
        # First, subset msh_t using indxs
        msh_t = pickle.load(open(msh_t_pickle_path, 'rb'))
        channels_gdf = channels_gdf.to_crs(msh_t.crs)
        # channels_sindex = channels_gdf.sindex
        # assuming that your triangles are a list of Shapely Polygon objects
        triangles_list = [Polygon([msh_t.vert2['coord'][i] for i in triangle_idx]) for triangle_idx in msh_t.tria3['index']]
        del msh_t
        gc.collect()
        triangles_gdf = gpd.GeoDataFrame(geometry=triangles_list, crs=channels_gdf.crs)

        # spatial index for faster operation
        channels_sindex = channels_gdf.sindex

        # we will store indexes of the triangles only partially within polygons here
        partial_overlap_idx = []
        intersections = []
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
        threshold_indexes = np.where((values > threshold) & np.isfinite(node_means[idxs]))[0]
        values[threshold_indexes] = node_means[idxs][threshold_indexes]


def find_narrow_channels(
        raster,
        min_ratio=0.1,
        threshold_size=None,
        min_area=np.finfo(np.float32).eps,
        cross_section_node_count=4,
        zmin=None,
        zmax=None,
        simplify_tolerance=None,
        resample_distance=None,
        ):
    if zmax is None:
        zmax = 0 if zmin is None else np.finfo(np.float64).max
    _old_rf = raster.resampling_factor
    raster.resampling_factor = None
    mp = Geom(raster, zmin=zmin, zmax=zmax).get_multipolygon()
    if mp is None:
        return gpd.GeoDataFrame(columns=['geometry'], crs=raster.crs)
    raster.resampling_factor = _old_rf
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

    return gpd.GeoDataFrame(final_patches, columns=['geometry'], crs=original_gdf.crs)



def make_plot(mesh_tempfile):
    logger.info('Drawing plot...')
    pickle.load(open(mesh_tempfile, 'rb')).make_plot()
    plt.show(block=True)


# def to_msh(args, msh_t):
#     logger.info('Write msh_t...')
#     savemsh(f'{args.to_msh.resolve()}', msh_t)


def to_pickle(args, mesh_tempfile):
    logger.info('Write pickle...')
    with open(args.to_pickle, 'wb') as fh:
        pickle.dump(pickle.load(mesh_tempfile, 'rb'), fh)
    logger.info('Done writing pickle...')


def to_gr3(args, mesh_tempfile):
    logger.info('write gr3 file...')
    pickle.load(open(mesh_tempfile, 'rb')).write(args.to_gr3, format='gr3')
    logger.info(f'Done writting gr3 file: {args.to_gr3}...')


def split_list(input_list, num_chunks):
    avg_len = len(input_list) / float(num_chunks)
    return [input_list[int(round(avg_len * i)): int(round(avg_len * (i + 1)))] for i in range(num_chunks)]


def main(args):
    """
    This program takes a pickle with a jigsaw_msh_t and interpolates the bathymetry
    as defined by the config file. It will also build the boundaries.
    The output can be directly saved as a model-ready "gr3" file.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hwinfo = lib.hardware_info(comm)
    unique_colors = hwinfo['color'].unique()
    local_color = hwinfo.iloc[rank]['color']
    color_sizes = hwinfo['color'].value_counts()
    local_comm = comm.Split(local_color, rank)
    cache_directory = args.cache_directory or Path(
            os.getenv("GEOMESH_TEMPDIR", Path.cwd() / '.tmp-geomesh')) / f'{Path(__file__).name}'
    if rank == 0:
        cache_directory.mkdir(exist_ok=True, parents=True)
        from geomesh.cli.build import BuildCli
        cli_config = BuildCli(args).config
        interp_config = cli_config.interpolate.interpolate_config.copy()
        interp_raster_window_requests = list(iter_raster_window_requests(interp_config))
        interp_config.pop('rasters', None)
        if len(unique_colors) > 1:
            interp_raster_window_requests = split_list(interp_raster_window_requests, len(unique_colors))
        else:
            interp_raster_window_requests = [interp_raster_window_requests]
        interp_config = {
                'use_aa': interp_config.get('use_aa', False),
                'min_ratio': interp_config.get('min_ratio', 0.1),
                'threshold_size': interp_config.get('threshold_size', None),
                'min_area': interp_config.get('min_area', np.finfo(np.float32).eps),
                'cross_section_node_count': interp_config.get('cross_section_node_count', 4),
                'zmin': interp_config.get('zmin', None),
                'zmax': interp_config.get('zmax', None),
                'simplify_tolerance': interp_config.get('simplify_tolerance', None),
                'resample_distance': interp_config.get('resample_distance', None),
                }
    else:
        interp_raster_window_requests = None
        interp_config = None
    interp_raster_window_requests = comm.bcast(interp_raster_window_requests, root=0)[local_color]
    interp_config = comm.bcast(interp_config, root=0)
    cpus_needed = 1
    jobs = []
    for global_k, ((i, raster_path, request_opts), (j, window)) in enumerate(interp_raster_window_requests):
        # ( func, args, cpus_needed )
        output_path = Path(tempfile.NamedTemporaryFile(dir=cache_directory).name)
        jobs.append((
                interpolate_raster_to_mesh,
                (
                    args.msh_t_pickle,
                    raster_path,
                    request_opts,
                    window,
                    output_path,
                    cpus_needed,
                    interp_config['use_aa'],
                    interp_config['min_ratio'],
                    interp_config['threshold_size'],
                    interp_config['min_area'],
                    interp_config['cross_section_node_count'],
                    interp_config['zmin'],
                    interp_config['zmax'],
                    interp_config['simplify_tolerance'],
                    interp_config['resample_distance'],
                ),
                cpus_needed,
            ))

    # Create a queue for jobs
    job_queue = queue.PriorityQueue()

    # Add jobs to queue, prioritized by number of processors needed
    for job, args, procs_needed in sorted(jobs, key=lambda x: x[2]):
        args = pickle.dumps(args)
        job_queue.put((-procs_needed, (job, args, procs_needed)))  # Negate procs_needed to sort in descending order

    # Create a list to hold futures
    futures = []
    results = []
    with MPICommExecutor(local_comm) as executor:
        if executor is not None:
            free_procs = color_sizes[local_color]
            while not job_queue.empty() or futures:
                # Remove finished futures and update the number of free processors
                for future in futures:
                    if future.done():
                        output_path, fprocs = future.result()
                        gc.collect()
                        free_procs += fprocs  # Assumes the job returns the number of processors it used
                        futures.remove(future)
                        if output_path is not None:
                            results.append(output_path)
                # ------- Option 1
                # Keep track of jobs we can't fit right now
                deferred_jobs = []

                # Try to fill all available processors
                while not job_queue.empty() and free_procs > 0:
                    _, (job, args, procs_needed) = job_queue.get()
                    if procs_needed <= free_procs:
                        args = pickle.loads(args)
                        future = executor.submit(job, *args)
                        futures.append(future)
                        free_procs -= procs_needed
                        from time import sleep
                        gc.collect()
                        sleep(2)
                    else:
                        # If a job can't be scheduled, defer it
                        deferred_jobs.append((-procs_needed, (job, args, procs_needed)))

                # Return deferred jobs back into the queue
                for job in deferred_jobs:
                    job_queue.put(job)
    outputs = [item for sublist in comm.allgather(results) for item in sublist]
    if rank == 0:
        print(outputs, flush=True)
    comm.barrier()
    exit()
    if comm.Get_rank() == 0:
        value = np.full(
                (msh_t.vert2['coord'].shape[0], 1),
                np.nan,
                dtype=jigsaw_msh_t.REALS_t
                )
        for indexes, values in interp_data:
            value[indexes, :] = values
        print('will do NaN padding now', flush=True)
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
    print('will do final bcast', flush=True)
    return comm.bcast(value, root=0)

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

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            # for task in finalization_tasks:
            #     executor.submit(task, mesh_tempfile).result()
            [_.result() for _ in [executor.submit(task, mesh_tempfile) for task in finalization_tasks]]


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
    main(args)


if __name__ == "__main__":
    entrypoint()
