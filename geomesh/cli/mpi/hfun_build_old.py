#!/usr/bin/env python
from functools import partial
from pathlib import Path
from time import time
import argparse
import hashlib
import json
import logging
import os
import pickle
import queue
import sys
import tempfile
import warnings

from colored_traceback.colored_traceback import Colorizer
from jigsawpy import jigsaw_msh_t, savemsh
from matplotlib.transforms import Bbox
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geomesh import Geom, Hfun
from geomesh import utils
from geomesh.cli.build import BuildCli
from geomesh.cli.mpi import lib
from geomesh.cli.schedulers.local import LocalCluster
from geomesh.cli.raster_opts import iter_raster_window_requests, get_raster_from_opts
from geomesh.hfun.raster import RasterHfun

warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
# warnings.filterwarnings("error")

logger = logging.getLogger(f'[rank: {MPI.COMM_WORLD.Get_rank()}]: {__name__}')


def mpiabort_excepthook(type, value, traceback):
    Colorizer('default').colorize_traceback(type, value, traceback)
    MPI.COMM_WORLD.Abort(-1)


def submit(
        executor,
        config_path,
        to_msh=None,
        to_pickle=None,
        dst_crs=None,
        show=False,
        cache_directory=None,
        log_level: str = None,
        **kwargs
        ):
    if isinstance(executor, LocalCluster):
        raise TypeError('LocalCluster is not supported, use MPICluster instead.')
    delete_pickle = False if to_pickle is not None else True
    if delete_pickle is True and cache_directory is None:
        cache_directory = cache_directory or Path(os.getenv('GEOMESH_TEMPDIR', Path.cwd() / f'.tmp-geomesh/{Path(__file__).stem}'))
        cache_directory.mkdir(parents=True, exist_ok=True)
    if delete_pickle is True:
        to_pickle = Path(tempfile.NamedTemporaryFile(dir=cache_directory, suffix='.pkl').name)
    build_cmd = get_cmd(
            config_path,
            to_msh=to_msh,
            to_pickle=to_pickle,
            dst_crs=dst_crs,
            show=show,
            cache_directory=cache_directory,
            log_level=log_level,
            )

    async def callback():
        await executor.submit(build_cmd, **kwargs)
        hfun = pickle.load(open(to_pickle, 'rb'))
        if delete_pickle:
            to_pickle.unlink()
        return hfun

    return callback()


def get_cmd(
        config_path,
        to_msh=None,
        to_pickle=None,
        dst_crs=None,
        show=False,
        cache_directory=None,
        log_level: str = None,
        ):
    build_cmd = [
            sys.executable,
            f'{Path(__file__).resolve()}',
            f'{Path(config_path).resolve()}',
            ]
    if to_msh is not None:
        build_cmd.append(f'--to-file={to_msh}')
    if to_pickle is not None:
        build_cmd.append(f'--to-pickle={to_pickle}')
    if show:
        build_cmd.append('--show')
    if cache_directory is not None:
        build_cmd.append(f'--cache-directory={cache_directory}')
    if dst_crs is not None:
        build_cmd.append(f'--dst-crs={dst_crs}')
    if log_level is not None:
        build_cmd.append(f'--log-level={log_level}')
    logger.debug(' '.join(build_cmd))
    return build_cmd


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
    # if int(log_level) < 40:
    #     logging.getLogger("geomesh").setLevel(log_level)
    # logging.Formatter.converter = lambda *args: datetime.now(tz=pytz.timezone("UTC")).timetuple()
    logging.captureWarnings(True)


def get_gdf_from_axes(axes, contour_opts, crs):

    # path_collection = axes.Collection[0]
    level = contour_opts['level']
    level_data = {}
    for i, (_level, path_collection) in enumerate(zip(axes.levels, axes.collections)):
        if _level != level:
            continue
        linestrings = []
        for path in path_collection.get_paths():
            try:
                linestrings.append(LineString(path.vertices))
            except ValueError as e:
                if "LineStrings must have at least 2 coordinate tuples" in str(e):
                    continue
                raise e
        level_data.update({level: MultiLineString(linestrings)})
        # break
        # print(i, level)
        # if level == args.level:
        #     level_data.update({level: MultiLineString(linestrings)})
    if len(level_data) == 0:
        return None

    return gpd.GeoDataFrame([{
        'geometry': MultiLineString(linestrings),
        'level': level,
        'expansion_rate': contour_opts.get('expansion_rate'),
        'target_size': contour_opts.get('target_size'),
        # 'color': rgb2hex(axes.collections[i].get_facecolor())
        } for i, (level, data) in enumerate(level_data.items())],
        crs=crs
        ).to_crs(CRS.from_epsg(4326))


def get_contours_gdf_from_raster(raster_path, request_opts, window, raster_hash, contour_request, cache_directory):
    cache_directory /= 'raster_window_contours'
    cache_directory.mkdir(exist_ok=True)
    normalized_contour_request = {
        'level': contour_request['level'],
        'expansion_rate': contour_request.get('expansion_rate'),
        'target_size': contour_request.get('target_size'),
        }
    normalized_contour_request.update(lib.get_normalized_raster_request_opts(raster_hash, request_opts, window))
    tmpfile = cache_directory / (hashlib.sha256(json.dumps(normalized_contour_request).encode()).hexdigest() + '.feather')
    if tmpfile.is_file():
        logger.debug(f"Load contours for {normalized_contour_request=} from {tmpfile=}")
        return gpd.read_feather(tmpfile)

    logger.debug(f"Building contours for {normalized_contour_request=} to {tmpfile=}")
    raster = get_raster_from_opts(raster_path, request_opts, window)
    gdf = []
    for xvals, yvals, zvals in raster:
        try:
            # plt.ioff()
            import matplotlib as mpl
            _old_backend = mpl.get_backend()
            plt.switch_backend('agg')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                ax = plt.contour(xvals, yvals, zvals[0, :], levels=[contour_request['level']])
            plt.close(plt.gcf())
            plt.switch_backend(_old_backend)
            _gdf = get_gdf_from_axes(ax, contour_request, raster.crs)
            # _gdf.to_crs(CRS.from_epsg(4326), inplace=True)
            if _gdf is not None:
                gdf.append(_gdf)
        except TypeError as err:
            if 'Input z must be at least a (2, 2) shaped array, but has shape' in str(err):
                plt.switch_backend(_old_backend)
                continue
            raise
    if len(gdf) == 0:
        return None
    gdf = pd.concat(gdf, ignore_index=True)
    # gdf = gdf.loc[gdf['geometry'].is_valid, :]
    # gdf = gdf.loc[~gdf['geometry'].is_empty, :]
    gdf.to_feather(tmpfile)
    return gdf


def get_hfun_contours_gdf(comm, hfun_raster_window_requests, bbox_gdf, cache_directory, raster_hashes):
    hfun_contours_requests = []
    if comm.Get_rank() == 0:
        for global_k, ((i, raster_path, request_opts), (j, window)) in enumerate(hfun_raster_window_requests):
            if 'contours' in request_opts:
                for contour_request in request_opts['contours']:
                    hfun_contours_requests.append(((global_k, ((i, raster_path, request_opts), (j, window))), contour_request))
        hfun_contours_requests = np.array(hfun_contours_requests, dtype=object)
        hfun_contours_requests = np.array_split(hfun_contours_requests, comm.Get_size())
    hfun_contours_requests = comm.bcast(hfun_contours_requests, root=0)[comm.Get_rank()]
    contours_gdf = []
    for (global_k, ((i, raster_path, request_opts), (j, window))), contour_request in hfun_contours_requests:
        local_gdf = get_contours_gdf_from_raster(
                raster_path,
                request_opts,
                window,
                raster_hashes[global_k],
                contour_request,
                cache_directory
                )
        if local_gdf is None:
            continue
        idxs = bbox_gdf.iloc[:global_k].sindex.query(local_gdf.unary_union)
        if len(idxs) > 0:
            local_gdf = local_gdf.difference(bbox_gdf.iloc[idxs].unary_union)
            local_gdf = gpd.GeoDataFrame([{'geometry': local_gdf.unary_union, **contour_request}], crs=bbox_gdf.crs)
        contours_gdf.append(local_gdf.to_crs('epsg:4326'))
    contours_gdf = pd.concat([item for sublist in comm.allgather(contours_gdf) for item in sublist], ignore_index=True)
    # verify
    # if comm.Get_rank() == 0:
    #     contours_gdf.plot(cmap='jet')
    #     plt.show(block=True)
    return contours_gdf


def collapse_by_required_columns(gdf):
    required_columns = set(['level', 'target_size', 'expansion_rate'])
    data = []
    for indexes in gdf.groupby(list(required_columns)).groups.values():
        # if len(indexes) == 0:
        #     continue
        rows = gdf.iloc[indexes]
        geometries = []
        for row in rows.itertuples():
            if row.geometry.is_empty:
                continue
            elif isinstance(row.geometry, LineString):
                geometries.append(row.geometry)
            elif isinstance(row.geometry, MultiLineString):
                for ls in row.geometry.geoms:
                    if not ls.is_empty:
                        geometries.append(ls)
        data.append({
            'geometry': MultiLineString(geometries),
            'level': gdf.iloc[indexes[0]].level,
            'target_size': gdf.iloc[indexes[0]].target_size,
            'expansion_rate': gdf.iloc[indexes[0]].expansion_rate,
            })
    return gpd.GeoDataFrame(data, crs=gdf.crs)


def get_local_msh_t_tmpfile(
        raster_path,
        request_opts,
        hfun_config,
        window,
        nprocs,
        local_contours_path,
        bbox_gdf,
        i,
        global_k,
        callback
        ):

    raster = get_raster_from_opts(raster_path, request_opts, window)

    base_mp = bbox_gdf.iloc[[global_k]]
    # if i > 0:
    others_gdf = bbox_gdf.iloc[:i]
    if len(others_gdf) > 0:
        others_rtree = others_gdf.sindex
        others_intersects = others_gdf.loc[others_rtree.query(base_mp.unary_union)]
        if len(others_intersects) > 0:
            base_mp = base_mp.difference(others_intersects.unary_union)
    del others_gdf

    base_unary_union = base_mp.unary_union
    if base_unary_union is None:
        return None, nprocs
    if base_unary_union.is_empty:
        # entirely covered
        return None, nprocs
    windows_geom = Geom(base_unary_union, crs=base_mp.crs)

    # verify windows_geom
    # if len(others_intersects) > 0:
    #     others_intersects.plot(ax=plt.gca(), facecolor='none', edgecolor='r', linestyle='--')
    #     windows_geom.make_plot(axes=plt.gca(), linestyle='.-')
    #     plt.title(f'{raster.bbox=}')
    #     plt.show(block=True)
    hfun = RasterHfun(
        raster=raster,
        hmin=hfun_config.get('hmin'),
        hmax=hfun_config.get('hmax'),
        verbosity=hfun_config.get('verbosity', 0),
        marche=hfun_config.get('marche', False),
        nprocs=nprocs,
        geom=windows_geom,
    )

    if local_contours_path is not None:
        local_contours = gpd.read_feather(local_contours_path)
        if len(local_contours) > 0:
            for k, row in local_contours.iterrows():
                row = row.copy()
                row.pop('level')
                row['feature'] = row.pop('geometry')
                # add as features bc theire precomputed + use additionals
                # logging.getLogger("geomesh").setLevel(logging.DEBUG)
                start = time()
                logger.info('start adding contour constraint ..')
                hfun.add_feature(**row)
                # logging.getLogger("geomesh").setLevel(logging.WARNING)
                logger.debug(f'adding contour constraints took {time()-start} on {nprocs=}')
        del local_contours
    for request_type, request_values in request_opts.items():
        # special handle for contours/features
        if request_type == 'contours':
            pass
        elif request_type == 'features':
            pass
            # raise NotImplementedError

        # intercept speecific request with line_profiler (for debugging)
        # elif request_type == 'narrow_channel_anti_aliasing':
        #     from line_profiler import LineProfiler
        #     lp = LineProfiler()
        #     lp_wrapper = lp(hfun.add_narrow_channel_anti_aliasing)

        #     if not isinstance(request_values, list):
        #         request_values = [request_values]
        #     for _request_values in request_values:
        #         lp_wrapper(**_request_values)
        #         lp.print_stats()

        # forward any additional requests
        else:
            if hasattr(hfun, f'add_{request_type}'):
                # logger.debug(f'{request_type=}')
                constraint_adder = getattr(hfun, f'add_{request_type}')
                if not isinstance(request_values, list):
                    request_values = [request_values]
                for _request_values in request_values:
                    logger.info(f'start adding {request_type} constraint to {raster_path}..')
                    start = time()
                    # no retry version
                    try:
                        constraint_adder(**_request_values)
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()  # get the traceback
                        logger.debug(f'adding {request_type} constraint failed for raster {raster_path}: {e}\n{tb}')
                        raise
                    # with retry
                    # retry_count = 0
                    # max_retries = 5
                    # while retry_count < max_retries:
                    #     try:
                    #         constraint_adder(**_request_values)
                    #         break
                    #     except OSError as e:
                    #         if str(e) == '[Errno 12] Cannot allocate memory':
                    #             retry_count += 1
                    #             if retry_count >= max_retries:
                    #                 raise
                    #     except Exception as e:
                    #         import traceback
                    #         tb = traceback.format_exc()  # get the traceback
                    #         logger.debug(f'adding {request_type} constraint failed for raster {raster_path}: {e}\n{tb}')
                    #         raise

                    logger.debug(f'adding {request_type} constraint took {time()-start}')
    logger.info('Generating msh_t')
    start = time()
    msh_t = hfun.msh_t(dst_crs=CRS.from_epsg(4326))
    logger.debug(f'Generating msh_t took: {time()-start}')
    return callback(msh_t), nprocs


def get_cpus_per_task(comm, hfun_raster_window_requests, hfun_config, local_contours_paths, bbox_gdf):

    cpus_per_task = hfun_config.get('cpus_per_task')
    max_cpus_per_task = hfun_config.get('max_cpus_per_task')
    min_cpus_per_task = hfun_config.get('min_cpus_per_task')

    if np.all([bool(cpus_per_task), np.any([bool(max_cpus_per_task), bool(min_cpus_per_task)])]):
        raise ValueError('Arguments cpus_per_task and max_cpus_per_task are mutually exclusive.')

    if cpus_per_task is not None:
        if not isinstance(cpus_per_task, int) or cpus_per_task < 1:
            raise ValueError(f'Argument cpus_per_task must be an int > 0 or None (for auto dist) but got {cpus_per_task=}.')

    if cpus_per_task is not None:
        return [cpus_per_task]*len(hfun_raster_window_requests)

    if all(local_contours_path is None for local_contours_path in local_contours_paths):
        if cpus_per_task is None:
            cpus_per_task = 1
        return [cpus_per_task]*len(hfun_raster_window_requests)

    nprocs_list = []
    local_contours_paths = np.array_split(local_contours_paths, comm.Get_size())[comm.Get_rank()]
    max_threads = lib.hardware_info(comm)['thread_count'].min()
    for local_contours_path in local_contours_paths:
        if local_contours_path is None:
            continue
        total_points = 0
        for row in gpd.read_feather(local_contours_path).itertuples():
            if hasattr(row.geometry, 'geoms'):
                for geom in row.geometry.geoms:
                    total_points += len(geom.coords)
            else:
                total_points += len(row.geometry.coords)

        nprocs_list.append(np.max([1, np.min([total_points % 30000 // 1000, max_threads])]))
        if max_cpus_per_task is not None and nprocs_list[-1] > max_cpus_per_task:
            nprocs_list[-1] = max_cpus_per_task
        if min_cpus_per_task is not None and nprocs_list[-1] < min_cpus_per_task:
            nprocs_list[-1] = min_cpus_per_task

    return [item for sublist in comm.allgather(nprocs_list) for item in sublist if item is not None]


def pickle_dump_callback(filepath, obj):
    pickle.dump(obj,  open(filepath, 'wb'))
    return filepath


def job_scheduler(jobs, comm):

    rank = comm.Get_rank()
    hwinfo = lib.hardware_info(comm)
    unique_colors = hwinfo['color'].unique()
    local_color = hwinfo.iloc[rank]['color']
    color_sizes = hwinfo['color'].value_counts()
    local_comm = comm.Split(local_color, rank)
    local_jobs = [job for _, job in lib.get_split_array_weighted(
            comm,
            jobs,
            [job[2] for job in jobs],
            len(unique_colors),
            )[local_color]]

    # Create a queue for jobs
    job_queue = queue.PriorityQueue()

    # Add jobs to queue, prioritized by number of processors needed
    for job, args, procs_needed in sorted(local_jobs, key=lambda x: x[2]):
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
                        sleep(5)
                    else:
                        # If a job can't be scheduled, defer it
                        deferred_jobs.append((-procs_needed, (job, args, procs_needed)))

                # Return deferred jobs back into the queue
                for job in deferred_jobs:
                    job_queue.put(job)
    return [item for sublist in comm.allgather(results) for item in sublist]

# def job_scheduler(jobs, comm):
#     # Create a queue for jobs
#     job_queue = queue.PriorityQueue()
#     # Add jobs to queue, prioritized by number of processors needed
#     for job, args, procs_needed in sorted(jobs, key=lambda x: x[2]):
#         args = pickle.dumps(args)
#         job_queue.put((-procs_needed, (job, args, procs_needed)))  # Negate procs_needed to sort in descending order
#     # Create a list to hold futures
#     futures = []
#     results = []
#     with MPICommExecutor(comm, root=0) as executor:
#         if executor is not None:
#             free_procs = comm.Get_size()
#             while not job_queue.empty() or futures:
#                 # Remove finished futures and update the number of free processors
#                 for future in futures:
#                     if future.done():
#                         output_path, fprocs = future.result()
#                         free_procs += fprocs  # Assumes the job returns the number of processors it used
#                         futures.remove(future)
#                         if output_path is not None:
#                             results.append(output_path)
#                 # ------- Option 1
#                 # # Keep track of jobs we can't fit right now
#                 # deferred_jobs = []
#                 # # Try to fill all available processors
#                 # while not job_queue.empty() and free_procs > 0:
#                 #     _, (job, args, procs_needed) = job_queue.get()
#                 #     if procs_needed <= free_procs:
#                 #         args = pickle.loads(args)
#                 #         future = executor.submit(job, *args)
#                 #         futures.append(future)
#                 #         free_procs -= procs_needed
#                 #     else:
#                 #         # If a job can't be scheduled, defer it
#                 #         deferred_jobs.append((-procs_needed, (job, args, procs_needed)))

#                 # # Return deferred jobs back into the queue
#                 # for job in deferred_jobs:
#                 #     job_queue.put(job)

#                 # Submit new jobs if there are enough free processors
#                 while not job_queue.empty():
#                     procs_needed = -job_queue.queue[0][0]  # Peek at the number of processors needed by the next job
#                     if procs_needed <= free_procs:
#                         _, (job, args, procs_needed) = job_queue.get()
#                         args = pickle.loads(args)
#                         future = executor.submit(job, *args)
#                         futures.append(future)
#                         free_procs -= procs_needed
#                     else:
#                         break
#     # Collect results
#     if comm.Get_rank() == 0:
#         results.extend([res for res in [f.result()[0] for f in futures] if res is not None])
#     return comm.bcast(results, root=0)


def get_out_msh_t(
        hfun_raster_window_requests,
        cpus_per_task,
        comm,
        hfun_config,
        local_contours_paths,
        bbox_gdf,
        cache_directory,
        raster_hashes,
        ):
    results = []
    jobs = []
    cache_directory /= 'window_msh_t'
    cache_directory.mkdir(exist_ok=True)

    hfun_constraints = [
            'constant_value',
            'contour',
            'feature',
            'gradient_delimiter',
            'narrow_channel_anti_aliasing',
            'gaussian_filter',
            ]
    for global_k, ((i, raster_path, request_opts), (j, window)) in enumerate(hfun_raster_window_requests):
        normalized_raster_opts = lib.get_normalized_raster_request_opts(raster_hashes[global_k], request_opts, window)
        normalized_hfun_request = {
                    'hmin': hfun_config.get('hmin'),
                    'hmax': hfun_config.get('hmax'),
                    'marche': hfun_config.get('marche'),
                    'normalized_raster_opts': normalized_raster_opts,
                }
        if local_contours_paths[global_k] is not None:
            normalized_hfun_request.update({
                        'local_contours_sha256': local_contours_paths[global_k].stem,
                    })
        # for requests in request_opts:
        for request_type, request_values in request_opts.items():
            if request_type in list(hfun_constraints):
                normalized_hfun_request.update({
                    request_type: request_values,
                    })
        tmpfile = cache_directory / (hashlib.sha256(
            json.dumps(
                normalized_hfun_request,
                ).encode()).hexdigest() + '.pkl')
        if tmpfile.exists():
            results.append(tmpfile)
        else:
            args = (
                    raster_path,
                    request_opts,
                    hfun_config,
                    window,
                    cpus_per_task[global_k],
                    local_contours_paths[global_k],
                    bbox_gdf,
                    i,
                    global_k,
                    partial(pickle_dump_callback, tmpfile),
                    )
            jobs.append((
                    get_local_msh_t_tmpfile,
                    args,
                    cpus_per_task[global_k]
                    ))
    results.extend(job_scheduler(jobs, comm))
    return results


def get_local_contours_paths(comm, hfun_raster_window_requests, bbox_gdf, cache_directory, raster_hashes):

    if comm.Get_rank() == 0:
        logger.info('Begin build raster contour requests...')

    contours_gdf = get_hfun_contours_gdf(comm, hfun_raster_window_requests, bbox_gdf, cache_directory, raster_hashes)

    # verify:
    if contours_gdf is None:
        return [None]*len(hfun_raster_window_requests)

    contours_hash = hashlib.sha256(str(contours_gdf).encode('utf-8')).hexdigest()

    cache_directory /= 'local_contours'
    cache_directory.mkdir(exist_ok=True)
    # hfun_raster_window_requests = np.array_split(
    #         np.array(hfun_raster_window_requests, dtype=object),
    #         comm.Get_size()
    #         )[comm.Get_rank()]
    hfun_raster_window_requests = lib.get_split_array(comm, hfun_raster_window_requests)[comm.Get_rank()]

    local_contours_paths = []
    for global_k, ((i, raster_path, raster_opts), (j, window)) in hfun_raster_window_requests:
        normalized_raster_request = lib.get_normalized_raster_request_opts(
                raster_hashes[global_k], raster_opts, window)
        normalized_raster_request['contours_hash'] = contours_hash
        feather = cache_directory / (hashlib.sha256(json.dumps(normalized_raster_request).encode()).hexdigest() + '.feather')
        if feather.exists():
            local_contours_paths.append(feather)
            continue
        base_mp = bbox_gdf.iloc[[global_k]]
        local_proj = CRS.from_user_input(
                ' '.join([
                        "+proj=aeqd",
                        "+R=6371000",
                        "+units=m",
                        f"+lat_0={base_mp.unary_union.centroid.y}",
                        f"+lon_0={base_mp.unary_union.centroid.x}",
                    ])
                )
        base_mp = base_mp.to_crs(local_proj)
        bbox_centroid = base_mp.unary_union.centroid
        bbox_bounds = Bbox.from_extents(*base_mp.unary_union.bounds)
        bbox_centroid = base_mp.unary_union.centroid
        bbox_radius = np.sqrt((bbox_bounds.xmax - bbox_centroid.x)**2+(bbox_bounds.ymax - bbox_centroid.y)**2)
        # NOTE: hardcoded to 2 times the radius
        clip_poly = bbox_centroid.buffer(2*bbox_radius)
        clip_poly = ops.transform(
                Transformer.from_crs(local_proj, contours_gdf.crs, always_xy=True).transform,
                clip_poly)
        local_contours = gpd.clip(contours_gdf, clip_poly).reset_index()
        if len(local_contours) > 0:
            local_contours = collapse_by_required_columns(local_contours)
        local_contours.to_feather(feather)
        local_contours_paths.append(feather)
    return [item for sublist in comm.allgather(local_contours_paths) for item in sublist]


def get_hfun_raster_window_requests(comm, hfun_config):
    if comm.Get_rank() == 0:
        logger.info('Loading hfun raster window requests...')
        hfun_raster_window_requests = list(iter_raster_window_requests(hfun_config))
        logger.debug('Done loading hfun raster window requests.')
    else:
        hfun_raster_window_requests = None
    hfun_raster_window_requests = comm.bcast(hfun_raster_window_requests, root=0)
    return hfun_raster_window_requests


def get_uncombined_hfuns(comm, hfun_config):

    rank = comm.Get_rank()

    hfun_raster_window_requests = get_hfun_raster_window_requests(comm, hfun_config)

    cache_directory = hfun_config['cache_directory']

    # Now that we have an expanded list of raster window requests, we nee to compute
    # a system-independent hash for each of the requests. The fisrt step is to compute the md5
    # of each of the rasteres in raw form. We then use this md5 as a salt for adding other requests to the
    # raster request. This should theoretically result in the same hash on any platform, becaue it based on
    # the file content instead of path.
    if rank == 0:
        logger.info('will get raster md5')
    raster_hashes = lib.get_raster_md5(comm, hfun_raster_window_requests)

    if rank == 0:
        logger.info('will get bbox gdf')
    bbox_gdf = lib.get_bbox_gdf(comm, hfun_raster_window_requests, cache_directory, raster_hashes)

    if rank == 0:
        logger.info('will get local contours paths')
    local_contours_paths = get_local_contours_paths(comm, hfun_raster_window_requests, bbox_gdf, cache_directory, raster_hashes)

    if rank == 0:
        logger.info('will get cpus_per_task')
    cpus_per_task = get_cpus_per_task(comm, hfun_raster_window_requests, hfun_config, local_contours_paths, bbox_gdf)

    if rank == 0:
        logger.info('will build msh_ts')
    out_msh_t_tempfiles = get_out_msh_t(
                hfun_raster_window_requests,
                cpus_per_task,
                comm,
                hfun_config,
                local_contours_paths,
                bbox_gdf,
                cache_directory,
                raster_hashes,
                )
    if rank == 0:
        logger.info(f'built {len(out_msh_t_tempfiles)} msh_ts')
    return out_msh_t_tempfiles


def combine_msh_t_list(msh_t_list):
    msh_t = jigsaw_msh_t()
    msh_t.mshID = "euclidean-mesh"
    msh_t.ndims = +2
    msh_t.crs = CRS.from_epsg(4326)
    for window_mesh in msh_t_list:
        window_mesh.tria3["index"] += len(msh_t.vert2)
        msh_t.tria3 = np.append(
                msh_t.tria3,
                window_mesh.tria3,
                axis=0
                )
        if not window_mesh.crs.equals(msh_t.crs):
            utils.reproject(window_mesh, msh_t.crs)
        msh_t.vert2 = np.append(
            msh_t.vert2,
            window_mesh.vert2,
            axis=0
        )
        msh_t.value = np.append(msh_t.value, window_mesh.value)
    return msh_t


def combine_msh_t_partition(msh_t_list, cache_directory):
    msh_t_names = [msh_t_path.name for msh_t_path in msh_t_list]
    tmpfile = cache_directory / (hashlib.sha256(json.dumps(msh_t_names).encode()).hexdigest() + ".pkl")
    msh_t_list = np.array(msh_t_list, dtype=object).flatten()
    msh_ts = []
    for msh_t_path in msh_t_list:
        # if msh_t_path.exists():
        try:
            with open(msh_t_path, 'rb') as fh:
                msh_ts.append(pickle.load(fh))
        except EOFError:
            pass
    msh_t = combine_msh_t_list(msh_ts)
    with open(tmpfile, 'wb') as fh:
        pickle.dump(msh_t, fh)
    return tmpfile


def distribute_buckets(buckets, size):
    """Distribute buckets into 'size' groups as evenly as possible."""
    result = [[] for _ in range(size)]
    index = 0
    for bucket in buckets:
        result[index].append(bucket)
        index = (index + 1) % size
    return result


def combine_hfuns(comm, out_msh_t_tempfiles, hfun_config):

    cache_directory = hfun_config["cache_directory"]
    cache_directory /= "msh_t_combined"
    cache_directory.mkdir(exist_ok=True)

    rank = comm.Get_rank()
    size = comm.Get_size()

    divisions = [len(out_msh_t_tempfiles)]
    while divisions[-1] > 1:
        divisions.append(int(np.floor(divisions[-1]/2)))

    for i, npartitions in enumerate(divisions[1:]):
        if rank == 0:
            if npartitions <= size:
                out_msh_t_tmpfiles = np.array(out_msh_t_tempfiles, dtype=object)
                buckets = [out_msh_t_tmpfiles[bucket] for bucket in np.array_split(
                    range(len(out_msh_t_tempfiles)), npartitions)]
                buckets = np.array(buckets + [[None]]*(size - len(buckets)), dtype=object)
            else:
                buckets = np.array_split(out_msh_t_tempfiles, npartitions)
                buckets = distribute_buckets(buckets, size)
        else:
            buckets = None

        bucket = comm.bcast(buckets, root=0)[rank]
        local_list = []
        for partition in bucket:
            if partition is None:
                continue
            if isinstance(partition, Path):
                local_list.append(partition)
            else:
                combined_msh_t_tempfile = combine_msh_t_partition(partition, cache_directory)
                local_list.append(combined_msh_t_tempfile)

        out_msh_t_tempfiles = [item for sublist in comm.allgather(local_list) for item in sublist]

    if rank == 0:
        combined_msh_t_tempfile = combine_msh_t_partition(out_msh_t_tempfiles, cache_directory)
    else:
        combined_msh_t_tempfile = None

    return comm.bcast(combined_msh_t_tempfile, root=0)


class ConfigPathAction(argparse.Action):

    def __call__(self, parser, namespace, config_path, option_string=None):
        if not config_path.is_file():
            raise FileNotFoundError(f'{config_path=} is not an existing file.')
        setattr(namespace, self.dest, config_path)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, action=ConfigPathAction)
    # parser.add_argument('--max-cpus-per-task', type=int)
    parser.add_argument('--to-msh', type=Path)
    parser.add_argument('--to-pickle', type=Path)
    parser.add_argument('--dst-crs', '--dst_crs', type=CRS.from_user_input, default=CRS.from_epsg(4326))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cache-directory', type=Path)
    parser.add_argument(
        "--log-level",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser


def make_plot(hfun_msh_t_tmpfile):
    logger.info('Drawing plot...')
    with open(hfun_msh_t_tmpfile, 'rb') as fh:
        msh_t = pickle.load(fh)
    utils.tricontourf(msh_t, axes=plt.gca(), cmap='jet')
    utils.triplot(msh_t, axes=plt.gca())
    plt.gca().axis('scaled')
    plt.show(block=True)
    return


def to_msh(args, hfun_msh_t_tmpfile):
    logger.info('Write msh_t...')
    with open(hfun_msh_t_tmpfile, 'rb') as fh:
        msh_t = pickle.load(fh)
    savemsh(f'{args.to_msh.resolve()}', Hfun(msh_t).msh_t(dst_crs=args.dst_crs))
    return


def to_pickle(args, hfun_msh_t_tmpfile):
    logger.info('Write pickle...')
    with open(hfun_msh_t_tmpfile, 'rb') as fh:
        msh_t = pickle.load(fh)
    with open(args.to_pickle, 'wb') as fh:
        pickle.dump(Hfun(msh_t), fh)
    logger.info('Done writing pickle...')
    return


def get_hfun_config_from_args(comm, args):
    rank = comm.Get_rank()
    # We begin by validating the user's request. The validation is done by the BuildCi class,
    # it does not coerce or change the data of the hfun_config dictionary.
    # The yaml configuration data needs to be expanded for parallelization,
    # and the expansion is done by the iter_raster_window_requests method.
    # TODO: Optimize (parallelize) the initial data loading.
    if rank == 0:
        logger.info('Validating user raster hfun requests...')
        hfun_config = BuildCli(args).config.hfun.hfun_config
        if hfun_config is None:
            raise RuntimeError(f'No hfun to process in {args.config}')
        tmpdir = hfun_config.get('cache_directory', Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp-geomesh')))
        hfun_config.setdefault("cache_directory", tmpdir)
        if args.cache_directory is not None:
            hfun_config['cache_directory'] = args.cache_directory
        hfun_config["cache_directory"].mkdir(exist_ok=True, parents=True)
    else:
        hfun_config = None

    return comm.bcast(hfun_config, root=0)


def main(args: argparse.Namespace, comm=None):
    """This program uses MPI and memoization. The memoization directory can be provided
    as en evironment variable GEOMESH_TEMPDIR, as the key 'cache_directory' in the yaml
    configuration file, or as the command line argument --cache-directory, and they
    superseed each other in this same order. The default is `os.getcwd() + '/.tmp'`.
    """

    comm = MPI.COMM_WORLD if comm is None else comm
    hfun_config = get_hfun_config_from_args(comm, args)

    uncombined_hfun_paths = get_uncombined_hfuns(comm, hfun_config)
    hfun_msh_t_tmpfile = combine_hfuns(comm, uncombined_hfun_paths, hfun_config)
    finalization_tasks = []

    if args.show:
        finalization_tasks.append(make_plot)

    if args.to_msh:
        finalization_tasks.append(partial(to_msh, args))

    if args.to_pickle:
        finalization_tasks.append(partial(to_pickle, args))

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            futures = [executor.submit(task, hfun_msh_t_tmpfile) for task in finalization_tasks]
            for future in futures:
                future.result()
            # executor.shutdown(wait=True)


def entrypoint():
    sys.excepthook = mpiabort_excepthook
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
    main(args)


if __name__ == "__main__":
    entrypoint()
