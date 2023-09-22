from functools import cached_property
from multiprocessing import cpu_count
from pathlib import Path
import asyncio
import hashlib
import json
import logging
import os
import pickle
import sys
import tempfile

from geomesh.cli.config_parser import ConfigParser
from geomesh.cli.mpi import hgrid_build


logger = logging.getLogger(__name__)


class BuildCli:

    def __init__(self, args):
        self.args = args

    def main(self):
        asyncio.get_event_loop().run_until_complete(self.main_async())

    async def main_async(self):
        hgrid = await self.get_output_hgrid()
        logger.info(f"Hgrid saved to {hgrid}")

    async def get_output_hgrid(self):
        msh_t = await self.get_msh_t()
        async with self.config.Scheduler() as executor:
            hfun_ntasks = self.args.hfun_ntasks or self.config.hfun.ntasks
            hgrid = await hgrid_build.submit(
                    executor,
                    self.args.config,
                    msh_t,
                    cwd=None,
                    log_level=None,
                    to_pickle=None,
                    ntasks=hfun_ntasks,
                    # cwd = self.config.
                    # **executor_kwargs
                    )
            print(hgrid)

        # topobathy_output_msh_t = await self.interpolate_mesh(output_msh_t_path)
        # hgrid = Hgrid(
        #     nodes=
        #     elements=
        #     )
        # hgrid.boundaries.auto_generate()
        # output_mesh = await self.generate_boundaries(output_mesh)
        return output_mesh

    async def get_msh_t(self):
        from geomesh.cli import msh_t_build
        geom_output_pkl, hfun_output_pkl = await self.get_geom_hfun_paths()
        msh_t_hash = hashlib.sha256(f"{geom_output_pkl.name}{hfun_output_pkl.name}".encode()).hexdigest()

       # call from the front-end so we can see progress
       # if logger.level <= logging.DEBUG:
        #     driver = JigsawDriver(
        #             geom=pickle.load(open(geom_output_pkl, 'rb')),
        #             hfun=pickle.load(open(hfun_output_pkl, 'rb')),
        #             # initial_mesh=args.initial_mesh,
        #             verbosity=1,
        #             # dst_crs=args.dst_crs,
        #             # sieve_area=args.sieve,
        #             # finalize=args.finalize,
        #             )
        #     return driver.msh_t()

        cache_directory = self.config.msh_t.msh_t_config.get(
                'cache_directory',
                Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp'))
                )
        cache_directory /= 'msh_t'
        msh_t_output_pkl = cache_directory / f'{msh_t_hash}.pkl'
        if msh_t_output_pkl.exists():
            logger.info("Loading mesh from cache")
            return pickle.load(open(msh_t_output_pkl, 'rb'))
        msh_t_output_pkl.parent.mkdir(parents=True, exist_ok=True)

        # Alternative way below, which is to call jigsaw binary directly
        # cwd = tempfile.TemporaryDirectory(dir=cache_directory)
        # geom = pickle.load(open(geom_output_pkl, 'rb'))
        # hfun_msh_t = pickle.load(open(hfun_output_pkl, 'rb'))
        # async with self.config.Scheduler(cwd=cwd.name) as executor:
        #     driver = JigsawDriver(
        #             geom=geom,
        #             hfun=hfun_msh_t,
        #             # initial_mesh=args.initial_mesh,
        #             verbosity=1,
        #             dst_crs=CRS.from_epsg(4326),
        #             sieve_area=True,
        #             finalize=True,
        #             )
        #     numthread = self.config.msh_t.msh_t_config['opts'].get('numthread', cpu_count())
        #     driver.opts.numthread = numthread
        #     # print(f"{numthread=}")
        #     output_msh_t = await driver.submit(
        #             executor,
        #             cpus_per_task=numthread,
        #             job_name='mesh-build',
        #             )
        # pickle.dump(output_msh_t, open(msh_t_output_pkl, 'wb'))
        # return output_msh_t
        numthread = self.config.msh_t.msh_t_config['opts'].get('numthread', cpu_count())
        msh_t_build_cmd = [
                sys.executable,
                str(Path(msh_t_build.__file__)),
                str(self.args.config),
                f"--geom-pickle={geom_output_pkl}",
                f"--hfun-pickle={hfun_output_pkl}",
                f"--numthread={numthread}",
                f"--log-level={self.args.log_level}",
                f"--to-pickle={msh_t_output_pkl}",
                "--verbosity=1",
                ]
        if quads_feather_path is not None:
            msh_t_build_cmd.append(f'--quads-feather-path={quads_feather_path}')
        async with self.config.Scheduler() as scheduler:
            await scheduler.submit(
                    msh_t_build_cmd,
                    cpus_per_task=numthread,
                    job_name='mesh-build'
                    )
        return pickle.load(open(msh_t_output_pkl, "rb"))

    async def get_geom_hfun_paths(self):

        geom_build_cmd, geom_output_pkl = self.get_geom_build_cmd()
        hfun_build_cmd, hfun_output_pkl = self.get_hfun_build_cmd()

        geom_ntasks = self.args.geom_ntasks or self.config.geom.ntasks
        hfun_ntasks = self.args.hfun_ntasks or self.config.hfun.ntasks

        async with self.config.Scheduler() as scheduler:
            # geom_output_pkl.unlink(missing_ok=True)
            if not geom_output_pkl.exists():
                geom_job = scheduler.submit(geom_build_cmd, ntasks=geom_ntasks, job_name='geom-build')
            else:
                geom_job = asyncio.sleep(0)

            # hfun_output_pkl.unlink(missing_ok=True)
            if not hfun_output_pkl.exists():
                hfun_job = scheduler.submit(hfun_build_cmd, ntasks=hfun_ntasks, job_name='hfun-build')
            else:
                hfun_job = asyncio.sleep(0)
            # print('waiting for geom and hfun jobs to complete')
            await asyncio.gather(geom_job, hfun_job)

        print(f'geom and hfun jobs complete - returning paths {geom_output_pkl=} {hfun_output_pkl=} ')
        return geom_output_pkl, hfun_output_pkl

    def get_geom_build_cmd(self):

        # avoid circular import
        from geomesh.cli.mpi import geom_build

        cache_directory = self.config.cache_directory / 'geom'
        cache_directory.mkdir(parents=True, exist_ok=True)

        # we need to create a unique hash from the config.geom attributes
        # and use that as the output file name, for memoization purposes.
        # geom_raster_requests = list(iter_raster_requests(self.config.geom.geom_config))

        # with Pool(cpu_count()) as pool:
        #     normalized_geom_request_hashes= pool.map(
        #             get_normalized_geom_request_hash,
        #             geom_raster_requests,
        #             )
        # raster_request_hash = hashlib.sha256(json.dumps(normalized_geom_request_hashes).encode()).hexdigest()
        # geom_output_pkl = cache_directory / f'{raster_request_hash}.pkl'
        geom_tmpfile = tempfile.NamedTemporaryFile(dir=cache_directory, suffix='.pkl')
        geom_output_pkl = Path(geom_tmpfile.name)
        geom_build_cmd = [
                sys.executable,
                f'{Path(geom_build.__file__).resolve()}',
                f'{self.args.config}',
                f'--log-level={self.args.log_level}',
                f'--to-pickle={geom_output_pkl}',
                ]
        return geom_build_cmd, geom_output_pkl

    def get_hfun_build_cmd(self):

        # avoid circular import
        from geomesh.cli.mpi import hfun_build

        cache_directory = self.config.cache_directory / 'hfun'
        cache_directory.mkdir(parents=True, exist_ok=True)

        # we need to create a unique hash from the config.hfun attributes
        # and use that as the output file name, for memoization purposes.
        # hfun_raster_requests = list(iter_raster_requests(self.config.hfun.hfun_config))

        # with Pool(cpu_count()) as pool:
        #     normalized_hfun_request_hashes= pool.map(
        #             get_normalized_hfun_request_hash,
        #             hfun_raster_requests,
        #             )
        # raster_request_hash = hashlib.sha256(json.dumps(normalized_hfun_request_hashes).encode()).hexdigest()
        hfun_tmpfile = tempfile.NamedTemporaryFile(dir=cache_directory, suffix='.pkl')
        hfun_output_pkl = Path(hfun_tmpfile.name)
        hfun_build_cmd = [
                sys.executable,
                f'{Path(hfun_build.__file__).resolve()}',
                f'{self.args.config}',
                f'--log-level={self.args.log_level}',
                f'--to-pickle={hfun_output_pkl}',
                ]
        return hfun_build_cmd, hfun_output_pkl

    @staticmethod
    def add_parser_arguments(parser):
        parser.add_argument('config', type=Path)
        parser.add_argument('--geom-ntasks', type=int)
        parser.add_argument('--hfun-ntasks', type=int)
        parser.add_argument('--show', type=int)
        parser.add_argument('--to-msh', type=Path)
        parser.add_argument('--to-pickle', type=Path)
        parser.add_argument('--to-gr3', type=Path)
        parser.add_argument(
            "--log-level",
            choices=["warning", "info", "debug"],
            default="warning",
        )

    @cached_property
    def config(self):
        return ConfigParser(self.args.config)


def get_normalized_geom_request_hash(raster_request):
    raster_path, geom_opts = raster_request
    hash_md5 = hashlib.md5()
    with open(raster_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    raster_hash = hash_md5.hexdigest()
    normalized_geom_request = {
            'zmin': geom_opts.get('zmin'),
            'zmax': geom_opts.get('zmax'),
            'sieve': geom_opts.get('sieve'),
            'raster_hash': raster_hash,
            'clip': geom_opts.get('clip'),
            'mask': geom_opts.get('mask'),
            }
    return hashlib.sha256(json.dumps(normalized_geom_request).encode('utf-8')).hexdigest()



# def get_normalized_hfun_request_hash(raster_request):
#     raster_path, hfun_opts = raster_request
#     hash_md5 = hashlib.md5()
#     with open(raster_path, "rb") as f:
#         for chunk in iter(lambda: f.read(4096), b""):
#             hash_md5.update(chunk)
#     raster_hash = hash_md5.hexdigest()
#     normalized_hfun_request = {
#             'raster_hash': raster_hash,
#             'clip': hfun_opts.get('clip'),
#             'mask': hfun_opts.get('mask'),
#             'hmin': hfun_opts.get('hmin'),
#             'hmax': hfun_opts.get('hmax'),
#             'marche': hfun_opts.get('marche'),
#             }
#     if local_contours_path is not None:
#         normalized_hfun_request.update({
#                     'local_contours_sha256': local_contours_path.stem,
#                     # 'normalized_hfun_request': normalized_hfun_request,
#                 })
#     return hashlib.sha256(json.dumps(normalized_hfun_request).encode('utf-8')).hexdigest()
