import asyncio
from collections import UserDict
from functools import cached_property
from glob import glob
import json
import logging
import os
from pathlib import Path
import pexpect
import sys
import tempfile
from typing import Dict, Generator, List, Union

import geopandas as gpd
import numpy as np
from shapely.geometry import box
import yaml

from .raster_opts import append_cmd_opts as append_raster_cmd_opts
from .tile_index import expand_tile_index

logger = logging.getLogger(__name__)


class SlurmConfig(UserDict):
    
    def get_preamble(self):
        return [
            'srun',
            f'-c{self.cpus_per_task}'
        ]

    @cached_property
    def cpus_per_task(self):
        return int(self.get('cpus_per_task', 1))

    @cached_property
    def mem_per_cpu(self):
        return self.get('mem_per_cpu')

class GeomConfigParser(UserDict):

    @cached_property
    def slurm(self) -> Union[SlurmConfig, None]:
        if 'slurm' in self:
            return SlurmConfig(self['slurm'])


    def launch_tasks(self, cache_directory):
        output_directory = cache_directory / 'geom'
        output_directory.mkdir(exist_ok=True)
        tasks = []
        if self.slurm is None:
            return tasks
        for path, geom_opts in self.iter_raster_requests():
            # TODO !!!!! Limit number of concurrent jobs with --dependency=singleton, --job-name=XXX
            tasks.append(asyncio.get_event_loop().create_task(self._await_geom_raster_request(path, geom_opts, output_directory)))
        for path, geom_opts in self.iter_feature_requests():
            tasks.append(asyncio.get_event_loop().create_task(self._await_geom_feature_request(path, geom_opts, output_directory)))
        return tasks
    
    async def _await_geom_raster_request(self, path, geom_opts, output_directory):
        output_filename = output_directory / f'{Path(path).name}.feather'
        await self._await_pexpect(self.get_raster_request_cmd(path, geom_opts, output_filename))   
        return output_filename

    async def _await_geom_feature_request(self, path, geom_opts, output_directory):
        output_filename = output_directory / f'{Path(path).name}.feather'
        await self._await_pexpect(self.get_feature_request_cmd(path, geom_opts, output_filename))   
        return output_filename

    def get_raster_request_cmd(self, path, geom_opts, output_filename)->List[str]:
        cmd = []
        if self.slurm is not None:
            cmd.extend([
                f'srun',
                f'-c1',
            ])
            if self.slurm.mem_per_cpu is not None:
                cmd.append(f'--mem-per-cpu={self.slurm.mem_per_cpu}')
        cmd.extend([
                f'{sys.executable}',
                f'{Path(__file__).parent.resolve() / "raster_geom_build.py"}',
                f"{Path(path).resolve()}",
                # f'--to-file={output_filename.resolve()}',
                f'--to-feather={output_filename.resolve()}',
                # f'--gdf-to-pickle={output_filename.resolve()}',
            ])
        if 'zmin' in geom_opts:
            cmd.append(f"--zmin={geom_opts['zmin']}")
        
        if 'zmax' in geom_opts:
            cmd.append(f"--zmax={geom_opts['zmax']}")
        append_raster_cmd_opts(cmd, geom_opts)
        # print(' '.join(cmd))
        # exit()
        return cmd
    
    def get_feature_request_cmd(self, path, geom_opts, output_filename)->List[str]:
        cmd = [
                f'{sys.executable}',
                f'{Path(__file__).parent.resolve() / "feature_geom_build.py"}',
                f"{Path(path).resolve()}",
                f'--to-feather={output_filename.resolve()}',
                # f'--to-file={output_filename.resolve()}',
            ]
        if 'vmin' in geom_opts:
            cmd.append(f"--vmin={geom_opts['vmin']}")
        
        if 'vmax' in geom_opts:
            cmd.append(f"--vmax={geom_opts['vmax']}")
        
        if 'crs' in geom_opts:
            cmd.append(f'--crs={geom_opts["crs"]}')

        return cmd
        
    def iter_raster_requests(self):
        for request in self.get('rasters', []):
            if 'path' in request:
                logger.info(f'Requested raster is a local path: {request["path"]}')
                for path, request in self._expand_raster_request_path(request):
                    yield path, request
            elif 'tile_index' in request:
                for path, request in expand_tile_index(self, request):
                    yield path, request
            else:
                raise TypeError(f'Unhandled type: {request}')
            
    def iter_feature_requests(self):
        for request in self.get('features', []):
            # print(request)
            if 'mesh' in request:
                logger.info(f'Requested feature is a mesh: {request["mesh"]}')
                for feature in self._expand_feature_request_path(request):
                    yield feature
            # elif 'tile_index' in request:
            #     for feature in self.expand_tile_index(request):
            #         yield feature
            else:
                raise TypeError(f'Unhandled type: {request}')
            
    def _expand_raster_request_path(self, request: Dict) -> Generator:
        request = request.copy()
        requested_paths = request.pop("path")
        if isinstance(requested_paths, str):
            requested_paths = [requested_paths]
        for requested_path in list(requested_paths):
            requested_path = os.path.expandvars(requested_path)
            if '*' in requested_path:
                paths = list(glob(str(Path(requested_path).resolve())))
                if len(paths) == 0:
                    raise ValueError(f'No rasters found on path {requested_path}')
                for path in paths:
                    yield path, request
                        
            else:
                yield requested_path, request
                
    def _expand_feature_request_path(self, request: Dict) -> Generator:
        request = request.copy()
        requested_paths = request.pop("mesh")
        if isinstance(requested_paths, str):
            requested_paths = [requested_paths]
        for requested_path in list(requested_paths):
            requested_path = os.path.expandvars(requested_path)
            if '*' in requested_path:
                paths = list(glob(str(Path(requested_path).resolve())))
                if len(paths) == 0:
                    raise ValueError(f'No features found on path {requested_path}')
                for path in paths:
                    yield path, request
                        
            else:
                yield requested_path, request
            


class HfunConfigParser(UserDict):

    def launch_tasks(self, cache_directory):
        output_directory = cache_directory / 'hfun'
        output_directory.mkdir(exist_ok=True)
        tasks = []
        if self.slurm is None:
            return tasks
        for path, hfun_opts in self.iter_raster_requests():
            tasks.append(asyncio.get_event_loop().create_task(self._await_hfun_raster_request(path, hfun_opts, output_directory)))
        for path, hfun_opts in self.iter_feature_requests():
              tasks.append(asyncio.get_event_loop().create_task(self._await_hfun_feature_request(path, hfun_opts, output_directory)))
        return tasks

    async def _await_hfun_raster_request(self, path, hfun_opts, output_directory):
        output_filename = output_directory / f'{Path(path).name}.pkl'
        await self._await_pexpect(self.get_raster_request_cmd(path, hfun_opts, output_filename))   
        return output_filename
    
    async def _await_hfun_feature_request(self, path, hfun_opts, output_directory):
        output_filename = output_directory / f'{Path(path).name}.pkl'
        await self._await_pexpect(self.get_feature_request_cmd(path, hfun_opts, output_filename))   
        return output_filename

    def get_raster_request_cmd(self, path, hfun_opts, output_filename):
        cmd = []
        if self.slurm is not None:
            cmd.extend([
                f'srun',
                f'-c{self.slurm.cpus_per_task}',
            ])
        cmd.extend([
                f'{sys.executable}',
                f'{Path(__file__).parent.resolve() / "raster_hfun_build.py"}',
                f"{Path(path).resolve()}",
            ])
        if 'hmin' in hfun_opts:
            cmd.append(f"--hmin={hfun_opts['hmin']}")
        
        if 'hmax' in hfun_opts:
            cmd.append(f"--hmax={hfun_opts['hmax']}")
            
        if 'nprocs' in hfun_opts:
            cmd.append(f"--nprocs={hfun_opts['nprocs']}")
        else:
            if self.slurm is not None:
                cmd.append(f"--nprocs={self.slurm.cpus_per_task}")
        # cmd.append("")
        if 'contours' in hfun_opts:
            for contour in hfun_opts['contours']:
                cmd.append(f"--contour='{json.dumps(contour)}'")
                
        # cmd.append(f"--to-msh={output_filename}")
        cmd.append(f"--to-pickle={output_filename}")

        append_raster_cmd_opts(cmd, hfun_opts)

        return cmd
    
    def get_feature_request_cmd(self, feat_type, hfun_opts, output_filename):
        path, hfun_opts = hfun_opts
        if feat_type == 'mesh':
            return self._get_mesh_request_cmd(path, hfun_opts, output_filename)
        raise NotImplementedError(f'feat_type: {feat_type} not implemented')
        # cmd = []
        # if self.slurm is not None:
        #     cmd.extend([
        #         f'srun',
        #         f'-c{self.slurm.cpus_per_task}',
        #     ])
        # cmd.extend([
        #         f'{sys.executable}',
        #         f'{Path(__file__).parent.resolve() / "raster_hfun_build.py"}',
        #         f"{Path(path).resolve()}",
        #     ])
        # if 'hmin' in hfun_opts:
        #     cmd.append(f"--hmin={hfun_opts['hmin']}")
        
        # if 'hmax' in hfun_opts:
        #     cmd.append(f"--hmax={hfun_opts['hmax']}")
            
        # if 'nprocs' in hfun_opts:
        #     cmd.append(f"--nprocs={hfun_opts['nprocs']}")
        # else:
        #     if self.slurm is not None:
        #         cmd.append(f"--nprocs={self.slurm.cpus_per_task}")
        # # cmd.append("")
        # if 'contours' in hfun_opts:
        #     for contour in hfun_opts['contours']:
        #         cmd.append(f"--contour='{json.dumps(contour)}'")
                
        # # cmd.append(f"--to-msh={output_filename}")
        # cmd.append(f"--to-pickle={output_filename}")

        # append_raster_cmd_opts(cmd, hfun_opts)

        return cmd

    # TODO: !!!!! repeated code!
    def iter_raster_requests(self):
        for request in self.get('rasters', []):
            if 'path' in request:
                logger.info(f'Requested raster is a local path: {request["path"]}')
                for raster in self._expand_raster_request_path(request):
                    yield raster
            elif 'tile_index' in request:
                for raster in expand_tile_index(self, request):
                    yield raster
            else:
                raise TypeError(f'Unhandled type: {request}')

    def _get_mesh_request_cmd(self, path, hfun_opts, output_filename):
        cmd = [
                f'{sys.executable}',
                f'{Path(__file__).parent.resolve() / "mesh_hfun_build.py"}',
                f"{Path(path).resolve()}",
                f'--to-pickle={output_filename.resolve()}',
                # f'--to-file={output_filename.resolve()}',
            ]
        if 'vmin' in hfun_opts:
            cmd.append(f"--hmin={hfun_opts['vmin']}")
        
        if 'vmax' in hfun_opts:
            cmd.append(f"--hmax={hfun_opts['vmax']}")
        
        if 'crs' in hfun_opts:
            cmd.append(f'--crs={hfun_opts["crs"]}')

        return cmd

    def iter_feature_requests(self):

        for request in self.get('features', []):
            # print(request)
            if 'mesh' in request:
                logger.info(f'Requested feature is a mesh: {request["mesh"]}')
                for feature in self._expand_feature_request_path(request):
                    yield 'mesh', feature
            # elif 'tile_index' in request:
            #     for feature in self.expand_tile_index(request):
            #         yield feature
            else:
                raise TypeError(f'Unhandled type: {request}')

    def _expand_raster_request_path(self, request: Dict) -> Generator:
        request = request.copy()
        requested_paths = request.pop("path")
        if isinstance(requested_paths, str):
            requested_paths = [requested_paths]
        for requested_path in list(requested_paths):
            requested_path = os.path.expandvars(requested_path)
            if '*' in requested_path:
                paths = list(glob(str(Path(requested_path).resolve())))
                if len(paths) == 0:
                    raise ValueError(f'No rasters found on path {requested_path}')
                for path in paths:
                    yield path, request
                        
            else:
                yield requested_path, request

    @cached_property
    def slurm(self) -> Union[SlurmConfig, None]:
        if 'slurm' in self:
            return SlurmConfig(self['slurm'])
        
    def _expand_feature_request_path(self, request: Dict) -> Generator:
        request = request.copy()
        requested_paths = request.pop("mesh")
        if isinstance(requested_paths, str):
            requested_paths = [requested_paths]
        for requested_path in list(requested_paths):
            requested_path = os.path.expandvars(requested_path)
            if '*' in requested_path:
                paths = list(glob(str(Path(requested_path).resolve())))
                if len(paths) == 0:
                    raise ValueError(f'No features found on path {requested_path}')
                for path in paths:
                    yield path, request
                        
            else:
                yield requested_path, request


class ConfigParser(UserDict):
  
    @cached_property
    def geom(self):
        if 'geom' in self:
            geom = GeomConfigParser(self['geom'])
            geom._await_pexpect = self._await_pexpect
            geom.config = self
            return geom

    @cached_property
    def hfun(self):
        if 'hfun' in self:
            hfun = HfunConfigParser(self['hfun'])
            hfun._await_pexpect = self._await_pexpect
            hfun.config = self
            return hfun
        
    @cached_property
    def root_directory(self):
        root = Path(self['root_directory'])
        root.mkdir(exist_ok=True, parents=True)
        return root
    
    @cached_property
    def cache_directory(self):
        self.__cache_tmpdir = tempfile.TemporaryDirectory(dir=self.root_directory)
        return Path(self.__cache_tmpdir.name)

    @classmethod
    def from_yml(cls, path):
        with open(path) as fh:
            yml = yaml.load(fh, Loader=yaml.SafeLoader)
        yml = {} if yml is None else yml
        if 'root_directory' not in yml:
            yml.update(root_directory=Path(path).parent)
        return cls(yml)

    async def _await_pexpect(self, cmd: List[str]) -> None:
        with pexpect.spawn(
                ' '.join(cmd),
                encoding='utf-8',
                timeout=None,
                # cwd=output_directory
        ) as p:
            p.logfile_read = sys.stdout
            await p.expect(pexpect.EOF, async_=True)

        if p.exitstatus != 0:
            raise Exception(p.before)

