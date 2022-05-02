import asyncio
import pickle
from pathlib import Path
import os
import sys
import tempfile


# import dask_geopandas
import geopandas as gpd
from jigsawpy import jigsaw_msh_t
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

from geomesh.driver import JigsawDriver
from geomesh.geom.shapely import MultiPolygonGeom
from geomesh.hfun.mesh import MeshHfun

from .configparser import ConfigParser
from .raster_opts import append_cmd_opts as append_raster_cmd_opts

class BuildCli:

    def __init__(self, yml_path: os.PathLike):
        self.config = ConfigParser.from_yml(yml_path)
        # self.yaml_path = yml_path

    def main(self):
        
        # preliminary checks to avoid errors after execution
        if self.config.mesh is not None:
            if self.config.mesh.boundaries is not None:
                if self.config.mesh.interpolate is None:
                    raise ValueError('need to interpolate in order to obtain boundaries')
        
        geom_tasks = []
        hfun_tasks = []
        geom_result = []
        hfun_result = []
        if getattr(self.config.geom, 'slurm', None) is not None:
            geom_tasks.extend(self.config.geom.launch_tasks(self.config.cache_directory))
        else:
            if self.config.geom is not None:
                for geom_job in self.config.geom.get_jobs():
                    geom_result.append(geom_job())

        if getattr(self.config.hfun, 'slurm', None) is not None:
            hfun_tasks.extend(self.config.hfun.launch_tasks(self.config.cache_directory))
        else:
            if self.config.hfun is not None:    
                for hfun_job in self.config.hfun.get_jobs():
                    hfun_result.append(hfun_job())
        
        if len(geom_tasks) > 0:
            geom_result.extend(asyncio.get_event_loop().run_until_complete(asyncio.gather(*geom_tasks)))
            if getattr(self.config.geom, 'slurm', None) is not None:
                geom_combine_task = asyncio.get_event_loop().create_task(self.srun_geom_combine(geom_result))

        if len(hfun_tasks) > 0:
            hfun_result.extend(asyncio.get_event_loop().run_until_complete(asyncio.gather(*hfun_tasks)))
            if getattr(self.config.hfun, 'slurm', None) is not None:
                hfun_combine_task = asyncio.get_event_loop().create_task(self.srun_hfun_combine(hfun_result))
            
        geom = None
        hfun = None
        if len(geom_tasks) > 0:
            geom = asyncio.get_event_loop().run_until_complete(asyncio.gather(geom_combine_task))[0]
        else:
            if len(geom_result) > 0:
                geom = self.finalize_geom(geom_result)
            
        if len(hfun_tasks) > 0:
            hfun = asyncio.get_event_loop().run_until_complete(asyncio.gather(hfun_combine_task))[0]
        else:
            if len(hfun_result) > 0:
                hfun = self.finalize_hfun(hfun_result)
        # geom.make_plot(show=True)
        # hfun.tricontourf(levels=256, show=True)
        # TODO: This part needs to be sent to srun when applicable (it may require significant RAM)
        driver = JigsawDriver(
            geom=geom,
            hfun=hfun,
            verbosity=1
        )
        mesh = driver.output_mesh
        if self.config.mesh.interpolate is not None:
            self.interpolate_mesh(mesh)
        
        
        
        # with open('tmpmesh.pkl', 'wb') as fh:
        #     pickle.dump(mesh, fh)
        
        # exit()
                
        # with open('tmpmesh.pkl', 'rb') as fh:
        #     mesh = pickle.load(fh)
            
        from geomesh import utils
        utils.finalize_mesh(mesh.msh_t)
        if self.config.mesh.boundaries is not None:
            self.build_boundaries(mesh)
            # mesh.boundaries.open.plot(facecolor='none')
            # ax = mesh.boundaries.gdf.plot('ibtype', facecolor='none')
            # mesh.boundaries.open.plot(ax=ax, facecolor='none')
            # mesh.nodes.gdf.loc[mesh.nodes.gdf['id'] == 3809].plot(ax=ax)
            # mesh.nodes.gdf[mesh.nodes.gdf.loc['id'==3809]].plot(ax=ax)
            # import matplotlib.pyplot as plt
            # plt.show()
        if self.config.mesh.outputs is not None:
            self.process_outputs(mesh)
        # mesh.write('output.2dm', overwrite=True)
        # mesh.values[:] = values
        # values = await self._get_mesh_interpolation(mesh)
        

    async def srun_geom_combine(self, filepaths):
        # first step is to geom-eat; should probably return a new filepaths_original
        
        await self._generate_geom_difference(filepaths)
        gdfs = []
        for fpath in filepaths:
            gdfs.append(gpd.read_feather(fpath))
        gdf = pd.concat(gdfs)
        # gdf.plot(facecolor='none')
        # import matplotlib.pyplot as plt
        # plt.show()
        # r_index = gdf.sindex
        # cnt = 1
        # while len(gdf) > 1:
        #     print(f'niter={cnt}; len(gdf)={len(gdf)}')
        #     used_indexes = set()
        #     neighbors = []
        #     for i, row in enumerate(gdf.itertuples()):
        #         used_indexes.add(i)
        #         row_indices = []
        #         for index in list(r_index.intersection(row.geometry.bounds)):
        #             if index not in used_indexes:
        #                 row_indices.append(index)
        #                 used_indexes.add(index)
        #         neighbors.append(row_indices)

        #     tasks = []
        #     for neighbor_group in neighbors:
        #         tasks.append(self._combine_geom_group(gdf_diff.iloc[neighbor_group]))
        #     gdfs = []
        #     for fpath in asyncio.gather(*tasks):
        #         gdfs.append(gpd.read_feather(fpath))
        #     gdf = pd.concat(gdfs)
        #     r_index = gdf.sindex
        #     cnt+=1
        # raise NotImplementedError('passed the srun_geom_combine')
            # filepaths_original = asyncio.gather(*tasks)
            # gdfs = []
            # for fpath in filepaths_original:
            #     gdfs.append(gpd.read_feather(fpath))
            # gdf = pd.concat(gdfs)
            # r_index = gdf.sindex


    
                    # raise NotImplementedError
        # gdf_diff = self._generate_geom_difference(gdf)
        # mp = gdf.iloc[0].geometry
        mp = gdf.unary_union
        if isinstance(mp, Polygon):
            mp = MultiPolygon([mp])
        return MultiPolygonGeom(mp, crs=gdf.crs)

    async def srun_hfun_combine(self, filepaths_to_combine):
        out_pkl_path = self.config.cache_directory / 'hfun_combined.pkl'
        cmd = []
        if getattr(self.config.hfun, 'slurm', None) is not None:
            nprocs = self.config.hfun.slurm.cpus_per_task
            cmd.extend([
                f'srun',
                f'-c{nprocs}',
            ])
        else:
            nprocs = self.config.hfun.nprocs
        cmd.extend([
                f'{sys.executable}',
                f'{Path(__file__).parent.resolve() / "hfun_pkl_combine.py"}',
            ])
        for filepath in filepaths_to_combine:
            cmd.append(f'{filepath.resolve()}')
        cmd.append(f'--nprocs={nprocs}')
        cmd.append(f'-o={out_pkl_path.resolve()}')
        # print(' '.join(cmd))
        # breakpoint()
        await self.config._await_pexpect(cmd)
        with open(out_pkl_path, 'rb') as fh:
            return MeshHfun(pickle.load(fh))

    async def _generate_geom_difference(self, filepaths):
        for i in range(len(filepaths)):
            cmd = []
            if getattr(self.config.geom, 'slurm', None) is not None:
                nprocs = self.config.geom.slurm.cpus_per_task
                cmd.extend([
                    f'srun',
                    f'-c{nprocs}',
                ])
            else:
                nprocs = self.config.geom.nprocs
            cmd.extend([
                    f'{sys.executable}',
                    f'{Path(__file__).parent.resolve() / "feather_geom_diff_filter.py"}',
                ])
            cmd.extend([f'{fpath.resolve()}' for fpath in filepaths[i:]])
            cmd.append(f'--nprocs={nprocs}')
            await self.config._await_pexpect(cmd)

    def interpolate_mesh(self, mesh):
        # print('getting interp?')
        # if self.config.mesh.interpolate is None:
        #     print('its none?')
        #     return
        # print('will iterate?')
        mesh_pkl_path = self.config.cache_directory / 'mesh.pkl'
        with open(mesh_pkl_path, 'wb') as f:
            pickle.dump(mesh, f)
        tasks = []
        for raster_path, raster_request in self.config.mesh.get_raster_interpolate_requests():
            tasks.append(
                asyncio.get_event_loop().create_task(
                    self._raster_to_mesh_interp(
                        mesh_pkl_path,
                        raster_path,
                        raster_request
                    )
                )
            )
        results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
        mesh_pkl_path.unlink()
        values = np.full(mesh.values.flatten().shape, np.nan)
        for result in results:
            values[result.indexes] = result.values
        mesh.msh_t.value = np.array(
            values.reshape((values.shape[0], 1)), dtype=jigsaw_msh_t.REALS_t
        )

    async def _raster_to_mesh_interp(self, mesh_pkl_path, raster_path, raster_request):
        out_pkl_path = self.config.cache_directory / f'{Path(tempfile.NamedTemporaryFile().name).name}.pkl'
        # print(self.config.cache_directory.resolve())
        # breakpoint()
        # print(out_pkl_path)
        # exit()
        cmd = []
        if getattr(self.config.mesh, 'slurm', None) is not None:
            cmd.extend([
                f'srun',
                f'-c1',
            ])
        cmd.extend([
                f'{sys.executable}',
                f'{Path(__file__).parent.resolve() / "distributed_raster_to_mesh_interp.py"}',
            ])
        cmd.append(f'{Path(raster_path).resolve()}')
        append_raster_cmd_opts(cmd, raster_request)
        cmd.append(f'{mesh_pkl_path.resolve()}')
        cmd.append(f'-o={out_pkl_path.resolve()}')
        # print(' '.join(cmd))
        await self.config._await_pexpect(cmd)
        # # breakpoint()
        # await self.config._await_pexpect(cmd)
        with open(out_pkl_path, 'rb') as fh:
            return pickle.load(fh)
        
    def build_boundaries(self, mesh):
        bound_opts = self.config.mesh.boundaries
        if bound_opts is True:
            mesh.boundaries.auto_generate()
        elif isinstance(bound_opts, dict):
            mesh.boundaries.auto_generate(**bound_opts)
        else:
            raise ValueError(f'Unhandled bound_opts: {bound_opts}')
        
        
    def process_outputs(self, mesh):
        for output_request in self.config.mesh.outputs:
            mesh.write(**output_request)
            

import numpy as np
 
class RasterToMeshInterpResult:
    raster_path: Path
    raster_opts: dict
    pkl_mesh_path: Path
    values: np.ndarray
    indexes: np.ndarray