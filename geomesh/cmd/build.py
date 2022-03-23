import asyncio
import pickle
from pathlib import Path
import os
import sys


# import dask_geopandas
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

from geomesh.driver import JigsawDriver
from geomesh.geom.shapely import MultiPolygonGeom
from geomesh.hfun.mesh import MeshHfun

from .configparser import ConfigParser

class BuildCli:

    def __init__(self, yml_path: os.PathLike):
        self.config = ConfigParser.from_yml(yml_path)

    def main(self):
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
        geom.make_plot(show=True)
        driver = JigsawDriver(
            geom=geom,
            hfun=hfun,
            verbosity=1
        )
        mesh = driver.output_mesh
        mesh.triplot(show=True)
        raise NotImplementedError('Driver is ready, save mesh now.')

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

        # should return geom diff gdf


        # len_geoms = len(gdf) 
        # data = []
        # geoms = [geom.geometry for geom in gdf.itertuples()]       
        # for i in range(len_geoms):
        #     if isinstance(geoms[i], Polygon):
        #         geoms[i] = MultiPolygon([geoms[i]])
        #     data.append({
        #         'geometry': geoms[i],
        #         })
        #     for j in range(i+1, len_geoms):
        #         geoms[j] = geoms[j].difference(geoms[i])

        # gdf = gpd.GeoDataFrame(data, crs=gdf.crs)
        # gdf.plot(facecolor='none', cmap='jet')
        # import matplotlib.pyplot as plt
        # plt.show()




            
        return 

    # async def _combine_geom_group(self, neighbor_group):

