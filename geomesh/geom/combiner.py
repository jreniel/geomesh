from copy import deepcopy
import logging
from multiprocessing import Pool
from typing import List, Union

from pyproj import CRS

from ..cmd.config.server import ServerConfig, SlurmConfig
from .base import BaseGeom
from .geom import Geom
from .shapely import MultiPolygonGeom

logger = logging.getLogger(__name__)

class GeomCombiner:
    
    def __init__(self, geom_list: List[BaseGeom], server_config: ServerConfig = None):
        self.geom_list = geom_list
        self.server_config = server_config

    def __call__(self) -> MultiPolygonGeom:
        return self.combine()
        
    def combine(self) -> MultiPolygonGeom:
        if isinstance(self.server_config, SlurmConfig):
            raise NotImplementedError('GeomCombiner with SlurmConfig is not implemented.')
        elif isinstance(self.server_config, ServerConfig):
            self._combine_parallel()
        else:
            raise ValueError(f"Unhandled argument type for self.server_config={self.server_config}.")

    def _combine_parallel(self):
        from time import time
        import geopandas as gpd
        import matplotlib.pyplot as plt

        geom_gdf = gpd.GeoDataFrame([{'geometry': geom.multipolygon, 'id': i} for i, geom in enumerate(self.geom_list)])
        rtree = geom_gdf.sindex

        for i, geom in enumerate(self.geom_list):
            geom_mp = geom.multipolygon
            for other_geom in self.geom_list[i:]:
                # print(geom_mp)
                geom_mp = geom_mp.difference(other_geom.multipolygon)
                gpd.GeoDataFrame([{'geometry': geom_mp}], crs='epsg:4326').plot()
                plt.show()
            breakpoint()
        # for i, geom in reversed(list(enumerate(self.geom_list)):
        #     rtree.intersection()



        logger.info('Start geom prioritization cropping for parallel geom combine.')
        start = time()

        for i, geom in reversed(list(enumerate(self.geom_list))):
            position = []
            job_args = []
            for j, other_geom in reversed(list(enumerate(self.geom_list[:i]))):
                position.append(j)
                job_args.append([geom, other_geom])
                # if geom.multipolygon.touches(other_geom.multipolygon):
                # start2 = time()
                # logger.info('Taking difference.')
                # self.geom_list[:i][j] = 
                # logger.info(f'Taking difference took: {time()-start2}')

            with Pool(processes=self.server_config.nprocs) as pool:
                results = pool.starmap(
                    self._compute_geom_difference,
                    job_args
                )
            pool.join()

            for pos in position:
                self._geom_list[pos] = results.pop()
           
        # logger.info(f'total cropping time: {time()-start}')


        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        # logger.info('Start geom prioritization cropping for parallel geom combine.')
        # start = time()

        # for i, geom in reversed(list(enumerate(self.geom_list))):
        #     position = []
        #     job_args = []
        #     for j, other_geom in reversed(list(enumerate(self.geom_list[i+1:]))):
        #         position.append(j+i)
        #         job_args.append([geom, other_geom])
        #         # if geom.multipolygon.touches(other_geom.multipolygon):
        #         # start2 = time()
        #         # logger.info('Taking difference.')
        #         # self.geom_list[:i][j] = 
        #         # logger.info(f'Taking difference took: {time()-start2}')

        #     with Pool(processes=self.server_config.nprocs) as pool:
        #         results = pool.starmap(
        #             self._compute_geom_difference,
        #             job_args
        #         )
        #     pool.join()

        #     for pos in position:
        #         self._geom_list[pos] = results.pop()
           
        # logger.info(f'total cropping time: {time()-start}')

        # logger.info('Start geom prioritization cropping for parallel geom combine.')
        # start = time()
        # for i, geom in reversed(list(enumerate(self.geom_list))):
        #     job_args = []
        #     for j, other_geom in reversed(list(enumerate(self.geom_list[:i]))):
        #         job_args.append([geom, other_geom])
        #         # if geom.multipolygon.touches(other_geom.multipolygon):
        #         # start2 = time()
        #         # logger.info('Taking difference.')
        #         # self.geom_list[:i][j] = 
        #         # logger.info(f'Taking difference took: {time()-start2}')

        #     with Pool(processes=self.server_config.nprocs) as pool:
        #         results = pool.starmap(
        #             self._compute_geom_difference,
        #             job_args
        #         )
        #     pool.join()
        #     for j, new_geom in enumerate(results):
        #         self.geom_list[:i][j] = new_geom
           
        # logger.info(f'total cropping time: {time()-start}')

        # import geopandas as gpd
        geom_gdf = gpd.GeoDataFrame([{'geometry': geom.multipolygon, 'id': i} for i, geom in enumerate(self.geom_list)])

        geom_gdf.plot('id', facecolor='none')
        plt.show(block=False)
        breakpoint()
        raise NotImplementedError('combine')
    
    @staticmethod
    def _compute_geom_difference(geom, other_geom):
        return Geom(geom.multipolygon.difference(other_geom.multipolygon), crs='epsg:4326')
       
    @property
    def geom_list(self) -> List[BaseGeom]:
        return self._geom_list
    
    @geom_list.setter
    def geom_list(self, geom_list: List[BaseGeom]):
        geom: BaseGeom
        for i, geom in enumerate(geom_list):
            if not isinstance(geom, BaseGeom):
                raise TypeError(f'Argument {i} of `geom_list` must be a derived type of {BaseGeom}, not {type(geom)}.')
        wgs84 = CRS.from_epsg(4326)
        for i, geom in list(enumerate(geom_list)):
            if not geom.crs.equals(wgs84):
                # self.geom_list[i] = Geom(geom.get_multipolygon(nprocs), crs=wgs84)
                raise ValueError('All Geom objects must be epsg:4326 for combining.')
        self._geom_list = geom_list
    

    @property
    def server_config(self) -> ServerConfig:
        return self._server_config
    
    @server_config.setter
    def server_config(self, server_config: Union[ServerConfig, None]):
        if server_config is None:
            server_config = ServerConfig(-1)
        if not isinstance(server_config, ServerConfig):
            raise ValueError(f'Argument `server_config` must be of type `{ServerConfig}` or `None`, not type {type(server_config)}.')
        self._server_config = server_config