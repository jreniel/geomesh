import asyncio
from functools import cached_property, partial
import hashlib
import logging
from multiprocessing import Pool
import pickle
import sys
import tempfile
from time import time
from typing import List, Union

import hashlib
import pexpect
from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import MultiPolygon

from geomesh import Raster
from geomesh.cmd.config.server import ServerConfig, SlurmConfig
from ... import db
from ...geom import Geom
from .yamlparser import YamlComponentParser

logger = logging.getLogger(__name__)

class GeomConfig(YamlComponentParser):
    """This class parses the "geom" component of a yaml configuration file."""


    def __call__(self) -> Union[Geom, None]:
        """Returns the fully built geom for this configuration."""

        if self.config is None:
            return
        if self.config.cache is None:
            raise NotImplementedError("Cache is disabled, what to do? -- Easy build an in-memory spatialite db")
        build_id = self.get_build_id()
        if build_id is None:
            return
        res = self.config.cache.geom.get(build_id, db.orm.Geom)
        if res is None:
            res = self.build_geom()
            self.config.cache.geom.add(build_id, db.orm.Geom)
        return res

    #     # geom_build_id = self.get_build_id()
    #     # geom = cache.geom.fetch_by_build_id(geom_build_id)

    @cached_property
    def geom_raster_config(self):

        geom_raster_config = self.config.yaml["geom"].get("rasters")

        if 'rasters' not in self.config.yaml["geom"]:
            geom_raster_config = None

        if not isinstance(geom_raster_config, list):
            raise ValueError(f"geom.raster entry must be of type list, not {type(geom_raster_config)}.")
        
        for item in geom_raster_config:
            if not isinstance(item, dict):
                raise ValueError("geom.raster entries must be a mapping.")
            
            if not (
                bool(item.get("path", False)) ^
                bool(item.get("tile-index", False)) ^
                bool(item.get("tile_index", False))):
                raise ValueError("geom.raster entries must contain only one of 'path' or 'tile_index' keys.")

        return geom_raster_config

    @cached_property
    def geom_feature_config(self):

        if 'features' not in self.config.yaml["geom"]:
            geom_feature_config = None
            
        geom_feature_config = self.config.yaml["geom"].get("features", [])
        if not isinstance(geom_feature_config, list):
            raise ValueError(f"geom.features entry must be of type list, not {type(geom_feature_config)}.")
        
        for item in geom_feature_config:
            if not isinstance(item, dict):
                raise ValueError("geom.features entries must be a mapping.")
            
            if not (
                bool(item.get("mesh", False)) ^
                bool(item.get("geometry", False))):
                raise ValueError("geom.features entries must contain only one of 'mesh' or 'geometry' keys.")
        
        return geom_feature_config


    def get_build_id(self) -> Union[str, None]:
        """Will generate an ID for the unique combination of user requests.M"""
        if isinstance(self.server_config, SlurmConfig):
            return self._get_build_id_from_srun()
        elif isinstance(self.server_config, ServerConfig):
            return self._get_build_id_parallel()
        else:
            raise Exception(f'Unhandled value for `server_config` of type {type(self.config.server_config)}')
        
    
    def build_geom(self) -> Geom:
        if isinstance(self.server_config, SlurmConfig):
            return self._build_geom_slurm()
        elif isinstance(self.server_config, ServerConfig):
            return self._build_geom_parallel()
        else:
            raise Exception(f'Unhandled value for `server_config` of type {type(self.config.server_config)}')
    
    def _build_geom_parallel(self) -> Geom:
        
        rasters_geom = self._get_rasters_geom_parallel()
        
        # logger.info(f'Building raster geoms took {time()-start}.')
        # logger.info('Combining raster geoms (this takes some time)...')
        # logger.debug('This part needs to be parallelized, and it will be in the near future.')
        print('WE HAVE THE GEOMS BUT NEED TO PARALLEL-COMBINE')
        exit()
        # return Geom(
        #     ops.unary_union([raster_geom.multipolygon for raster_geom in raster_geoms]), 
        #     crs="epsg:4326"
        # )
        
    
    def _get_rasters_geom_parallel(self, start_index: int = None, end_index: int = None)-> List[MultiPolygon]:
        
        # end_index = if end_index == -1
        
        job_args = []
        # raster_geom_opts = []
        for i, ((raster_path, raster_opts), geom_opts) in enumerate(self._get_raster_iter()):
            if start_index is not None:
                if i < start_index:
                    continue
            job_args.append((
                raster_path,
                raster_opts,
                self.config.rasters.apply_opts,
                geom_opts,
            ))
            if end_index is not None:
                if i == end_index:
                    break
            # raster_geom_opts.append(geom_opts)
        logger.info('Start building raster geoms.')
        start = time()
        with Pool(processes=self.server_config.nprocs) as pool:
            # NOTE: All raster geoms will be stored in lat/lon coords.
            raster_geoms = pool.starmap(
                self._build_raster_geom,
                job_args
            )
        pool.join()
        return raster_geoms

    # def _build_geom_parallel_srun(self, start_index, end_index, tmpfiles):
    #     job_args = []
    #     # raster_geom_opts = []
    #     for i, ((raster_path, raster_opts), geom_opts) in enumerate(self._get_raster_iter()):
    #         !!!!!!!!!!!!!!!!!!
    #         job_args.append((
    #             raster_path,
    #             raster_opts,
    #             self.config.rasters.apply_opts,
    #             geom_opts,
    #         ))
    #         # raster_geom_opts.append(geom_opts)
    #     logger.info('Start building raster geoms.')
    #     start = time()
    #     with Pool(processes=self.server_config.nprocs) as pool:
    #         # NOTE: All raster geoms will be stored in lat/lon coords.
    #         raster_geoms = pool.starmap(
    #             self._build_raster_geom,
    #             job_args
    #         )
    #     pool.join()
    #     logger.info(f'Building raster geoms took {time()-start}.')
    #     logger.info('Combining raster geoms (this takes some time)...')
    #     logger.debug('This part needs to be parallelized, and it will be in the near future.')
    #     print('WE HAVE THE GEOMS BUT NEED TO PARALLEL-COMBINE')
    #     exit()
    #     return Geom(
    #         ops.unary_union([raster_geom.multipolygon for raster_geom in raster_geoms]), 
    #         crs="epsg:4326"
    #     )
    
    @staticmethod
    def _build_raster_geom(
        raster_path,
        raster_opts,
        apply_opts,
        geom_opts,
    ) -> Geom:
        """Geoms generated by this function are all returned with crs="epsg:4326".
        This is so that all resulting geoms can be merged at the end of the algorithm on the same CRS.
        """
        original_geom = Geom(apply_opts(Raster(raster_path), raster_opts), **geom_opts)
        original_crs = original_geom.crs
        wgs84_crs = CRS.from_epsg(4326)
        if not original_crs.equals(wgs84_crs):
            wgs84_mp = ops.transform(
                Transformer.from_crs(
                    original_crs,
                    wgs84_crs,
                    always_xy=True
                ).transform,
                original_geom.get_multipolygon()
            )
            del original_geom
            return Geom(wgs84_mp, crs=wgs84_crs)
        return Geom(original_geom.get_multipolygon(), crs=original_crs)
        
    def _build_geom_slurm(self):
        # https://stackoverflow.com/questions/42231161/asyncio-gather-vs-asyncio-wait



        tmpfiles = []
        futures = []
        start_index = 0
        for i, ((raster_request, raster_opts), geom_opts) in enumerate(self._get_raster_iter()): 
            tmpfiles.append(tempfile.NamedTemporaryFile())
            if (i+1) % self.server_config.cpus_per_task == 0:
                futures.append(asyncio.gather(self._get_rasters_geom_from_srun(start_index, i, tmpfiles)))
                start_index = i + 1
        raster_geom_paths = asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
        print(raster_geom_paths)
        exit()
                  
        
        tasks = []
        tmpfiles = []
        for (raster_request, raster_opts), geom_opts in self._get_raster_iter():
            tmpfiles.append(tempfile.NamedTemporaryFile())
            tasks.append(
                partial(
                    self._get_raster_geom_from_srun, 
                        raster_request,
                        raster_opts,
                        geom_opts,
                        tmpfiles[-1]
                    )
                )
        raster_geom_paths = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*[func() for func in tasks])
        )
        print(raster_geom_paths)
        exit()

        # tmpfiles = []
        # futures = []
        # tasks = []
        # job_args = []
        # for i, ((raster_request, raster_opts), geom_opts) in enumerate(self._get_raster_iter()): 
        #     tmpfiles.append(tempfile.NamedTemporaryFile())
        #     job_args.append((
        #         raster_request,
        #         raster_opts,
        #         geom_opts,
        #         tmpfiles[-1]
        #     ))
        #     if i % self.server_config.cpus_per_task == 0:
        #         tasks.append(
        #             partial(
        #                 self._get_raster_geom_from_srun, 
        #                 job_args
        #             )
        #         )
        #         futures.append(asyncio.gather(*[func() for func in tasks]))
        #         tasks = []
        #         job_args = []
        # raster_geom_paths = asyncio.get_event_loop().run_until_complete(asyncio.gather(asyncio.gather(*futures)))
        # print(raster_geom_paths)
        # exit()
         
        # rasters_data = []
        # features_data = []

        # tmpfiles = []
        # futures = []
        # tasks = []
        # job_args = []
        # start_index = 0
        # for i, ((raster_request, raster_opts), geom_opts) in enumerate(self._get_raster_iter()): 
        #     tmpfiles.append(tempfile.NamedTemporaryFile())
            # job_args.append((
            #     raster_request,
            #     raster_opts,
            #     geom_opts,
            #     tmpfiles[-1]
            # ))
            # breakpoint()
            # if (i+1) % self.server_config.cpus_per_task == 0:
            #     futures.append(asyncio.gather(self._get_raster_build_id_from_srun(start_index=start_index, end_index=i)))
            #     # job_args = []
            #     start_index = i + 1
                
        # print(futures)
        # print('will launch event loop')
        # raster_geom_paths = asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
        # print(raster_geom_paths)
        # exit()

    
    async def _get_rasters_geom_from_srun(self, start_index, end_index, tmp_outputs):
        
        
        wrap = [
            # "from geomesh import Geom, Raster",
            # "from geomesh.cmd.config.raster import RasterConfig",
            # "from pyproj import CRS, Transformer",
            # "from shapely import ops",
            # "import geopandas as gpd",
            # f"original_geom = Geom(Raster('{raster_path}'), **{geom_opts})",
            # "original_crs = original_geom.crs",
            # "wgs84_crs = CRS.from_epsg(4326)",
            # "if not original_crs.equals(wgs84_crs):",
            # "\twgs84_mp = ops.transform(",
            # "\t\tTransformer.from_crs(",
            # "\t\t\toriginal_crs,",
            # "\t\t\twgs84_crs,",
            # "\t\t\talways_xy=True",
            # "\t\t).transform,",
            # "\t\toriginal_geom.get_multipolygon()",
            # "\t)",
            # f"\tgpd.GeoDataFrame(wgs84_mp, crs=wgs84_crs).to_feather('{tmp_output.name}')",
            # "else:",
            # f"\tgpd.GeoDataFrame(original_geom.get_multipolygon(), crs=original_crs).to_feather('{tmp_output.name}')"
            
            "import pickle",
            "from geomesh.cmd.config.yamlparser import YamlParser",
            "from geomesh.cmd.config.server import ServerConfig",
            f"yamlparser = YamlParser('{self.config.path.resolve()}')",
            f"yamlparser.geom.server_config = ServerConfig({self.server_config.cpus_per_task})",
            f"raster_geoms = yamlparser.geom._get_rasters_geom_parallel(start_index={start_index}, end_index={end_index})",
            "print(raster_geoms)"
            # f"with open('{tmp_output.name}', 'wb') as f:",
            # f"\tpickle.dump(, f)"
            
        ]

        tmp_script = tempfile.NamedTemporaryFile()
        with open(tmp_script.name, 'w') as f:
            f.write('\n'.join(wrap).replace('\t', 4*" "))

        cmd = [
            "srun",
            f"-c{self.server_config.cpus_per_task}",
            f"{sys.executable} {tmp_script.name}"
        ]
        with pexpect.spawn(' '.join(cmd), encoding='utf-8', timeout=None) as p:
            await p.expect(pexpect.EOF, async_=True)

        if p.exitstatus != 0:
            raise Exception(p.before)

        # with open(tmp_output.name, 'rb') as f:
        #     return pickle.load(f)
    
    
        
        
        
    #     rasters_geoms = self.build_rasters_geoms()
    #     features_geoms = self.build_features_geoms()
    #     res = ops.unary_union(
    #         [
    #             *[rg.multipolygon for rg in rasters_geoms],
    #             *[fg.multipolygon for fg in features_geoms],
    #         ]
    #     )
    #     return Geom(res)

    # def build_rasters_geoms(self):
    #     rasters = list(self.rasters)
    #     geoms = len(rasters) * [None]
    #     for i, feat_build_id in enumerate(self.iter_raster_build_ids()):
    #         res = self.config.cache.geom.get(feat_build_id, db.orm.GeomCollection)
    #         if res is not None:
    #             geoms[i] = res
    #     jobs = [rasters[i] for i, val in enumerate(geoms) if val is None]
    #     if len(jobs) > 0:
    #         breakpoint()
    #     return []

    # def build_features_geoms(self):
    #     return [
    #         Geom(feature.multipolygon, crs=feature.crs) for feature, _ in self.features
    #     ]

    @staticmethod
    def _build_raster_geom_():
        pass
    
    def _get_build_id_parallel(self):
        return self._compute_build_id(
            self._get_rasters_md5_parallel(),
            self._get_features_md5_parallel()
            )
    

    def _get_build_id_from_srun(self):
        """
        Three different paradigms were considered for this part. Below they are listed starting from the fastest running to the slowest.
        Inverse to speed here is the RAM requirement. The slowest methodologies are more bigmem friendly, but they might be an overkill
        for this function
        
            (fastest) - Do a single call to srun that wraps all rasters into a single `multiprocessing.Pool` with `cpus_per_tasks` value. This is the
            equivalent of an OpenMP application. It consumes a lot of RAM, but adjusting `chunk_size` on the largest rasters (usually the base rasters
            makes a difference. Otherwise you have to decrease the number of `cpus_per_task`

            (medium) - Do multiple asynchronous calls to srun each with 1 individual raster. This is robust and can distribute the job over multiple nodes,
            however, it has the disadvantage that it is harder to control how many processors are allocated in a single node, therefore fully
            suscribing a node can end up draining the entire RAM.

            (slowest) - Do multiple asynchrounous calls to srun that groups rasters into lists of len(rasters) <= cpus_per_tasks for a single call
            to multiprocessing.Pool. Most RAM tolerant, but also the slowest.
            
        It was decided that for this function, the first one of this three paradigms would be implemented, so this particular function behaves
        similar to an OMP (multithreaded) app.

        """



        # tmpfiles = []
        # futures = []
        # tasks = []
        # job_args = []
        # for i, ((raster_request, raster_opts), geom_opts) in enumerate(self._get_raster_iter()): 
        #     tmpfiles.append(tempfile.NamedTemporaryFile())
        #     job_args.append((
        #         raster_request,
        #         raster_opts,
        #         geom_opts,
        #         tmpfiles[-1]
        #     ))
        #     if i % self.server_config.cpus_per_task == 0:
        #         tasks.append(
        #             partial(
        #                 self._get_raster_build_id_from_srun, 
        #                 job_args
        #             )
        #         )
        #         futures.append(asyncio.gather(*[func() for func in tasks]))
        #         tasks = []
        # raster_geom_paths = asyncio.get_event_loop().run_until_complete(asyncio.gather(asyncio.gather(*futures)))
        # print(raster_geom_paths)
        # exit()
         
        # rasters_data = []
        # features_data = []

        # tmpfiles = []
        # futures = []
        # tasks = []
        # job_args = []
        # start_index = 0
        # for i, ((raster_request, raster_opts), geom_opts) in enumerate(self._get_raster_iter()): 
        #     tmpfiles.append(tempfile.NamedTemporaryFile())
            # job_args.append((
            #     raster_request,
            #     raster_opts,
            #     geom_opts,
            #     tmpfiles[-1]
            # ))
            # breakpoint()
            # if (i+1) % self.server_config.cpus_per_task == 0:
            #     futures.append(asyncio.gather(self._get_raster_build_id_from_srun(start_index=start_index, end_index=i)))
            #     # job_args = []
            #     start_index = i + 1
                
        # print(futures)
        # print('will launch event loop')
        # raster_geom_paths = asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
        # print(raster_geom_paths)
        # exit()
            
        # for 
        rasters_data, features_data = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(
                self._get_rasters_md5_from_srun(),
                self._get_features_md5_from_srun(),
            )
        )
        return self._compute_build_id(rasters_data, features_data)
    
    async def _get_raster_build_id_from_srun(self):
        tmp_output = tempfile.NamedTemporaryFile()
        wrap = [
            "import pickle",
            "from geomesh.cmd.config.yamlparser import YamlParser",
            "from geomesh.cmd.config.server import ServerConfig",
            f"yamlparser = YamlParser('{self.config.path.resolve()}')",
            f"yamlparser.geom.server_config = ServerConfig({self.server_config.cpus_per_task})",
            f"with open('{tmp_output.name}', 'wb') as f:",
            f"\tpickle.dump(yamlparser.geom._get_rasters_md5_parallel(), f)"
        ]

        tmp_script = tempfile.NamedTemporaryFile()
        with open(tmp_script.name, 'w') as f:
            f.write('\n'.join(wrap))

        cmd = [
            "srun",
            f"-c{self.server_config.cpus_per_task}",
            f"{sys.executable} {tmp_script.name}"
        ]

        with pexpect.spawn(' '.join(cmd), encoding='utf-8', timeout=None) as p:
            await p.expect(pexpect.EOF, async_=True)
        print(p.before)
        if p.exitstatus != 0:
            raise Exception(p.before)
        with open(tmp_output.name, 'rb') as f:
            return pickle.load(f)
        

    @staticmethod
    def _compute_build_id(rasters_data, features_data):
        # assert len(rasters_md5) == len(features_md5) == len(geoms_opts)
        f: List[str] = []
        for raster_md5, geom_opts in zip(*rasters_data):
            zmin = geom_opts.get("zmin")
            zmax = geom_opts.get("zmax")
            zmin = "" if zmin is None else f"{zmin}"
            zmax = "" if zmax is None else f"{zmax}"
            f.append(f"{raster_md5}{zmin}{zmax}")
        for feat_md5, feat_geom_opts in zip(*features_data):
            f.append(f"{feat_md5}")
        if len(f) == 0:
            return None
        return hashlib.md5("".join(f).encode("utf-8")).hexdigest()
        
    def _get_features_md5_parallel(self):
        with Pool(processes=self.server_config.nprocs) as pool:
            result = pool.starmap(
                self._get_feature_md5,
                [(feature_request) for feature_request, _ in self._get_feature_iter(yield_type='path')]
            )
        pool.join()
        return result 

    @staticmethod
    def _get_feature_md5(feature_path):
        hash_md5 = hashlib.md5()
        with open(feature_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_feature_iter(self, yield_type='path'):
        for geom_feature_request in self.geom_feature_config:
            for feature, geom_opts in self.config.features.from_request(geom_feature_request, yield_type):
                yield feature, geom_opts
    
    def _get_rasters_md5_parallel(self):
        job_args = []
        raster_geom_opts = []
        for (raster_request, raster_opts), geom_opts in self._get_raster_iter():
            job_args.append((
                raster_request,
                raster_opts,
                self.config.rasters.apply_opts,
            ))
            raster_geom_opts.append(geom_opts)
        with Pool(processes=self.server_config.nprocs) as pool:
            result = pool.starmap(
                self._get_raster_md5,
                job_args
            )
        pool.join()
        return result, raster_geom_opts
    
    @staticmethod
    def _get_raster_md5(raster_path, raster_opts, apply_opts):
        return apply_opts(Raster(raster_path), raster_opts).md5
    
    def _get_raster_iter(self):
        for geom_raster_request in self.geom_raster_config:
            for raster, raster_opts in self.config.rasters.from_request(geom_raster_request, 'path'):
                if raster is not None:
                    geom_opts = {
                        'zmin': geom_raster_request.get('zmin'),
                        'zmax': geom_raster_request.get('zmax'),
                    }
                    yield (raster, raster_opts), geom_opts
   

   
    # def _get_build_id(self):
    #     # TODO: This could be replaced for an async version.
    #     f: List[str] = []
    #     for raster, raster_opts, geom_opts in self._get_raster_iter(yield_type='path'):
    #         zmin = geom_opts.get("zmin")
    #         zmax = geom_opts.get("zmax")
    #         zmin = "" if zmin is None else f"{zmin}"
    #         zmax = "" if zmax is None else f"{zmax}"
    #         f.append(f"{raster.md5}{zmin}{zmax}")
    #     for feat, geom_opts in self.features:
    #         f.append(f"{feat.md5}")
    #     # NOTE: Considered sorting so that the input order of items doesn't matter
    #     # but on second thought, order does matter in some cases and it's hard to distiguish
    #     # a-priory on which cases it matters and when it doesn't, although it's not impossible to know.
    #     # f.sort()
    #     if len(f) == 0:
    #         return None
    #     return hashlib.md5("".join(f).encode("utf-8")).hexdigest()
    

    # async def _get_rasters_md5_parallel_srun(self, job_args):
    #     wrap = [
    #         "import pickle",
    #         "from geomesh.cmd.config.yamlparser import YamlParser",
    #         "from geomesh.cmd.config.server import ServerConfig",
    #         f"yamlparser = YamlParser('{self.config.path.resolve()}')",
    #         f"yamlparser.geom.server_config = ServerConfig({self.server_config.cpus_per_task})",
    #         f"with open('{tmp_output.name}', 'wb') as f:",
    #         "\tpickle.dump(yamlparser.geom._get_rasters_md5_parallel(), f)"
    #     ]

    #     with Pool(processes=self.server_config.nprocs) as pool:
    #         result = pool.starmap(
    #             self._get_raster_md5,
    #             job_args
    #         )
    #     pool.join()
    #     return result, raster_geom_opts
    
    
    async def _get_rasters_md5_from_srun(self):
        
        if len(self.geom_raster_config) == 0:
            return []

        tmp_output = tempfile.NamedTemporaryFile()
        wrap = [
            "import pickle",
            "from geomesh.cmd.config.yamlparser import YamlParser",
            "from geomesh.cmd.config.server import ServerConfig",
            f"yamlparser = YamlParser('{self.config.path.resolve()}')",
            f"yamlparser.geom.server_config = ServerConfig({self.server_config.cpus_per_task})",
            f"with open('{tmp_output.name}', 'wb') as f:",
            f"\tpickle.dump(yamlparser.geom._get_rasters_md5_parallel(), f)"
        ]

        tmp_script = tempfile.NamedTemporaryFile()
        with open(tmp_script.name, 'w') as f:
            f.write('\n'.join(wrap))

        cmd = [
            "srun",
            f"-c{self.server_config.cpus_per_task}",
            f"{sys.executable} {tmp_script.name}"
        ]
        with pexpect.spawn(' '.join(cmd), encoding='utf-8', timeout=None) as p:
            await p.expect(pexpect.EOF, async_=True)

        if p.exitstatus != 0:
            raise Exception(p.before)
        with open(tmp_output.name, 'rb') as f:
            return pickle.load(f)
        
    async def _get_features_md5_from_srun(self):

        if len(self.geom_feature_config) == 0:
            return []

        tmp_output = tempfile.NamedTemporaryFile()
        wrap = [
            "import pickle",
            "from geomesh.cmd.config.yamlparser import YamlParser",
            "from geomesh.cmd.config.server import ServerConfig",
            f"yamlparser = YamlParser('{self.config.path.resolve()}')",
            f"yamlparser.geom.server_config = ServerConfig({self.server_config.cpus_per_task})",
            f"with open('{tmp_output.name}', 'wb') as f:",
            "\tpickle.dump(yamlparser.geom._get_features_md5_parallel(), f)"
        ]
        tmp_script = tempfile.NamedTemporaryFile()
        with open(tmp_script.name, 'w') as f:
            f.write('\n'.join(wrap))
        cmd = [
            "srun",
            f"-c{self.server_config.cpus_per_task}",
            f"{sys.executable} {tmp_script.name}"
        ]
        with pexpect.spawn(' '.join(cmd), encoding='utf-8', timeout=None) as p:
            await p.expect(pexpect.EOF, async_=True)
        if p.exitstatus != 0:
            raise Exception(p.before)
        with open(tmp_output.name, 'rb') as f:
            return pickle.load(f)
    
    @property
    def key(self):
        return "geom"
    
    @cached_property
    def server_config(self):
        has_slurm = bool(self.config.yaml["geom"].get("slurm", False))
        has_nprocs = bool(self.config.yaml["geom"].get("nprocs", False))
        if has_slurm and has_nprocs:
            raise ValueError('Yaml entry `geom` must only contain one of `slurm` or `nprocs`, but not both.')
        
        # user did not specify, assumes full power local parallelization
        if not has_nprocs and not has_slurm:
            return ServerConfig(-1)
        
        if has_slurm:
            return SlurmConfig(
                **self.config.yaml["geom"]["slurm"],
                # modules=self.modules_config
                )
        if has_nprocs:
            return ServerConfig(
                int(self.config.yaml["geom"]["nprocs"]),
                # modules=self.modules_config
                )
        
    @cached_property
    def modules_config(self):
        raise NotImplementedError('modules_config')
            

            




    # @property
    # def rasters(self):
    #     """
    #     Returns a generator of (Raster, geom_opts) tuple.
    #     """
    #     for geom_raster_request in self.geom_raster_config:
    #         for raster in self.config.rasters.from_request(geom_raster_request):
    #             if raster is not None:
    #                 yield raster, {
    #                     'zmin': geom_raster_request.get('zmin'),
    #                     'zmax': geom_raster_request.get('zmax'),
    #                 }

    # @property
    # def features(self):
    #     for geom_feature_request in self.geom_features_config:
    #         for feature in self.config.features.from_request(geom_feature_request):
    #             if feature is not None:
    #                 yield feature, {
    #                     # 'zmin': geom_feature_request.get('zmin'),
    #                     # 'zmax': geom_feature_request.get('zmax'),
    #                 }

        # config = self.config.yaml.get("feature", [])
        # if isinstance(config, dict):
        #     config = [config]
        # for request in config:
        #     for feature in self.config.features.from_request(request):
        #         yield feature, request

    # @staticmethod
    # def _get_raster_md5(path, ops):
    #     return GeomConfig._apply_ops_to_raster(Raster(path)).md5

    # def iter_raster_build_ids(self) -> Generator:
    #     for raster, request in self.rasters:
    #         zmin = request.get("zmin")
    #         zmax = request.get("zmax")
    #         zmin = "" if zmin is None else f"{zmin:G}"
    #         zmax = "" if zmax is None else f"{zmax:G}"
    #         yield hashlib.md5(f"{raster.md5}{zmin}{zmax}".encode("utf-8")).hexdigest()

    # def iter_features_build_ids(self):
    #     for feat, request in self.features:
    #         yield feat.md5
