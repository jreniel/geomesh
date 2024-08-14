from copy import copy
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union
import logging
import os
import tempfile

from jigsawpy import libsaw, jigsaw_msh_t, jigsaw_jig_t
from pyproj import CRS
import jigsawpy
import numpy as np
import pandas as pd

from geomesh import utils
from geomesh.geom.base import BaseGeom
from geomesh.geom.geom import Geom
from geomesh.hfun.base import BaseHfun
from geomesh.hfun.hfun import Hfun
from geomesh.mesh.base import BaseMesh
from geomesh.mesh.mesh import Mesh


logger = logging.getLogger(__name__)


class JigsawDriverArgumentError(Exception):
    pass


class JigsawDriver:

    def __init__(
            self,
            geom: Union[BaseGeom, jigsaw_msh_t, None] = None,
            hfun: Union[BaseHfun, jigsaw_msh_t, None] = None,
            initial_mesh: Union[BaseMesh, jigsaw_msh_t, None] = None,
            verbosity: int = 0,
            dst_crs=None,
            sieve=None,
            finalize: bool = True,
            geom_feat: bool = None,
    ):
        has_geom = True if geom is not None else False
        has_hfun = True if hfun is not None else False
        has_initial_mesh = True if initial_mesh is not None else False
        if not (has_geom | has_hfun | has_initial_mesh):
            raise JigsawDriverArgumentError(
                "Arguments to JigsawDriver must contain at least one of `geom`"
                ", `hfun` or `initial_mesh`")
        self.geom = geom
        self.hfun = hfun
        self.initial_mesh = initial_mesh
        self.verbosity = verbosity
        self.dst_crs = dst_crs
        self.sieve = sieve
        self.finalize = bool(finalize)
        if geom_feat is not None:
            self.opts.geom_feat = bool(geom_feat)

    async def submit(self, executor, **kwargs):
        msh_t_build_cmd, output_msh_t_path, local_crs = self._get_msh_t_build_cmd(
                binary=kwargs.pop("binary", "jigsaw"),
                cwd=executor.cwd,
                )
        print(kwargs, flush=True)
        await executor.submit(
                msh_t_build_cmd,
                cwd=self.__tmpworkdir.name,
                **kwargs
                )
        output_msh_t = jigsaw_msh_t()
        jigsawpy.loadmsh(str(output_msh_t_path), output_msh_t)
        del self.__tmpworkdir
        output_msh_t = self._finalize_msh_t(output_msh_t)
        if local_crs is not None:
            output_msh_t.crs = local_crs
            utils.reproject(output_msh_t, self.dst_crs)
        else:
            output_msh_t.crs = None
        return output_msh_t

    def _get_projected_msh_t_set(self, dst_crs=None):
        print('get projected msh_t set')
        local_azimuthal_projection = None
        if dst_crs is not None:
            self.dst_crs = dst_crs
        if self.geom is not None:
            geom = self.geom.msh_t(dst_crs=self.dst_crs)
        else:
            geom = None
        if self.hfun is not None:
            hfun = self.hfun.msh_t(dst_crs=self.dst_crs)
        else:
            hfun = None

        if self.initial_mesh is not None:
            if isinstance(self.initial_mesh, jigsaw_msh_t):
                init = self.initial_mesh
            else:
                init = self.initial_mesh.msh_t(dst_crs=self.dst_crs)
        else:
            init = None

        if self.dst_crs.is_geographic:
            y0, x0 = 2*[float('inf')]
            y1, x1 = 2*[-float('inf')]
            if geom is not None:
                y0 = np.min([y0, np.min(geom.vert2['coord'][:, 1])])
                y1 = np.max([y1, np.max(geom.vert2['coord'][:, 1])])
                x0 = np.min([x0, np.min(geom.vert2['coord'][:, 0])])
                x1 = np.max([x1, np.max(geom.vert2['coord'][:, 0])])

            if hfun is not None:
                y0 = np.min([y0, np.min(hfun.vert2['coord'][:, 1])])
                y1 = np.max([y1, np.max(hfun.vert2['coord'][:, 1])])
                x0 = np.min([x0, np.min(hfun.vert2['coord'][:, 0])])
                x1 = np.max([x1, np.max(hfun.vert2['coord'][:, 0])])

            if init is not None:
                y0 = np.min([y0, np.min(init.vert2['coord'][:, 1])])
                y1 = np.max([y1, np.max(init.vert2['coord'][:, 1])])
                x0 = np.min([x0, np.min(init.vert2['coord'][:, 0])])
                x1 = np.max([x1, np.max(init.vert2['coord'][:, 0])])

            local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={(y0 + y1)/2} +lon_0={(x0 + x1)/2}"
            local_crs = CRS.from_user_input(local_azimuthal_projection)
            if geom is not None:
                utils.reproject(geom, local_crs)
            if hfun is not None:
                utils.reproject(hfun, local_crs)
            if init is not None:
                utils.reproject(init, local_crs)
        print('done with geting the msh_t')
        return geom, hfun, init, local_crs

    def _get_msh_t_build_cmd(self, binary='jigsaw', cwd=None):
        cwd = cwd or Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp'))
        self.__tmpworkdir = tempfile.TemporaryDirectory(dir=cwd)
        geom_msh_t, hfun_msh_t, init_msh_t, local_crs = self._get_projected_msh_t_set()
        output_msh_t_path = Path(self.__tmpworkdir.name) / 'mesh.msh'
        opts = copy(self.opts)
        opts.mesh_file = str(output_msh_t_path)
        if geom_msh_t is not None:
            geom_file = Path(self.__tmpworkdir.name) / 'geom.msh'
            jigsawpy.savemsh(str(geom_file), geom_msh_t)
            opts.geom_file = str(geom_file)
        if hfun_msh_t is not None:
            hfun_file = Path(self.__tmpworkdir.name) / 'hfun.msh'
            jigsawpy.savemsh(str(hfun_file), hfun_msh_t)
            opts.hfun_file = str(hfun_file)
        if init_msh_t is not None:
            init_file = Path(self.__tmpworkdir.name) / 'init.msh'
            jigsawpy.savemsh(str(init_file), init_msh_t)
            opts.init_file = str(init_file)
            jigsawpy.savemsh(str(Path(self.__tmpworkdir.name) / 'init.msh'), init_msh_t)
        jigsawpy.savejig(str(Path(self.__tmpworkdir.name) / 'opts.jig'), opts)
        msh_t_build_cmd = [str(binary), 'opts.jig']
        # TODO: Add memoization to the build_msh_t command
        return msh_t_build_cmd, output_msh_t_path, local_crs

    def msh_t(self, dst_crs=None):
        logger.info('Generating msh_t object...')
        geom, hfun, init, local_crs = self._get_projected_msh_t_set()
        output_mesh = jigsaw_msh_t()
        output_mesh.mshID = 'euclidean-mesh'
        output_mesh.ndims = 2
        logger.info('Launching libsaw.jigsaw...')
        libsaw.jigsaw(self.opts, geom, output_mesh, hfun=hfun, init=init)
        logger.info('Finalizing mesh...')
        if local_crs is not None:
            output_mesh.crs = local_crs
            utils.reproject(output_mesh, self.dst_crs)
        else:
            output_mesh.crs = self.dst_crs
        if self.finalize:
            utils.finalize_mesh(output_mesh, sieve=self.sieve)
        return output_mesh

    def run(self):
        return self.output_mesh

    def make_plot(self, *args, **kwargs):
        utils.triplot(self.output_mesh)

    @property
    def opts(self):
        if not hasattr(self, '_opts'):
            opts = jigsaw_jig_t()
            opts.mesh_dims = +2
            opts.hfun_scal = 'absolute'
            opts.optm_tria = True
            self._opts = opts
        return self._opts

    @property
    def output_mesh(self):
        return Mesh(self.msh_t(
            dst_crs=self.dst_crs
            ))

    @property
    def geom(self) -> BaseGeom:
        return self._geom

    @geom.setter
    def geom(self, geom: Union[BaseGeom, jigsaw_msh_t]):
        if isinstance(geom, jigsaw_msh_t):
            geom = Geom(geom)
        self._geom = geom

    @property
    def hfun(self) -> BaseHfun:
        return self._hfun

    @hfun.setter
    def hfun(self, hfun: Union[BaseHfun, jigsaw_msh_t]):
        if isinstance(hfun, jigsaw_msh_t):
            hfun = Hfun(hfun)
        print('setting hfun_min, hfun_max')
        self.opts.hfun_hmin = np.min(hfun.values)
        self.opts.hfun_hmax = np.max(hfun.values)
        self._hfun = hfun

    @property
    def initial_mesh(self):
        return self._initial_mesh

    @initial_mesh.setter
    def initial_mesh(self, initial_mesh):
        if initial_mesh is None:
            if hasattr(self.geom, '_quads_gdf'):
                if len(self.geom._quads_gdf) > 0:
                    initial_mesh = self.geom._get_initial_msh_t()
        #             self.opts.mesh_rad2 = 1000.
        self._initial_mesh = initial_mesh

    @property
    def verbosity(self):
        return self.opts.verbosity

    @verbosity.setter
    def verbosity(self, verbosity: int):
        self.opts.verbosity = int(verbosity)

    @property
    def dst_crs(self):
        return self._dst_crs

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        if dst_crs is None:
            crs_set = set()
            if self.geom is not None:
                crs_set.add(self.geom.crs)
                dst_crs = self.geom.crs
            if self.hfun is not None:
                crs_set.add(self.hfun.crs)
                dst_crs = self.hfun.crs
            if self.initial_mesh is not None:
                crs_set.add(self.initial_mesh.crs)
                dst_crs = self.initial_mesh.crs

            if len(crs_set) > 1:
                dst_crs = CRS.from_epsg(4326)
                # raise ValueError(
                #     'Argument dst_crs must not be None if the inputs (geom/hfun/initial_mesh) are in different CRS\'s.'
                # )
        if dst_crs is None:
            dst_crs = CRS.from_epsg(4326)
        self._dst_crs = dst_crs



def test_meshgen_for_Harlem_River():
    import pickle
    from pathlib import Path
    from geomesh import Geom, Hfun, Raster, JigsawDriver
    import matplotlib.pyplot as plt
    from appdirs import user_data_dir

    rootdir = user_data_dir('geomesh')
    raster = Raster(
            f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/'
            'raster2/elevation/NCEI_ninth_Topobathy_2014_8483/'
            'northeast_sandy/ncei19_n41x00_w074x00_2015v1.tif',
            )
    # quads = Quads.from_raster(
    #                         raster,
    #                         # min_quad_length=10.,
    #                         max_quad_length=500.,
    #                         resample_distance=100.,
    #                         max_quad_width=500.,
    #                         threshold_size=1500.,
    #                         # threshold_size=500.,
    #                         # min_quad_width=10.,
    #                         )
    # quads = Quads.from_raster(
    #                         raster,
    #                         threshold_size=1500.,
    #                         # min_quad_length=10.,
    #                         max_quad_length=500.,
    #                         resample_distance=100.,
    #                         zmin=0.,
    #                         zmax=10.,
    #                         max_quad_width=500.,
    #                         # min_quad_width=10.,
    #                         previous=quads,
    #                         )
    raster.resampling_factor = 0.2
    geom = Geom(
            raster,
            zmax=10.,
            )
    hfun = Hfun(
            raster,
            nprocs=cpu_count(),
            verbosity=1.,
            )
    hfun.add_contour(
            0.,
            target_size=100.,
            expansion_rate=0.007
            )
    hfun.add_contour(
            10.,
            target_size=100.,
            expansion_rate=0.007
            )
    hfun.add_constant_value(
            value=500.,
            # lower_bound=0.
            )
    driver = JigsawDriver(
            geom=geom,
            hfun=hfun,
            verbosity=1,
            # sieve_area=True,
            # finalize=False,
            )
    # driver.opts.geom_feat = True
    old_msh_t = driver.msh_t()
    # # pickle.dump(old_msh_t, open("the_old_msh_t.pkl", "wb"))
    # old_msh_t = pickle.load(open("the_old_msh_t.pkl", "rb"))
    # # raise NotImplementedError("ready")
    # # new_msh_t = old_msh_t
    # new_msh_t = quads(old_msh_t)

    # # utils.split_quad4_to_tria3(new_msh_t)

    # from geomesh.cli.mpi.hgrid_build import interpolate_raster_to_mesh
    # raster.resampling_factor = None
    # new_msh_t.value = np.full((new_msh_t.vert2['coord'].shape[0], 1), np.nan)
    # from time import time
    # logger.debug('interpolating raster to mesh...')
    # start = time()
    # idxs, values = interpolate_raster_to_mesh(
    #         new_msh_t,
    #         raster,
    #         )
    # new_msh_t.value[idxs] = values.reshape((values.size, 1)).astype(jigsaw_msh_t.REALS_t)
    # logger.debug(f'Done interpolating raster to mesh, took: {time()-start}...')
    # if np.all(np.isnan(new_msh_t.value)):
    #     raise ValueError('All values are NaN!')
    # from scipy.interpolate import griddata
    # if np.any(np.isnan(new_msh_t.value)):
    #     value = new_msh_t.value.flatten()
    #     non_nan_idxs = np.where(~np.isnan(value))[0]
    #     nan_idxs = np.where(np.isnan(value))[0]
    #     value[nan_idxs] = griddata(
    #             new_msh_t.vert2['coord'][non_nan_idxs, :],
    #             value[non_nan_idxs],
    #             new_msh_t.vert2['coord'][nan_idxs, :],
    #             method='nearest'
    #             )
    #     new_msh_t.value = value.reshape((value.size, 1)).astype(jigsaw_msh_t.REALS_t)
    # from geomesh import Mesh
    # the_mesh = Mesh(new_msh_t)
    # # print("writting output file")
    # # the_mesh.write('test_output_no_bnd.grd', overwrite=True)
    # # the_mesh.make_plot(ax=plt.gca(), elements=True)
    # # plt.show(block=True)
    # # pickle.dump(the_mesh, open("the_quad_mesh.pkl", "wb"))

    # # raise NotImplementedError("Ready for auto_bndgen")
    # # exit()
    # # import pickle
    # # the_mesh = pickle.load(open("the_quad_mesh.pkl", "rb"))
    # # the_mesh.make_plot(ax=plt.gca(), elements=True)
    # # plt.show(block=True)
    # # raise
    # # the_mesh.wireframe(ax=plt.gca())
    # # plt.show(block=False)
    # # breakpoint()
    # # raise
    # the_mesh.boundaries.auto_generate(
    #         # min_open_bound_length=10000.
    #         )
    # the_mesh.boundaries.open.plot(ax=plt.gca(), color='b')
    # # Add text labels to the centroid of each LineString
    # for idx, row in the_mesh.boundaries.open.iterrows():
    #     centroid = row['geometry'].centroid
    #     plt.gca().text(centroid.x, centroid.y, f"{idx+1=}", color='red')
    # the_mesh.boundaries.land.plot(ax=plt.gca(), color='g')
    # the_mesh.boundaries.interior.plot(ax=plt.gca(), color='r')
    # the_mesh.make_plot(ax=plt.gca(), elements=True)
    # # the_mesh.wireframe(ax=plt.gca())
    # logger.debug('begin making mesh triplot')
    # plt.title(f'node count: {len(new_msh_t.vert2["coord"])}')
    # plt.gca().axis('scaled')
    # plt.show(block=False)
    # the_mesh.write('test_with_schism/test_output.grd', overwrite=True)
    # print('done writting test file')
    # breakpoint()
