import logging
from typing import Union

from jigsawpy import libsaw, jigsaw_msh_t, jigsaw_jig_t
import numpy as np
from pyproj import CRS

from geomesh import utils
from geomesh.geom.base import BaseGeom
from geomesh.geom.geom import Geom
from geomesh.hfun.base import BaseHfun
from geomesh.hfun.hfun import Hfun
from geomesh.mesh.base import BaseMesh
from geomesh.mesh.mesh import Mesh


logger = logging.getLogger(__name__)


class JigsawDriver:

    def __init__(
            self,
            geom: Union[BaseGeom, jigsaw_msh_t] = None,
            hfun: Union[BaseHfun, jigsaw_msh_t] = None,
            initial_mesh: Union[BaseMesh, jigsaw_msh_t] = None,
            verbosity: int = 0,
            dst_crs=None,
    ):
        has_geom = True if geom is not None else False
        has_hfun = True if hfun is not None else False
        has_initial_mesh = True if initial_mesh is not None else False

        if not (has_geom | has_hfun | has_initial_mesh):

            class JigsawDriverArgumentError(Exception):
                pass

            raise JigsawDriverArgumentError(
                "Arguments to JigsawDriver must contain at least one of `geom`, `hfun` of `initial_mesh`"
            )
        self.geom = geom
        self.hfun = hfun
        self.initial_mesh = initial_mesh
        self.verbosity = verbosity
        self.dst_crs = dst_crs

    def msh_t(self, dst_crs=None):
        logger.info('Generating msh_t object...')
        output_mesh = jigsaw_msh_t()
        output_mesh.mshID = 'euclidean-mesh'
        output_mesh.ndims = 2
        local_azimuthal_projection = None
        if dst_crs is not None:
            self.dst_crs = dst_crs
        if self.geom is not None:
            geom = self.geom.msh_t(dst_crs=self.dst_crs)

        if self.hfun is not None:
            hfun = self.hfun.msh_t(dst_crs=self.dst_crs)

        if self.initial_mesh is not None:
            initial_mesh = self.initial_mesh.msh_t(dst_crs=self.dst_crs)

        if self.dst_crs.is_geographic:
            y0, x0 = 2*[float('inf')]
            y1, x1 = 2*[-float('inf')]
            if self.geom is not None:
                y0 = np.min([y0, np.min(geom.vert2['coord'][:, 1])])
                y1 = np.max([y1, np.max(geom.vert2['coord'][:, 1])])
                x0 = np.min([x0, np.min(geom.vert2['coord'][:, 0])])
                x1 = np.max([x1, np.max(geom.vert2['coord'][:, 0])])

            if self.hfun is not None:
                y0 = np.min([y0, np.min(hfun.vert2['coord'][:, 1])])
                y1 = np.max([y1, np.max(hfun.vert2['coord'][:, 1])])
                x0 = np.min([x0, np.min(hfun.vert2['coord'][:, 0])])
                x1 = np.max([x1, np.max(hfun.vert2['coord'][:, 0])])

            if self.initial_mesh is not None:
                y0 = np.min([y0, np.min(initial_mesh.vert2['coord'][:, 1])])
                y1 = np.max([y1, np.max(initial_mesh.vert2['coord'][:, 1])])
                x0 = np.min([x0, np.min(initial_mesh.vert2['coord'][:, 0])])
                x1 = np.max([x1, np.max(initial_mesh.vert2['coord'][:, 0])])

            local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={(y0 + y1)/2} +lon_0={(x0 + x1)/2}"
            local_crs = CRS.from_user_input(local_azimuthal_projection)
            if self.geom is not None:
                utils.reproject(geom, local_crs)
            if self.hfun is not None:
                utils.reproject(hfun, local_crs)
            if self.initial_mesh is not None:
                utils.reproject(initial_mesh, local_crs)

        logger.info('Launching libsaw.jigsaw...')
        libsaw.jigsaw(self.opts, geom, output_mesh, hfun=hfun)
        print('WARNING: Mesh cleanup is missing.')
        if local_azimuthal_projection is not None:
            output_mesh.crs = local_crs
            utils.reproject(output_mesh, self.dst_crs)
        else:
            output_mesh.crs = self.dst_crs
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
        self.opts.hfun_hmin = np.min(hfun.values)
        self.opts.hfun_hmax = np.max(hfun.values)
        self._hfun = hfun

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
















#         """
#         geom can be SizeFunction or PlanarStraightLineGraph instance.
#         """
#         self._geom = geom
#         self._hfun = hfun
#         self._initial_mesh = initial_mesh
#         self._interface = 'libsaw'

#     def run(self, sieve=None):
#         self.logger.debug("run()")
#         if self._interface == 'cmdsaw':
#             self._run_cmdsaw()
#         else:
#             self.jigsaw(
#                 self.opts,
#                 self.geom,
#                 self.output_mesh,
#                 self.initial_mesh,
#                 self.hfun
#             )

#         # post process
#         msg = 'ERROR: Jigsaw returned empty mesh.'
#         assert self.output_mesh.tria3['index'].shape[0] > 0, msg
#         if self.verbosity > 0:
#             print('Finalizing mesh...', end='', flush=True)
#         utils.finalize_mesh(self.output_mesh, sieve)
#         if self.verbosity > 0:
#             print('done!')
#         return mesh.Mesh.from_msh_t(self.output_mesh, crs=self.dst_crs)

#     @property
#     def geom(self):
#         return self._geom

#     @property
#     def hfun(self):
#         return self._hfun

#     @property
#     def initial_mesh(self):
#         return self._initial_mesh

#     @property
#     def output_mesh(self):
#         return self._output_mesh

#     @property
#     def jigsaw(self):
#         return jigsawpy.lib.jigsaw

#     @property
#     def opts(self):
#         return self._opts

#     @property
#     def verbosity(self):
#         return self.opts.verbosity

#     @property
#     def hfun_hmin(self):
#         return self.opts.hfun_hmin

#     @property
#     def hfun_hmax(self):
#         return self.opts.hfun_hmax

#     @property
#     def hfun_scal(self):
#         return self.opts.hfun_scal

#     @property
#     def optm_qlim(self):
#         return self.opts.optm_qlim

#     @property
#     def mesh_top1(self):
#         return self.opts.mesh_top1

#     @property
#     def geom_feat(self):
#         return self.opts.geom_feat

#     @property
#     def dst_crs(self):
#         return self._dst_crs

#     @property
#     def logger(self):
#         try:
#             return self.__logger
#         except AttributeError:
#             self.__logger = logging.getLogger(
#                 __name__ + '.' + self.__class__.__name__)
#             return self.__logger

#     @property
#     def hmin_is_absolute_limit(self):
#         try:
#             return self.__hmin_is_absolute_limit
#         except AttributeError:
#             # Uses the data's hmin limit by default
#             return False

#     @property
#     def hmax_is_absolute_limit(self):
#         try:
#             return self.__hmax_is_absolute_limit
#         except AttributeError:
#             # Uses the data's hmax limit by default
#             return False

#     @verbosity.setter
#     def verbosity(self, verbosity):
#         self._verbosity = verbosity

#     @hfun_hmin.setter
#     def hfun_hmin(self, hfun_hmin):
#         self.opts.hfun_hmin = float(hfun_hmin)

#     @hfun_hmax.setter
#     def hfun_hmax(self, hfun_hmax):
#         self.opts.hfun_hmax = float(hfun_hmax)

#     @hmin_is_absolute_limit.setter
#     def hmin_is_absolute_limit(self, hmin_is_absolute_limit):
#         assert isinstance(hmin_is_absolute_limit, bool)
#         self.__hmin_is_absolute_limit = hmin_is_absolute_limit

#     @hmax_is_absolute_limit.setter
#     def hmax_is_absolute_limit(self, hmax_is_absolute_limit):
#         assert isinstance(hmax_is_absolute_limit, bool)
#         self.__hmax_is_absolute_limit = hmax_is_absolute_limit

#     @hfun_scal.setter
#     def hfun_scal(self, hfun_scal):
#         assert hfun_scal in ["absolute", "relative"]
#         self.opts.hfun_scal = hfun_scal

#     @optm_qlim.setter
#     def optm_qlim(self, optm_qlim):
#         optm_qlim = float(optm_qlim)
#         assert optm_qlim > 0 and optm_qlim < 1
#         self.opts.optm_qlim = optm_qlim

#     @mesh_top1.setter
#     def mesh_top1(self, mesh_top1):
#         assert isinstance(mesh_top1, bool)
#         self.opts.mesh_top1 = mesh_top1

#     @geom_feat.setter
#     def geom_feat(self, geom_feat):
#         assert isinstance(geom_feat, bool)
#         self.opts.geom_feat = geom_feat

#     def _run_cmdsaw(self):
#         msg = f'_run_cmdsaw()'
#         self.logger.debug(msg)

#         # init tmpfiles
#         self.logger.debug(f'init tmpfiles')
#         mesh_file = tempfile.NamedTemporaryFile(
#             prefix=tmpdir, suffix='.msh')
#         hmat_file = tempfile.NamedTemporaryFile(
#             prefix=tmpdir, suffix='.msh')
#         geom_file = tempfile.NamedTemporaryFile(
#             prefix=tmpdir, suffix='.msh')
#         jcfg_file = tempfile.NamedTemporaryFile(
#             prefix=tmpdir, suffix='.jig')

#         # dump data to tempfiles
#         jigsawpy.savemsh(hmat_file.name, self.hfun)
#         jigsawpy.savemsh(geom_file.name, self.geom)

#         # init opts
#         opts = jigsaw_jig_t()
#         opts.mesh_file = mesh_file.name
#         opts.hfun_file = hmat_file.name
#         opts.geom_file = geom_file.name
#         opts.jcfg_file = jcfg_file.name

#         # additional configuration options
#         opts.verbosity = self.verbosity
#         opts.mesh_dims = +2  # NOTE: Hardcoded value
#         opts.hfun_scal = 'absolute'
#         opts.optm_tria = True  # NOTE: Hardcoded value
#         opts.optm_qlim = self.optm_qlim

#         if self.hmin_is_absolute_limit:
#             opts.hfun_hmin = self.hmin
#         else:
#             opts.hfun_hmin = np.min(self.hfun.value)

#         if self.hmax_is_absolute_limit:
#             opts.hfun_hmax = self.hmax
#         else:
#             opts.hfun_hmax = np.max(self.hfun.value)

#         # init outputmesh
#         mesh = jigsaw_msh_t()

#         # call jigsaw
#         self.logger.debug('call cmdsaw')
#         jigsawpy.cmd.jigsaw(opts, mesh)

#         # cleanup temporary files
#         for tmpfile in (mesh_file, hmat_file, geom_file, jcfg_file):
#             del(tmpfile)

#         self.__output_mesh = mesh

#     @property
#     def _geom(self):
#         return self.__geom

#     @property
#     def _hfun(self):
#         return self.__hfun

#     @property
#     def _initial_mesh(self):
#         return self.__initial_mesh

#     @property
#     def _dst_crs(self):
#         return self.__dst_crs

#     @property
#     def _mesh_dims(self):
#         return self.__mesh_dims

#     @property
#     @lru_cache(maxsize=None)
#     def _opts(self):
#         return jigsaw_jig_t()

#     @property
#     def _verbosity(self):
#         return self.__verbosity

#     @property
#     def _output_mesh(self):
#         try:
#             return self.__output_mesh
#         except AttributeError:
#             self.__output_mesh = jigsaw_msh_t()
#             return self.__output_mesh

#     @_geom.setter
#     def _geom(self, geom):
#         if isinstance(geom, SizeFunction):
#             self._hfun = geom
#         else:
#             assert isinstance(geom, Geom)
#         self._dst_crs = geom._dst_crs
#         self._mesh_dims = geom.ndims
#         self.__geom = geom.geom

#     @_hfun.setter
#     def _hfun(self, hfun):
#         if hfun is not None:
#             assert isinstance(hfun, SizeFunction)
#             # set scaling
#             self.hfun_scal = hfun.scaling

#             # use hmin limits
#             if hfun.hmin_is_absolute_limit:
#                 self.hfun_hmin = hfun.hmin
#             else:
#                 self.hfun_hmin = np.min(hfun.values)

#             # set hmax limits
#             if hfun.hmax_is_absolute_limit:
#                 self.hfun_hmax = hfun.hmax
#             else:
#                 self.hfun_hmax = np.max(hfun.values)

#             # push jigsaw_msh_t object
#             hfun = hfun.hfun

#             self.__hfun = hfun
#         else:
#             if not hasattr(self, f"_{__class__.__name__}__hfun"):
#                 self.__hfun = hfun

#     @_initial_mesh.setter
#     def _initial_mesh(self, initial_mesh):
#         if initial_mesh is not None:
#             msg = f"initial_mesh must be of type {jigsaw_msh_t}, "
#             msg += f"got type: {type(initial_mesh)}"
#             assert isinstance(initial_mesh, jigsaw_msh_t), msg
#         self.__initial_mesh = initial_mesh

#     @_verbosity.setter
#     def _verbosity(self, verbosity):
#         assert verbosity in [0, 1, 2, 3]
#         self.opts.verbosity = verbosity

#     @_dst_crs.setter
#     def _dst_crs(self, dst_crs):
#         self.__dst_crs = dst_crs

#     @_mesh_dims.setter
#     def _mesh_dims(self, mesh_dims):
#         self.opts.mesh_dims = mesh_dims

#     @_opts.setter
#     def _opts(self, opts):
#         assert isinstance(opts, jigsaw_jig_t)
#         self.__opts = opts
