import os
from typing import Union

from jigsawpy import jigsaw_msh_t

from geomesh.hfun.base import BaseHfun
from geomesh.mesh.base import BaseMesh
from geomesh.mesh.mesh import Mesh


class MeshHfun(BaseHfun):

    def __init__(self, mesh: Union[BaseMesh, str, os.PathLike, jigsaw_msh_t]):
        self._mesh = mesh

    @property
    def mesh(self) -> BaseMesh:
        return self._mesh

    @property
    def _mesh(self):
        return self.__mesh

    @_mesh.setter
    def _mesh(self, mesh: Union[BaseMesh, str, os.PathLike, jigsaw_msh_t]):

        if isinstance(mesh, (str, os.PathLike)):
            mesh = Mesh.open(mesh)

        elif isinstance(mesh, jigsaw_msh_t):
            mesh = Mesh(mesh)

        if not isinstance(mesh, BaseMesh):
            raise TypeError(
                f"Argument mesh must be of type {Mesh}, {str} "
                f"or {os.PathLike}, not type {type(mesh)}"
            )

        self.__mesh = mesh
