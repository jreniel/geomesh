import os
from typing import Union

from jigsawpy import jigsaw_msh_t

from geomesh.figures import figure
from geomesh import utils
from geomesh.hfun.base import BaseHfun
from geomesh.mesh.base import BaseMesh
from geomesh.mesh.mesh import Mesh


class MeshHfun(BaseHfun):

    def __init__(self, mesh: Union[BaseMesh, str, os.PathLike, jigsaw_msh_t]):
        self._mesh = mesh

    def msh_t(self, dst_crs=None) -> jigsaw_msh_t:
        msh_t = self.mesh.msh_t
        if dst_crs is not None:
            utils.reproject(msh_t, dst_crs)
        return msh_t

    @property
    def mesh(self) -> BaseMesh:
        return self._mesh

    @property
    def values(self):
        return self.mesh.values

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
        
    @property
    def crs(self):
        return self.mesh.crs

    def tricontourf(
        self,
        axes=None,
        show=False,
        figsize=None,
        extend="both",
        elements=True,
        cmap="jet",
        color="k",
        linewidth=0.07,
        **kwargs,
    ):
        msh_t = self.msh_t()
        axes.tricontourf(
            msh_t.vert2["coord"][:, 0],
            msh_t.vert2["coord"][:, 1],
            msh_t.tria3["index"],
            msh_t.value.flatten(),
            **kwargs,
        )
        if elements is True:
            axes.triplot(
                msh_t.vert2["coord"][:, 0],
                msh_t.vert2["coord"][:, 1],
                msh_t.tria3["index"],
                color=color,
                linewidth=linewidth,
            )
        return axes

    @figure
    def triplot(
        self,
        axes=None,
        show=False,
        figsize=None,
        color="k",
        linewidth=0.07,
        **kwargs,
    ):
        msh_t = self.msh_t()
        axes.triplot(
            msh_t.vert2["coord"][:, 0],
            msh_t.vert2["coord"][:, 1],
            msh_t.tria3["index"],
            color=color,
            linewidth=linewidth,
            **kwargs,
        )
        return axes

