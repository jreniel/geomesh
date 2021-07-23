import os
from typing import Union

# from jigsawpy import jigsaw_msh_t  # type: ignore[import]
# import matplotlib.pyplot as plt  # type: ignore[import]
# import mpl_toolkits.mplot3d as m3d  # type: ignore[import]
# import numpy as np  # type: ignore[import]
# from shapely import ops  # type: ignore[import]

from .base import BaseGeom
from geomesh.mesh.base import BaseMesh
from geomesh.mesh.mesh import Mesh


class MeshGeom(BaseGeom):
    def __init__(self, mesh: Union[BaseMesh, str, os.PathLike]):
        """
        Input parameters
        ----------------
        mesh:
            Input object used to compute the output mesh hull.
        """
        self.mesh = mesh  # type: ignore[assignment]

    def get_multipolygon(self):
        return self.mesh.hull.multipolygon()

    @property
    def mesh(self) -> BaseMesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Union[BaseMesh, str, os.PathLike]):

        if isinstance(mesh, (str, os.PathLike)):
            mesh = Mesh.open(mesh)

        if not isinstance(mesh, BaseMesh):
            raise TypeError(
                f"Argument mesh must be of type {Mesh}, {str} "
                f"or {os.PathLike}, not type {type(mesh)}"
            )

        self._mesh = mesh

    @property
    def crs(self):
        return self._mesh.crs
