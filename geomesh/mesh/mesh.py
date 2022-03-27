from collections import defaultdict
from functools import lru_cache, cached_property
from itertools import permutations
from multiprocessing import Pool, cpu_count
import os
import pathlib
import tempfile
from typing import Union, List
import warnings

from attr import has

import geopandas as gpd
from jigsawpy import jigsaw_msh_t, savemsh, loadmsh, savevtk
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pyproj import CRS, Transformer
import requests
from scipy.interpolate import RectBivariateSpline, griddata
from shapely.geometry import Polygon, LineString, LinearRing, MultiPolygon, box, Point

from geomesh import utils
from geomesh.figures import figure, get_topobathy_kwargs
from geomesh.raster import Raster
from geomesh.mesh.base import BaseMesh
from geomesh.mesh.parsers import grd, sms2dm

from .boundaries import Boundaries


class Rings:
    def __init__(self, mesh: "EuclideanMesh"):
        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self):
        tri = self.mesh.elements.triangulation()
        idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
        boundary_edges = []
        for i, j in idxs:
            boundary_edges.append((tri.triangles[i, j], tri.triangles[i, (j + 1) % 3]))
        sorted_rings = sort_rings(edges_to_rings(boundary_edges), self.mesh.coord)
        data = []
        for bnd_id, rings in sorted_rings.items():
            coords = self.mesh.coord[rings["exterior"][:, 0], :]
            geometry = LinearRing(coords)
            data.append({"geometry": geometry, "bnd_id": bnd_id, "type": "exterior"})
            for interior in rings["interiors"]:
                coords = self.mesh.coord[interior[:, 0], :]
                geometry = LinearRing(coords)
                data.append(
                    {"geometry": geometry, "bnd_id": bnd_id, "type": "interior"}
                )
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self):
        return self().loc[self()["type"] == "exterior"]

    def interior(self):
        return self().loc[self()["type"] == "interior"]
    
    @lru_cache(maxsize=1)
    def sorted(self):
        tri = self.mesh.elements.triangulation()
        idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
        boundary_edges = []
        for i, j in idxs:
            boundary_edges.append((tri.triangles[i, j], tri.triangles[i, (j + 1) % 3]))
        return sort_rings(edges_to_rings(boundary_edges), self.mesh.coord)




class Edges:
    def __init__(self, mesh: "EuclideanMesh"):
        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self) -> gpd.GeoDataFrame:
        data = []
        for ring in self.mesh.hull.rings().itertuples():
            coords = ring.geometry.coords
            for i in range(1, len(coords)):
                data.append(
                    {
                        "geometry": LineString([coords[i - 1], coords[i]]),
                        "bnd_id": ring.bnd_id,
                        "type": ring.type,
                    }
                )
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self):
        return self().loc[self()["type"] == "exterior"]

    def interior(self):
        return self().loc[self()["type"] == "interior"]


class Hull:
    def __init__(self, mesh: "EuclideanMesh"):
        self.mesh = mesh
        self.rings = Rings(mesh)
        self.edges = Edges(mesh)

    @lru_cache(maxsize=1)
    def __call__(self):
        data = []
        for bnd_id in np.unique(self.rings()["bnd_id"].tolist()):
            exterior = self.rings().loc[
                (self.rings()["bnd_id"] == bnd_id)
                & (self.rings()["type"] == "exterior")
            ]
            interiors = self.rings().loc[
                (self.rings()["bnd_id"] == bnd_id)
                & (self.rings()["type"] == "interior")
            ]
            data.append(
                {
                    "geometry": Polygon(
                        exterior.iloc[0].geometry.coords,
                        [row.geometry.coords for _, row in interiors.iterrows()],
                    ),
                    "bnd_id": bnd_id,
                }
            )
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self):
        data = []
        for exterior in (
            self.rings().loc[self.rings()["type"] == "exterior"].itertuples()
        ):
            data.append({"geometry": Polygon(exterior.geometry.coords)})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def interior(self):
        data = []
        for interior in (
            self.rings().loc[self.rings()["type"] == "interior"].itertuples()
        ):
            data.append({"geometry": Polygon(interior.geometry.coords)})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def implode(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "geometry": MultiPolygon(
                    [polygon.geometry for polygon in self().itertuples()]
                )
            },
            crs=self.mesh.crs,
        )

    def multipolygon(self) -> MultiPolygon:
        mp = self.implode().iloc[0].geometry
        if isinstance(mp, Polygon):
            mp = MultiPolygon([mp])
        return mp

    def triangulation(self):
        triangles = self.mesh.msh_t.tria3["index"].tolist()
        for quad in self.mesh.msh_t.quad4["index"]:
            triangles.extend([[quad[0], quad[1], quad[3]], [quad[1], quad[2], quad[3]]])
        return Triangulation(self.mesh.coord[:, 0], self.mesh.coord[:, 1], triangles)


class Nodes:
    def __init__(self, mesh: "EuclideanMesh"):
        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self):
        # return {
        #     i + 1: (coord, self.mesh.value[i][0] if len(self.mesh.value[i]) <= 1 else self.mesh.value[i])
        #     for i, coord in enumerate(self.coords())
        # }
        return {i + 1: (coord, self.mesh.value[i]) for i, coord in enumerate(self.coords())}

    def id(self):
        return list(self().keys())

    def index(self):
        return np.arange(len(self()))

    def coords(self):
        return self.mesh.coord.tolist()

    def values(self):
        return self.mesh.value

    def get_index_by_id(self, id):
        return self.id_to_index[id]

    def get_id_by_index(self, index: int):
        return self.index_to_id[index]

    @property
    def id_to_index(self):
        if not hasattr(self, "_id_to_index"):
            self._id_to_index = {
                node_id: index for index, node_id in enumerate(self().keys())
            }
        return self._id_to_index

    @property
    def index_to_id(self):
        if not hasattr(self, "_index_to_id"):
            self._index_to_id = {
                index: node_id for index, node_id in enumerate(self().keys())
            }
        return self._index_to_id

    def get_indexes_around_index(self, index):
        if not hasattr(self, '_indexes_around_index'):
            def append(geom):
                for simplex in geom:
                    for i, j in permutations(simplex, 2):
                        indexes_around_index[i].add(j)
            indexes_around_index = defaultdict(set)
            append(self.mesh.elements.triangles())
            append(self.mesh.elements.quads())
            self._indexes_around_index = indexes_around_index
        return list(self._indexes_around_index[index])
    
    @property
    def gdf(self):
        if not hasattr(self, '_gdf'):
            data = []
            for i, (coord, value) in self().items():
                data.append({
                    'id': i,
                    'geometry': Point(coord),
                    'value': value,
                })
                
            self._gdf = gpd.GeoDataFrame(data, crs=self.mesh.crs)
        return self._gdf
        


class Elements:
    def __init__(self, mesh: "EuclideanMesh"):
        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self):
        elements = {
            i + 1: index + 1 for i, index in enumerate(self.mesh.msh_t.tria3["index"])
        }
        elements.update(
            {
                i + len(elements) + 1: index + 1
                for i, index in enumerate(self.mesh.msh_t.quad4["index"])
            }
        )
        return elements

    @lru_cache(maxsize=1)
    def id(self):
        return list(self().keys())

    @lru_cache(maxsize=1)
    def index(self):
        return np.arange(len(self()))

    def array(self):
        rank = int(max(map(len, self().values())))
        array = np.full((len(self()), rank), -1)
        for i, element in enumerate(self().values()):
            row = np.array(list(map(self.mesh.nodes.get_index_by_id, element)))
            array[i, : len(row)] = row
        return np.ma.masked_equal(array, -1)

    @lru_cache(maxsize=1)
    def triangles(self):
        return np.array(
            [
                list(map(self.mesh.nodes.get_index_by_id, element))
                for element in self().values()
                if len(element) == 3
            ]
        )

    @lru_cache(maxsize=1)
    def quads(self):
        return np.array(
            [
                list(map(self.mesh.nodes.get_index_by_id, element))
                for element in self().values()
                if len(element) == 4
            ]
        )

    def triangulation(self):
        triangles = self.triangles().tolist()
        for quad in self.quads():
            # TODO: Not tested.
            triangles.append([quad[0], quad[1], quad[3]])
            triangles.append([quad[1], quad[2], quad[3]])
        return Triangulation(self.mesh.coord[:, 0], self.mesh.coord[:, 1], triangles)

    def geodataframe(self):
        data = []
        for id, element in self().items():
            data.append(
                {
                    "geometry": Polygon(
                        self.mesh.coord[
                            list(map(self.mesh.nodes.get_index_by_id, element))
                        ]
                    ),
                    "id": id,
                }
            )
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)


class EuclideanMesh(BaseMesh):
    def __init__(self, mesh: jigsaw_msh_t):
        if not isinstance(mesh, jigsaw_msh_t):
            raise TypeError(
                f"Argument mesh must be of type {jigsaw_msh_t}, "
                f"not type {type(mesh)}."
            )
        if mesh.mshID != "euclidean-mesh":
            raise ValueError(
                f"Argument mesh has property mshID={mesh.mshID}, "
                "but expected 'euclidean-mesh'."
            )
        if not hasattr(mesh, "crs"):
            warnings.warn("Input mesh has no CRS information.")
            mesh.crs = None
        else:
            if not isinstance(mesh.crs, CRS):
                raise ValueError(
                    f"crs property must be of type {CRS}, not "
                    f"type {type(mesh.crs)}."
                )

        self._msh_t = mesh

    def write(
        self,
        path: Union[str, os.PathLike],
        overwrite: bool = False,
        format=None,
    ):
        path = pathlib.Path(path)
        fname = path.name
        if format is None:
            if fname.endswith('grd'):
                format = 'grd'
            if fname.endswith('gr3'):
                format = 'grd'
            if fname.endswith('2dm'):
                format = '2dm'
            if fname.endswith('msh'):
                format = 'msh'
            if fname.endswith('vtk'):
                format = 'vtk'
        else:
            format = 'grd'
        if path.exists() and overwrite is not True:
            raise IOError(f"File {str(path)} exists and overwrite is not True.")
        if format == "grd":
            grd_dict = utils.msh_t_to_grd(self.msh_t)
            # if hasattr(self, "_boundaries") and self._boundaries.data:
            grd_dict.update(boundaries=self.boundaries.data)
            grd.write(grd_dict, path, overwrite)

        elif format == "2dm":
            sms2dm.writer(utils.msh_t_to_2dm(self.msh_t), path, overwrite)

        elif format == "msh":
            savemsh(str(path), self.msh_t)

        elif format == "vtk":
            savevtk(str(path), self.msh_t)

        else:
            raise ValueError(f"Unhandled format {format}.")

    @property
    def tria3(self):
        return self.msh_t.tria3

    @property
    def triangles(self):
        return self.msh_t.tria3["index"]

    @property
    def quad4(self):
        return self.msh_t.quad4

    @property
    def quads(self):
        return self.msh_t.quad4["index"]

    @property
    def crs(self):
        return self.msh_t.crs

    @property
    def hull(self):
        if not hasattr(self, "_hull"):
            self._hull = Hull(self)
        return self._hull

    @property
    def nodes(self):
        if not hasattr(self, "_nodes"):
            self._nodes = Nodes(self)
        return self._nodes

    @property
    def elements(self):
        if not hasattr(self, "_elements"):
            self._elements = Elements(self)
        return self._elements


class EuclideanMesh2D(EuclideanMesh):
    def __init__(self, mesh: jigsaw_msh_t):
        super().__init__(mesh)
        if mesh.ndims != +2:
            raise ValueError(
                f"Argument mesh has property ndims={mesh.ndims}, "
                "but expected ndims=2."
            )

        if len(self.msh_t.value) == 0:
            self.msh_t.value = np.array(
                np.full((self.vert2["coord"].shape[0], 1), np.nan)
            )

    def get_bbox(
        self, crs: Union[str, CRS] = None, return_type='matplotlib',
    ) -> Bbox:
        if return_type not in ['shapely', 'matplotlib']:
            raise ValueError(f'Argument `return_type` must be "shapely" or "matplotlib", not {return_type}.')
        xmin, xmax = np.min(self.coord[:, 0]), np.max(self.coord[:, 0])
        ymin, ymax = np.min(self.coord[:, 1]), np.max(self.coord[:, 1])
        crs = self.crs if crs is None else crs
        if crs is not None:
            if not self.crs.equals(crs):
                transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
                (xmin, xmax), (ymin, ymax) = transformer.transform(
                    (xmin, xmax), (ymin, ymax)
                )
        if return_type == 'shapely':
            return box(xmin, ymin, xmax, ymax)
        elif return_type == 'matplotlib':
            return Bbox([[xmin, ymin], [xmax, ymax]])
        else:
            raise Exception(f'Unhandled return_type={return_type}.')


    @cached_property
    def boundaries(self):
        return Boundaries(self)

    @figure
    def make_plot(
        self,
        axes=None,
        vmin=None,
        vmax=None,
        title=None,
        extent=None,
        cbar_label=None,
        elements=False,
        **kwargs
    ):
        if vmin is None:
            vmin = np.min(self.values)
        if vmax is None:
            vmax = np.max(self.values)
        kwargs.update(**get_topobathy_kwargs(self.values, vmin, vmax))
        kwargs.pop('col_val')
        levels = kwargs.pop('levels')
        if vmin != vmax:
            self.tricontourf(
                axes=axes,
                levels=levels,
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )
        else:
            self.tripcolor(axes=axes, **kwargs)
        if elements is True:
            utils.triplot(self.msh_t, axes=axes)
        self.quadface(axes=axes, **kwargs)
        axes.axis('scaled')
        if extent is not None:
            axes.axis(extent)
        if title is not None:
            axes.set_title(title)
        mappable = ScalarMappable(cmap=kwargs['cmap'])
        mappable.set_array([])
        mappable.set_clim(vmin, vmax)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("bottom", size="2%", pad=0.5)
        cbar = plt.colorbar(
            mappable,
            cax=cax,
            orientation='horizontal'
        )
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([np.around(vmin, 2), np.around(vmax, 2)])
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        return axes

    def tricontourf(self, **kwargs):
        return utils.tricontourf(self.msh_t, **kwargs)

    def triplot(self, **kwargs):
        return utils.triplot(self.msh_t, **kwargs)

    def quadface(self, **kwargs):
        return utils.quadface(self.msh_t, **kwargs)      

    def interpolate(
        self, raster: Union[Raster, List[Raster]], method="nearest", nprocs=None
    ):

        if isinstance(raster, Raster):
            raster = [raster]

        if len(raster) > 1:
            nprocs = -1 if nprocs is None else nprocs
            nprocs = cpu_count() if nprocs == -1 else nprocs
            with Pool(processes=nprocs) as pool:
                res = pool.starmap(
                    _mesh_interpolate_worker,
                    [
                        (self.vert2["coord"], self.crs, _raster.tmpfile, _raster.chunk_size)
                        for _raster in raster
                    ],
                )
            pool.join()
            values = self.msh_t.value.flatten()
            for idxs, _values in res:
                values[idxs] = _values

        else:
            idxs, _values = _mesh_interpolate_worker(self.vert2["coord"], self.crs, raster[0].tmpfile, raster[0].chunk_size)
            values = self.msh_t.value.flatten()
            values[idxs] = _values
        
        nan_idxs = np.where(np.isnan(values))
        q_non_nan = np.where(~np.isnan(values))
        values[nan_idxs] = griddata(
            (self.vert2["coord"][q_non_nan, 0].flatten(), self.vert2["coord"][q_non_nan, 1].flatten()),
            values[q_non_nan],
            (self.vert2["coord"][nan_idxs, 0].flatten(), self.vert2["coord"][nan_idxs, 1].flatten()),
            method='nearest',
        )

        self.msh_t.value = np.array(
            values.reshape((values.shape[0], 1)), dtype=jigsaw_msh_t.REALS_t
        )

    @property
    def vert2(self):
        return self.msh_t.vert2

    @property
    def value(self):
        return self.msh_t.value

    @property
    def values(self):
        return self.msh_t.value

    @property
    def bbox(self):
        return self.get_bbox()

    @staticmethod
    def edges_to_rings(edges):
        return edges_to_rings(edges)

    @staticmethod
    def sort_rings(index_rings, vertices):
        return sort_rings(index_rings, vertices)

    @staticmethod
    def signed_polygon_area(vertices):
        return signed_polygon_area(vertices)


class Mesh(BaseMesh):
    """Mesh factory"""

    def __new__(self, mesh: jigsaw_msh_t):

        if not isinstance(mesh, jigsaw_msh_t):
            raise TypeError(
                f"Argument mesh must be of type {jigsaw_msh_t}, "
                f"not type {type(mesh)}."
            )

        if mesh.mshID == "euclidean-mesh":
            if mesh.ndims == 2:
                return EuclideanMesh2D(mesh)
            else:
                raise NotImplementedError(
                    f"mshID={mesh.mshID} + mesh.ndims={mesh.ndims} not " "handled."
                )

        else:
            raise NotImplementedError(f"mshID={mesh.mshID} not handled.")

    @staticmethod
    def open(path, crs=None):

        try:
            response = requests.get(path)
            response.raise_for_status()
            tmpfile = tempfile.NamedTemporaryFile()
            with open(tmpfile.name, "w") as fh:
                fh.write(response.text)
            return Mesh.open(tmpfile.name, crs=crs)

        except requests.exceptions.MissingSchema:
            pass

        try:
            msh_t = utils.grd_to_msh_t(grd.read(path, crs=crs))
            msh_t.value = np.negative(msh_t.value)
            return Mesh(msh_t)
        except Exception as e:
            if "not a valid grd file" in str(e):
                pass
            else:
                raise e
        try:
            msh_t = jigsaw_msh_t()
            loadmsh(path, msh_t)
            msh_t.crs = crs
            return Mesh(msh_t)
        except Exception:
            pass

        try:
            return Mesh(utils.sms2dm_to_msh_t(sms2dm.read(path, crs=crs)))
        except ValueError: # TODO: We also have KeyError: 'ND' File "geomesh/utils.py", line 586, in sms2dm_to_msh_t in enumerate(_sms2dm['ND'].keys())}
            pass


        raise TypeError(f"Unable to automatically determine file type for {str(path)}.")


def edges_to_rings(edges):
    if len(edges) == 0:
        return edges
    # start ordering the edges into linestrings
    edge_collection = list()
    ordered_edges = [edges.pop(-1)]
    if len(edges) == 0:
        return [tuple(ordered_edges)]
    e0, e1 = [list(t) for t in zip(*edges)]
    while len(edges) > 0:
        if ordered_edges[-1][1] in e0:
            idx = e0.index(ordered_edges[-1][1])
            ordered_edges.append(edges.pop(idx))
        elif ordered_edges[0][0] in e1:
            idx = e1.index(ordered_edges[0][0])
            ordered_edges.insert(0, edges.pop(idx))
        elif ordered_edges[-1][1] in e1:
            idx = e1.index(ordered_edges[-1][1])
            ordered_edges.append(list(reversed(edges.pop(idx))))
        elif ordered_edges[0][0] in e0:
            idx = e0.index(ordered_edges[0][0])
            ordered_edges.insert(0, list(reversed(edges.pop(idx))))
        else:
            edge_collection.append(tuple(ordered_edges))
            idx = -1
            ordered_edges = [edges.pop(idx)]
        e0.pop(idx)
        e1.pop(idx)
    # finalize
    if len(edge_collection) == 0 and len(edges) == 0:
        edge_collection.append(tuple(ordered_edges))
    else:
        edge_collection.append(tuple(ordered_edges))
    return edge_collection


def sort_rings(index_rings, vertices):
    """Sorts a list of index-rings.

    Takes a list of unsorted index rings and sorts them into an "exterior" and
    "interior" components. Any doubly-nested rings are considered exterior
    rings.

    TODO: Refactor and optimize. Calls that use :class:matplotlib.path.Path can
    probably be optimized using shapely.
    """

    # sort index_rings into corresponding "polygons"
    areas = list()
    for index_ring in index_rings:
        e0, e1 = [list(t) for t in zip(*index_ring)]
        areas.append(float(Polygon(vertices[e0, :]).area))

    # maximum area must be main mesh
    idx = areas.index(np.max(areas))
    exterior = index_rings.pop(idx)
    areas.pop(idx)
    _id = 0
    _index_rings = dict()
    _index_rings[_id] = {"exterior": np.asarray(exterior), "interiors": []}
    e0, e1 = [list(t) for t in zip(*exterior)]
    path = Path(vertices[e0 + [e0[0]], :], closed=True)
    while len(index_rings) > 0:
        # find all internal rings
        potential_interiors = list()
        for i, index_ring in enumerate(index_rings):
            e0, e1 = [list(t) for t in zip(*index_ring)]
            if path.contains_point(vertices[e0[0], :]):
                potential_interiors.append(i)
        # filter out nested rings
        real_interiors = list()
        for i, p_interior in reversed(list(enumerate(potential_interiors))):
            _p_interior = index_rings[p_interior]
            check = [
                index_rings[k]
                for j, k in reversed(list(enumerate(potential_interiors)))
                if i != j
            ]
            has_parent = False
            for _path in check:
                e0, e1 = [list(t) for t in zip(*_path)]
                _path = Path(vertices[e0 + [e0[0]], :], closed=True)
                if _path.contains_point(vertices[_p_interior[0][0], :]):
                    has_parent = True
            if not has_parent:
                real_interiors.append(p_interior)
        # pop real rings from collection
        for i in reversed(sorted(real_interiors)):
            _index_rings[_id]["interiors"].append(np.asarray(index_rings.pop(i)))
            areas.pop(i)
        # if no internal rings found, initialize next polygon
        if len(index_rings) > 0:
            idx = areas.index(np.max(areas))
            exterior = index_rings.pop(idx)
            areas.pop(idx)
            _id += 1
            _index_rings[_id] = {"exterior": np.asarray(exterior), "interiors": []}
            e0, e1 = [list(t) for t in zip(*exterior)]
            path = Path(vertices[e0 + [e0[0]], :], closed=True)
    return _index_rings


def signed_polygon_area(vertices):
    # https://code.activestate.com/recipes/578047-area-of-polygon-using-shoelace-formula/
    n = len(vertices)  # of vertices
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
        return area / 2.0


# def sort_rings(index_rings, vertices):
#     """Sorts a list of index-rings.

#     Takes a list of unsorted index rings and sorts them into an "exterior" and
#     "interior" components. Any doubly-nested rings are considered exterior
#     rings.

#     TODO: Refactor and optimize. Calls that use :class:matplotlib.path.Path can
#     probably be optimized using shapely.
#     """

#     # sort index_rings into corresponding "polygons"
#     areas = list()
#     for index_ring in index_rings:
#         e0, e1 = [list(t) for t in zip(*index_ring)]
#         areas.append(float(Polygon(vertices[e0, :]).area))

#     # maximum area must be main mesh
#     idx = areas.index(np.max(areas))
#     exterior = index_rings.pop(idx)
#     areas.pop(idx)
#     _id = 0
#     _index_rings = dict()
#     _index_rings[_id] = {"exterior": np.asarray(exterior), "interiors": []}
#     e0, e1 = [list(t) for t in zip(*exterior)]
#     path = Path(vertices[e0 + [e0[0]], :], closed=True)
#     while len(index_rings) > 0:
#         # find all internal rings
#         potential_interiors = list()
#         for i, index_ring in enumerate(index_rings):
#             e0, e1 = [list(t) for t in zip(*index_ring)]
#             if path.contains_point(vertices[e0[0], :]):
#                 potential_interiors.append(i)
#         # filter out nested rings
#         real_interiors = list()
#         for i, p_interior in reversed(list(enumerate(potential_interiors))):
#             _p_interior = index_rings[p_interior]
#             check = [
#                 index_rings[k]
#                 for j, k in reversed(list(enumerate(potential_interiors)))
#                 if i != j
#             ]
#             has_parent = False
#             for _path in check:
#                 e0, e1 = [list(t) for t in zip(*_path)]
#                 _path = Path(vertices[e0 + [e0[0]], :], closed=True)
#                 if _path.contains_point(vertices[_p_interior[0][0], :]):
#                     has_parent = True
#             if not has_parent:
#                 real_interiors.append(p_interior)
#         # pop real rings from collection
#         for i in reversed(sorted(real_interiors)):
#             _index_rings[_id]["interiors"].append(np.asarray(index_rings.pop(i)))
#             areas.pop(i)
#         # if no internal rings found, initialize next polygon
#         if len(index_rings) > 0:
#             idx = areas.index(np.max(areas))
#             exterior = index_rings.pop(idx)
#             areas.pop(idx)
#             _id += 1
#             _index_rings[_id] = {"exterior": np.asarray(exterior), "interiors": []}
#             e0, e1 = [list(t) for t in zip(*exterior)]
#             path = Path(vertices[e0 + [e0[0]], :], closed=True)
#     return _index_rings


# def signed_polygon_area(vertices):
#     # https://code.activestate.com/recipes/578047-area-of-polygon-using-shoelace-formula/
#     n = len(vertices)  # of vertices
#     area = 0.0
#     for i in range(n):
#         j = (i + 1) % n
#         area += vertices[i][0] * vertices[j][1]
#         area -= vertices[j][0] * vertices[i][1]
#         return area / 2.0


def _mesh_interpolate_worker(coords, coords_crs, raster_path, chunk_size):
    coords = np.array(coords)
    raster = Raster(raster_path)
    idxs = []
    values = []
    for window in raster.iter_windows(chunk_size=chunk_size, overlap=2):

        if not raster.crs.equals(coords_crs):
            transformer = Transformer.from_crs(coords_crs, raster.crs, always_xy=True)
            coords[:, 0], coords[:, 1] = transformer.transform(
                coords[:, 0], coords[:, 1]
            )
        xi = raster.get_x(window)
        yi = raster.get_y(window)
        zi = raster.get_values(window=window)
        f = RectBivariateSpline(
            xi,
            np.flip(yi),
            np.flipud(zi).T,
            kx=3,
            ky=3,
            s=0,
            # bbox=[min(x), max(x), min(y), max(y)]  # ??
        )
        _idxs = np.where(
            np.logical_and(
                np.logical_and(np.min(xi) <= coords[:, 0], np.max(xi) >= coords[:, 0]),
                np.logical_and(np.min(yi) <= coords[:, 1], np.max(yi) >= coords[:, 1]),
            )
        )[0]
        _values = f.ev(coords[_idxs, 0], coords[_idxs, 1])

        idxs.append(_idxs)
        values.append(_values)

    return (np.hstack(idxs), np.hstack(values))
