from collections import defaultdict
from functools import lru_cache, cached_property
from itertools import permutations
from multiprocessing import Pool, cpu_count
from typing import Union, List
import logging
import os
import pathlib
import tempfile
import warnings


from jigsawpy import jigsaw_msh_t, savemsh, loadmsh, savevtk
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS, Transformer
from scipy.interpolate import RectBivariateSpline, griddata
from shapely.geometry import Polygon, MultiPolygon, box, Point, MultiLineString, LineString
from shapely.ops import linemerge, polygonize
from scipy.spatial import KDTree
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests

from .boundaries import Boundaries
from geomesh import utils
from geomesh.figures import get_topobathy_kwargs
from geomesh.mesh.base import BaseMesh
from geomesh.mesh.parsers import grd, sms2dm
from geomesh.raster import Raster

logger = logging.getLogger(__name__)


class Rings:
    def __init__(self, mesh: "EuclideanMesh"):
        self.mesh = mesh

    @lru_cache
    def __call__(self):
        # Step 1: Construct a list of all edges
        tria3 = self.mesh.msh_t.tria3['index']
        quad4 = self.mesh.msh_t.quad4['index']
        tria3_edges = np.hstack((tria3[:, [0, 1]], tria3[:, [1, 2]], tria3[:, [2, 0]])).reshape(-1, 2)
        quad4_edges = np.hstack((quad4[:, [0, 1]], quad4[:, [1, 2]], quad4[:, [2, 3]], quad4[:, [3, 0]])).reshape(-1, 2)
        all_edges = np.vstack([tria3_edges, quad4_edges])
        sorted_edges = np.sort(all_edges, axis=1)
        unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        boundary_rings = linemerge(MultiLineString([LineString(x) for x in self.mesh.msh_t.vert2['coord'][boundary_edges]]))
        if isinstance(boundary_rings, LineString):
            boundary_rings = MultiLineString(boundary_rings)
        gpd.GeoDataFrame(geometry=[ls for ls in boundary_rings.geoms], crs=self.mesh.crs).plot(
                facecolor='none',
                cmap='tab20',
                ax=plt.gca(),
                )
        plt.show(block=True)
        breakpoint()
        polygons = list(polygonize(boundary_rings))

        gpd.GeoDataFrame(geometry=polygons, crs=self.mesh.crs).plot(facecolor='none', ax=plt.gca())
        plt.show(block=True)
        breakpoint()
        raise

        outer_polygons = []
        outer_polygon, remaining = utils.filter_polygons(polygons)
        outer_polygons.append(outer_polygon)
        while len(remaining) > 0:
            outer_polygon, remaining = utils.filter_polygons(remaining)
            outer_polygons.append(outer_polygon)
        mp = MultiPolygon(outer_polygons)
        data = []
        for bnd_id, polygon in enumerate(mp.geoms):
            data.append({"geometry": polygon.exterior, "bnd_id": bnd_id, "type": "exterior"})
            for interior in polygon.interiors:
                data.append(
                    {"geometry": interior, "bnd_id": bnd_id, "type": "interior"}
                )
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self):
        return self().loc[self()["type"] == "exterior"]

    def interior(self):
        return self().loc[self()["type"] == "interior"]

    @lru_cache(maxsize=1)
    def sorted(self):
        mp = utils.geom_to_multipolygon(self.mesh.msh_t)
        sorted_rings = {}
        tree = KDTree(self.mesh.coord)
        # on_edge = np.where(on_edge)
        for id, polygon in enumerate(mp.geoms):
            _, ii = tree.query(polygon.exterior.coords)
            # e1 = [ii[(i+1) % ii.size] for i in range(ii.size)]
            # exterior = np.vstack([np.array(ii), np.array(e1)]).T
            exterior = np.vstack([ii, np.roll(ii, -1)]).T
            interiors = []
            for interior in polygon.interiors:
                _, ii = tree.query(interior.coords)
                # e1 = [ii[(i+1) % ii.size] for i in range(ii.size)]
                # interior = np.vstack([np.array(ii), np.array(e1)]).T
                interior = np.vstack([ii, np.roll(ii, -1)]).T
                interiors.append(interior)
            sorted_rings[id] = {'exterior': exterior, 'interiors': interiors}
        # print('will verify')
        # for id, ring_data in sorted_rings.items():
        #     gpd.GeoDataFrame([{'geometry': LineString(
        #         [
        #             Point(*self.mesh.coord[e0, :]),
        #             Point(*self.mesh.coord[e1, :])
        #         ])}
        #         for e0, e1 in ring_data['exterior']]).plot(ax=plt.gca(), color='k')
        #     for interior in ring_data['interiors']:
        #         gpd.GeoDataFrame([{'geometry': LineString(
        #             [
        #                 Point(*self.mesh.coord[e0, :]),
        #                 Point(*self.mesh.coord[e1, :])
        #             ])}
        #             for e0, e1 in interior]).plot(ax=plt.gca(), color='r')
        # print('will show...')
        # plt.show(block=True)

        return sorted_rings


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
        mp = self.implode().unary_union
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
        return {i + 1: (coord, self.mesh.value[i]) for i, coord in enumerate(self.mesh.coord)}

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

    def make_plot(
        self,
        ax=None,
        vmin=None,
        vmax=None,
        title=None,
        extent=None,
        cbar_label=None,
        elements=False,
        **kwargs
    ):
        ax = ax or plt.gca()
        if vmin is None:
            vmin = np.min(self.values)
        if vmax is None:
            vmax = np.max(self.values)
        kwargs.update(**get_topobathy_kwargs(self.values, vmin, vmax))
        kwargs.pop('col_val')
        levels = kwargs.pop('levels')
        # self.quadface(ax=ax, **kwargs)
        if vmin != vmax:
            self.tricontourf(
                ax=ax,
                levels=levels,
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )
        else:
            self.tripcolor(ax=ax, **kwargs)
        if elements is True:
            utils.triplot(self.msh_t, ax=ax, linewidth=0.3)
            utils.quadplot(self.msh_t, ax=ax, linewidth=0.3)
        if extent is not None:
            ax.axis(extent)
        if title is not None:
            ax.set_title(title)
        mappable = ScalarMappable(cmap=kwargs['cmap'])
        mappable.set_array([])
        mappable.set_clim(vmin, vmax)
        divider = make_axes_locatable(ax)
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
        ax.axis('scaled')
        return ax

    def tricontourf(self, **kwargs):
        return utils.tricontourf(self.msh_t, **kwargs)

    def triplot(self, **kwargs):
        return utils.triplot(self.msh_t, **kwargs)

    def quadplot(self, **kwargs):
        return utils.quadplot(self.msh_t, **kwargs)

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



# from numba import jit
# @jit(nopython)
# def delete_workaround(arr, num):
#     mask = np.zeros(arr.shape[0], dtype=np.int64) == 0
#     mask[np.where(arr == num)[0]] = False
#     return arr[mask]


# def edges_to_rings(edges):
#     """
#     https://stackoverflow.com/questions/64960368/how-to-order-tuples-by-matching-the-first-and-last-values-of-each-a-b-b-c
#     """



    # while len(edges) > 0:
    #     if ordered_edges[-1][1] in e0:
    #         idx = e0.index(ordered_edges[-1][1])
    #         ordered_edges.append(edges.pop(idx))
    #     elif ordered_edges[0][0] in e1:
    #         idx = e1.index(ordered_edges[0][0])
    #         ordered_edges.insert(0, edges.pop(idx))
    #     elif ordered_edges[-1][1] in e1:
    #         idx = e1.index(ordered_edges[-1][1])
    #         ordered_edges.append(list(reversed(edges.pop(idx))))
    #     elif ordered_edges[0][0] in e0:
    #         idx = e0.index(ordered_edges[0][0])
    #         ordered_edges.insert(0, list(reversed(edges.pop(idx))))
    #     else:
    #         edge_collection.append(tuple(ordered_edges))
    #         idx = -1
    #         ordered_edges = [edges.pop(idx)]
    #     e0.pop(idx)
    #     e1.pop(idx)
    # # finalize
    # if len(edge_collection) == 0 and len(edges) == 0:
    #     edge_collection.append(tuple(ordered_edges))
    # else:
    #     edge_collection.append(tuple(ordered_edges))
    # return edge_collection



    # import networkx as nx

    # G = nx.DiGraph(edges)
    # print('start nx call')
    # print(list(nx.topological_sort(nx.line_graph(G))))

    # graph = Graph()
    # for e0, e1 in edges:
    #     graph.add_edge(e0, e1)
    # vertices_topo_sorted = graph.topological_sort()
    # edge_tuples = [(u, v) for u, v in zip(vertices_topo_sorted[0:], vertices_topo_sorted[1:])]
    # print(np.array(edge_tuples))
    # Create an adjacency matrix to find the next value fast 
    # adjacency_matrix = {pair[0]: pair for pair in edges}
    # The first element can be found being the first element of the pair not
    # present in the second elements
    # first_key = set(pair[0] for pair in edges).difference(pair[1] for pair in edges)


    # while adjacency_matrix:
    #     # sorted_pairs[-1][1] takes the second element of 
    #     # the last pair inserted
    #     try:
    #         sorted_pairs.append(adjacency_matrix.pop(sorted_pairs[-1][1]))
    #     except KeyError:
    #         sorted_pairs.append(adjacency_matrix.pop(list(adjacency_matrix.keys())[0]))
    #     # print(len(adjacency_matrix))
    # print(sorted_pairs)
    # exit()




    # # e0_to_e1 = {e0: (i, e1) for i, (e0, e1) in enumerate(edges)}
    # # e1_to_e0 = {e1: (i, e0) for i, (e0, e1) in enumerate(edges)}
    # # e0 = list(e0_to_e1.keys())
    # # e1 = list(e1_to_e0.keys())
    # # def index_getter(edge):
    #     # row_e0 = edge[0]
    #     row_e1 = edge[1]
    #     index_pos, _ = e1_to_e0[row_e1]
    #     return index_pos

        # offset = current_pos + 1
        # if row_e1 in e0[offset:]:
        #     return offset + e0[offset:].index(row_e1)
        # elif 
        # else:
        #     raise ValueError(f'{row_e1} not in reamining e0')
        # elif edge[
    #         return e0.index(ordered_edges[-1][1])
    #         ordered_edges.append(edges.pop(idx))
    #     elif ordered_edges[0][0] in e1:
    #         idx = e1.index(ordered_edges[0][0])
    #         ordered_edges.insert(0, edges.pop(idx))
    #     elif ordered_edges[-1][1] in e1:
    #         idx = e1.index(ordered_edges[-1][1])
    #         ordered_edges.append(list(reversed(edges.pop(idx))))
    #     elif ordered_edges[0][0] in e0:
    #         idx = e0.index(ordered_edges[0][0])
    #         ordered_edges.insert(0, list(reversed(edges.pop(idx))))
        


    # edges.sort(key=index_getter)
    # print(edges)
    # e0, e1 = [list(t) for t in zip(*edges)]
    # def sorting_function(row):
    #     if ordered_edges[-1][1] in e0:
    #         return e0.index(ordered_edges[-1][1])
    #         ordered_edges.append(edges.pop(idx))
    #     elif ordered_edges[0][0] in e1:
    #         idx = e1.index(ordered_edges[0][0])
    #         ordered_edges.insert(0, edges.pop(idx))
    #     elif ordered_edges[-1][1] in e1:
    #         idx = e1.index(ordered_edges[-1][1])
    #         ordered_edges.append(list(reversed(edges.pop(idx))))
    #     elif ordered_edges[0][0] in e0:
    #         idx = e0.index(ordered_edges[0][0])
    #         ordered_edges.insert(0, list(reversed(edges.pop(idx))))
    #     return position

    # edges = sorted(edges, key=sorting_function)

    # print(edges)
    # if len(edges) == 0:
    #     return edges
    # # start ordering the edges into linestrings
    # edge_collection = list()
    # ordered_edges = [edges.pop(-1)]
    # if len(edges) == 0:
    #     return [tuple(ordered_edges)]
    # e0, e1 = [list(t) for t in zip(*edges)]





# @jit(nopython=True)
# def edges_to_rings(edges):
#     if len(edges) == 0:
#         return edges
#     edges = np.array(edges)
#     edges = edges.reshape(int(edges.size / 2), 2)
#     # start ordering the edges into linestrings
#     edge_collection = list()
#     ordered_edges = [edges[-1, :]]
#     edges = np.delete(edges, -1, 0)
#     if len(edges) == 0:
#         return [ordered_edges]
#     while len(edges) > 0:
#         # print(ordered_edges[-1][1])
#         # exit()
#         print(np.any(np.in1d(edges[:, 0], ordered_edges[-1][1])))
#         if np.any(np.in1d(edges[:, 0], ordered_edges[-1][1])):
#             idx = np.where(edges[:, 0] == ordered_edges[-1][1])[0]
#             ordered_edges.append(edges[idx, :])
#             edges = np.delete(edges, idx, 0)
#         elif np.any(np.in1d(edges[:, 1], [ordered_edges[0][0]])):
#             idx = np.where(edges[:, 1] == ordered_edges[0][0])[0]
#             ordered_edges.insert(0, edges[idx, :])
#             edges = np.delete(edges, idx, 0)
#         elif np.any(np.in1d(edges[:, 1], ordered_edges[-1][1])):
#             idx = np.where(edges[:, 1] == ordered_edges[-1][1])[0]
#             ordered_edges.append(edges[idx, :].reversed())
#             edges = np.delete(edges, idx, 0)
#         elif np.any(np.in1d(edges[:, 0], ordered_edges[0][0])):
#             idx = np.where(edges[:, 0] == ordered_edges[0][0])[0]
#             # idx = e0.index(ordered_edges[0][0])
#             ordered_edges.insert(0, edges[idx, :].reversed())
#             edges = np.delete(edges, idx, 0)
#         else:
#             edge_collection.append(ordered_edges)
#             idx = -1
#             ordered_edges = [edges[idx, :]]
#             print(ordered_edges)
#             edges = np.delete(edges, idx, 0)
#         # e0.pop(idx)
#         # e1.pop(idx)
#     # finalize
#     if len(edge_collection) == 0 and len(edges) == 0:
#         edge_collection.append(ordered_edges)
#     else:
#         edge_collection.append(ordered_edges)
#     return edge_collection

# @jit(nopython=True)
# def edges_to_rings(edges):
#     if len(edges) == 0:
#         return edges
#     edges = np.array(edges)
#     edges = edges.reshape(int(edges.size / 2), 2)
#     # start ordering the edges into linestrings
#     edge_collection = list()
#     ordered_edges = [edges[-1, :]]
#     edges = np.delete(edges, -1, 0)
#     if len(edges) == 0:
#         return [ordered_edges]
#     while len(edges) > 0:
#         # print(ordered_edges[-1][1])
#         # exit()
#         if np.any(np.in1d(edges[:, 0], ordered_edges[-1][1])):
#             idx = np.where(edges[:, 0] == ordered_edges[-1][1])[0]
#             ordered_edges.append(edges[idx, :])
#             edges = np.delete(edges, idx, 0)
#         elif np.any(np.in1d(edges[:, 1], ordered_edges[0][0])):
#             idx = np.where(edges[:, 1] == ordered_edges[0][0])[0]
#             ordered_edges.insert(0, edges[idx, :])
#             edges = np.delete(edges, idx, 0)
#         elif np.any(np.in1d(edges[:, 1], ordered_edges[-1][1])):
#             idx = np.where(edges[:, 1] == ordered_edges[-1][0])[0]
#             ordered_edges.append(edges[idx, :].reversed())
#             edges = np.delete(edges, idx, 0)
#         elif np.any(np.in1d(edges[:, 0], ordered_edges[0][0])):
#             idx = np.where(edges[:, 0] == ordered_edges[0][0])[0]
#             # idx = e0.index(ordered_edges[0][0])
#             ordered_edges.insert(0, edges[idx, :].reversed())
#             edges = np.delete(edges, idx, 0)
#         else:
#             edge_collection.append(ordered_edges)
#             idx = -1
#             ordered_edges = [edges[idx, :]]
#             edges = np.delete(edges, idx, 0)
#         # e0.pop(idx)
#         # e1.pop(idx)
#     # finalize
#     if len(edge_collection) == 0 and len(edges) == 0:
#         edge_collection.append(ordered_edges)
#     else:
#         edge_collection.append(ordered_edges)
#     return edge_collection


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


# def edges_to_rings(edges):
#     if len(edges) == 0:
#         return edges
#     # start ordering the edges into linestrings
#     edge_collection = list()
#     ordered_edges = [edges.pop(-1)]
#     if len(edges) == 0:
#         return [tuple(ordered_edges)]
#     e0, e1 = [list(t) for t in zip(*edges)]
#     while len(edges) > 0:
#         if ordered_edges[-1][1] in e0:
#             idx = e0.index(ordered_edges[-1][1])
#             ordered_edges.append(edges.pop(idx))
#         elif ordered_edges[0][0] in e1:
#             idx = e1.index(ordered_edges[0][0])
#             ordered_edges.insert(0, edges.pop(idx))
#         elif ordered_edges[-1][1] in e1:
#             idx = e1.index(ordered_edges[-1][1])
#             ordered_edges.append(list(reversed(edges.pop(idx))))
#         elif ordered_edges[0][0] in e0:
#             idx = e0.index(ordered_edges[0][0])
#             ordered_edges.insert(0, list(reversed(edges.pop(idx))))
#         else:
#             edge_collection.append(tuple(ordered_edges))
#             idx = -1
#             ordered_edges = [edges.pop(idx)]
#         e0.pop(idx)
#         e1.pop(idx)
#     # finalize
#     if len(edge_collection) == 0 and len(edges) == 0:
#         edge_collection.append(tuple(ordered_edges))
#     else:
#         edge_collection.append(tuple(ordered_edges))
#     return edge_collection


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
        points = list()
        for i, index_ring in enumerate(index_rings):
            e0, e1 = [list(t) for t in zip(*index_ring)]
            points.append(vertices[e0[0], :])

            if path.contains_point(vertices[e0[0], :]):
                potential_interiors.append(i)
        # potential_interiors = np.array(index_rings)[
        #         np.where(path.contains_points(points))]
        # print(potential_interiors)
        # exit()

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
    # for window in raster.iter_windows(chunk_size=chunk_size, overlap=2):
    for xi, yi, zi in raster:
        zi = zi[0, :]
        if not raster.crs.equals(coords_crs):
            transformer = Transformer.from_crs(coords_crs, raster.crs, always_xy=True)
            coords[:, 0], coords[:, 1] = transformer.transform(
                coords[:, 0], coords[:, 1]
            )
        # xi = raster.get_x(window)
        # yi = raster.get_y(window)
        # zi = raster.get_values(window=window)
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
