from collections import defaultdict
from itertools import permutations
from time import time
from typing import Dict, Union
import inspect
import logging
import typing


from inpoly import inpoly2
from jigsawpy import jigsaw_msh_t
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from numpy.linalg import norm
from pyproj import CRS, Transformer
from scipy.interpolate import RectBivariateSpline, griddata
from shapely.geometry import MultiPolygon, Polygon, LinearRing
from shapely.ops import linemerge, polygonize, LineString, MultiLineString
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def mesh_to_tri(mesh):
    """
    mesh is a jigsawpy.jigsaw_msh_t() instance.
    """
    return Triangulation(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'])


def get_mapping(mapping, x):
    return list(map(lambda x: mapping[x], x))


def compute_msh_t_tria3_angles(msh_t):
    vert2, tria3 = msh_t.vert2['coord'], msh_t.tria3['index']
    # Get the vertices for each triangle
    v1 = vert2[tria3[:, 0]]
    v2 = vert2[tria3[:, 1]]
    v3 = vert2[tria3[:, 2]]

    # Compute the vectors of the sides of the triangles
    vec1 = v2 - v1
    vec2 = v3 - v1
    vec3 = v3 - v2

    # Compute the lengths of the vectors
    len1 = np.linalg.norm(vec1, axis=-1)
    len2 = np.linalg.norm(vec2, axis=-1)
    len3 = np.linalg.norm(vec3, axis=-1)

    # Compute the interior angles using the law of cosines
    angle1 = np.arccos(np.einsum('ij,ij->i', vec1, -vec2) / (len1 * len2))
    angle2 = np.arccos(np.einsum('ij,ij->i', -vec1, vec3) / (len1 * len3))
    angle3 = np.arccos(np.einsum('ij,ij->i', vec2, -vec3) / (len2 * len3))

    # Put the angles in an nx3 array, where n is the number of triangles
    tria3_angles = np.column_stack((angle1, angle2, angle3))

    return tria3_angles


def remove_duplicate_triangles(msh_t):
    # remove duplicate triangles (if any)
    # Sort each row
    sorted_arr = np.sort(msh_t.tria3['index'], axis=1)

    # Identify and remove duplicates
    _, indices = np.unique(sorted_arr, axis=0, return_index=True)

    # Extract unique rows (preserving the first occurrence)
    msh_t.tria3 = msh_t.tria3.take(indices)


def cleanup_isolates(mesh):
    node_indexes = np.arange(mesh.vert2['coord'].shape[0])
    used_indexes, inverse = np.unique(mesh.tria3['index'], return_inverse=True)
    isin = np.isin(node_indexes, used_indexes, assume_unique=True)
    if np.all(isin):
        return
    vert2_idxs = np.where(isin)[0]
    df = pd.DataFrame(index=node_indexes).iloc[vert2_idxs].reset_index()
    mapping = {v: k for k, v in df.to_dict()['index'].items()}
    tria3 = np.array([mapping[x] for x in used_indexes])[inverse].reshape(mesh.tria3['index'].shape)
    mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)
    if len(mesh.value) > 0:
        mesh.value = mesh.value.take(vert2_idxs)
    mesh.tria3 = np.asarray(
        [(tuple(indices), mesh.tria3['IDtag'][i])
         for i, indices in enumerate(tria3)],
        dtype=jigsaw_msh_t.TRIA3_t)


def put_edge2(mesh):
    tri = Triangulation(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'])
    mesh.edge2 = tri.edges.astype(jigsaw_msh_t.EDGE2_t)


def _check_if_is_interior_worker(polygon, _msh_t):
    in_poly, on_edge = inpoly2(np.array(polygon.exterior.coords), _msh_t.vert2['coord'], _msh_t.edge2['index'])
    if np.all(in_poly):
        return True
    return False



def filter_polygons(polygons, crs) -> typing.List[Polygon]:
    exteriors_gdf = gpd.GeoDataFrame(geometry=[Polygon(poly.exterior) for poly in polygons], crs=crs)
    joined = gpd.sjoin(exteriors_gdf, exteriors_gdf, how='left', predicate='within')
    joined = joined[joined.index != joined.index_right]
    filtered_polygons = exteriors_gdf.loc[~exteriors_gdf.index.isin(joined.index.unique())]
    within_indices = filtered_polygons.index.unique()
    new_polygons = [polygons[index] for index in filtered_polygons.index.unique()]
    within_counts = joined.index.value_counts()
    multiple_within = within_counts[within_counts > 1].index.tolist()
    if len(multiple_within) == 0:
        return new_polygons
    else:
        nested_polygons = [polygons[i] for i in multiple_within]
        processed_nested_polygons = filter_polygons(nested_polygons, crs)
        new_polygons.extend(processed_nested_polygons)
        return new_polygons

def cleanup_pinched_nodes(mesh, e0_unique, e0_count, e1_unique, e1_count):
    mesh.tria3 = mesh.tria3.take(
        np.where(
            np.logical_or(
                ~np.any(np.isin(mesh.tria3['index'], e0_unique[e0_count > 1]), axis=1),
                ~np.any(np.isin(mesh.tria3['index'], e1_unique[e1_count > 1]), axis=1),
                )
            )[0],
        axis=0
        )


def geom_to_multipolygon(mesh):
    logger.warning('Deprecation warning: Use get_geom_msh_t_from_msh_t_as_mp instead.')
    return get_geom_msh_t_from_msh_t_as_mp(mesh)


def get_geom_msh_t_from_msh_t_as_msh_t(msh_t) -> jigsaw_msh_t:
    mp = get_geom_msh_t_from_msh_t_as_mp(msh_t)
    return multipolygon_to_jigsaw_msh_t(mp)

def get_geom_msh_t_from_msh_t(mesh):
    from geomesh.geom.base import multipolygon_to_jigsaw_msh_t
    return multipolygon_to_jigsaw_msh_t(get_geom_msh_t_from_msh_t_as_mp(mesh))

def multipolygon_to_jigsaw_msh_t(mp):
    from geomesh.geom.base import multipolygon_to_jigsaw_msh_t
    return multipolygon_to_jigsaw_msh_t(mp)

def get_geom_msh_t_from_msh_t_as_mp(msh_t) -> MultiPolygon:
    from shapely.validation import explain_validity

    tria3 = msh_t.tria3['index']
    quad4 = msh_t.quad4['index']
    tria3_edges = np.hstack((tria3[:, [0, 1]], tria3[:, [1, 2]], tria3[:, [2, 0]])).reshape(-1, 2)
    quad4_edges = np.hstack((quad4[:, [0, 1]], quad4[:, [1, 2]], quad4[:, [2, 3]], quad4[:, [3, 0]])).reshape(-1, 2)
    all_edges = np.vstack([tria3_edges, quad4_edges])
    sorted_edges = np.sort(all_edges, axis=1)
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    boundary_rings = linemerge([LineString(x) for x in msh_t.vert2['coord'][boundary_edges]])
    if isinstance(boundary_rings, LineString):
        boundary_rings = MultiLineString(boundary_rings)
    # verify
    # gpd.GeoDataFrame(geometry=[LineString(x) for x in boundary_rings.geoms]).plot(ax=plt.gca(), color=np.random.rand(3,), linewidth=1.3)
    # plt.show(block=False)
    # breakpoint()
    # boundary_rings = [LinearRing(x) for x in boundary_rings.geoms]
    # verify:
    invalids = []
    valids = []
    for i, ring in enumerate(boundary_rings.geoms):
        try:
            ring = LinearRing(ring)
        except Exception as err:
            invalids.append((i, ring, str(err)))
        if not np.all([ring.is_ring, ring.is_valid]):
            invalids.append((i, ring, explain_validity(ring)))
    if len(invalids) > 0:
        # # verify:
        wireframe(msh_t, ax=plt.gca(), linewidth=0.1)
        icoll = []
        for i, ring, validity_text in invalids:
            print(validity_text)
            icoll.append(str(i))
            gpd.GeoDataFrame(geometry=[ring]).plot(ax=plt.gca(), color=np.random.rand(3,), linewidth=1.3)
        plt.title(f"Cannot build Not rings at {', '.join(icoll)}")
        plt.show(block=False)
        breakpoint()
        raise
    #     try:
    #         print(ring)
    #         print(LinearRing(ring))
    #         print(Polygon(ring))
    #     except:
    #         gpd.GeoDataFrame(geometry=[ring]).plot(ax=plt.gca(), color='r', linewidth=1.3)
    #         plt.show(block=False)
    #         breakpoint()
    #         raise
    # fig, ax = plt.subplots()
    polygons = list(polygonize(boundary_rings))
    mp =  MultiPolygon(filter_polygons(polygons, msh_t.crs))
    # verify
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    # gpd.GeoDataFrame(geometry=boundary_rings).explode().plot(ax=axes[0], cmap='tab20')
    # gpd.GeoDataFrame(geometry=polygons).plot(ax=axes[1], cmap='jet')
    # plt.show(block=False)
    # breakpoint()
    # raise
    # TODO: Use the new shapely functions and avoid the manual filtering
    # import shapely
    # res = shapely.multipolygons(shapely.get_parts(polygons))
    # # breakpoint()
    # # verify
    # gpd.GeoDataFrame(geometry=[mp], crs=msh_t.crs).plot(ax=plt.gca())
    # plt.show(block=False)
    # breakpoint()
    return mp



def needs_sieve(mesh, area=None):
    # lp = LineProfiler()
    # mp = lp(geom_to_multipolygon)(mesh, nprocs)
    # lp.print_stats()
    # exit()
    # from time import time
    # start = time()
    print('start geom to mp')
    mp = geom_to_multipolygon(mesh)
    # print(f'end geom_to_mp: {time()-start}')
    areas = [polygon.area for polygon in mp.geoms]
    if area in [None, True]:
        remove = np.where(areas < np.max(areas))[0].tolist()
        if len(remove) > 0:
            return True, mp, remove
    else:
        remove = list()
        for idx, patch_area in enumerate(areas):
            if patch_area <= area:
                remove.append(idx)
                if len(remove) > 0:
                    return True, mp, remove
    # if len(remove) > 0:
    #     return True
    # else:
    return False, mp, remove


def put_IDtags(mesh):
    # start enumerating on 1 to avoid issues with indexing on fortran models
    mesh.vert2 = np.array(
        [(coord, id+1) for id, coord in enumerate(mesh.vert2['coord'])],
        dtype=jigsaw_msh_t.VERT2_t
        )
    mesh.tria3 = np.array(
        [(index, id+1) for id, index in enumerate(mesh.tria3['index'])],
        dtype=jigsaw_msh_t.TRIA3_t
        )
    mesh.quad4 = np.array(
        [(index, id+1) for id, index in enumerate(mesh.quad4['index'])],
        dtype=jigsaw_msh_t.QUAD4_t
        )
    mesh.hexa8 = np.array(
        [(index, id+1) for id, index in enumerate(mesh.hexa8['index'])],
        dtype=jigsaw_msh_t.HEXA8_t
        )


def check_pinched_nodes(mesh):
    print('start cleanup pinched nodes')
    start = time()
    tri = mesh_to_tri(mesh)
    # TODO: can probably be faster if using inpoly2 instead
    idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
    e0_coll = list()
    e1_coll = list()
    for i, j in idxs:
        e0_coll.append(tri.triangles[i, j])
        e1_coll.append(tri.triangles[i, (j+1) % 3])
    e0_unique, e0_count = np.unique(e0_coll, return_counts=True)
    e1_unique, e1_count = np.unique(e1_coll, return_counts=True)

    if len(e0_unique[e0_count > 1]) > 0 or len(e1_unique[e1_count > 1]) > 0:
        return True, e0_unique, e0_count, e1_unique, e1_count
    else:
        return False, None, None, None, None


def get_msh_t_tria3_LinearRing_gdf(msh_t):
    return gpd.GeoDataFrame(
        geometry=[LinearRing(msh_t.vert2['coord'][tria, :]) for tria in msh_t.tria3['index']],
        crs=msh_t.crs
    )


def get_msh_t_tria3_Polygon_gdf(msh_t):
    return gpd.GeoDataFrame(geometry=[Polygon(msh_t.vert2['coord'][tria, :]) for tria in msh_t.tria3['index']], crs=msh_t.crs)


def get_split_quad4_as_tria3_index(msh_t):
    new_tri = []
    for quad in msh_t.quad4['index']:
        new_tri.append([quad[0], quad[1], quad[3]])
        new_tri.append([quad[1], quad[2], quad[3]])
    return np.array([(new_idx, 0) for new_idx in new_tri], dtype=jigsaw_msh_t.TRIA3_t)


def get_tria3_from_split_quad4(msh_t):
    return np.hstack([
        msh_t.tria3.copy(),
        get_split_quad4_as_tria3_index(msh_t)
        ])


def split_quad4_to_tria3(msh_t):
    msh_t.tria3 = get_tria3_from_split_quad4(msh_t)


def remove_quad4_from_tria3(msh_t):
    # Get the indexes of triangles created by splitting quads
    split_tri_indexes = get_split_quad4_as_tria3_index(msh_t)
    split_tri_indexes = set(tuple(index[0]) for index in split_tri_indexes)

    # Filter the existing tria3 to exclude those that were created by splitting quads
    msh_t.tria3 = np.array([
        (index, IDtag) for index, IDtag in msh_t.tria3
        if tuple(index) not in split_tri_indexes
    ], dtype=jigsaw_msh_t.TRIA3_t)


# def remove_degenerate_triangles(msh_t, min_area=1e-1):
#     tria3_poly_gdf = get_msh_t_tria3_Polygon_gdf(msh_t)
#     tria_areas = tria3_poly_gdf.to_crs("EPSG:6933").area
#     non_degenerate_indices = np.where(tria_areas > min_area)[0]
#     msh_t.tria3 = msh_t.tria3.take(non_degenerate_indices)


def remove_flat_triangles(msh_t):
    row_has_repeated_nodes = np.array([len(np.unique(msh_t.tria3['index'][row, :])) != 3 for row in range(msh_t.tria3['index'] .shape[0])])
    msh_t.tria3 = msh_t.tria3.take(np.where(~row_has_repeated_nodes)[0])
    # degenerate_mask = check_degenerate(msh_t)
    # if np.any(degenerate_mask):
    #     msh_t.tria3 = msh_t.tria3.take(np.where(~degenerate_mask)[0])


def check_degenerate(msh_t, max_tol=1e-8):
    """
    Vectorized check for degenerate triangles in msh_t
    """

    # Extract all vertex coords for all triangles
    tri_verts = msh_t.vert2['coord'][msh_t.tria3['index']]

    # Calculate vertex vectors for each triangle
    v1 = tri_verts[:, 1, :] - tri_verts[:, 0, :]
    v2 = tri_verts[:, 2, :] - tri_verts[:, 0, :]

    # Calculate cross products
    cross = np.cross(v1, v2)

    # Histogram of cross product magnitudes
    hist, bins = np.histogram(np.log10(np.abs(cross)))
    gaps = np.diff(bins)
    tol = 10**((bins[np.argmax(gaps)] - gaps.max()/2))

    # Enforce max tolerance
    tol = min(tol, max_tol)

    # Check for degenerates
    return np.abs(cross) < tol

# def cleanup_flat_triangles2(mesh):
#     areas = []
#     for vertices in mesh.vert2['coord'][mesh.tria3['index'], :]:
#         areas.append(signed_polygon_area(vertices))
#     areas = np.array(areas)
#     # remove very small areas and
#     remove_vert2_indexes = np.where(np.logical_and(areas <= 0, areas > -np.finfo(np.float16).eps))

    


# def remove_duplicate_coord_with_round(msh_t, decimal):
#     # Round the coordinates to the specified decimal place
#     coords = np.round(msh_t.vert2['coord'], decimals=decimal)

#     unique_coords = np.unique(coords)

#     tria3 = msh_t.tria3['index']



#     breakpoint()



# def cleanup_flat_triangles(mesh):
#     # The problem is that there are non-unique vertices in the node
#     # table, but they are referenced correctly (as unique points) in
#     # the element table. This leads to triangles with a pair of repeated
#     # coordinates (e.g. ((0., 0.), (1., 1.), (0., 0.)) -> [0, 1, 2]
#     # where nodes 0 and 2 have the same coordinates.


#     areas = []
#     for vertices in mesh.vert2['coord'][mesh.tria3['index'], :]:
#         areas.append(signed_polygon_area(vertices))
#     areas = np.array(areas)
#     print(areas)
#     # remove very small areas and
#     remove_vert2_indexes = np.where(np.logical_and(areas <= 0, areas > -np.finfo(np.float16).eps))
#     print(remove_vert2_indexes)


def remove_duplicate_vert2(msh_t, rounding_decimals: int = None):

    vert2_coord = msh_t.vert2['coord'].copy()
    if rounding_decimals is not None:
        vert2_coord = np.around(msh_t.vert2['coord'], decimals=rounding_decimals)

    # Use numpy unique to find unique rows and keep their first occurrences
    unique_coords, indices, inverse_mapping = np.unique(vert2_coord, axis=0, return_index=True, return_inverse=True)

    # Create a mapping to adjust the tria3_index accordingly
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(indices)}

    # Adjust the indices in tria3_index
    adjusted_tria3_index = np.array([list(map(index_mapping.get, indices)) for indices in msh_t.tria3['index']])

    # Zip over the original vert2 coordinates using the indices
    original_coords = msh_t.vert2['coord'][indices]

    msh_t.vert2 = np.array([(coord, IDtag) for coord, IDtag in zip(original_coords, msh_t.vert2['IDtag'][indices])],
                            dtype=jigsaw_msh_t.VERT2_t)

    msh_t.tria3 = np.array([(tria3_indices, IDtag) for tria3_indices, IDtag in zip(adjusted_tria3_index, msh_t.tria3['IDtag'])],
                            dtype=jigsaw_msh_t.TRIA3_t)

#     logger.info('Checking for flat triangles.')
#     node_indexes = np.arange(mesh.vert2['coord'].shape[0])
#     new_vert2, vert2_idxs, inverse = np.unique(
#             np.around(mesh.vert2['coord'], decimals=3),
#             return_index=True,
#             return_inverse=True,
#             axis=0
#             )
#     tree = KDTree(new_vert2)
#     index_difference = list(set(node_indexes).difference(set(vert2_idxs)))
#     _, new_index_of_repeated_value = tree.query(mesh.vert2['coord'][index_difference])
#     # _, new_index_of_repeated_value = tree.query(new_vert2[index_difference])
#     breakpoint()
#     repeated_value_mapping = {repeated: new_index_of_repeated_value[i] for i, repeated in enumerate(index_difference)}
#     df = pd.DataFrame(
#             index=list(range(mesh.vert2['coord'].shape[0]))
#             ).iloc[vert2_idxs].reset_index()

#     mapping = {v: k for k, v in df.to_dict()['index'].items()}
#     mapping.update(repeated_value_mapping)
#     tria3 = np.array([[mapping[node_id] for node_id in tria3_row] for tria3_row in mesh.tria3['index']])
#     # now remove rows from tria3 that have repeated values (row-wise)
#     _tria3 = np.sort(tria3, axis=1)
#     tria3_row_unique_count = (_tria3[:,1:] != _tria3[:,:-1]).sum(axis=1)+1
#     tria3_keep_index = np.where(tria3_row_unique_count == 3)[0]
#     tria3 = tria3[tria3_keep_index, :]

#     # makes sure they all have same orientation

#     # for i, vertices in enumerate(new_vert2[tria3, :]):
#     #     if not LinearRing(vertices).is_ccw:
#     #         tria3[i, :] = np.flip(tria3[i, :])
#     #     # debug
#     #     _vertices = new_vert2[tria3[i, :], :]
#     #     if not LinearRing(_vertices).is_ccw:
#     #         raise Exception(f'I Flipped\n{vertices=}\nas\n{_vertices=}\nbut still they are still clockwise.')

#     # exit()






#     # xtri = mesh.vert2['coord'][mesh.tria3['index'], 0]
#     # ytri = mesh.vert2['coord'][mesh.tria3['index'], 1]
#     # areas = []
#     # for vertices in new_vert2['coord'][tria3, :]:
#     #     areas.append(signed_polygon_area(vertices))
#     # areas = np.array(areas)
#     # tria3 = tria3[np.where(areas != 0)[0], :]
#     # areas = []
#     # for vertices in mesh.vert2['coord'][tria3, :]:
#     #     areas.append(signed_polygon_area(vertices))
#     # areas = np.array(areas)
#     # _flip_idx = np.where(areas > 0)[0]
#     # tria3[_flip_idx] = np.fliplr(tria3[_flip_idx])


#     # print(areas)


#     mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)

#     if len(mesh.value) > 0:
#         mesh.value = mesh.value.take(vert2_idxs)

#     mesh.tria3 = np.array(
#         [(tuple(indices), mesh.tria3['IDtag'][i])
#          for i, indices in enumerate(tria3)],
#         dtype=jigsaw_msh_t.TRIA3_t
#         )


# def signed_polygon_area(x,y):
#     # coordinate shift
#     x_ = x - x.mean()
#     y_ = y - y.mean()
#     correction = x_[-1] * y_[0] - y_[-1]* x_[0]
#     main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
#     return 0.5*(main_area + correction)


def cleanup_pinched_nodes_iter(msh_t):
    # split_quad4_to_tria3(msh_t)
    logger.debug('Checking for pinched nodes...')
    has_pinched_nodes, e0_unique, e0_count, e1_unique, e1_count = check_pinched_nodes(msh_t)
    while has_pinched_nodes:
        logger.debug('cleaning up pinched nodes...')
        start = time()
        cleanup_pinched_nodes(msh_t, e0_unique, e0_count, e1_unique, e1_count)
        logger.debug(f'cleanup pinched took={time()-start}')
        logger.debug('checking for additional pinched nodes...')
        has_pinched_nodes, e0_unique, e0_count, e1_unique, e1_count = check_pinched_nodes(msh_t)
        if has_pinched_nodes is False:
            break
    logger.debug('No additional pinched nodes found, continuing...')
    # remove_quad4_from_tria3(msh_t)


def split_bad_quality_quads(
        msh_t,
        cutoff=0.5,
        ):

    def angle_between(v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude_product = norm(v1) * norm(v2)
        # Check if magnitudes are zero to avoid division by zero error
        if magnitude_product == 0:
            return 0
        else:
            # Clip the value to avoid RuntimeWarning in arccos
            clipped_value = np.clip(dot_product / magnitude_product, -1.0, 1.0)
            angle = np.arccos(clipped_value)
            return np.degrees(angle)

    to_split = []
    for quad_indices, quad_id in msh_t.quad4:
        coords = msh_t.vert2['coord'][quad_indices]
        angles = []
        for i in range(4):
            prev_point = np.array(coords[(i-2) % 4])
            curr_point = np.array(coords[(i-1) % 4])
            next_point = np.array(coords[i])
            prev_vector = curr_point - prev_point
            next_vector = next_point - curr_point
            angle = angle_between(prev_vector, next_vector)
            angles.append(angle)
        min_angle = min(angles)
        max_angle = max(angles)
        to_split.append(min_angle / max_angle <= cutoff)
    to_split = np.array(to_split, dtype=bool).flatten()
    # New triangles container
    new_triangles = []

    # Getting the quads to split
    quads_to_split = msh_t.quad4[to_split]

    def calculate_triangle_quality(triangle):
        # Get the coordinates of the vertices
        coords = msh_t.vert2['coord'][triangle]
        # Define a function to calculate the angle between three points

        def angle(a, b, c):
            v1 = a - b
            v2 = c - b
            return angle_between(v1, v2)
        # Calculate the angles of the triangle
        angles = [
            angle(coords[0], coords[1], coords[2]),
            angle(coords[1], coords[2], coords[0]),
            angle(coords[2], coords[0], coords[1]),
        ]

        # Define a function to calculate how close the triangle is to being equilateral
        def equilateral_quality(angles):
            return -sum((angle - 60)**2 for angle in angles)

        # Calculate and return the equilateral quality of the triangle
        return equilateral_quality(angles)

    # Loop through the quads to be split and create triangles
    for quad_indices, quad_id in quads_to_split:
        # Create two triangles by splitting the quad along its diagonals
        triangle1 = [quad_indices[0], quad_indices[1], quad_indices[2]]
        triangle2 = [quad_indices[2], quad_indices[3], quad_indices[0]]
        triangle_set1 = [triangle1, triangle2]
        triangle3 = [quad_indices[0], quad_indices[1], quad_indices[3]]
        triangle4 = [quad_indices[3], quad_indices[1], quad_indices[2]]
        triangle_set2 = [triangle3, triangle4]
        # Calculate the quality of each set of triangles by summing the quality of the individual triangles in the set
        quality_set1 = sum(calculate_triangle_quality(triangle) for triangle in triangle_set1)
        quality_set2 = sum(calculate_triangle_quality(triangle) for triangle in triangle_set2)

        # Select the set of triangles with the highest total quality
        if quality_set1 > quality_set2:
            new_triangles.append(triangle_set1[0])
            new_triangles.append(triangle_set1[1])
        else:
            new_triangles.append(triangle_set2[0])
            new_triangles.append(triangle_set2[1])

    # Assuming msh_t.tri3 is the container for triangles, add the new triangles to it
    new_triangles = np.array([(el, 0) for el in new_triangles], dtype=jigsaw_msh_t.TRIA3_t)
    msh_t.tria3 = np.hstack([msh_t.tria3, new_triangles])

    # Remove the split quads from the quad4 container
    msh_t.quad4 = msh_t.quad4[~to_split]    # New triangles container


def wireframe(msh_t, ax=None, triplot_kwargs=None, quadplot_kwargs=None, **kwargs):
    triplot_kwargs = triplot_kwargs or {}
    quadplot_kwargs = quadplot_kwargs or {}

    # Filter kwargs for triplot
    triplot_params = inspect.signature(triplot).parameters
    triplot_args = {k: v for k, v in kwargs.items() if k in triplot_params}
    combined_triplot_kwargs = {**triplot_args, **triplot_kwargs}  # triplot_kwargs takes precedence
    combined_triplot_kwargs.setdefault('ax', ax or plt.gca())
    triplot(msh_t, **combined_triplot_kwargs)

    # Filter kwargs for quadplot
    quadplot_params = inspect.signature(quadplot).parameters
    quadplot_args = {k: v for k, v in kwargs.items() if k in quadplot_params}
    combined_quadplot_kwargs = {**quadplot_args, **quadplot_kwargs}  # quadplot_kwargs takes precedence
    combined_quadplot_kwargs.setdefault('ax', ax or plt.gca())
    quadplot(msh_t, **combined_quadplot_kwargs)

    return ax


def finalize_mesh(mesh, sieve=True):
    # remove flat triangles

    cleanup_pinched_nodes_iter(mesh)
    # cleanup_flat_triangles(mesh)
    # breakpoint()
    mp = geom_to_multipolygon(mesh)

    from geomesh.geom.base import multipolygon_to_jigsaw_msh_t  # TODO: Circular import

    if sieve is True:
        areas = [poly.area for poly in mp.geoms]
        polygon = mp.geoms[areas.index(max(areas))]
        mp = MultiPolygon([polygon])
    elif sieve is None or sieve is False:
        pass
    else:
        # TODO:
        raise NotImplementedError(f'Unhandled sieve: expected None or bool but got {sieve}')

    _msh_t = multipolygon_to_jigsaw_msh_t(mp)

    in_poly, on_edge = inpoly2(mesh.vert2['coord'], _msh_t.vert2['coord'], _msh_t.edge2['index'])

    tria3_mask = np.any(~in_poly[mesh.tria3['index']], axis=1)

    # tria3_idxs_take = np.where(~tria3_mask)[0]
    tria3_index = mesh.tria3['index'][~tria3_mask, :]
    # tria3_index = mesh.tria3['index'].take(tria3_idxs_take)
    used_indexes, inverse = np.unique(tria3_index, return_inverse=True)
    node_indexes = np.arange(mesh.vert2['coord'].shape[0])
    isin = np.isin(node_indexes, used_indexes)
    # tria3_idxs = np.where(~isin)[0]
    vert2_idxs = np.where(isin)[0]

    df = pd.DataFrame(index=node_indexes).iloc[vert2_idxs].reset_index()
    mapping = {v: k for k, v in df.to_dict()['index'].items()}
    tria3_index = np.array([mapping[x] for x in used_indexes])[inverse].reshape(tria3_index.shape)
    mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)

    tria3_IDtag = mesh.tria3['IDtag'][~tria3_mask]

    # update value
    if len(mesh.value) > 0:
        mesh.value = mesh.value.take(vert2_idxs)

    mesh.tria3 = np.array(
        [(tuple(indices), tria3_IDtag[i])
         for i, indices in enumerate(tria3_index)],
        dtype=jigsaw_msh_t.TRIA3_t)

    print('Checking for pinched nodes...')
    has_pinched_nodes, e0_unique, e0_count, e1_unique, e1_count = check_pinched_nodes(mesh)
    while has_pinched_nodes:
        print('cleaning up pinched nodes...')
        start = time()
        cleanup_pinched_nodes(mesh, e0_unique, e0_count, e1_unique, e1_count)
        print(f'cleanup pinched took={time()-start}')
        print('checking for additional pinched nodes...')
        has_pinched_nodes, e0_unique, e0_count, e1_unique, e1_count = check_pinched_nodes(mesh)
        if has_pinched_nodes is False:
            print('No additional pinched nodes found, continuing...')
            break

    # cleanup_flat_triangles(mesh)

    # areas = []
    # xtri = mesh.vert2['coord'][mesh.tria3['index'], 0]
    # ytri = mesh.vert2['coord'][mesh.tria3['index'], 1]
    # for x, y in zip(xtri, ytri):
    #     areas.append(signed_polygon_area(x, y))
    # areas = np.array(areas)
    # print(f'3 - index of triangles with zero or negative area: {np.where(areas <= 0)}')
    # mesh.tria3 = mesh.tria3.take(np.where(areas != 0)[0])
    # # negative area index
    # _idx = np.where(areas < 0)[0]
    # mesh.tria3['index'][_idx] = np.fliplr(mesh.tria3['index'][_idx])
    # areas = []
    # xtri = mesh.vert2['coord'][mesh.tria3['index'], 0]
    # ytri = mesh.vert2['coord'][mesh.tria3['index'], 1]
    # for x, y in zip(xtri, ytri):
    #     areas.append(signed_polygon_area(x, y))
    # areas = np.array(areas)
    # print(f'4 - index of triangles with zero or negative area: {np.where(areas <= 0)}')

    # remove flat triangles
    # areas = []
    # xtri = mesh.vert2['coord'][mesh.tria3['index'], 0]
    # ytri = mesh.vert2['coord'][mesh.tria3['index'], 1]
    # for x, y in zip(xtri, ytri):
    #     areas.append(signed_polygon_area(x, y))
    # areas = np.array(areas)
    # mesh.tria3 = mesh.tria3.take(np.where(areas != 0)[0])
    # print('plotting from utils.finalize_mesh end')
    # triplot(mesh)
    # import contextily as cx
    # cx.add_basemap(plt.gca(), crs=mesh.crs)
    # plt.show(block=True)
    return


    # gpd.GeoDataFrame([{'geometry': poly} for poly in mp.geoms], crs=mesh.crs).plot(facecolor='none', cmap='jet')
    # plt.show(block=True)



    # boundary_edges = list()
    # print('build mesh_to_tri')
    # tri = mesh_to_tri(mesh)
    # print('build element neighbors array')
    # idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
    # print('build boundary edges array')
    # start = time()
    # boundary_edges = build_edges_from_triangles(tri.triangles, idxs)
    # print('call linemerge')
    # start = time()
    # boundary_lines = linemerge(MultiLineString([LineString(x) for x in mesh.vert2['coord'][boundary_edges]]))
    # print(f'linemerge took {time()-start}')



    # print('polygonizing ... ')
    # start = time()
    # polygons = list(polygonize(boundary_lines))
    # print(f'polygonizing took {time()-start}')

    # gpd.GeoDataFrame([{'geometry': polygon} for polygon in polygons]).plot(facecolor='none', ax=plt.gca(), cmap='jet')
    # plt.show(block=True)
    # breakpoint()

    # print(f'edges took {time()-start}')
    # from geomesh.mesh.mesh import sort_rings, edges_to_rings
    # start = time()
    # print('start edges_to_rings')
    # linear_rings = edges_to_rings(boundary_edges)
    # print(f'edges to rings took {time()-start}')

    # start = time()
    # print('start  sorting rings')
    # sorted_rings = sort_rings(linear_rings, mesh.vert2['coord'])
    # print(f'sorting rings took {time()-start}')
    # data = []
    # for bnd_id, rings in sorted_rings.items():
    #     coords = mesh.vert2['coord'][rings["exterior"][:, 0], :]
    #     geometry = LinearRing(coords)
    #     data.append({"geometry": geometry, "bnd_id": bnd_id, "type": "exterior"})
    #     for interior in rings["interiors"]:
    #         coords = mesh.vert2['coord'][interior[:, 0], :]
    #         geometry = LinearRing(coords)
    #         data.append(
    #             {"geometry": geometry, "bnd_id": bnd_id, "type": "interior"}
    #         )
    # print('exit in utils.finalize_mesh')
    # gdf = gpd.GeoDataFrame(data, crs=mesh.crs)
    # gdf.plot(plt.gca())
    # plt.show(block=True)
    # exit()


# def finalize_mesh(mesh):

#     boundary_edges = list()
#     print('build mesh_to_tri')
#     tri = mesh_to_tri(mesh)
#     print('build element neighbors array')
#     idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
#     print('build boundary edges array')
#     for i, j in idxs:
#         boundary_edges.append((
#             tri.triangles[i, j],
#             tri.triangles[i, (j+1) % 3]
#             ))

#     print('call linemerge')
#     start = time()
#     boundary_rings = linemerge(MultiLineString([LineString(x) for x in mesh.vert2['coord'][boundary_edges]]))
#     print(f'linemerge took {time()-start}')

#     print('polygonizing ... ')
#     start = time()
#     polygon = list(polygonize(boundary_rings))
#     print(f'polygonizing took {time()-start}')

#     # just take the largest polygon
#     areas = [p.area for p in polygon]
#     polygon = polygon.pop(areas.index(max(areas)))

#     from geomesh.geom.base import multipolygon_to_jigsaw_msh_t

#     _msh_t = multipolygon_to_jigsaw_msh_t(MultiPolygon([polygon]))
#     in_poly, on_edge = inpoly2(
#                 mesh.vert2['coord'],
#                 _msh_t.vert2['coord'],
#                 _msh_t.edge2['index']
#             )

#     tria3_mask = np.any(~in_poly[mesh.tria3['index']], axis=1)

#     # tria3_idxs_take = np.where(~tria3_mask)[0]
#     tria3_index = mesh.tria3['index'][~tria3_mask, :]
#     # tria3_index = mesh.tria3['index'].take(tria3_idxs_take)
#     used_indexes, inverse = np.unique(tria3_index, return_inverse=True)
#     node_indexes = np.arange(mesh.vert2['coord'].shape[0])
#     isin = np.isin(node_indexes, used_indexes)
#     # tria3_idxs = np.where(~isin)[0]
#     vert2_idxs = np.where(isin)[0]

#     df = pd.DataFrame(index=node_indexes).iloc[vert2_idxs].reset_index()
#     mapping = {v: k for k, v in df.to_dict()['index'].items()}
#     tria3_index = np.array([mapping[x] for x in used_indexes])[inverse].reshape(tria3_index.shape)
#     mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)

#     tria3_IDtag = mesh.tria3['IDtag'][~tria3_mask]

#     # update value
#     if len(mesh.value) > 0:
#         mesh.value = mesh.value.take(vert2_idxs)

#     mesh.tria3 = np.array(
#         [(tuple(indices), tria3_IDtag[i])
#          for i, indices in enumerate(tria3_index)],
#         dtype=jigsaw_msh_t.TRIA3_t)
#     # cleanup_pinched_nodes(mesh)
#     return



def index_ring_collection(mesh):

    # find boundary edges using triangulation neighbors table,
    # see: https://stackoverflow.com/a/23073229/7432462
    boundary_edges = list()
    tri = mesh_to_tri(mesh)
    idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
    for i, j in idxs:
        boundary_edges.append((
            tri.triangles[i, j],
            tri.triangles[i, (j+1) % 3]
            ))


    # import geopandas as gpd
    # from shapely.ops import linemerge, polygonize, LineString, MultiLineString
    # ls = MultiLineString([LineString(x) for x in mesh.vert2['coord'][boundary_edges]])
    # lines = linemerge(ls)
    # lines = [line for line in lines if len(line) >= 3]

    # import matplotlib.pyplot as plt
    # # plt.show()
    # # exit()
    # # poly = list(polygonize(lines))
    # # breakpoint()
    # polygons = list(polygonize(lines))
    # areas = [p.area for p in polygons]
    # outer_polygon = polygons.pop(areas.index(max(areas)))

    # gpd.GeoDataFrame([{'geometry': outer_polygon}]).plot(facecolor='none', cmap='jet')
    # plt.show()
    # exit()
    from shapely.geometry import Polygon, Point

    # from pyproj import CRS
    # gpd.GeoDataFrame([{'geometry': poly} for poly in polygonize(lines)], crs=CRS.from_epsg(4326)).plot(cmap='jet')
    # plt.show()
    # exit()
    # index_ring_collection = [np.array(linestring.coords) for linestring in lines]
    # areas = [Polygon(linestring).areas for linestring in lines]
    # print('sort edges')
    index_ring_collection = sort_edges(boundary_edges)
    # sort index_rings into corresponding "polygons"
    # print('make polygons')
    # data = []
    areas = list()
    vertices = mesh.vert2['coord']
    for index_ring in index_ring_collection:
        e0, e1 = [list(t) for t in zip(*index_ring)]
        # data.append({'geometry': Polygon(mesh.vert2['coord'][e0, :])})
        areas.append(float(Polygon(vertices[e0, :]).area))
    # gdf = gpd.GeoDataFrame(data, crs=mesh.crs)
    # gdf['area'] = gdf.apply(lambda x: x.geometry.area, axis=1)
    # rtee = gdf.sindex
    # maximum area must be main mesh
    idx = areas.index(np.max(areas))
    exterior = index_ring_collection.pop(idx)
    areas.pop(idx)
    _id = 0
    _index_ring_collection = dict()
    _index_ring_collection[_id] = {
        'exterior': np.asarray(exterior),
        'interiors': []
        }
    e0, e1 = [list(t) for t in zip(*exterior)]
    path = Path(vertices[e0 + [e0[0]], :], closed=True)
    while len(index_ring_collection) > 0:
        # find all internal rings
        potential_interiors = list()
        for i, index_ring in enumerate(index_ring_collection):
            e0, e1 = [list(t) for t in zip(*index_ring)]
            # point_bbox = Point(vertices[e0[0], :]).buffer(np.finfo(np.float32).eps).bounds
            # print(point_bbox)
            # exit()
            if path.contains_point(vertices[e0[0], :]):
                potential_interiors.append(i)
        # filter out nested rings
        real_interiors = list()
        for i, p_interior in reversed(list(enumerate(potential_interiors))):
            _p_interior = index_ring_collection[p_interior]
            check = [index_ring_collection[_]
                     for j, _ in reversed(list(enumerate(potential_interiors)))
                     if i != j]
            has_parent = False
            for _path in check:
                e0, e1 = [list(t) for t in zip(*_path)]
                e0.append(e0[0])
                _path = Path(vertices[e0, :], closed=True)
                if _path.contains_point(vertices[_p_interior[0][0], :]):
                    has_parent = True
            if not has_parent:
                real_interiors.append(p_interior)
        # pop real rings from collection
        for i in reversed(sorted(real_interiors)):
            _index_ring_collection[_id]['interiors'].append(
                np.asarray(index_ring_collection.pop(i)))
            areas.pop(i)
        # if no internal rings found, initialize next polygon
        if len(index_ring_collection) > 0:
            idx = areas.index(np.max(areas))
            exterior = index_ring_collection.pop(idx)
            areas.pop(idx)
            _id += 1
            _index_ring_collection[_id] = {
                'exterior': np.asarray(exterior),
                'interiors': []
                }
            e0, e1 = [list(t) for t in zip(*exterior)]
            e0.append(e0[0])
            path = Path(vertices[e0, :], closed=True)
    return _index_ring_collection


def outer_ring_collection(mesh):
    _index_ring_collection = index_ring_collection(mesh)
    outer_ring_collection = defaultdict()
    for key, ring in _index_ring_collection.items():
        outer_ring_collection[key] = ring['exterior']
    return outer_ring_collection


def inner_ring_collection(mesh):
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # _index_ring_collection = lp(index_ring_collection)(mesh)
    # lp.print_stats()
    # exit()
    _index_ring_collection = index_ring_collection(mesh)
    inner_ring_collection = defaultdict()
    for key, rings in _index_ring_collection.items():
        inner_ring_collection[key] = rings['interiors']
    return inner_ring_collection


def signed_polygon_area(vertices):
    # https://code.activestate.com/recipes/578047-area-of-polygon-using-shoelace-formula/
    n = len(vertices)  # of vertices
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return area / 2.0


def vertices_around_vertex(mesh):
    if mesh.mshID == 'euclidean-mesh':
        def append(geom):
            for simplex in geom['index']:
                for i, j in permutations(simplex, 2):
                    vertices_around_vertex[i].add(j)
        vertices_around_vertex = defaultdict(set)
        append(mesh.tria3)
        append(mesh.quad4)
        append(mesh.hexa8)
        return vertices_around_vertex
    else:
        msg = f"Not implemented for mshID={mesh.mshID}"
        raise NotImplementedError(msg)


# https://en.wikipedia.org/wiki/Polygon_mesh#Summary_of_mesh_representation
# V-V     All vertices around vertex
# E-F     All edges of a face
# V-F     All vertices of a face
# F-V     All faces around a vertex
# E-V     All edges around a vertex
# F-E     Both faces of an edge
# V-E     Both vertices of an edge
# Flook   Find face with given vertices


def must_be_euclidean_mesh(f):
    def decorator(mesh):
        if mesh.mshID.lower() != 'euclidean-mesh':
            msg = f"Not implemented for mshID={mesh.mshID}"
            raise NotImplementedError(msg)
        return f(mesh)
    return decorator


@must_be_euclidean_mesh
def elements(mesh):
    elements_id = list()
    elements_id.extend(list(mesh.tria3['IDtag']))
    elements_id.extend(list(mesh.quad4['IDtag']))
    elements_id.extend(list(mesh.hexa8['IDtag']))
    elements_id = range(1, len(elements_id)+1) \
        if len(set(elements_id)) != len(elements_id) else elements_id
    elements = list()
    elements.extend(list(mesh.tria3['index']))
    elements.extend(list(mesh.quad4['index']))
    elements.extend(list(mesh.hexa8['index']))
    elements = {
        elements_id[i]: indexes for i, indexes in enumerate(elements)}
    return elements


@must_be_euclidean_mesh
def faces_around_vertex(mesh):
    # _elements = elements(mesh)
    # length = max(map(len, _elements.values()))
    # y = np.array([xi+[-99999]*(length-len(xi)) for xi in _elements.values()])
    # print(y)
    faces_around_vertex = defaultdict(set)
    for i, coord in enumerate(mesh.vert2['index']):
        np.isin(i, axis=0)
        faces_around_vertex[i].add()

    faces_around_vertex = defaultdict(set)


def has_pinched_nodes(mesh):

    boundary_edges = list()
    tri = mesh_to_tri(mesh)
    idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
    for i, j in idxs:
        boundary_edges.append(tri.triangles[i, j])
    unique, count = np.unique(boundary_edges, return_counts=True)
    # _inner_ring_collection = inner_ring_collection(mesh)
    # all_nodes = list()
    # for inner_rings in _inner_ring_collection.values():
    #     for ring in inner_rings:
    #         all_nodes.extend(np.asarray(ring)[:, 0].tolist())
    # _outer_ring_collection = outer_ring_collection(mesh)
    # for outer_ring in _outer_ring_collection.values():
    #     all_nodes.extend(np.asarray(outer_ring)[:, 0].tolist())
    # u, c = np.unique(all_nodes, return_counts=True)
    if len(unique[count > 1]) > 0:
        return True, unique, count
    else:
        return False, unique, count



def interpolate_hmat(mesh, hmat, method='spline', kx=1, ky=1, **kwargs):
    assert isinstance(mesh, jigsaw_msh_t)
    assert isinstance(hmat, jigsaw_msh_t)
    assert method in ['spline', 'linear', 'nearest']
    kwargs.update({'kx': kx, 'ky': ky})
    if method == 'spline':
        values = RectBivariateSpline(
            hmat.xgrid,
            hmat.ygrid,
            hmat.value.T,
            **kwargs
            ).ev(
            mesh.vert2['coord'][:, 0],
            mesh.vert2['coord'][:, 1])
        mesh.value = np.array(
            values.reshape((values.size, 1)),
            dtype=jigsaw_msh_t.REALS_t)
    else:
        raise NotImplementedError("Only 'spline' method is available")


def tricontourf(
    mesh,
    ax=None,
    show=False,
    figsize=None,
    extend='both',
    **kwargs
):
    ax = ax or plt.gca()
    split_tria3 = get_tria3_from_split_quad4(mesh)
    ax.tricontourf(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        split_tria3['index'],
        mesh.value.flatten(),
        **kwargs)
    return ax


def triplot(
    mesh,
    ax=None,
    show=False,
    figsize=None,
    color='k',
    linewidth=0.3,
    **kwargs
):
    ax = ax or plt.gca()
    ax.triplot(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'],
        color=color,
        linewidth=linewidth,
        **kwargs)
    return ax


def quadplot(
    msh_t,
    ax=None,
    facecolor="none",
    edgecolor="k",
    linewidth=0.3,
    **kwargs,
):
    ax = ax or plt.gca()
    if not hasattr(msh_t, 'quad4'):
        return ax
    if len(msh_t.quad4) > 0:
        pc = PolyCollection(
            msh_t.vert2['coord'][msh_t.quad4['index']],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs,
        )
        ax.add_collection(pc)
    return ax


def quadface(
    mesh,
    ax=None,
    show=False,
    figsize=None,
    extend='both',
    **kwargs
):
    # if len(self.quads) > 0:
    pc = PolyCollection(
        mesh.vert2['coord'][mesh.quad4['index']],
        # facecolor=facecolor,
        # edgecolor=edgecolor,
        linewidth=0.07
    )
    quad_value = np.mean(mesh.value.flatten()[mesh.quad4['index']], axis=1)
    pc.set_array(quad_value)
    ax = ax or plt.gca()
    ax.add_collection(pc)
    return ax


def msh_t_to_grd(msh: jigsaw_msh_t) -> Dict:

    src_crs = msh.crs if hasattr(msh, 'crs') else None
    coords = msh.vert2['coord']
    if src_crs is not None:
        EPSG_4326 = CRS.from_epsg(4326)
        if not src_crs.equals(EPSG_4326):
            transformer = Transformer.from_crs(
                src_crs, EPSG_4326, always_xy=True)
            coords = np.vstack(
                transformer.transform(coords[:, 0], coords[:, 1])).T

    desc = "EPSG:4326"
    nodes = {i + 1: [tuple(p.tolist()), v] for i, (p, v) in enumerate(zip(coords, -msh.value))}
    elements = {i + 1: v + 1 for i, v in enumerate(msh.tria3['index'])}
    offset = len(elements)
    elements.update({offset + i + 1: v + 1 for i, v in enumerate(msh.quad4['index'])})

    return {'description': desc,
            'nodes': nodes,
            'elements': elements}


def grd_to_msh_t(_grd: Dict) -> jigsaw_msh_t:

    msh = jigsaw_msh_t()
    msh.ndims = +2
    msh.mshID = 'euclidean-mesh'
    id_to_index = {node_id: index for index, node_id
                   in enumerate(_grd['nodes'].keys())}
    triangles = [list(map(lambda x: id_to_index[x], element)) for element
                 in _grd['elements'].values() if len(element) == 3]
    quads = [list(map(lambda x: id_to_index[x], element)) for element
             in _grd['elements'].values() if len(element) == 4]
    msh.vert2 = np.array([(coord, 0) for coord, _ in _grd['nodes'].values()],
                         dtype=jigsaw_msh_t.VERT2_t)
    msh.tria3 = np.array([(index, 0) for index in triangles],
                         dtype=jigsaw_msh_t.TRIA3_t)
    msh.quad4 = np.array([(index, 0) for index in quads],
                         dtype=jigsaw_msh_t.QUAD4_t)
    value = [value for _, value in _grd['nodes'].values()]
    msh.value = np.array(np.array(value).reshape((len(value), 1)),
                         dtype=jigsaw_msh_t.REALS_t)
    crs = _grd.get('crs')
    if crs is not None:
        msh.crs = CRS.from_user_input(crs)
    return msh


def msh_t_to_2dm(msh: jigsaw_msh_t):
    coords = msh.vert2['coord']
    src_crs = msh.crs if hasattr(msh, 'crs') else None
    if src_crs is not None:
        EPSG_4326 = CRS.from_epsg(4326)
        if not src_crs.equals(EPSG_4326):
            transformer = Transformer.from_crs(
                src_crs, EPSG_4326, always_xy=True)
            coords = np.vstack(
                transformer.transform(coords[:, 0], coords[:, 1])).T
    return {
            'ND': {i+1: (coord, msh.value[i, 0] if not
                         np.isnan(msh.value[i, 0]) else -99999)
                   for i, coord in enumerate(coords)},
            'E3T': {i+1: index+1 for i, index
                    in enumerate(msh.tria3['index'])},
            'E4Q': {i+1: index+1 for i, index
                    in enumerate(msh.quad4['index'])}
        }


def sms2dm_to_msh_t(_sms2dm: Dict) -> jigsaw_msh_t:
    msh = jigsaw_msh_t()
    msh.ndims = +2
    msh.mshID = 'euclidean-mesh'
    id_to_index = {node_id: index for index, node_id
                   in enumerate(_sms2dm['ND'].keys())}
    if 'E3T' in _sms2dm:
        triangles = [list(map(lambda x: id_to_index[x], element)) for element
                     in _sms2dm['E3T'].values()]
        msh.tria3 = np.array([(index, 0) for index in triangles],
                             dtype=jigsaw_msh_t.TRIA3_t)
    if 'E4Q' in _sms2dm:
        quads = [list(map(lambda x: id_to_index[x], element)) for element
                 in _sms2dm['E4Q'].values()]
        msh.quad4 = np.array([(index, 0) for index in quads],
                             dtype=jigsaw_msh_t.QUAD4_t)
    msh.vert2 = np.array([(coord, 0) for coord, _ in _sms2dm['ND'].values()],
                         dtype=jigsaw_msh_t.VERT2_t)
    value = [value for _, value in _sms2dm['ND'].values()]
    msh.value = np.array(np.array(value).reshape((len(value), 1)),
                         dtype=jigsaw_msh_t.REALS_t)
    crs = _sms2dm.get('crs')
    if crs is not None:
        msh.crs = CRS.from_user_input(crs)
    return msh


# def reproject_parallel_worker(coord, src_crs, dst_crs):
#     transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
#     return transformer.transform(coord[0], coord[1])


def reproject(
        mesh: jigsaw_msh_t,
        dst_crs: Union[str, CRS],
        # nprocs=None,
):
    src_crs = mesh.crs
    if not isinstance(dst_crs, CRS):
        dst_crs = CRS.from_user_input(dst_crs)
    if not isinstance(src_crs, CRS):
        # breakpoint()
        src_crs = CRS.from_user_input(src_crs)

    start = time()
    logger.debug(f'Begin transforming points from {mesh.crs} to {dst_crs}.')
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = transformer.transform(
        mesh.vert2['coord'][:, 0], mesh.vert2['coord'][:, 1])
    logger.debug(f'Transforming points took {time() - start}.')
    mesh.vert2['coord'][:] = np.vstack([x, y]).T
    mesh.crs = dst_crs


def interpolate(src: jigsaw_msh_t, dst: jigsaw_msh_t, **kwargs):
    if src.mshID == 'euclidean-grid' and dst.mshID == 'euclidean-mesh':
        interpolate_euclidean_grid_to_euclidean_mesh(src, dst, **kwargs)
    elif src.mshID == 'euclidean-mesh' and dst.mshID == 'euclidean-mesh':
        interpolate_euclidean_mesh_to_euclidean_mesh(src, dst, **kwargs)
    else:
        raise NotImplementedError(
            f'Not implemented type combination: source={src.mshID}, '
            f'dest={dst.mshID}')


def interpolate_euclidean_mesh_to_euclidean_mesh(
        src: jigsaw_msh_t,
        dst: jigsaw_msh_t,
        method='linear',
        fill_value=np.nan
):
    values = griddata(
        src.vert2['coord'],
        src.value.flatten(),
        dst.vert2['coord'],
        method=method,
        fill_value=fill_value
    )
    dst.value = np.array(
        values.reshape(len(values), 1), dtype=jigsaw_msh_t.REALS_t)


def interpolate_euclidean_grid_to_euclidean_mesh(
        src: jigsaw_msh_t,
        dst: jigsaw_msh_t,
        bbox=[None, None, None, None],
        kx=3,
        ky=3,
        s=0
):
    values = RectBivariateSpline(
        src.xgrid,
        src.ygrid,
        src.value.T,
        bbox=bbox,
        kx=kx,
        ky=ky,
        s=s
        ).ev(
        dst.vert2['coord'][:, 0],
        dst.vert2['coord'][:, 1])
    dst.value = np.array(
        values.reshape((values.size, 1)),
        dtype=jigsaw_msh_t.REALS_t)


def test_filter_polygons_nested():
    from shapely.geometry import Polygon
    outer_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    hole1 = Polygon([(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)])
    polygon2 = Polygon([(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)])
    hole2 = Polygon([(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)])
    polygon3 = Polygon([(4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5), (4.5, 4.5)])
    nested_polygon = Polygon(shell=outer_polygon.exterior.coords, holes=[hole1.exterior.coords, hole2.exterior.coords])
    polygons = [nested_polygon, outer_polygon, hole1, polygon2, hole2, polygon3]
    polygons = filter_polygons(polygons, CRS.from_epsg(3857))
    gpd.GeoDataFrame(geometry=polygons, crs=CRS.from_epsg(3857)).plot(ax=plt.gca(), alpha=0.3, edgecolor='k')
    plt.show(block=True)

