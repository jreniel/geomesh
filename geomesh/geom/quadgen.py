import hashlib
import logging
import math
import os
import tempfile
import typing
import warnings
from functools import cached_property, partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Union
from typing import Tuple

import centerline.exceptions
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import PythonCDT as cdt
import scipy.spatial
import shapely.errors
from centerline import exceptions
from centerline.geometry import Centerline
from inpoly import inpoly2
from jigsawpy import jigsaw_msh_t
from matplotlib.transforms import Bbox
from matplotlib.tri import Triangulation
from mpi4py.futures import MPICommExecutor
from numpy.linalg import norm
from pyproj import CRS, Transformer
from scipy.spatial import KDTree
from shapely import simplify
from shapely import equals_exact, ops, wkb
from shapely.geometry import (GeometryCollection, LinearRing, LineString,
                              MultiLineString, MultiPoint, MultiPolygon, Point,
                              Polygon, box, polygon)

from geomesh import utils

pd.options.mode.chained_assignment = 'raise'

if typing.TYPE_CHECKING:
    from geomesh.raster.raster import Raster

logger = logging.getLogger(__name__)


# --- We begin by applying some patches to the Centerline class to make it more useful for our purposes.

# -- patch to add a new exception that can someimes happen:
class InputGeometryEnvelopeIsPointError(exceptions.CenterlineError):
    default_message = "Input geometry envelope is a point."


exceptions.InputGeometryEnvelopeIsPointError = InputGeometryEnvelopeIsPointError


# patch centerline.Centerline.__init__ to allow for interpolation_distance to be None,
# which should use the original input geometry instead of forcing a resample
def _centerline_init(self, input_geometry, interpolation_distance=None, **attributes):
    self._input_geometry = input_geometry
    if interpolation_distance is None:
        self._interpolation_distance = None
    else:
        self._interpolation_distance = abs(interpolation_distance)

    if not self.input_geometry_is_valid():
        raise exceptions.InvalidInputTypeError

    if isinstance(self._input_geometry.envelope, Point):
        raise exceptions.InputGeometryEnvelopeIsPointError

    self._min_x, self._min_y = self._get_reduced_coordinates()
    self.assign_attributes_to_instance(attributes)

    self.geometry = MultiLineString(lines=self._construct_centerline())


Centerline.__init__ = _centerline_init


# also need to patch _get_interpolated_boundary to use the original input geometry
def _get_interpolated_boundary(self, boundary):
    if self._interpolation_distance is None:
        return [self._create_point_with_reduced_coordinates(x, y) for x, y in zip(boundary.xy[0], boundary.xy[1])]

    else:
        line = LineString(boundary)
        first_point = self._get_coordinates_of_first_point(line)
        last_point = self._get_coordinates_of_last_point(line)
        intermediate_points = self._get_coordinates_of_interpolated_points(
            line
        )
        return [first_point] + intermediate_points + [last_point]


Centerline._get_interpolated_boundary = _get_interpolated_boundary

# --- End module level patches for Centerline


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


def is_convex(coords):
    polygon = Polygon(coords)
    convex_hull = polygon.convex_hull
    return polygon.equals(convex_hull)


def get_quad_length_ratio(quad):
    coords = list(quad.coords[:-1])
    side_lengths = []
    for i in range(4):
        curr_point = np.array(coords[i % 4])
        next_point = np.array(coords[(i + 1) % 4])
        side_length = np.linalg.norm(curr_point - next_point)
        side_lengths.append(side_length)
    min_side_length = np.min(side_lengths)
    max_side_length = np.max(side_lengths)
    return max_side_length / min_side_length


def check_angle_cutoff(quad, min_cutoff_deg=None, max_cutoff_deg=None):
    coords = quad.coords
    for i in range(4):
        prev_point = np.array(coords[(i-2) % 4])
        curr_point = np.array(coords[(i-1) % 4])
        next_point = np.array(coords[i])
        prev_vector = curr_point - prev_point
        next_vector = next_point - curr_point
        angle = angle_between(prev_vector, next_vector)
        if min_cutoff_deg is not None and angle <= min_cutoff_deg:
            return False
        if max_cutoff_deg is not None and angle >= max_cutoff_deg:
            return False
    return True


def split_bad_quality_quads(
        quad,
        cutoff,
        ):

    coords = list(quad.coords[:-1])

    angles = []
    for i in range(4):
        prev_point = np.array(coords[(i-2) % 4])
        curr_point = np.array(coords[(i-1) % 4])
        next_point = np.array(coords[i])
        prev_vector = curr_point - prev_point
        next_vector = next_point - curr_point
        angle = angle_between(prev_vector, next_vector)
        # if not min_cutoff_deg <= angle <= max_cutoff_deg:
        #     return False
        angles.append(angle)
    min_angle = min(angles)
    max_angle = max(angles)
    if min_angle == 0 or max_angle == 0:
        return False
    return min_angle / max_angle >= cutoff


def compute_perpendicular_vector(point1, point2):
    vector = np.array(point2) - np.array(point1)
    perp_vector = np.array([-vector[1], vector[0]])
    return perp_vector / np.linalg.norm(perp_vector)


def compute_distances(point, patch):
    return Point(point).distance(patch.boundary)


def compute_new_points(point, distance, unit_vector):
    point_left = point - distance * unit_vector
    point_right = point + distance * unit_vector
    # point_left = np.around(point_left, 7)
    # point_right = np.around(point_right, 7)
    return Point(point_left), Point(point_right)


def compute_initial_points(p1, distance_p1, perp_vector):
    return compute_new_points(p1, distance_p1, perp_vector)


def interpolate_centerline(centerline, max_quad_length):
    # Replace this with the correct implementation.
    return LineString([
        centerline.interpolate(distance) for distance in np.arange(0, centerline.length, max_quad_length)
        ])


def extract_nodes(centerlines):
    if isinstance(centerlines, MultiLineString):
        return [Point(x, y) for linestring in centerlines for x, y in linestring.coords]
    elif isinstance(centerlines, LineString):
        return [Point(x, y) for x, y in centerlines.coords]


# def find_downstream_node(initial_node, centerline: LineString, length):
#     distance_along_line = centerline.project(initial_node)
#     return centerline.interpolate(distance_along_line + length)

#     # Ensure that the downstream_node is farther along the centerline than initial_node
#     downstream_distance = centerline.project(downstream_node)
#     if downstream_distance > distance_along_line + length:
#         # If downstream_node is not farther along the centerline, return the last coordinate
#         # return Point(centerline.coords[-1])
#         return None
#     else:
#         return downstream_node

# def subdivide_quadrilateral(ray_vector, linearring, npieces) -> List[LinearRing]:
#     if len(linearring.coords) != 5:
#         return []

#     coords = list(linearring.coords[:-1])

#     vectors = [np.array(coords[(i+1) % 4]) - np.array(coords[i]) for i in range(len(coords))]
#     vectors = [v/np.linalg.norm(v) if np.any(v) else v for v in vectors]

#     angles = [math.acos(np.clip(np.dot(v, ray_vector), -1.0, 1.0)) for v in vectors]
#     most_parallel_idx = np.argmin(angles)
#     opposite_idx = (most_parallel_idx + 2) % 4

#     def interpolate_points(p1, p2, npieces):
#         return [(p1[0] + i/npieces*(p2[0]-p1[0]), p1[1] + i/npieces*(p2[1]-p1[1])) for i in range(npieces+1)]

#     points_most_parallel_side = interpolate_points(coords[most_parallel_idx], coords[(most_parallel_idx+1) % 4], npieces)
#     points_opposite_side = interpolate_points(coords[opposite_idx], coords[(opposite_idx+1) % 4], npieces)

#     if most_parallel_idx % 2 != 0:
#         points_most_parallel_side = list(reversed(points_most_parallel_side))
#     if opposite_idx % 2 == 0:
#         points_opposite_side = list(reversed(points_opposite_side))

#     new_quads = [LinearRing(np.around([
#         points_most_parallel_side[i],
#         points_most_parallel_side[i+1],
#         points_opposite_side[i+1],
#         points_opposite_side[i]
#         ], 7)) for i in range(npieces)]

#     return new_quads

def subdivide_quadrilateral(p1_left: Point, p1_right: Point, linearring, npieces) -> List[LinearRing]:
    if len(linearring.coords) != 5:
        return []

    coords = list(linearring.coords[:-1])

    # Convert Points to coordinate tuples for comparison
    p1_left_coords = (p1_left.x, p1_left.y)
    p1_right_coords = (p1_right.x, p1_right.y)

    # Find the index of the side containing the p1_left and p1_right
    most_parallel_idx = None
    for i in range(4):
        if (coords[i] == p1_left_coords and coords[(i+1) % 4] == p1_right_coords) or \
           (coords[i] == p1_right_coords and coords[(i+1) % 4] == p1_left_coords):
            most_parallel_idx = i
            break

    if most_parallel_idx is None:
        return []

    opposite_idx = (most_parallel_idx + 2) % 4

    def interpolate_points(p1, p2, npieces):
        return [(p1[0] + i/npieces*(p2[0]-p1[0]), p1[1] + i/npieces*(p2[1]-p1[1])) for i in range(npieces+1)]

    points_most_parallel_side = interpolate_points(coords[most_parallel_idx], coords[(most_parallel_idx+1) % 4], npieces)
    points_opposite_side = interpolate_points(coords[opposite_idx], coords[(opposite_idx+1) % 4], npieces)

    if most_parallel_idx % 2 != 0:
        points_most_parallel_side = list(reversed(points_most_parallel_side))
    if opposite_idx % 2 == 0:
        points_opposite_side = list(reversed(points_opposite_side))

    new_quads = [LinearRing(np.around([
        points_most_parallel_side[i],
        points_most_parallel_side[i+1],
        points_opposite_side[i+1],
        points_opposite_side[i]
        ], 7)) for i in range(npieces)]

    return new_quads


def compute_bisector_or_perpendicular(p1: Union[np.ndarray, typing.Sequence[float]],
                                      p2: Union[np.ndarray, typing.Sequence[float]],
                                      p3: Union[np.ndarray, typing.Sequence[float]]) -> np.ndarray:
    try:
        return compute_bisector_vector(p1, p2, p3)
    except ValueError:  # handle collinearity
        return compute_perpendicular(p1, p2, p3)


def compute_perpendicular(p1: Union[np.ndarray, typing.Sequence[float]],
                          p2: Union[np.ndarray, typing.Sequence[float]],
                          p3: Union[np.ndarray, typing.Sequence[float]]) -> np.ndarray:
    v1 = p1 - p2
    v2 = p3 - p2
    # Check for direction to make sure the vector is always oriented the same way
    direction = np.sign(v2[0]*v1[1] - v2[1]*v1[0])
    # Return the perpendicular vector
    return direction * np.array([v1[1], -v1[0]]) / np.linalg.norm(v1)


def compute_bisector_vector(p1: Union[np.ndarray, typing.Sequence[float]],
                            p2: Union[np.ndarray, typing.Sequence[float]],
                            p3: Union[np.ndarray, typing.Sequence[float]]) -> np.ndarray:
    # Compute vectors
    v1 = p1 - p2
    v2 = p3 - p2

    # Check if vectors are zero
    if np.all(v1 == 0) or np.all(v2 == 0):
        raise ValueError('Input points should not be identical')

    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm != 0:
        v1 = v1 / v1_norm
    if v2_norm != 0:
        v2 = v2 / v2_norm

    # Compute bisector and check if vectors are opposite
    bisector = v1 + v2
    bisector_norm = np.linalg.norm(bisector)
    if bisector_norm == 0:
        raise ValueError('Input points should not be collinear')
    else:
        bisector = bisector / bisector_norm
    return bisector


def sort_points_clockwise(points):
    # Get the centroid of the points
    centroid = np.mean(points, axis=0)

    # Create a new array that contains the angle of each point from the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points by these angles
    sorted_points = points[np.argsort(angles)]

    return sorted_points

def build_last_base_quad(
        initial_point,
        downstream_point,
        bounding_ring,
        previous_base_quad,
        previous_tail_points,
        cross_distance_factor
        ) -> typing.Tuple[LinearRing, np.ndarray]:
    p0 = np.array(initial_point.coords).flatten()
    p1 = np.array(downstream_point.coords).flatten()
    if previous_base_quad is None:
        normal_vector = np.subtract(p1, p0)
        normal_vector = [-normal_vector[1], normal_vector[0]]
        normal_vector /= np.linalg.norm(normal_vector)
        distance_p0 = cross_distance_factor * compute_distances(p0, bounding_ring)
        p0_left, p0_right = compute_new_points(p0, distance_p0, normal_vector)
        p0_left = np.array(p0_left.coords).flatten()
        p0_right = np.array(p0_right.coords).flatten()
    else:
        # coords = list(previous_base_quad.coords)
        # p0_left = None
        # for i in range(len(coords) - 1):
        #     edge = LineString([coords[i], coords[i+1]])
        #     if edge.intersects(initial_point.buffer(np.finfo(np.float32).eps)):
        #         p0_left = np.array(coords[i])
        #         p0_right = np.array(coords[i+1])
        #         break
        # if p0_left is None:
        #     return LinearRing(), np.array([np.nan, np.nan])
        p0_left, p0_right = previous_tail_points
        normal_vector = np.subtract(np.array(p0_right.coords), np.array(p0_left.coords))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message="invalid value encountered in divide")
            normal_vector /= np.linalg.norm(normal_vector)
    distance_p1 = cross_distance_factor * compute_distances(p1, bounding_ring)
    p1_left, p1_right = compute_new_points(p1, distance_p1, normal_vector)
    try:
        points = np.array([p0_left, p0_right, np.array(p1_right.coords).flatten(), np.array(p1_left.coords).flatten()]).reshape((4, 2))
    except Exception:
        return LinearRing(), (Point(), Point())
    this_quad = LinearRing(points)
    return this_quad, (p1_left, p1_right)


def compute_last_perpendicular(p0, p1):
    v1 = p1 - p0
    direction = v1 / np.linalg.norm(v1)
    return direction * np.array([v1[1], -v1[0]]) / np.linalg.norm(v1)


def build_base_quad(
        initial_point,
        downstream_point,
        next_point,
        bounding_ring,
        previous_base_quad=None,
        previous_tail_points=None,
        cross_distance_factor: float = 0.95
        ) -> typing.Tuple[LinearRing, np.ndarray]:
    p0 = np.array(initial_point.coords).flatten()
    p1 = np.array(downstream_point.coords).flatten()
    p2 = np.array(next_point.coords).flatten()
    normal_vector = compute_bisector_or_perpendicular(p0, p1, p2)
    if previous_base_quad is not None:
        p0_left, p0_right = previous_tail_points
        # coords = list(previous_base_quad.coords)
        # p0_left = None
        # for i in range(len(coords) - 1):
        #     edge = LineString([coords[i], coords[i+1]])
        #     if edge.intersects(initial_point.buffer(np.finfo(np.float32).eps)):
        #         p0_left = Point(coords[i])
        #         p0_right = Point(coords[i+1])
        #         break
        # if p0_left is None:
        #     distance_p0 = cross_distance_factor * compute_distances(p0, bounding_ring)
        #     p0_left, p0_right = compute_new_points(p0, distance_p0, normal_vector)
    else:
        distance_p0 = cross_distance_factor * compute_distances(p0, bounding_ring)
        p0_left, p0_right = compute_new_points(p0, distance_p0, normal_vector)
    distance_p1 = cross_distance_factor * compute_distances(p1, bounding_ring)
    p1_left, p1_right = compute_new_points(p1, distance_p1, normal_vector)
    if previous_base_quad is not None:
        if Polygon(previous_base_quad).contains(p1_left) or Polygon(previous_base_quad).contains(p1_right):
            normal_vector = compute_perpendicular(p0, p1, p2)
            p1_left, p1_right = compute_new_points(p1, distance_p1, normal_vector)
    points = np.array([np.array(p.coords)[0] for p in [p0_left, p0_right, p1_left, p1_right]])
    this_quad = [Point(p) for p in sort_points_clockwise(points)]
    this_base_quad = LinearRing([np.array(p.coords[0]) for p in this_quad])
    return this_base_quad, (p1_left, p1_right)


def get_perpedicular_segment_distances(quads_coll):
    perpendicular_segments_distances = []
    for this_quad, (p1_left, p1_right) in quads_coll:
        perpendicular_segments_distances.append(p1_left.distance(p1_right))
    return perpendicular_segments_distances


def is_collapsed(coords):
    unique_rows = np.unique(coords, axis=0)
    if len(unique_rows) != 4:
        return True
    return False


def split_base_quads(
        quads_coll,
        min_quad_width=None,
        max_quad_width=None,
        min_cross_section_node_count=None
        ):
    """
    Splits or subdivides quadrilaterals based on specified width and node count constraints.

    Parameters:
    - quads_coll (list of tuples): A list of tuples, each containing a quadrilateral and its normal vector.
    - min_quad_width (float, optional): The minimum allowable width for any quadrilateral. Quadrilaterals smaller than this will not be subdivided.
    - max_quad_width (float, optional): The maximum allowable width for any quadrilateral. Quadrilaterals larger than this will be subdivided.
    - min_cross_section_node_count (int, optional): The minimum number of nodes that should exist along the cross-section of the quadrilaterals.

    Returns:
    - list of tuples: A new list of tuples, each containing a subdivided quadrilateral and its normal vector.

    Notes:
    - The function uses the perpendicular distances of the quadrilaterals to make decisions about subdivision.
    - If min_quad_width, max_quad_width, and min_cross_section_node_count are all None or zero, the function returns the input quadrilaterals as-is.
    """
    # Calculate perpendicular distances (assuming this function exists)
    perp_dists = get_perpedicular_segment_distances(quads_coll)
    max_perp_seg_dist = np.max(perp_dists)
    min_perp_seg_dist = np.min(perp_dists)

    # Set max_quad_width and min_quad_width to their defaults if not provided
    max_quad_width = max_quad_width or max_perp_seg_dist
    min_quad_width = min_quad_width or min_perp_seg_dist

    # Initialize new quadrilateral collection
    new_quads_coll = []

    # Handle cases based on max_perp_seg_dist
    if max_perp_seg_dist > max_quad_width:
        npieces = int(np.ceil(max_perp_seg_dist / max_quad_width))

        # Enforce minimum number of cross-section nodes
        if min_cross_section_node_count is not None:
            if npieces < min_cross_section_node_count - 1:
                npieces = min_cross_section_node_count - 1

    elif max_perp_seg_dist < min_quad_width:
        # return [([this_quad], this_normal_vector) for this_quad, this_normal_vector in quads_coll]
        return []

    else:
        # If you reach this point, then min_quad_width <= max_perp_seg_dist <= max_quad_width
        # This could mean that you might want to do nothing, or enforce a minimum number of pieces
        if min_cross_section_node_count is not None:
            npieces = min_cross_section_node_count - 1
        else:
            # If min_cross_section_node_count is None, you might want to return the quads as-is
            return [([this_quad], this_normal_vector) for this_quad, this_normal_vector in quads_coll]

    # Subdivide the quadrilaterals
    for i, (this_quad, (p1_left, p1_right)) in enumerate(quads_coll):
        quads_row = subdivide_quadrilateral(p1_left, p1_right, this_quad, npieces)
        new_quads_coll.append((quads_row, (p1_left, p1_right)))

    return new_quads_coll


def compute_quads_from_centerline(
        bounding_ring: LinearRing,
        centerline: LineString,
        max_quad_length: float,
        min_quad_length=None,
        shrinkage_factor=0.9,
        cross_distance_factor=0.9,
        min_cross_section_node_count=4,
        min_quad_width=None,
        max_quad_width=None,
        ) -> List[typing.Tuple[LinearRing, np.ndarray]]:
    (
        bounding_ring,
        centerline,
        max_quad_length,
        min_cross_section_node_count
    ) = _check_compute_quads_from_centerline_args(
            bounding_ring,
            centerline,
            max_quad_length,
            # ltc_ratio,
            min_cross_section_node_count
            )
    # min_quad_length = min_quad_length or max_quad_length / 10.
    min_quad_length = min_quad_length or 0.
    # min_quad_length = min_quad_length or max_quad_length
    # min_quad_width = min_quad_width or min_quad_length
    # max_quad_width = max_quad_width or max_quad_length / 10.
    # max_quad_width = max_quad_width or max_quad_length

    # if 2.*centerline.length < min_quad_length:
    #     return []

    # if not bounding_ring.is_valid:
    #     from shapely import make_valid
    #     bounding_ring = make_valid(bounding_ring)
    initial_p0 = Point(centerline.coords[0])
    p0 = Point(centerline.coords[0])
    p1 = centerline.interpolate(max_quad_length)
    if p1 == Point(centerline.coords[-1]):
        p2 = p1
    else:
        distance_along_line = centerline.project(p1)
        # distance_along_line = initial_p0.distance(p1)
        p2 = centerline.interpolate(distance_along_line + max_quad_length)
    quads_coll = []
    current_quad_length = max_quad_length
    base_quad = None
    previous_tail_points = None
    # logger.debug(f"initial conditions: {p0=} {p1=} {p2=}")
    cnt = 0
    while not equals_exact(p1, p2, tolerance=np.finfo(np.float32).eps):
        new_base_quad, new_tail_points = build_base_quad(p0, p1, p2, bounding_ring, base_quad, previous_tail_points, cross_distance_factor)
        contained = bounding_ring.contains(new_base_quad)
        if not contained:
            if current_quad_length <= min_quad_length:
                break
            current_quad_length *= shrinkage_factor
            if current_quad_length < min_quad_length:
                current_quad_length = min_quad_length
        else:
            quads_coll.append((new_base_quad, new_tail_points))
            base_quad = new_base_quad
            previous_tail_points = new_tail_points
            current_quad_length = max_quad_length
            p0 = p1
        distance_along_line = centerline.project(p0)
        # distance_along_line = initial_p0.distance(p0)
        p1 = centerline.interpolate(distance_along_line + current_quad_length)
        if equals_exact(p1, Point(centerline.coords[-1]), tolerance=np.finfo(np.float32).eps):
            break
        else:
            distance_along_line = centerline.project(p1)
            # distance_along_line = initial_p0.distance(p1)
            p2 = centerline.interpolate(distance_along_line + current_quad_length)
        cnt += 1
        # logger.debug(f"iter {cnt=}: {p0=} {p1=} {p2=}")
        if equals_exact(p2, Point(centerline.coords[0]), tolerance=np.finfo(np.float32).eps):
            break
        if cnt > 1000:
            # TODO: This is very heuristic, me no like.
            return []



    if base_quad is not None:
        if p0 != p1 and p1 == p2:
            d = p0.distance(p1)
            if np.around(max_quad_length, 0) >= np.around(d, 0) >= np.around(min_quad_length, 0):
            # if max_quad_length >= d >= min_quad_length:
                quads_coll.append(
                        build_last_base_quad(
                            p0,
                            p1,
                            bounding_ring,
                            base_quad,
                            previous_tail_points,
                            cross_distance_factor
                            )
                        )
            # elif max_quad_length > d:
            #     logger.debug(
            else:
                # rare but happens,
                # logger.debug(f'Condition A-1: {type(base_quad)=} {p0=} {p1=} {p2=} {max_quad_length=} >= {d=} >= {min_quad_length=} is False so passed')
                pass
                # raise NotImplementedError(' A-1 Presumbaly unreachable: {p0.distance(p1)}')
        else:
            # rare, but it happens (all points are presumably different)
            # logger.debug(f'Condition A-2: {type(base_quad)=} {p0=} {p1=} {p2=} All must be different so we added two quad rows')
            d01 = p0.distance(p1)
            d12 = p1.distance(p2)
            d02 = p0.distance(p2)
            if (d01 < min_quad_length or d12 < min_quad_length) and min_quad_length <= d02 <= max_quad_length:
                quads_coll.append(
                        build_last_base_quad(
                            p0,
                            p2,
                            bounding_ring,
                            base_quad,
                            previous_tail_points,
                            cross_distance_factor
                            )
                        )
            # elif min_quad_length <= d02 <= np.around(max_quad_length, 0):
            else:
                pass
                # raise NotImplementedError(f'Condition A-2: Presumably unreachable: {d01=} {d12=} {d02=} {min_quad_length=} {max_quad_length=} {base_quad}')

            # d = p0.distance(p1)
            # if np.around(max_quad_length, 0) >= np.around(d, 0) >= np.around(min_quad_length, 0):
            #     quads_coll.append(
            #             build_last_base_quad(
            #                 p0,
            #                 p1,
            #                 bounding_ring,
            #                 base_quad,
            #                 cross_distance_factor
            #                 )
            #             )
            # else:
            #     # normally means d < min_quad_length so we pass
            #     # TODO: Some times the difference is marrginal aka. ~5 meters so maybe
            #     # we wanto to check proximity instead of rounding
            #     pass
            #     # raise NotImplementedError(f'A-2-1 Presumbaly unreachable {d=}')
            # d = p1.distance(p2)
            # if np.around(max_quad_length, 0) >= np.around(d, 0) >= np.around(min_quad_length, 0):
            #     quads_coll.append(
            #             build_last_base_quad(
            #                 p1,
            #                 p2,
            #                 bounding_ring,
            #                 base_quad,
            #                 cross_distance_factor
            #                 )
            #             )
            # else:
            #     # TODO: Some times the difference is marrginal aka. ~5 meters so maybe
            #     # we wanto to check proximity instead of rounding
            #     pass
            #     # raise NotImplementedError(f'A-2-2 Presumbaly unreachable {d=}')

    else:  # base_quad is None
        if p0 == p1 == p2:
            # rare, but it happens
            # logger.debug(f'Condition B-1: {type(base_quad)=} {p0=} {p1=} {p2=} all equal and base_quad is None so we passed')
            pass

        elif p0 != p1 and p1 == p2:
            # This condition is the second most common
            # logger.debug(f'Condition B-2-I: {type(base_quad)=} {p0=} {p1=} {p2=} (base_quad is None and p0!=p1==p2) we built last quad...')
            d = p0.distance(p1)
            if np.around(max_quad_length, 0) >= np.around(p0.distance(p1), 0) >= np.around(min_quad_length, 0):
                quads_coll.append(
                        build_last_base_quad(
                            p0,
                            p1,
                            bounding_ring,
                            base_quad,
                            previous_tail_points,
                            cross_distance_factor
                            )
                        )
            else:
                # usually means that d < min_quad_length so we pass
                pass

        else:
            # rare, but happens. all of them are presumed different in this scenario. Can lead to narrow quads.
            # logger.debug(f'Condition B-2-II: {type(base_quad)=} {p0=} {p1=} {p2=} (base_quad is None and p0!=p1==p2 IS NOT TRUE) we passed...')
            d01 = p0.distance(p1)
            d12 = p1.distance(p2)
            d02 = p0.distance(p2)
            if (d01 < min_quad_length or d12 < min_quad_length) and min_quad_length <= d02 <= max_quad_length:
                # if max_quad_length >= p0.distance(p1) >= min_quad_length:
                new_base_quad, new_normal_vector = build_base_quad(
                        p0,
                        p1,
                        p2,
                        bounding_ring,
                        base_quad,
                        cross_distance_factor
                        )
                quads_coll.append((new_base_quad, new_normal_vector))
            else:
                raise NotImplementedError(f'Condition B-2-II-1: Presumably unreachable: {d01=} {d12=} {d02=} {min_quad_length=} {max_quad_length=}')
            # if max_quad_length >= p1.distance(p2) >= min_quad_length:
            #     new_base_quad, new_normal_vector = build_last_base_quad(
            #             p1,
            #             p2,
            #             bounding_ring,
            #             new_base_quad,
            #             cross_distance_factor
            #             )
            #     quads_coll.append((new_base_quad, new_normal_vector))
            # else:
            #     raise NotImplementedError('B-2-II-2 Presumbaly unreachable')
    # filter out empty
    quads_coll = [
            (
                base_ring,
                array
            )
            for base_ring, array in quads_coll if not base_ring.is_empty
        ]
    # filter out concave
    quads_coll = [
            (
                base_ring,
                array
            )
            for base_ring, array in quads_coll if is_convex(base_ring.coords)
        ]
    # filter out collpased
    quads_coll = [
            (
                base_ring,
                array
            )
            for base_ring, array in quads_coll if not is_collapsed(base_ring.coords)
        ]

    if len(quads_coll) > 0:
        quads_coll = split_base_quads(quads_coll, min_quad_width, max_quad_width, min_cross_section_node_count)
        quads_coll = [
                (new_quads_row, normal_vector)
                for quad_row, normal_vector in quads_coll
                for new_quads_row in [[quad for quad in quad_row if is_convex(quad.coords)]]
                if len(new_quads_row) > 0
            ]

        # quads_coll = [
        #         (new_quads_row, normal_vector)
        #         for quad_row, normal_vector in quads_coll
        #         for new_quads_row in [[quad for quad in quad_row if check_quad_quality(
        #             quad,
        #             split_quad_cutoff=None,
        #             # side_length_ratio=20.,
        #             )]]
        #         if len(new_quads_row) > 0
        #     ]

        quads_coll = [
                (new_quads_row, normal_vector)
                for quad_row, normal_vector in quads_coll
                for new_quads_row in [[quad for quad in quad_row if not is_collapsed(quad.coords)]]
                if len(new_quads_row) > 0
            ]
        quads_coll = [
            (new_quads_row, normal_vector)
            for quad_row, normal_vector in quads_coll
            for new_quads_row in [
                [quad for quad in quad_row if all(length >= 1. for length in get_side_lengths(Polygon(quad)))]
            ]
            if len(new_quads_row) > 0
        ]
        # quads_coll = [
        #         (new_quads_row, normal_vector)
        #         for quad_row, normal_vector in quads_coll
        #         for new_quads_row in [[quad for quad in quad_row if check_angle_cutoff(
        #             quad,
        #             min_cutoff_deg=90-45,
        #             max_cutoff_deg=90+45,
        #             )]]
        #         if len(new_quads_row) > 0
        #     ]
    return quads_coll


def _check_compute_quads_from_centerline_args(
        bounding_poly: Polygon,
        centerlines: Union[LineString, MultiLineString],
        max_quad_length: float,
        # ltc_ratio,
        cross_section_node_count: int
        ):
    if not isinstance(bounding_poly, Polygon):
        raise TypeError(f"Expected bounding_poly to be a Polygon but got {type(bounding_poly)}")
    if not isinstance(centerlines, LineString):
        raise TypeError(
                f"Expected centerlines to be of type LineString but got {type(centerlines)}"
                )
    if float(max_quad_length) <= 0:
        raise ValueError(f"Expected max_quad_length to be a float > 0, got {max_quad_length=}")
    # if float(ltc_ratio) <= 0:
    #     raise ValueError(f"Expected longshore to cross-shore ratio to be a float > 0, got {ltc_ratio=}")
    if int(cross_section_node_count) < 4:
        raise ValueError(f"Expected cross_section_node_count to be an int >= 4, got {cross_section_node_count=}")
    # if not bounding_poly.is_valid:
    #     raise ValueError(
    #             f'compute_quads_from_centerline received an invalid polygon:\n{explain_validity(bounding_poly)}'
    #             )
    # if not centerlines.is_valid:
    #     raise ValueError(
    #             f'compute_quads_from_centerline received an invalid polygon:\n{explain_validity(centerlines)}'
    #             )
    # if not centerlines.within(bounding_poly):
    #     raise ValueError('Centerlines are not entirely contained within the given polygon.')
    # return bounding_poly, centerlines, float(ltc_ratio), int(cross_section_node_count)
    return bounding_poly, centerlines, float(max_quad_length), int(cross_section_node_count)


def resample_polygon(polygon, segment_length):
    exterior = polygon.exterior
    interiors = polygon.interiors

    # Resample the exterior linear ring
    resampled_exterior = resample_linear_ring(exterior, segment_length)
    if resampled_exterior is None:
        return Polygon()

    # Resample each interior linear ring
    resampled_interiors = [resample_linear_ring(ring, segment_length) for ring in interiors]
    resampled_interiors = [ring for ring in resampled_interiors if not ring.is_empty]
    return Polygon(shell=resampled_exterior, holes=resampled_interiors)


def resample_multipolygon(multipolygon, segment_length):
    resampled_polygons = [resample_polygon(polygon, segment_length) for polygon in multipolygon.geoms]
    resampled_polygons = [polygon for polygon in resampled_polygons if not polygon.is_empty]
    return MultiPolygon(resampled_polygons)


def resample_linear_ring(linear_ring, segment_length):
    total_length = linear_ring.length
    num_segments = int(np.ceil(total_length / segment_length))
    resampled_points = [
        linear_ring.interpolate(i * total_length / num_segments)
        for i in range(num_segments)
    ]
    try:
        return LinearRing([point.coords[0] for point in resampled_points])
    except ValueError as err:
        if str(err) == "A linearring requires at least 4 coordinates.":
            return LinearRing()
        raise err


def get_centerlines(patch, centerline_kwargs) -> MultiLineString:
    try:
        centerlines = Centerline(patch, **centerline_kwargs)
    except centerline.exceptions.InputGeometryEnvelopeIsPointError:
        return MultiLineString()
    except centerline.exceptions.TooFewRidgesError:
        # we could recursively increase interpolation_distance but that may be slow and an overkill
        # plus requires implementing a stop condition, but might be necessary for capturing thin areas.
        return MultiLineString()
    except shapely.errors.GEOSException:
        return MultiLineString()
    except scipy.spatial._qhull.QhullError:
        return MultiLineString()
    centerlines.geometry = ops.linemerge(sorted(centerlines.geometry.geoms, key=lambda geom: geom.length, reverse=True))
    if isinstance(centerlines.geometry, LineString):
        centerlines.geometry = MultiLineString([centerlines.geometry])
    return MultiLineString(list(sorted(centerlines.geometry.geoms, key=lambda geom: geom.length, reverse=True)))


def filter_linestrings(linestrings: List[LineString], min_branch_length=None):
    if isinstance(linestrings, GeometryCollection) and linestrings.is_empty:
        return []
    gdf = gpd.GeoDataFrame(geometry=linestrings)
    G = nx.Graph()

    for idx, row in gdf.iterrows():
        start_point = Point(row['geometry'].coords[0])
        end_point = Point(row['geometry'].coords[-1])
        G.add_edge(start_point, end_point, geometry=row['geometry'], length=row['geometry'].length)

    # Identify nodes of degree 1 (branch tips)
    branch_tips = [node for node, degree in G.degree() if degree == 1]

    # Identify branches: edges where one node is a branch tip and the other node is of degree 3
    branches = [(u, v) for u, v in G.edges() if ((u in branch_tips and G.degree(v) == 3)
                or (v in branch_tips and G.degree(u) == 3))]

    # If min_branch_length is provided, filter out branches that are shorter than this length
    if min_branch_length is not None:
        branches = [(u, v) for u, v in branches if G.edges[u, v]['length'] < min_branch_length]

    # Check if there are any branches to remove
    if branches:
        # Remove branches from the graph
        G.remove_edges_from(branches)

        # Extract the remaining linestrings
        linestrings = [G.edges[edge]['geometry'] for edge in G.edges]

        # Apply linemerge to the linestrings
        linestrings = ops.linemerge(linestrings)

        # If the type of the result is MultiLineString, convert it to a list of LineStrings
        if isinstance(linestrings, MultiLineString):
            linestrings = list(linestrings.geoms)
        if isinstance(linestrings, LineString):
            linestrings = [linestrings]
        # Recurse the function with the new set of linestrings
        return filter_linestrings(linestrings, min_branch_length)
    else:
        # If there are no more branches to remove, return the final set of linestrings
        linestrings = ops.linemerge(linestrings)
        if isinstance(linestrings, MultiLineString):
            linestrings = list(linestrings.geoms)
        return linestrings


# def match_centerlines_to_patches(centerlines: List[LineString], final_patches: List[Polygon]):
#     # Create GeoDataFrames
#     gdf_centerlines = gpd.GeoDataFrame(centerlines, columns=['geometry'])
#     gdf_patches = gpd.GeoDataFrame(final_patches, columns=['geometry'])

#     # Ensure that the GeoDataFrames have the same CRS (if not already)
#     # gdf_centerlines.set_crs(gdf_patches.crs, inplace=True)

#     # Perform the spatial join
#     joined = gpd.sjoin(gdf_centerlines, gdf_patches, how="inner", predicate='within')

#     # Extract pairs of centerlines and patches
#     matched_centerlines_and_patches = list(zip(joined['geometry_left'], joined['geometry_right']))

#     return matched_centerlines_and_patches

def match_centerlines_to_patches(centerlines: List[LineString], final_patches: List[Polygon], local_crs):
    # Create GeoDataFrames
    gdf_centerlines = gpd.GeoDataFrame(centerlines, columns=['geometry'], crs=local_crs)
    gdf_patches = gpd.GeoDataFrame(final_patches, columns=['geometry'], crs=local_crs)

    # Ensure that the GeoDataFrames have the same CRS (if not already)
    gdf_centerlines.set_crs(gdf_patches.crs, inplace=True)

    # Perform the spatial join
    joined = gpd.sjoin(gdf_centerlines, gdf_patches, how="inner", predicate='intersects')

    # Extract pairs of centerlines and patches
    matched_centerlines_and_patches = [(centerline, gdf_patches.loc[idx_right].geometry) for centerline, idx_right in zip(joined.geometry, joined.index_right)]

    # Get the patches containing a centerline
    # final_patches = gdf_patches.loc[joined.index_right.unique()]

    return matched_centerlines_and_patches

def build_thalweg_multipolygon_from_window(
        xval,
        yval,
        zvals,
        window,
        raster_crs,
        pad_width: int = 0,
        lower_bound=None,
        upper_bound=None,
        geom_mask=None,
        ):
    pad_width = pad_width or 0
    from geomesh.geom.raster import get_multipolygon_from_axes
    pad_width_2d = ((pad_width, pad_width), (pad_width, pad_width))
    # Compute the step size at the edges
    xval_diff_before = xval[1] - xval[0]
    xval_diff_after = xval[-1] - xval[-2]
    yval_diff_before = yval[1] - yval[0]
    yval_diff_after = yval[-1] - yval[-2]

    # Generate padding for xval and yval
    xval_pad_before = xval[0] - np.array(range(1, pad_width+1)) * xval_diff_before
    xval_pad_after = xval[-1] + np.array(range(1, pad_width+1)) * xval_diff_after
    yval_pad_before = yval[0] - np.array(range(1, pad_width+1)) * yval_diff_before
    yval_pad_after = yval[-1] + np.array(range(1, pad_width+1)) * yval_diff_after

    # Concatenate the padding and the original arrays
    xval = np.concatenate([xval_pad_before[::-1], xval, xval_pad_after])
    yval = np.concatenate([yval_pad_before[::-1], yval, yval_pad_after])

    # Pad zvals as before
    zvals = np.pad(zvals, pad_width_2d, mode='edge')
    if not np.ma.is_masked(zvals):
        zvals = np.ma.masked_array(zvals)

    zvals[zvals > 0.] = np.nan
    if raster_crs.is_geographic:
        logger.debug(
            "CRS is geographic, transforming points to local projection."
        )
        local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m "\
            f"+lat_0={np.median(yval)} +lon_0={np.median(xval)}"
        local_crs = CRS.from_user_input(local_azimuthal_projection)
        transformer = Transformer.from_crs(raster_crs, local_crs, always_xy=True)
        geographic_to_local = transformer.transform
        x0, x1 = np.min(xval), np.max(xval)
        y0, y1 = np.min(yval), np.max(yval)
        (x0, x1), (y0, y1) = geographic_to_local([x0, x1], [y0, y1])
        dx = np.diff(np.linspace(x0, x1, int(window.width)))[0]
        dy = np.diff(np.linspace(y0, y1, int(window.height)))[0]
    else:
        dx = np.mean(np.diff(xval))
        dy = np.mean(np.diff(yval))
    dzdx, dzdy = np.gradient(zvals, dx, dy)
    mask = np.logical_and(np.abs(dzdx) < 0.1, np.abs(dzdy) < 0.1)
    bathy = zvals.copy()
    non_saddle_values = zvals.copy()
    non_saddle_values[mask] = np.nan
    bathy[~mask] = np.nan
    bathy[bathy > np.nanmean(non_saddle_values)] = np.nan
    binary_image = zvals.copy()
    binary_image[~np.isnan(bathy)] = 1
    binary_image[np.isnan(bathy)] = 0
    if lower_bound is not None:
        binary_image[bathy < lower_bound] = 0
    if upper_bound is not None:
        binary_image[bathy > upper_bound] = 0

    from scipy.ndimage import binary_dilation, binary_erosion
    binary_image = binary_erosion(binary_dilation(binary_image))
    # TODO: Might use binary erosion/dilation to cleanup tiny pieces
    # verify
    # plt.contourf(xval, yval, binary_image)
    # plt.show()

    # if not np.any(zvals.mask):
    #     if zmin <= np.min(zvals) and zmax >= np.max(zvals):
    #         return MultiPolygon([box(np.min(xval), np.min(yval), np.max(xval), np.max(yval))])
    #     elif zmax < np.min(zvals) or zmin > np.max(zvals):
    #         return
    plt.ioff()
    original_backend = plt.get_backend()
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    multipolygon = get_multipolygon_from_axes(ax.contourf(xval, yval, binary_image, levels=[0.5, 1]))
    plt.close(fig)
    plt.switch_backend(original_backend)
    plt.ion()

    if geom_mask is not None:
        # multipolygon = multipolygon.difference(geom_mask)
        multipolygon = geom_mask.difference(multipolygon)

    if isinstance(multipolygon, Polygon):
        multipolygon = MultiPolygon([multipolygon])

    return multipolygon

def build_multipolygon_from_window(xval, yval, zvals, zmin, zmax, pad_width: int = 20):
    from geomesh.geom.raster import get_multipolygon_from_axes
    pad_width_2d = ((pad_width, pad_width), (pad_width, pad_width))
    # Compute the step size at the edges
    xval_diff_before = xval[1] - xval[0]
    xval_diff_after = xval[-1] - xval[-2]
    yval_diff_before = yval[1] - yval[0]
    yval_diff_after = yval[-1] - yval[-2]

    # Generate padding for xval and yval
    xval_pad_before = xval[0] - np.array(range(1, pad_width+1)) * xval_diff_before
    xval_pad_after = xval[-1] + np.array(range(1, pad_width+1)) * xval_diff_after
    yval_pad_before = yval[0] - np.array(range(1, pad_width+1)) * yval_diff_before
    yval_pad_after = yval[-1] + np.array(range(1, pad_width+1)) * yval_diff_after

    # Concatenate the padding and the original arrays
    xval = np.concatenate([xval_pad_before[::-1], xval, xval_pad_after])
    yval = np.concatenate([yval_pad_before[::-1], yval, yval_pad_after])

    # Pad zvals as before
    zvals = np.pad(zvals, pad_width_2d, mode='edge')
    if zmin is None:
        zmin = -1.e16
    if zmax is None:
        zmax = 1.e16
    if not np.ma.is_masked(zvals):
        zvals = np.ma.masked_array(zvals)
    if not np.any(zvals.mask):
        if zmin <= np.min(zvals) and zmax >= np.max(zvals):
            return MultiPolygon([box(np.min(xval), np.min(yval), np.max(xval), np.max(yval))])
        elif zmax < np.min(zvals) or zmin > np.max(zvals):
            return

    plt.ioff()
    original_backend = plt.get_backend()
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    multipolygon = get_multipolygon_from_axes(ax.contourf(xval, yval, zvals, levels=[zmin, zmax]))
    plt.close(fig)
    plt.switch_backend(original_backend)
    plt.ion()


    if isinstance(multipolygon, Polygon):
        multipolygon = MultiPolygon([multipolygon])

    return multipolygon

def get_thalwegs_multipolygon_for_raster(
        raster,
        raster_opts=None,
        window=None,
        pad_width=0,
        lower_bound=None,
        upper_bound=None,
        threshold_size=None,
        ):
    from geomesh import Geom, Raster
    from geomesh.cli.raster_opts import get_raster_from_opts
    if not isinstance(raster, Raster):
        raster = get_raster_from_opts(
                raster_path=raster,
                window=window,
                **raster_opts
                )

    if pad_width is None:
        geom_mp = Geom(raster, zmax=0.).get_multipolygon()
    else:
        window_multipolygon = []
        for xval, yval, zvals in raster:
            zvals = zvals[0, :]
            window_multipolygon.append(
                    build_multipolygon_from_window(xval, yval, zvals, None, 0., pad_width=pad_width)
                    )
        geom_mp = ops.unary_union(window_multipolygon)
    if geom_mp is not None and threshold_size is not None:
        buffered_geom = geom_mp.buffer(-threshold_size).buffer(threshold_size)
    else:
        buffered_geom = None
    window_multipolygon = []
    for xval, yval, zvals in raster:
        zvals = zvals[0, :]
        window_multipolygon.append(
                build_thalweg_multipolygon_from_window(
                    xval,
                    yval,
                    zvals,
                    raster.window,
                    raster.crs,
                    pad_width=pad_width,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    geom_mask=buffered_geom
                    )
                )
    return ops.unary_union(window_multipolygon), raster.crs

def get_multipolygon_for_raster(raster, zmin=None, zmax=None, raster_opts=None, window=None, pad_width=20):
    from geomesh import Geom, Raster
    from geomesh.cli.raster_opts import get_raster_from_opts
    if zmax is None:
        zmax = 0 if zmin is None else np.finfo(np.float64).max
    if not isinstance(raster, Raster):
        raster = get_raster_from_opts(
                raster_path=raster,
                window=window,
                **raster_opts
                )

    if pad_width is None:
        geom = Geom(raster, zmin=zmin, zmax=zmax)
        return geom.get_multipolygon(), raster.crs
    else:
        window_multipolygon = []
        for xval, yval, zvals in raster:
            zvals = zvals[0, :]
            window_multipolygon.append(
                    build_multipolygon_from_window(xval, yval, zvals, zmin, zmax, pad_width=20)
                    )
        return ops.unary_union(window_multipolygon), raster.crs


def get_quad_group_data(
        patch,
        this_centerline,
        group_id,
        max_quad_length,
        min_cross_section_node_count,
        min_quad_length,
        min_quad_width,
        max_quad_width,
        shrinkage_factor,
        cross_distance_factor
        ):
    total_length = this_centerline.length
    num_segments = np.ceil(total_length / max_quad_length)  # rounds up to ensure all segments are within max length
    adjusted_max_length = total_length / num_segments
    this_quads = compute_quads_from_centerline(
            patch,
            this_centerline,
            adjusted_max_length,
            min_quad_length,
            shrinkage_factor,
            cross_distance_factor,
            min_cross_section_node_count,
            min_quad_width,
            max_quad_width,
            )
    quad_groups = []
    # for quad_row_id, quad_row in enumerate(this_quads):
    for quad_row_id, (quad_row, normal_vector) in enumerate(this_quads):
        # if quad_row is None:
        #     continue
        for quad_id, quad in enumerate(quad_row):
            # print(quad)
            quad_groups.append({
                'quad_group_id': group_id,
                'quad_id': quad_id,
                'quad_row_id': quad_row_id,
                'geometry': quad,
                # 'normal_vector': (normal_vector[0], normal_vector[1]),
                })
    # verify
    # gpd.GeoDataFrame(quad_groups).plot(ax=plt.gca())
    # gpd.GeoDataFrame(geometry=[patch]).plot(ax=plt.gca(), facecolor='none', edgecolor='blue')
    # plt.show(block=True)
    return quad_groups

# def process_row(geometry, node_mappings):
#     # Process a single row
#     element = polygon.orient(geometry, sign=1.).exterior
#     element_indices = []
#     for p in element.coords[:-1]:
#         if p not in node_mappings:
#             node_mappings[p] = len(node_mappings)
#         element_indices.append(node_mappings[p])
#     return element_indices


# def poly_gdf_to_elements(poly_gdf):
#     import multiprocessing
#     manager = multiprocessing.Manager()
#     node_mappings = manager.dict()
#     connectivity_table = []
#     print("poly_gdf_to_elements processing rows in parallel")
#     with multiprocessing.Pool(cpu_count()) as pool:
#         results = pool.starmap(process_row, [(geometry, node_mappings) for geometry in poly_gdf.geometry])

#     connectivity_table.extend(results)

#     return list(node_mappings.keys()), connectivity_table

from collections import defaultdict


def polygon_orient_wrapper(poly):
    return polygon.orient(poly, sign=1.)

def poly_gdf_to_elements(poly_gdf):
    # print("make sure orientation is ccw", flush=True)
    # poly_gdf = poly_gdf.copy()
    # print("map polygon.orient", flush=True)
    # with Pool(cpu_count()) as pool:
    #     poly_gdf.geometry = list(pool.map(polygon_orient_wrapper, poly_gdf.geometry))
        # poly_gdf.geometry.map(lambda x: polygon.orient(x, sign=1.))
    # print("map is_ccw", flush=True)
    # poly_gdf['is_ccw'] = poly_gdf.geometry.map(lambda x: x.exterior.is_ccw)
    # cw_indexes = poly_gdf[~poly_gdf.is_ccw].index
    # print("map polygon.orient", flush=True)
    # poly_gdf.loc[cw_indexes, 'geometry'] = poly_gdf.loc[cw_indexes, 'geometry'].map(lambda x: polygon.orient(x, sign=1.0))
    # poly_gdf.geometry.map(lambda x: polygon.orient(x, sign=1.))
    print("begin creating node mappings", flush=True)
    # node_mappings = defaultdict(lambda: len(node_mappings))
    # element_coords = [list(row.geometry.exterior.coords[:-1]) for row in poly_gdf.itertuples()]
    # element_indices = [[node_mappings[tuple(tuple(x) for x in p)] for p in element_coords]]
    # return list(node_mappings.keys()), element_indices
    node_mappings = defaultdict(lambda: len(node_mappings))
    connectivity_table = []
    for row in poly_gdf.itertuples():
        element_coords = row.geometry.exterior.coords[:-1]
        element_indices = [node_mappings[p] for p in element_coords]
        connectivity_table.append(element_indices)
    print("done with node mappings", flush=True)
    return list(node_mappings.keys()), connectivity_table

# def poly_gdf_to_elements(poly_gdf):
#     node_mappings = {}
#     connectivity_table = []
#     for row in poly_gdf.itertuples():
#         # element: LinearRing = row.geometry.exterior
#         element = polygon.orient(row.geometry, sign=1.).exterior
#         element_indices = []
#         for i in range(len(element.coords) - 1):
#             p = element.coords[i]
#             if p not in node_mappings:
#                 node_mappings[p] = len(node_mappings)
#             element_indices.append(node_mappings[p])
#         connectivity_table.append(element_indices)
#     return list(node_mappings.keys()), connectivity_table


def quads_gdf_to_elements(quads_gdf):
    node_mappings = {}
    connectivity_table = []

    for row in quads_gdf.itertuples():
        quad: LinearRing = row.geometry
        quad_indices = []
        for i in range(len(quad.coords) - 1):
            p = quad.coords[i]
            if p not in node_mappings:
                node_mappings[p] = len(node_mappings)
            quad_indices.append(node_mappings[p])
        connectivity_table.append(quad_indices)

    nodes = np.array(list(node_mappings.keys()))
    connectivity_table = np.array(connectivity_table)
    return nodes, connectivity_table


def quads_gdf_to_edges(quads_gdf, split: bool = False):
    node_mappings = {}
    edges = []

    for row in quads_gdf.itertuples():
        quad: LinearRing = row.geometry
        for i in range(len(quad.coords)):
            p0 = quad.coords[i]
            p1 = quad.coords[(i+1) % len(quad.coords)]
            # p0 = tuple(np.round(np.array(quad.coords[i]), 7))
            # p1 = tuple(np.round(np.array(quad.coords[(i+1) % len(quad.coords)]), 7))
            if p0 not in node_mappings:
                node_mappings[p0] = len(node_mappings)
            if p1 not in node_mappings:
                node_mappings[p1] = len(node_mappings)
            edges.append((node_mappings[p0], node_mappings[p1]))
        if split:
            edges.append((node_mappings[p0], node_mappings[p1]))
    nodes = np.array(list(node_mappings.keys()))
    edges = np.array(edges)
    return nodes, edges


def decompose_gdf_into_Point_rows(gdf):
    gdf['index_before_decompose'] = gdf.index

    def extract_coords(geom):
        if isinstance(geom, Point):
            return [(geom.x, geom.y)]
        elif isinstance(geom, Polygon):
            return list(geom.exterior.coords) + [pt for interior in geom.interiors for pt in interior.coords]
        else:
            return list(geom.coords)

    gdf_geometry_coords = gdf.geometry.apply(extract_coords)

    gdf_geometry_counts = gdf_geometry_coords.apply(len)
    expanded_gdf = pd.DataFrame(
        np.repeat(gdf.values, gdf_geometry_counts, axis=0), columns=gdf.columns
    )

    expanded_gdf['geometry'] = [
        Point(x, y) for coords in gdf_geometry_coords for x, y in coords
    ]

    return gpd.GeoDataFrame(expanded_gdf, geometry='geometry', crs=gdf.crs)


def slope_collinearity_test(p0, p1, p2):
    if p0 == p1 or p1 == p2 or p0 == p2:
        return True
    # If x2-x1 is zero, then it's a vertical line, and we need to check if x3-x2 is also zero.
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    if np.isclose(x2 - x1, 0):
        return np.isclose(x3 - x2, 0)

    # Otherwise, calculate the two slopes and compare them using isclose
    slope_AB = (y2 - y1) / (x2 - x1)
    slope_BC = (y3 - y2) / (x3 - x2)
    return np.isclose(slope_AB, slope_BC)

def angle(a, b, c):
    """Returns the angle at vertex b in degrees."""
    if a == b or b == c or a == c:
        return 0.0
    ba = ((a[0] - b[0]), (a[1] - b[1]))
    bc = ((c[0] - b[0]), (c[1] - b[1]))
    cosine_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = math.acos(cosine_angle)
    return math.degrees(angle)

def find_ears(linear_ring):
    ears = []
    n = len(linear_ring.coords)
    for i in range(n-1):  # Adjusted the loop range to avoid adding the duplicate first point in the last triangle
        triangle = [linear_ring.coords[i], linear_ring.coords[(i + 1) % n], linear_ring.coords[(i + 2) % n]]
        if slope_collinearity_test(*triangle):
            continue
        if len(set(triangle)) < 3:  # Check that all vertices are distinct
            continue
        if Polygon(linear_ring).contains(Polygon(triangle)):
            ears.append(triangle)
    return ears

def manually_triangulate(
        linear_ring,
        # mesh_subset
        ):
    triangles = []
    coords_list = list(linear_ring.coords)[:-1]  # Excluding the duplicate first point
    while len(coords_list) > 3:
        ears = find_ears(linear_ring)
        if ears:
            # Find the ear with the minimum smallest angle
            best_ear = max(ears, key=lambda ear: max(angle(*ear), angle(ear[1], ear[2], ear[0]), angle(ear[2], ear[0], ear[1])))
            triangles.append(LinearRing(best_ear))
            idx = coords_list.index(best_ear[1])
            coords_list.pop(idx)
            linear_ring = LinearRing(coords_list)
        else:
            # raise NotImplementedError("Unreachable: No ears found.")
            break

    if len(coords_list) == 3:
        triangles.append(LinearRing(coords_list))

    return triangles
# def manually_triangulate(linear_ring):

#     triangles = []
#     coords_list = list(linear_ring.coords)

#     while len(coords_list) > 3:
#         ears = find_ears(linear_ring)
#         if ears:
#             # Find ear with max largest angle
#             best_ear = max(ears, key=lambda ear: max(angle(*ear), angle(ear[1], ear[2], ear[0]), angle(ear[2], ear[0], ear[1])))
#             triangles.append(LinearRing(best_ear))
#             # Remove ear point by creating a new linear ring
#             ear_point = best_ear[1]
#             new_coords = [c for c in coords_list if c != ear_point]
#             linear_ring = LinearRing(new_coords)

#         else:
#             raise NotImplementedError("Unreachable: No ears found.")
#             # Fallback triangulation if no ears
#             # ...

#     coords_list = list(linear_ring.coords)

#     # Add final triangle
#     if len(coords_list) == 3:
#         triangles.append(LinearRing(coords_list))

#     return triangles


def point_on_line_segment(point, line_segment):
    # Extract coordinates
    px, py = point
    (x1, y1), (x2, y2) = line_segment.coords

    # Check if point is within the bounding box of the segment
    if (min(x1, x2) <= px <= max(x1, x2)) and (min(y1, y2) <= py <= max(y1, y2)):
        # Use cross product to check collinearity
        cross_product = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)

        # If cross product is near zero, the point lies on the line
        if abs(cross_product) < np.finfo(float).eps:
            # Use dot product to check if point lies on the segment
            dot_product = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
            squared_length = (x2 - x1)**2 + (y2 - y1)**2
            if 0 <= dot_product <= squared_length:
                return True

    return False

def are_triangle_vertices_collinear(point1, point2, point3, eps=None):
    eps = eps or np.finfo(np.float32).eps
    # Extract coordinates
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    # Calculate vectors
    vec1 = (x2 - x1, y2 - y1)
    vec2 = (x3 - x1, y3 - y1)

    # Calculate the cross product
    cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

    # If the cross product is near zero, the points are collinear
    return abs(cross_product) <= eps



def vector_from_linestring(line):
    """Returns the vector representation of a linestring."""
    coords = list(line.coords)
    return np.array([coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]])


# def angle_between(v1, v2):
#     """Returns the angle in radians between vectors 'v1' and 'v2'."""
#     cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#     return np.arccos(np.clip(cos_theta, -1, 1))

def is_near_collinear(line1, line2, tolerance=np.radians(5)):
    """Checks if two linestrings are nearly collinear within a certain angle tolerance."""
    v1 = vector_from_linestring(line1)
    v2 = vector_from_linestring(line2)
    angle = angle_between(v1, v2)
    return abs(angle) < tolerance or abs(angle - np.pi) < tolerance

def get_intersecting_edges(group) -> List[LineString]:

    from math import atan2, degrees


    def calculate_angle(line):
        p1, p2 = line.coords[:2]
        angle_rad = atan2(p2[1] - p1[1], p2[0] - p1[0])
        return degrees(angle_rad) % 360

    def angle_difference(angle1, angle2):
        # Calculate the absolute difference and modulo 180 to find the smallest angle difference
        return min(np.abs(angle1 - angle2), 360 - np.abs(angle1 - angle2))


    intersecting_edges = set()
    for row in group[group['element_type'] == 'tria'].itertuples():
        tria = row.geometry
        tria_edges = [
            LineString([tria.exterior.coords[i], tria.exterior.coords[(i+1)%3]])
            for i in range(3)
        ]

        for row in group[group['element_type'] == 'quad'].itertuples():
            quad: Polygon = row.geometry
            quad_edges = [
                LineString([quad.exterior.coords[i], quad.exterior.coords[(i+1)%4]])
                for i in range(4)
            ]

            for tria_edge in tria_edges:
                tria_angle = calculate_angle(tria_edge)
                for quad_edge in quad_edges:
                    quad_angle = calculate_angle(quad_edge)
                    if tria_edge.buffer(np.finfo(np.float32).eps).intersects(quad_edge):
                        if angle_difference(tria_angle, quad_angle) < 1:  # Tolerance in degrees
                            intersecting_edges.add(quad_edge)

    if len(intersecting_edges) == 0:
        # mesh_gdf.loc[group.index].plot(ax=plt.gca(), facecolor='none', edgecolor='r')
        # mesh_gdf.loc[group.index_right].plot(ax=plt.gca(), facecolor='none', edgecolor='b')
        group.plot(ax=plt.gca(), facecolor='none')
        plt.title('no intersecting edges')
        plt.show(block=False)
        breakpoint()
        raise ValueError("No intersecting edges found.")

    return list(intersecting_edges)


# def get_intersecting_edges(group, mesh_gdf) -> List[LineString]:

#     def are_collinear(p1, p2, p3):
#         x1, y1 = p1
#         x2, y2 = p2
#         x3, y3 = p3
#         determinant = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
#         return np.isclose(determinant, 0, atol=np.finfo(np.float32).eps)

#     def is_between(p, start, end):
#         return (min(start[0], end[0]) < p[0] < max(start[0], end[0]) and
#                 min(start[1], end[1]) < p[1] < max(start[1], end[1]))


#     unique_indexes = group['index_right'].unique()
#     intersecting_edges = set()
#     for index in unique_indexes:
#         tria = mesh_gdf.loc[index].geometry
#         tria_edges = [
#             LineString([tria.exterior.coords[i], tria.exterior.coords[(i+1)%3]])
#             for i in range(3)
#         ]
#         for _, row in group.loc[group.index.unique()].iterrows():
#             quad = row.geometry
#             quad_edges = [
#                 LineString([quad.exterior.coords[i], quad.exterior.coords[(i+1)%4]])
#                 for i in range(4)
#             ]
#             for tria_edge in tria_edges:
#                 tria_edge_buffered = tria_edge.buffer(np.finfo(np.float32).eps)
#                 for quad_edge in quad_edges:
#                     # ULTRA heuristic and prone to errors; unsure about how to better approach this
#                     quad_edge_buffered = quad_edge.buffer(np.finfo(np.float32).eps)
#                     if quad_edge.within(tria_edge_buffered) or tria_edge.within(quad_edge_buffered):
#                         intersecting_edges.add(quad_edge)
#                         continue
#                     if ((are_collinear(tria_edge.coords[0], quad_edge.coords[0], quad_edge.coords[1]) and
#                          is_between(tria_edge.coords[0], quad_edge.coords[0], quad_edge.coords[1])) or
#                         (are_collinear(tria_edge.coords[1], quad_edge.coords[0], quad_edge.coords[1]) and
#                          is_between(tria_edge.coords[1], quad_edge.coords[0], quad_edge.coords[1]))):
#                             # if quad_edge_buffered.intersects(tria):
#                             intersecting_edges.add(quad_edge)
#                             continue
#                     if quad_edge.within(tria):
#                         intersecting_edges.add(quad_edge)
#                         continue
#     intersecting_edges = list(intersecting_edges)
#     if len(intersecting_edges) == 0:
#         # gpd.GeoDataFrame(
#         mesh_gdf.loc[group.index].plot(ax=plt.gca(), facecolor='none', edgecolor='r')
#         mesh_gdf.loc[group.index_right].plot(ax=plt.gca(), facecolor='none', edgecolor='b')
#         plt.title('no intersecting edges')
#         plt.show(block=False)
#         breakpoint()
#         raise
#     return list(intersecting_edges)

# def get_intersecting_edges(group, mesh_gdf) -> List[LineString]:

#     trias = mesh_gdf.loc[group.index_right.unique()]
#     quads = mesh_gdf.loc[group.index.unique()]
#     intersecting_edges = set()

#     quad_edges = []
#     for quad in quads.geometry:
#         quad_edges.extend([
#             LineString([quad.exterior.coords[i], quad.exterior.coords[(i+1)%4]])
#             for i in range(4)
#         ])
#     tria_edges = []
#     for tria in trias.geometry:
#         tria_edges.extend([
#             LineString([tria.exterior.coords[i], tria.exterior.coords[(i+1)%3]])
#             for i in range(3)
#         ])

#     quad_edges_gdf = gpd.GeoDataFrame(geometry=quad_edges, crs=mesh_gdf.crs)
#     tria_edges_gdf = gpd.GeoDataFrame(geometry=tria_edges, crs=mesh_gdf.crs)

#     def get_normalized_vectors(gdf):
#         # Extract start and end points for the linestrings
#         start_points = np.array(list(gdf.geometry.apply(lambda line: np.array(line.coords[0]))))
#         end_points = np.array(list(gdf.geometry.apply(lambda line: np.array(line.coords[-1]))))

#         # Calculate the direction vectors
#         direction_vectors = end_points - start_points

#         # Normalize the vectors
#         norms = np.linalg.norm(direction_vectors, axis=1).reshape(-1, 1)
#         normalized_vectors = direction_vectors / norms

#         return normalized_vectors

#     # Assuming the linestrings are in the 'geometry' column
#     quad_direction_vectors = get_normalized_vectors(quad_edges_gdf)
#     tria_direction_vectors = get_normalized_vectors(tria_edges_gdf)

#     # Calculate the dot product between each pair of vectors in a vectorized manner
#     # Here we're computing the dot product of each quad with every tria, resulting in a 2D array
#     dot_products = np.dot(quad_direction_vectors, tria_direction_vectors.T)

#     # Calculate the angles using arccos (note: the output will be in radians)
#     angles = np.arccos(dot_products.clip(-1, 1))  # clip values to avoid errors due to floating point inaccuracies

#     # Convert angles to degrees
#     angles_deg = np.degrees(angles)

#     # Define your threshold for "close to zero" in degrees
#     angle_threshold = 5

#     # Filter quads that have any angle less than the threshold with trias
#     # We are looking for the minimum angle each quad makes with any of the trias
#     min_angles = np.min(angles_deg, axis=1)
#     close_to_parallel_indices = np.where(min_angles <= angle_threshold)[0]

#     # Create a new GeoDataFrame with the selected quad edges
#     potential_edges = quad_edges_gdf.iloc[close_to_parallel_indices]
#     # verify

#     # mesh_gdf.loc[group.index].plot(ax=plt.gca(), facecolor='none', edgecolor='r')
#     # mesh_gdf.loc[group.index_right].plot(ax=plt.gca(), facecolor='none', edgecolor='b')
#     # potential_edges.plot(ax=plt.gca(), edgecolor='green')
#     # plt.title('potential edges')
#     # plt.show(block=False)
#     # breakpoint()
#     # raise

#     # potential_edges_buffered = potential_edges.copy()
#     # potential_edges_buffered.geometry = potential_edges.geometry.buffer(np.finfo(np.float32).eps)


#     joined = gpd.sjoin(potential_edges, trias, how='inner', predicate='intersects')
#     intersecting_edges = joined.geometry.unique().tolist()
#     # verify

#     # mesh_gdf.loc[group.index].plot(ax=plt.gca(), facecolor='none', edgecolor='r')
#     # mesh_gdf.loc[group.index_right].plot(ax=plt.gca(), facecolor='none', edgecolor='b')
#     # joined.plot(ax=plt.gca(), edgecolor='green')
#     # plt.title('potential edges')
#     # plt.show(block=True)
#     # breakpoint()
#     # raise
#     # intersecting_edges = list(intersecting_edges)
#     if len(intersecting_edges) == 0:
#         # gpd.GeoDataFrame(
#         mesh_gdf.loc[group.index].plot(ax=plt.gca(), facecolor='none', edgecolor='r')
#         mesh_gdf.loc[group.index_right].plot(ax=plt.gca(), facecolor='none', edgecolor='b')
#         plt.title('no intersecting edges')
#         plt.show(block=False)
#         breakpoint()
#         raise
#     return list(intersecting_edges)


# def get_intersecting_edges(group, mesh_gdf):
#     # Extracting unique indexes
#     unique_indexes = group['index_right'].unique()
#     intersecting_edges = set()
#     for index in unique_indexes:
#         tria = mesh_gdf.loc[index].geometry
#         for _, row in group.loc[group.index.unique()].iterrows():
#             quad = row['geometry']
#             quad_edges = MultiLineString([
#                 LineString([quad.exterior.coords[i], quad.exterior.coords[(i+1)%4]])
#                 for i in range(4)
#             ])
#             # this_intrs = tria.intersection(quad_edges)
#             # if isinstance(this_intrs, Point):
#             #     for quad_edge in quad_edges.geoms:
#             #         tria_to_quad_edge_intersection = tria.intersection(quad_edge)
#             #         if tria_to_quad_edge_intersection.is_empty:
#             #             continue
#             #         elif isinstance(tria_to_quad_edge, Point):

#             #         else:
#             #         print(tria_to_quad_edge_intersection)
#             #     breakpoint()
#             # else:
#             #     raise Exception(f"Unreachable: Expected intersection of types LineString or Point but got {this_intrs=}")

#             # breakpoint()
#             for quad_edge in quad_edges:
#                 tria_to_quad_edge_intersection = tria.intersection(quad_edge)
#                 if tria_to_quad_edge_intersection.is_empty:
#                     logger.info(f"{tria=}")
#                     logger.info(f"{quad_edge=}")
#                     continue  # null case (no intersection)
#                 if isinstance(tria_to_quad_edge_intersection, LineString):
#                     intersecting_edges.add(quad_edge)
#                 elif isinstance(tria_to_quad_edge_intersection, Point):
#                     tria_points_intersect = 0
#                     for tria_point in tria.exterior.coords[:-1]:
#                         if point_on_line_segment(tria_point, quad_edge):
#                             tria_points_intersect += 1
#                         if tria_points_intersect > 1:
#                             intersecting_edges.add(quad_edge)
#                 else:
#                     raise Exception(f"Unreachable: Expected intersection of types LineString or Point but got {tria_to_quad_edge_intersection=}")
#     return list(intersecting_edges)


# def determine_connections(point, graph, num_connections=2):
#     """
#     Connects a missing point to the closest nodes in the graph.

#     :param point: Tuple representing the missing point's coordinates.
#     :param graph: A dictionary representation of a graph where keys are nodes (points) and values are lists of neighbor nodes.
#     :param num_connections: Number of closest nodes to connect the missing point to.
#     :return: List of nodes (points) to which the missing point should connect.
#     """
#     distances = [(node, euclidean_distance(point, node)) for node in graph]
#     distances.sort(key=lambda x: x[1])

#     return [node for node, _ in distances[:num_connections]]

# def euclidean_distance(point1, point2):
#     """
#     Computes the Euclidean distance between two points.

#     :param point1: First point.
#     :param point2: Second point.
#     :return: Euclidean distance between the points.
#     """
#     return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


# def build_graph(edges, missing_points):
#     from collections import defaultdict
#     graph = defaultdict(list)
#     for edge in edges:
#         graph[edge.coords[0]].append(edge.coords[1])
#         graph[edge.coords[1]].append(edge.coords[0])

#     for point in missing_points:
#         # This assumes you have some logic to determine which nodes a missing point should connect to
#         connected_nodes = determine_connections(point, graph)
#         for node in connected_nodes:
#             graph[point].append(node)
#             graph[node].append(point)

#     return graph

# def traverse_graph(graph):
#     rings = []
#     visited_edges = set()
#     last_end_node = None
#     current_node = list(graph.keys())[0]
#     traversed_starting_nodes = set()
#     print(f"Starting traversal from node: {current_node}")
#     while graph:
#         # If current node has been deleted or has no neighbors, pick another node
#         while current_node not in graph or not graph[current_node] or current_node in traversed_starting_nodes:
#             current_node = next(iter(graph.keys()))
#         initial_start_node = current_node  # track the initial starting point
#         print(f"Current node: {current_node}, Neighbors: {graph[current_node]}")

#         ring = [current_node]
#         visited_nodes = {current_node}
#         while True:
#             for neighbor in graph[current_node]:
#                 edge = tuple(sorted([current_node, neighbor]))
#                 if edge not in visited_edges:
#                     visited_edges.add(edge)
#                     current_node = neighbor
#                     print(f"Chosen neighbor: {neighbor}")

#                     # Check if we're revisiting a node within this traversal
#                     if current_node in visited_nodes:
#                         # Find the index of the first occurrence of this node in the ring
#                         idx = ring.index(current_node)
#                         ring = ring[:idx + 1]  # Keep only nodes up to the repeated node
#                         break
#                     visited_nodes.add(current_node)
#                     ring.append(current_node)
#             else:
#                 continue
#             break
#         print(f"Formed ring: {ring}")
#         if len(ring) < 3:
#             print("Invalid ring formed. Restarting traversal...")
#             current_node = next(iter(graph.keys()))
#             continue
#         last_end_node = ring[-1]
#         the_ring = LinearRing(ring)
#         if not the_ring.is_valid:
#             gpd.GeoDataFrame(geometry=[the_ring]).plot(ax=plt.gca())
#             plt.show(block=False)
#             breakpoint()
#             raise NotImplementedError("Unreachable: Ended up with an invalid ring while traversing graph.")
#         rings.append(the_ring)

#         # Remove visited edges from the graph
#         for node1, node2 in visited_edges:
#             print(f"Removing edge: {node1, node2}")
#             if node2 in graph[node1]:
#                 graph[node1].remove(node2)
#             if node1 in graph[node2]:
#                 graph[node2].remove(node1)


#         # Remove nodes with no remaining edges:
#         nodes_to_delete = [node for node, neighbors in graph.items() if not neighbors]
#         for node in nodes_to_delete:
#             print(f"Deleting node with no remaining edges: {node}")
#             del graph[node]

#         # Clear visited_edges for the next iteration
#         visited_edges.clear()
#     print("returning rings...")
#     return rings

def find_connected_components_trias_edges(trias_to_drop: gpd.GeoDataFrame, intersecting_edges: gpd.GeoDataFrame):
    """
    Find connected components of triangles and edges that mutually intersect.

    Parameters:
    - trias_to_drop (gpd.GeoDataFrame): A GeoDataFrame of triangles.
    - intersecting_edges (gpd.GeoDataFrame): A GeoDataFrame of edges.

    Returns:
    - List[Set[str]]: A list of sets, where each set contains identifiers for triangles and edges that mutually intersect.
    """

    # Join based on intersection
    joined = gpd.sjoin(trias_to_drop, intersecting_edges, how='inner', predicate='intersects')

    # Create a graph
    G = nx.Graph()

    # Add nodes for each triangle and edge
    for idx, row in trias_to_drop.iterrows():
        G.add_node(f'triangle_{idx}', type='triangle')

    for idx, row in intersecting_edges.iterrows():
        G.add_node(f'edge_{idx}', type='edge')

    # Add edges for intersections
    for idx, row in joined.iterrows():
        G.add_edge(f'triangle_{row.index}', f'edge_{row.index_right}')

    # Find connected components
    connected_components = list(nx.connected_components(G))

    return connected_components

def get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges) -> typing.List[Point]:
    edges_mls = MultiLineString(intersecting_edges)
    missing_points = set()
    for tria in trias_to_drop.geometry:
        for tria_point in tria.exterior.coords[:-1]:
            if not Point(tria_point).buffer(np.finfo(np.float32).eps).intersects(edges_mls):
                missing_points.add(tria_point)
    for edge in edges_mls.geoms:
        for coord in edge.coords:
            missing_points.add(coord)
    return [Point(x) for x in missing_points]

def get_linear_rings_from_trias_and_edges(mesh_subset, trias_to_drop, intersecting_edges) -> typing.List[LinearRing]:
    participant_points = get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges) # List[Point]
    convex_hull = MultiPoint(participant_points).convex_hull
    the_patches = convex_hull.difference(mesh_subset.unary_union)
    if isinstance(the_patches, Polygon):
        the_patches = MultiPolygon([the_patches])
    if not isinstance(the_patches, MultiPolygon):
        raise ValueError(f"Unreachable: Expected a Polygon or MultiPolygon but got {the_patches=}")
    the_rings = []
    for geom in the_patches.geoms:
        boundary = geom.boundary
        if isinstance(boundary, LineString):
            boundary = MultiLineString([boundary])
        if not isinstance(boundary, MultiLineString):
            raise ValueError("Unreachable: Expected LineString or MultiLineString")
        the_rings.extend(boundary.geoms)

    return the_rings

# def get_linear_rings_from_trias_and_edges(mesh_subset, trias_to_drop, intersecting_edges) -> typing.List[LinearRing]:

#     edges_merged = ops.linemerge(intersecting_edges)
#     missing_points = set()
#     linear_rings = []
#     for tria in trias_to_drop.geometry:
#         for tria_point in tria.exterior.coords[:-1]:
#             if not Point(tria_point).buffer(np.finfo(np.float32).eps).intersects(edges_merged):
#                 missing_points.add(tria_point)
#     if isinstance(edges_merged, LineString):
#         reference_point = Point(edges_merged.coords[-1])
#         def distance_to_reference(p):
#             return reference_point.distance(Point(p))
#         missing_points_sorted = sorted(missing_points, key=distance_to_reference)
#         merged_coords = list(edges_merged.coords) + missing_points_sorted
#         linear_ring = LinearRing(merged_coords)
#     elif isinstance(edges_merged, MultiLineString):
#         all_coords = []
#         remaining_lines = list(edges_merged.geoms)

#         # Start with the first line segment
#         current_line = remaining_lines.pop(0)
#         reference_point = Point(current_line.coords[-1])

#         # Function to measure distance to the reference point
#         def distance_to_reference(p):
#             return reference_point.distance(Point(p))

#         # Sort the missing points based on their distance to the reference point
#         missing_points_sorted = sorted(missing_points, key=distance_to_reference)

#         all_coords.extend(current_line.coords)
#         all_coords.extend(missing_points_sorted)

#         while remaining_lines:
#             # Find the next closest line segment
#             next_line_idx = min(range(len(remaining_lines)), key=lambda i: reference_point.distance(Point(remaining_lines[i].coords[0])))
#             next_line = remaining_lines.pop(next_line_idx)
#             all_coords.extend(next_line.coords)
#             reference_point = Point(next_line.coords[-1])
#         linear_ring = LinearRing(all_coords)
#     else:
#         raise NotImplementedError(f"Unreachable: Expected LineString or MultiLineString but got {type(edges_merged)=} {intersecting_edges=}")

#     if not linear_ring.is_valid:

#         def custom_group(intersection_gdf):
#             graph = {}
#             for idx, row in intersection_gdf.iterrows():
#                 graph[idx] = graph.get(idx, []) + [row['index_right']]
#                 graph[row['index_right']] = graph.get(row['index_right'], []) + [idx]
#             components = find_connected_components(graph)
#             grouped_rows = []
#             for component in components:
#                 group = intersection_gdf[intersection_gdf.index.isin(component) | intersection_gdf['index_right'].isin(component)]
#                 grouped_rows.append(group)
#             return grouped_rows


#         def get_inner_groups():
#             intersecting_edges_gdf = gpd.GeoDataFrame(geometry=intersecting_edges, crs=trias_to_drop.crs)
#             intersecting_edges_buffered_gdf = intersecting_edges_gdf.copy()
#             intersecting_edges_buffered_gdf.geometry = intersecting_edges_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
#             joined = gpd.sjoin(
#                     intersecting_edges_buffered_gdf,
#                     trias_to_drop.drop(columns=["index_right"]),
#                     how='inner',
#                     predicate='intersects',
#                     )
#             joined.geometry = intersecting_edges_gdf.loc[joined.index].geometry
#             return custom_group(joined)
#         for inner_group in get_inner_groups():
#             this_trias_to_drop = trias_to_drop.loc[inner_group.index_right]
#             this_intersecting_edges = list(inner_group.geometry)
#             edges_merged = ops.linemerge(this_intersecting_edges)
#             if isinstance(edges_merged, LineString):
#                 reference_point = Point(edges_merged.coords[-1])
#                 def distance_to_reference(p):
#                     return reference_point.distance(Point(p))
#                 missing_points_sorted = sorted(missing_points, key=distance_to_reference)
#                 merged_coords = list(edges_merged.coords) + missing_points_sorted
#                 linear_ring = LinearRing(merged_coords)
#             elif isinstance(edges_merged, MultiLineString):
#                 all_coords = []
#                 remaining_lines = list(edges_merged.geoms)

#                 # Start with the first line segment
#                 current_line = remaining_lines.pop(0)
#                 reference_point = Point(current_line.coords[-1])

#                 # Function to measure distance to the reference point
#                 def distance_to_reference(p):
#                     return reference_point.distance(Point(p))

#                 # Sort the missing points based on their distance to the reference point
#                 missing_points_sorted = sorted(missing_points, key=distance_to_reference)

#                 all_coords.extend(current_line.coords)
#                 all_coords.extend(missing_points_sorted)

#                 while remaining_lines:
#                     # Find the next closest line segment
#                     next_line_idx = min(range(len(remaining_lines)), key=lambda i: reference_point.distance(Point(remaining_lines[i].coords[0])))
#                     next_line = remaining_lines.pop(next_line_idx)
#                     all_coords.extend(next_line.coords)
#                     reference_point = Point(next_line.coords[-1])
#                 linear_ring = LinearRing(all_coords)
#             else:
#                 raise NotImplementedError(f"Unreachable: Expected LineString or MultiLineString but got {type(edges_merged)=} {intersecting_edges=}")
#             linear_rings.append(linear_ring)
#     return linear_rings
#         # edges_merged = ops.linemerge(list(inner_group.geometry))
#         # reference_point = Point(edges_merged.coords[-1])
#         # def distance_to_reference(p):
#         #     return reference_point.distance(Point(p))
#         # missing_points_sorted = sorted(missing_points, key=distance_to_reference)
#         # merged_coords = list(edges_merged.coords) + missing_points_sorted
#         # linear_ring = LinearRing(merged_coords)
#         # plt.title("get new tria poly list from group and mesh_gdf")
#         # plt.show(block=False)
#         # breakpoint()
#         # raise

#         # this_triangles = ops.triangulate(inner_group)




# def get_linear_rings_from_trias_and_edges(mesh_subset, trias_to_drop, intersecting_edges) -> typing.List[LinearRing]:
#     edges_merged = ops.linemerge(intersecting_edges)
#     missing_points = set()
#     linear_rings = []
#     for tria in trias_to_drop.geometry:
#         for tria_point in tria.exterior.coords[:-1]:
#             if not Point(tria_point).buffer(np.finfo(np.float32).eps).intersects(edges_merged):
#                 missing_points.add(tria_point)
#     if isinstance(edges_merged, LineString):
#         reference_point = Point(edges_merged.coords[-1])
#         def distance_to_reference(p):
#             return reference_point.distance(Point(p))
#         missing_points_sorted = sorted(missing_points, key=distance_to_reference)
#         merged_coords = list(edges_merged.coords) + missing_points_sorted
#         linear_ring = LinearRing(merged_coords)
#     elif isinstance(edges_merged, MultiLineString):
#         all_coords = []
#         remaining_lines = list(edges_merged.geoms)

#         # Start with the first line segment
#         current_line = remaining_lines.pop(0)
#         reference_point = Point(current_line.coords[-1])

#         # Function to measure distance to the reference point
#         def distance_to_reference(p):
#             return reference_point.distance(Point(p))

#         # Sort the missing points based on their distance to the reference point
#         missing_points_sorted = sorted(missing_points, key=distance_to_reference)

#         all_coords.extend(current_line.coords)
#         all_coords.extend(missing_points_sorted)

#         while remaining_lines:
#             # Find the next closest line segment
#             next_line_idx = min(range(len(remaining_lines)), key=lambda i: reference_point.distance(Point(remaining_lines[i].coords[0])))
#             next_line = remaining_lines.pop(next_line_idx)
#             all_coords.extend(next_line.coords)
#             reference_point = Point(next_line.coords[-1])
#         linear_ring = LinearRing(all_coords)
#     else:
#         raise NotImplementedError(f"Unreachable: Expected LineString or MultiLineString but got {type(edges_merged)=} {intersecting_edges=}")

#     if not linear_ring.is_valid:
#         linear_ring_convex_hull = linear_ring.convex_hull
#         possible_matches_index = list(mesh_subset.sindex.intersection(linear_ring_convex_hull.bounds))
#         mesh_subset = mesh_subset.iloc[possible_matches_index]
#         result = linear_ring_convex_hull.difference(mesh_subset.geometry.unary_union)
#         if isinstance(result, MultiPolygon):
#             for this_poly in result.geoms:
#                 if this_poly.is_valid:
#                     linear_rings.append(this_poly.exterior)
#                 else:
#                     raise Exception("Unreachable: All polygons here should be valid.")
#         elif isinstance(result, Polygon):
#             if result.is_valid:
#                 linear_rings.append(result.exterior)
#             else:
#                 raise Exception("Unreachable: All polygons here should be valid.")

#     else:
#         linear_rings.append(linear_ring)
#     return linear_rings


# def elements_are_conforming(element_left, element_right) -> bool:
#     if not isinstance(element_left, LinearRing):
#         raise ValueError(f"Argument element_left must be a LinearRing but got {element_left=}")

#     if not isinstance(element_right, LinearRing):
#         raise ValueError(f"Argument element_right must be a LinearRing but got {element_right=}")

#     eps = np.finfo(np.float32).eps

#     # Create list of edges for element_left
#     mod_left = len(element_left.coords) - 1
#     element_left_edges = [
#         LineString([element_left.coords[i], element_left.coords[(i+1)%mod_left]])
#         for i in range(mod_left)
#     ]

#     # Create list of edges for element_right
#     mod_right = len(element_right.coords) - 1
#     element_right_edges = [
#         LineString([element_right.coords[i], element_right.coords[(i+1)%mod_right]])
#         for i in range(mod_right)
#     ]

#     # Cross-reference all edges
#     for edge_left in element_left_edges:
#         for edge_right in element_right_edges:
#             # If two edges are collinear

#             if is_near_collinear(edge_left, edge_right, tolerance=np.radians(5)):
#             # if edge_left.is_parallel(edge_right):
#                 # but not equals_exact, then they are not conforming
#                 if not (equals_exact(edge_left, edge_right, tolerance=eps) or equals_exact(edge_left, edge_right.reverse(), tolerance=eps)):
#                     return False

#     # If no non-conforming edges found, return True
#     return True


def elements_share_an_edge(element_left_edges, element_right_edges, tolerance=np.finfo(np.float32).eps):
    for element_left_edge in element_left_edges:
        for element_right_edge in element_right_edges:
            if equals_exact(element_left_edge, element_right_edge, tolerance=tolerance):
                return True
            elif equals_exact(element_left_edge.reverse(), element_right_edge, tolerance=tolerance):
                return True
    return False

def elements_have_collinear_non_conforming_edges(element_left_edges, element_right_edges):
    for element_left_edge in element_left_edges:
        for element_right_edge in element_right_edges:
            if element_left_edge.within(element_right_edge.buffer(np.finfo(np.float32).eps)):
                return True
            if element_right_edge.within(element_left_edge.buffer(np.finfo(np.float32).eps)):
                return True
    return False



def elements_are_conforming(element_left, element_right) -> bool:
    if not isinstance(element_left, LinearRing):
        raise ValueError(f"Argument element_left must be a LinearRing but got {element_left=}")
    if not isinstance(element_right, LinearRing):
        raise ValueError(f"Argument element_right must be a LinearRing but got {element_right=}")

    eps = np.finfo(np.float32).eps
    if Polygon(element_left).buffer(eps).contains(Polygon(element_right).buffer(-eps)):
        return False
    if Polygon(element_right).buffer(eps).contains(Polygon(element_left).buffer(-eps)):
        return False

    def create_edges(element):
        return [LineString([element.coords[i], element.coords[(i + 1) % (len(element.coords) - 1)]]) for i in range(len(element.coords) - 1)]

    element_left_edges = create_edges(element_left)
    element_right_edges = create_edges(element_right)

    if elements_share_an_edge(element_left_edges, element_right_edges):
        return True
    elif elements_have_collinear_non_conforming_edges(element_left_edges, element_right_edges):
        return False

    intersection = element_right.intersection(element_left)
    if isinstance(intersection, Point):
        from shapely.geometry import MultiPoint
        element_left_points = MultiPoint([Point(x) for x in element_left.coords]).buffer(eps)
        element_right_points = MultiPoint([Point(x) for x in element_right.coords]).buffer(eps)
        intersection_buffered = intersection.buffer(eps)
        if intersection_buffered.intersects(element_left_points) and intersection_buffered.intersects(element_right_points):
            return True

    return False



def filter_triangle_candidates_mut(triangle_candidates, mesh_subset):
    possible_matches_index = list(mesh_subset.sindex.intersection(MultiLineString(triangle_candidates).bounds))
    mesh_subset = mesh_subset.iloc[possible_matches_index]
    triangle_candidates_gdf = gpd.GeoDataFrame(geometry=triangle_candidates, crs=mesh_subset.crs)

    def is_conforming(row):
        ls_right = LinearRing(mesh_subset.loc[row.index_right].geometry.exterior.coords)
        return elements_are_conforming(row.geometry, ls_right)

    joined = gpd.sjoin(
            triangle_candidates_gdf,
            mesh_subset,
            how='inner',
            predicate='intersects'
            )
    joined = joined[~joined.apply(is_conforming, axis=1)]
    indexes_to_drop = joined.index.unique()
    # if len(indexes_to_drop) == len(triangle_candidates):
    #      dropping all
    for index in reversed(sorted(joined.index.unique())):
        triangle_candidates.pop(index)
    # if len(triangle_candidates) == 0:


def point_is_between_points(point: Point, edge: LineString) -> bool:
    point_buffered = point.buffer(np.finfo(np.float32).eps)
    if point_buffered.intersects(edge):
        edge_endpoints = MultiPoint([Point(x) for x in edge.coords])
        if not point_buffered.intersects(edge_endpoints):
            return True
    return False

# def get_new_tria_poly_list_from_group_and_mesh_gdf(group, mesh_gdf) -> List[Polygon]:
#     group_id, group = group
#     intersecting_edges = get_intersecting_edges(group) # List[LineString]
#     intersecting_edges_gdf = gpd.GeoDataFrame(geometry=intersecting_edges, crs=mesh_gdf.crs)
#     trias_to_drop = group[group["element_type"] == "tria"]
#     trias_to_drop_buffered_gdf = trias_to_drop.copy()
#     trias_to_drop_buffered_gdf.drop(columns=["index_right"], inplace=True)
#     trias_to_drop_buffered_gdf.geometry = trias_to_drop.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
#     joined = gpd.sjoin(
#             trias_to_drop_buffered_gdf,
#             intersecting_edges_gdf,
#             how='inner',
#             predicate='intersects',
#             )
#     joined.geometry = trias_to_drop.loc[joined.index].geometry
#     out_trias = []
#     for index_right, this_group in joined.groupby("index_right"):
#         this_trias = list(this_group.geometry)
#         this_edge = list(intersecting_edges_gdf.loc[[index_right]].geometry)
#         convex_hull = GeometryCollection([*this_trias, *this_edge]).convex_hull
#         out_trias.extend(ops.triangulate(convex_hull))
#         # for this_tria in this_trias:
#         #     intersection = this_tria.intersection(this_edge)
#         #     if isinstance(intersection, Point):
#         #         this_group.plot(ax=plt.gca(), facecolor='r', edgecolor='r', alpha=0.3)
#         #         intersecting_edges_gdf.loc[[index_right]].plot(ax=plt.gca(), edgecolor='b')
#         #         gpd.GeoDataFrame(geometry=[intersection]).plot(ax=plt.gca(), color='g')
#         #         plt.title("intersection is point")
#         #         plt.show(block=False)
#         #         breakpoint()
#         #         raise


#         #     elif isinstance(intersection, LineString):
#         #         this_group.plot(ax=plt.gca(), facecolor='r', edgecolor='r', alpha=0.3)
#         #         intersecting_edges_gdf.loc[[index_right]].plot(ax=plt.gca(), edgecolor='b')
#         #         gpd.GeoDataFrame(geometry=[intersection]).plot(ax=plt.gca(), edgecolor='g')
#         #         plt.title("intersection is ls")
#         #         plt.show(block=False)
#         #         breakpoint()
#         #         raise
#         #         raise NotImplementedError("LineString intersection")

#         #     else:
#         #         raise ValueError(f"Unreachable: Expected intersection to be Point or LineString but got {intersection=}")
#             # new_tria_points = []
#             # for point in this_tria.coords[:-1]:
#             #     if point_is_between_points(point):

#             #         new_tria_points.append(new_point)
#             #     else:
#             #         new_tria_points.append(point)



#     return out_trias

def do_debug_plot(mesh_subset, trias_to_drop, new_trias, title):
    mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
    trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
    gpd.GeoDataFrame(geometry=new_trias).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
    plt.title(title)
    plt.show(block=False)
    breakpoint()
    raise


def select_nodes_to_move(bad_tria, mesh_quad_edges_gdf, mesh_quad_edges_buffered_gdf):
    selected_node_indices = []
    breakpoint()
    # mesh_quad_edge_points_gdf =
    bad_tria_points_gdf = gpd.GeoDataFrame(geometry=[Point(x) for x in bad_tria.coords[:-1]], crs=mesh_quad_edges_gdf.crs)
    joined = gpd.sjoin(
            bad_tria_points_gdf,
            mesh_quad_edges_buffered_gdf,
            how='inner',
            predicate='intersects',
            )








def do_extended_tria_method(mesh_subset, trias_to_drop, conforming_trias, reason) -> List[Polygon]:

    conforming_trias_gdf = gpd.GeoDataFrame(geometry=conforming_trias, crs=mesh_subset.crs)

    mesh_subset_quad_edges = []
    for row in mesh_subset[mesh_subset['element_type'] == 'quad'].itertuples():
        quad: Polygon = row.geometry
        mesh_subset_quad_edges.extend([
            LineString([quad.exterior.coords[i], quad.exterior.coords[(i+1)%4]])
            for i in range(4)
        ])

    mesh_subset_quad_edges_gdf = gpd.GeoDataFrame(geometry=mesh_subset_quad_edges, crs=mesh_subset.crs)
    mesh_subset_quad_edges_buffered_gdf = mesh_subset_quad_edges_gdf.copy()
    mesh_subset_quad_edges_buffered_gdf.geometry = mesh_subset_quad_edges_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))

    def get_groups():
        conforming_trias_uu = ops.unary_union(conforming_trias)
        conforming_trias_uu_buffered = conforming_trias_uu.buffer(np.finfo(np.float32).eps)
        trias_to_modify = []
        # TODO: Highly vectorizable
        for row in trias_to_drop.itertuples():
            if not row.geometry.within(conforming_trias_uu_buffered):
                trias_to_modify.append(row.geometry)
        trias_to_mod_gdf = gpd.GeoDataFrame(geometry=trias_to_modify, crs=mesh_subset.crs)

        def is_conforming(row) -> bool:
            lr_left = LinearRing(row.geometry.boundary)  # tria to mod
            lr_right = LinearRing(conforming_trias_gdf.loc[row.index_right].geometry.boundary)
            return elements_are_conforming(lr_left, lr_right)

        conforming_trias_buffered_gdf = conforming_trias_gdf.copy()
        conforming_trias_buffered_gdf.geometry = conforming_trias_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
        joined = gpd.sjoin(
                trias_to_mod_gdf,
                conforming_trias_buffered_gdf,
                how='left',
                predicate='intersects',
                )

        is_conforming_bool_series = joined.apply(is_conforming, axis=1)
        always_conforming = is_conforming_bool_series.groupby(is_conforming_bool_series.index).all()
        indices_always_conforming = always_conforming[always_conforming].index
        to_check_against_quads = trias_to_mod_gdf.loc[indices_always_conforming]
        all_indices = trias_to_mod_gdf.index
        indices_not_always_conforming = all_indices.difference(indices_always_conforming)
        to_check_against_trias = trias_to_mod_gdf.loc[indices_not_always_conforming]
        return to_check_against_trias, to_check_against_quads

    to_check_against_trias, to_check_against_quads = get_groups()

    against_trias_joined = gpd.sjoin(
            conforming_trias_gdf,
            to_check_against_trias,
            how='inner',
            predicate='intersects'
            )

    the_bad_trias = to_check_against_trias.loc[against_trias_joined.index_right.unique()]
    logger.debug(f"{the_bad_trias=}")

    for bad_tria_index in the_bad_trias.index:
        bad_tria = the_bad_trias.loc[bad_tria_index]
        logger.debug(f"{bad_tria=}")

        conforming_intersections = against_trias_joined[against_trias_joined.index == bad_tria_index]
        logger.debug(f"{conforming_intersections=}")

        breakpoint()
        for _, intersecting_conforming in conforming_intersections.iterrows():
            conforming_tria = conforming_trias_gdf.loc[intersecting_conforming.index_right]

            # Iterate over each node of the conforming triangle
            for node in conforming_tria.geometry.exterior.coords:
                # Assuming 'select_node_to_move' returns the index of the node in the bad triangle to move
                node_indexes = select_nodes_to_move(bad_tria, mesh_subset_quad_edges_gdf, mesh_subset_quad_edges_buffered_gdf)
                # if len(node_indexes) > 1:
                #     break
                # Move the bad triangle's node to the position of the conforming triangle's node
                # and create a new triangle to test for conformity
                test_tria = move_node(bad_tria, node_index, node)

                # Check if the new triangle is conforming
                if is_conforming(test_tria, conforming_tria):
                    # If it is conforming, update the bad triangle geometry
                    to_check_against_trias.at[bad_tria_index, 'geometry'] = test_tria.geometry
                    break  # Break if you only want the first conforming match, otherwise remove this line to test all nodes


    against_quad_edges_joined = gpd.sjoin(
            mesh_subset_quad_edges_buffered_gdf,
            to_check_against_quads,
            how='inner',
            predicate='intersects'
            )





    breakpoint()
    # find the



def get_new_tria_poly_list_from_group_and_mesh_gdf(group, mesh_gdf) -> List[Polygon]:
    group_id, group = group
    epsilon = np.finfo(np.float32).eps
    intersecting_edges = get_intersecting_edges(group) # List[LineString]
    trias_to_drop = group[group["element_type"] == "tria"]
    def get_mesh_subset():
        trias_to_drop_buffered_gdf = trias_to_drop.copy()
        trias_to_drop_buffered_gdf.drop(inplace=True, columns=["index_right"])
        trias_to_drop_buffered_gdf.geometry = trias_to_drop.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
        possible_matches_index = list(mesh_gdf.sindex.intersection(trias_to_drop_buffered_gdf.total_bounds))
        joined = gpd.sjoin(
                trias_to_drop_buffered_gdf,
                mesh_gdf.iloc[possible_matches_index],
                how='inner',
                predicate='intersects'
                )
        return mesh_gdf.iloc[joined.index_right.unique()].drop(index=trias_to_drop.index)
    mesh_subset = get_mesh_subset()
    linear_rings = get_linear_rings_from_trias_and_edges(mesh_subset, trias_to_drop, intersecting_edges)
    triangle_candidates = []
    outside_triangles = []
    for linear_ring in linear_rings:
        if linear_ring.is_valid:
            polygon_buffered = Polygon(linear_ring).buffer(epsilon)
            for tria in ops.triangulate(linear_ring):
                if tria.buffer(-epsilon).within(polygon_buffered):
                    triangle_candidates.append(tria)
                else:
                    outside_triangles.append(tria)
    triangle_candidates_gdf = gpd.GeoDataFrame(geometry=triangle_candidates, crs=mesh_gdf.crs)
    triangle_candidates_buffered_gdf = triangle_candidates_gdf.copy()
    triangle_candidates_buffered_gdf.geometry = triangle_candidates_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
    joined = gpd.sjoin(
            triangle_candidates_buffered_gdf,
            mesh_subset,
            how='inner',
            predicate='intersects',
            )
    # reset the joined geometries to non-buffered poly
    joined.geometry = triangle_candidates_gdf.loc[joined.index].geometry

    def is_conforming(row):
        ls_left = LinearRing(row.geometry.exterior.coords)
        ls_right = LinearRing(mesh_gdf.loc[row.index_right].geometry.exterior.coords)
        return elements_are_conforming(ls_left, ls_right)

    is_conforming_bool_series = joined.apply(is_conforming, axis=1)

    if np.all(is_conforming_bool_series.values):
        logger.debug(f"No problems with {group_id=}, they were all conforming!")
        return triangle_candidates

    # attempt #2
    always_conforming = is_conforming_bool_series.groupby(is_conforming_bool_series.index).all()
    indices_always_conforming = always_conforming[always_conforming].index
    candidates_conforming = triangle_candidates_gdf.loc[indices_always_conforming]
    triangle_candidates = list(candidates_conforming.geometry)

    # mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
    # gpd.GeoDataFrame(geometry=triangle_candidates).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
    # trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
    # gpd.GeoDataFrame(geometry=linear_rings).plot(edgecolor='magenta', alpha=0.5, ax=plt.gca())
    # plt.title(f"{group_id=} are not all conforming after the first try (based on linear ring)")
    # plt.show(block=False)
    # breakpoint()
    # raise


    tc_uu = ops.unary_union(MultiPolygon(triangle_candidates))
    tc_uu_buffered = tc_uu.buffer(np.finfo(np.float32).eps)
    # if not trias_to_drop.unary_union.within(tc_uu_buffered):
    logger.debug(f"{group_id=} failed once, trying to fix...")
    linear_rings_as_polys = MultiPolygon([Polygon(x) for x in linear_rings])
    try:
        what_were_missing = linear_rings_as_polys.difference(tc_uu)

        if isinstance(what_were_missing, Polygon):
            what_were_missing = MultiPolygon([what_were_missing])
        if not isinstance(what_were_missing, MultiPolygon):
            raise ValueError(f"Unreachable: Expected Polygon or MultiPolygon but got {what_were_missing=}")
        poly_coll = []
        for item in what_were_missing.geoms:
            this_triangles = ops.triangulate(item)
            for tria in this_triangles:
                if not tria.within(tc_uu_buffered):
                    poly_coll.append(tria)
        new_candidates_gdf = gpd.GeoDataFrame(geometry=poly_coll, crs=mesh_gdf.crs)
        triangle_candidates_buffered_gdf = new_candidates_gdf.copy()
        triangle_candidates_buffered_gdf.geometry = new_candidates_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
        joined = gpd.sjoin(
                triangle_candidates_buffered_gdf,
                mesh_subset,
                how='inner',
                predicate='intersects',
                )
        # reset the joined geometries to non-buffered poly
        joined.geometry = new_candidates_gdf.loc[joined.index].geometry

        is_conforming_bool_series = joined.apply(is_conforming, axis=1)

        always_conforming = is_conforming_bool_series.groupby(is_conforming_bool_series.index).all()
        indices_always_conforming = always_conforming[always_conforming].index
        candidates_conforming = new_candidates_gdf.loc[indices_always_conforming]
        triangle_candidates.extend(list(candidates_conforming.geometry))

        tc_uu = ops.unary_union(MultiPolygon(triangle_candidates))
        tc_uu_buffered = tc_uu.buffer(np.finfo(np.float32).eps)
        if not trias_to_drop.unary_union.within(tc_uu_buffered):
            return do_extended_tria_method(mesh_subset, trias_to_drop, triangle_candidates, reason='second attempt failed')
            # # return triangle_candidates
            # mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
            # gpd.GeoDataFrame(geometry=triangle_candidates).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
            # trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
            # new_candidates_gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='green')
            # # gpd.GeoDataFrame(geometry=linear_rings).plot(edgecolor='magenta', alpha=0.5, ax=plt.gca())
            # # participant_points = get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges) # List[Point]
            # # gpd.GeoDataFrame(geometry=participant_points, crs=mesh_subset.crs).plot(ax=plt.gca(), color='green')
            # plt.title(f"{group_id=} second attempt failed")
            # plt.show(block=False)
            # breakpoint()
            # raise
        return triangle_candidates
    except shapely.errors.GEOSException as err:
        if "Nested shells" in str(err):
            return do_extended_tria_method(mesh_subset, trias_to_drop, triangle_candidates, reason='nested shells')
            # return triangle_candidates
            # mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
            # gpd.GeoDataFrame(geometry=triangle_candidates).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
            # trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
            # # new_candidates_gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='green')
            # gpd.GeoDataFrame(geometry=triangle_candidates).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
            # # gpd.GeoDataFrame(geometry=linear_rings).plot(edgecolor='magenta', alpha=0.5, ax=plt.gca())
            # # participant_points = get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges) # List[Point]
            # # gpd.GeoDataFrame(geometry=participant_points, crs=mesh_subset.crs).plot(ax=plt.gca(), color='green')
            # plt.title(f"{group_id=} it has nested shells failed")
            # plt.show(block=False)
            # breakpoint()
            # raise
            # gpd.GeoDataFrame(geometry=[linear_rings_as_polys]).plot(ax=plt.gca(), facecolor='none', edgecolor='r')
            # gpd.GeoDataFrame(geometry=[tc_uu]).plot(ax=plt.gca(), facecolor='none', edgecolor='g')
            # plt.show(block=False)
            # breakpoint()
        else:
            raise





        # .plot(ax=plt.gca(), facecolor='none', edgecolor='magenta', alpha=0.3)
        # plt.show(block=False)
        # breakpoint()






        # triangle_candidates_gdf = gpd.GeoDataFrame(geometry=triangle_candidates, crs=mesh_gdf.crs)
        # triangle_candidates_buffered_gdf = triangle_candidates_gdf.copy()
        # triangle_candidates_buffered_gdf.geometry = triangle_candidates_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
        # this_uu = ops.unary_union([triangle_candidates_buffered_gdf.unary_union, mesh_subset.unary_union])
        # outside_triangles_gdf = gpd.GeoDataFrame(geometry=outside_triangles)
        # outside_triangles_uu = outside_triangles_gdf.unary_union
        # hole_to_pad = outside_triangles_uu.buffer(np.finfo(np.float32).eps).difference(this_uu)
        # the_new_trias = []
        # for tria in ops.triangulate(hole_to_pad):
        #     if tria.buffer(-epsilon).within(hole_to_pad):
        #         the_new_trias.append(tria)
        #     # else:
        #     #     outside_triangles.append(tria)

        # triangle_candidates_gdf = gpd.GeoDataFrame(geometry=the_new_trias, crs=mesh_gdf.crs)
        # triangle_candidates_buffered_gdf = triangle_candidates_gdf.copy()
        # triangle_candidates_buffered_gdf.geometry = triangle_candidates_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
        # joined = gpd.sjoin(
        #         triangle_candidates_buffered_gdf,
        #         mesh_subset,
        #         how='inner',
        #         predicate='intersects',
        #         )
        # # reset the joined geometries to non-buffered poly
        # joined.geometry = triangle_candidates_gdf.loc[joined.index].geometry
        # is_conforming_bool_series = joined.apply(is_conforming, axis=1)
        # always_conforming = is_conforming_bool_series.groupby(is_conforming_bool_series.index).all()
        # indices_always_conforming = always_conforming[always_conforming].index
        # candidates_conforming = triangle_candidates_gdf.loc[indices_always_conforming]
        # triangle_candidates.extend(list(candidates_conforming.geometry))

        # attempt #3
        # if not trias_to_drop.unary_union.within(ops.unary_union(MultiPolygon(triangle_candidates).buffer(np.finfo(np.float32).eps))):
            # find the trias to drop not entirely within the triangle candidate, extend their non-conforming edges,
            # then take the difference and retriangulate




            # mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
            # # gpd.GeoDataFrame(geometry=the_new_trias).plot(ax=plt.gca(), facecolor='magenta', edgecolor='magenta', alpha=0.3)
            # gpd.GeoDataFrame(geometry=triangle_candidates).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
            # trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
            # # gpd.GeoDataFrame(geometry=new_triangle_candidates).plot(facecolor='green', edgecolor='g', alpha=0.5, ax=plt.gca())
            # plt.title(f"{group_id=} still fails")
            # # participant_points = get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges) # List[Point]
            # # gpd.GeoDataFrame(geometry=participant_points, crs=mesh_subset.crs).plot(ax=plt.gca(), color='green')
            # plt.show(block=False)
            # breakpoint()
            # raise




            # # participant_points = get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges) # List[Point]
            # # new_triangle_candidates = ops.triangulate(MultiPoint(participant_points))
            # # previous_triangle_candidates_gdf = gpd.GeoDataFrame(geometry=triangle_candidates, crs=mesh_gdf.crs)
            # # new_mesh_subset = pd.concat([mesh_subset, previous_triangle_candidates_gdf])
            # # new_triangle_candidates_gdf = gpd.GeoDataFrame(geometry=new_triangle_candidates, crs=mesh_gdf.crs)
            # # new_triangle_candidates_buffered_gdf = new_triangle_candidates_gdf.copy()
            # # new_triangle_candidates_buffered_gdf.geometry = new_triangle_candidates_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
            # # joined = gpd.sjoin(
            # #         new_triangle_candidates_buffered_gdf,
            # #         new_mesh_subset,
            # #         how='inner',
            # #         predicate='intersects',
            # #         )
            # # joined.geometry = new_triangle_candidates_gdf.loc[joined.index].geometry
            # # is_conforming_bool_series = joined.apply(is_conforming, axis=1)
            # # always_conforming = is_conforming_bool_series.groupby(is_conforming_bool_series.index).all()
            # # indices_always_conforming = always_conforming[always_conforming].index
            # # new_candidates_conforming = new_triangle_candidates_gdf.loc[indices_always_conforming]
            # # triangle_candidates.extend(list(new_candidates_conforming.geometry))


            # # new_triangles.extend(triangle_candidates)
            #     # triangle_candidates = manually_triangulate(linear_ring)
            #     # filter_triangle_candidates_mut(triangle_candidates, mesh_gdf.drop(index=trias_to_drop.index))
            #     # new_triangles.extend([Polygon(x) for x in triangle_candidates])





            # participant_points_gdf = gpd.GeoDataFrame(geometry=participant_points, crs=mesh_gdf.crs)
            # participant_points_buffered_gdf = participant_points_gdf.copy()
            # participant_points_buffered_gdf.geometry = participant_points_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
            # triangle_candidates_gdf = gpd.GeoDataFrame(geometry=triangle_candidates, crs=mesh_gdf.crs)
            # intersects_triangle_candidates = gpd.sjoin(
            #     participant_points_buffered_gdf,
            #     triangle_candidates_gdf,
            #     how='inner',
            #     predicate='intersects'
            # )
            # non_intersecting_points = participant_points_buffered_gdf[
            #     ~participant_points_buffered_gdf.index.isin(intersects_triangle_candidates.index)
            # ]
            # mesh_subset_edges = []
            # for poly in mesh_subset.geometry:
            #     exterior_ring = poly.exterior
            #     num_coords = len(exterior_ring.coords)
            #     for i in range(num_coords - 1):  # minus 1 to avoid repeating the first coordinate
            #         edge = LineString([exterior_ring.coords[i], exterior_ring.coords[i + 1]])
            #         mesh_subset_edges.append(edge)
            # mesh_subset_edges_gdf = gpd.GeoDataFrame(geometry=mesh_subset_edges, crs=mesh_subset.crs)
            # # Proceed with the original join operation using the filtered points
            # joined = gpd.sjoin(
            #     mesh_subset_edges_gdf,
            #     non_intersecting_points,
            #     how='inner',
            #     predicate='intersects'
            # )
            # joined.drop(columns=["index_right"], inplace=True)
            # joined = gpd.sjoin(
            #     joined,
            #     triangle_candidates_gdf,
            #     how='inner',
            #     predicate='intersects'
            # )
            # # the_new_trias = []
            # # breakpoint()
            # # ops.triangulate(MultiLineString(list(joined.geometry.unique())))
            # # for tria in ops.triangulate(joined.geometry):
            # #     # if tria.buffer(-epsilon).within(hole_to_pad):
            # #     the_new_trias.append(tria)
            #     # else:
            #     #     outside_triangles.append(tria)
            # the_new_trias = ops.triangulate(MultiLineString(list(joined.geometry.unique())))
            # triangle_candidates_gdf = gpd.GeoDataFrame(geometry=the_new_trias, crs=mesh_gdf.crs)
            # triangle_candidates_buffered_gdf = triangle_candidates_gdf.copy()
            # triangle_candidates_buffered_gdf.geometry = triangle_candidates_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
            # joined = gpd.sjoin(
            #         triangle_candidates_buffered_gdf,
            #         mesh_subset,
            #         how='inner',
            #         predicate='intersects',
            #         )
            # # reset the joined geometries to non-buffered poly
            # joined.geometry = triangle_candidates_gdf.loc[joined.index].geometry
            # is_conforming_bool_series = joined.apply(is_conforming, axis=1)
            # always_conforming = is_conforming_bool_series.groupby(is_conforming_bool_series.index).all()
            # indices_always_conforming = always_conforming[always_conforming].index
            # candidates_conforming = triangle_candidates_gdf.loc[indices_always_conforming]
            # new_triangle_candidates = list(candidates_conforming.geometry)
            # new_modified = [*triangle_candidates, *new_triangle_candidates]
            # # if not trias_to_drop.unary_union.within(ops.unary_union(MultiPolygon(new_modified).buffer(np.finfo(np.float32).eps))):
            # #     new_triangle_candidates = ops.triangulate(MultiPoint(participant_points))
            # #     previous_triangle_candidates_gdf = gpd.GeoDataFrame(geometry=triangle_candidates, crs=mesh_gdf.crs)
            # #     new_mesh_subset = pd.concat([mesh_subset, previous_triangle_candidates_gdf])
            # #     new_triangle_candidates_gdf = gpd.GeoDataFrame(geometry=new_triangle_candidates, crs=mesh_gdf.crs)
            # #     new_triangle_candidates_buffered_gdf = new_triangle_candidates_gdf.copy()
            # #     new_triangle_candidates_buffered_gdf.geometry = new_triangle_candidates_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
            # #     joined = gpd.sjoin(
            # #             new_triangle_candidates_buffered_gdf,
            # #             new_mesh_subset,
            # #             how='inner',
            # #             predicate='intersects',
            # #             )
            # #     joined.geometry = new_triangle_candidates_gdf.loc[joined.index].geometry
            # #     is_conforming_bool_series = joined.apply(is_conforming, axis=1)
            # #     always_conforming = is_conforming_bool_series.groupby(is_conforming_bool_series.index).all()
            # #     indices_always_conforming = always_conforming[always_conforming].index
            # #     new_candidates_conforming = new_triangle_candidates_gdf.loc[indices_always_conforming]
            # #     triangle_candidates.extend(list(new_candidates_conforming.geometry))
            # #     mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
            # #     # gpd.GeoDataFrame(geometry=the_new_trias).plot(ax=plt.gca(), facecolor='magenta', edgecolor='magenta', alpha=0.3)
            # #     gpd.GeoDataFrame(geometry=triangle_candidates).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
            # #     trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
            # #     gpd.GeoDataFrame(geometry=new_triangle_candidates).plot(facecolor='green', edgecolor='g', alpha=0.5, ax=plt.gca())
            # #     plt.title(f"{group_id=} still fails")
            # #     # participant_points = get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges) # List[Point]
            # #     # gpd.GeoDataFrame(geometry=participant_points, crs=mesh_subset.crs).plot(ax=plt.gca(), color='green')
            # #     plt.show(block=False)
            # #     breakpoint()
            # #     raise
            # # else:
            # #     triangle_candidates.extend(new_triangle_candidates)

        # # else:
            # # logger.debug(f"{group_id=} with centroid at {trias_to_drop.centroid=} passed on first fix attempt")



    # return triangle_candidates
    # I think this part is not necessary, tria.within handles this.

    # triangle_candidates = ops.triangulate(MultiPoint(participant_points))  # List[Polygon]
    # return triangle_candidates
    triangle_candidates_gdf = gpd.GeoDataFrame(geometry=triangle_candidates, crs=mesh_gdf.crs)
    triangle_candidates_buffered_gdf = triangle_candidates_gdf.copy()
    triangle_candidates_buffered_gdf.geometry = triangle_candidates_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))

    def get_mesh_subset():
        triangle_candidates_mp = MultiPolygon(triangle_candidates)
        possible_matches_index = list(mesh_gdf.sindex.intersection(triangle_candidates_mp.bounds))
        joined = gpd.sjoin(
                triangle_candidates_buffered_gdf,
                mesh_gdf.iloc[possible_matches_index],
                how='inner',
                predicate='intersects'
                )
        return mesh_gdf.iloc[joined.index_right.unique()]
    mesh_subset = get_mesh_subset()
    mesh_subset.drop(index=trias_to_drop.index, inplace=True)
    # verify
    # mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
    # gpd.GeoDataFrame(geometry=participant_points).plot(ax=plt.gca(), color='g')
    # gpd.GeoDataFrame(geometry=triangle_candidates).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
    # trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
    # plt.title("get new tria poly list from group and mesh_gdf")
    # plt.show(block=False)
    # breakpoint()
    # raise
    joined = gpd.sjoin(
            triangle_candidates_buffered_gdf,
            mesh_subset,
            how='inner',
            predicate='intersects',
            )
    # reset the joined geometries to non-buffered poly
    joined.geometry = triangle_candidates_gdf.loc[joined.index].geometry

    def is_conforming(row):
        # element_left = row.geometry
        # element_right = mesh_gdf.loc[row.index_right].geometry
        # if element_left.buffer(-np.finfo(np.float32).eps).overlaps(element_right.buffer(-np.finfo(np.float32).eps)):
        #     return False
        ls_left = LinearRing(row.geometry.exterior.coords)
        ls_right = LinearRing(mesh_gdf.loc[row.index_right].geometry.exterior.coords)
        return elements_are_conforming(ls_left, ls_right)

    is_conforming_bool_series = joined.apply(is_conforming, axis=1)

    if np.all(is_conforming_bool_series.values):
        return triangle_candidates
    else:
        return get_new_tria_poly_list_from_group_and_mesh_gdf2(group, mesh_gdf)

    # mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
    # gpd.GeoDataFrame(geometry=intersecting_edges).plot(ax=plt.gca(), color='g')
    # gpd.GeoDataFrame(geometry=triangle_candidates).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
    # trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
    # plt.title("This triangle set is not fully conforming")
    # plt.show(block=False)
    # breakpoint()
    # raise

    always_conforming = is_conforming_bool_series.groupby(is_conforming_bool_series.index).all()
    indices_always_conforming = always_conforming[always_conforming].index
    candidates_conforming = triangle_candidates_gdf.loc[indices_always_conforming]
    out_trias = candidates_conforming.geometry

    # if len(out_trias) == 0:
    #     return get_new_tria_poly_list_from_group_and_mesh_gdf2(group, mesh_gdf)
        # mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
        # gpd.GeoDataFrame(geometry=participant_points).plot(ax=plt.gca(), color='g')
        # gpd.GeoDataFrame(geometry=triangle_candidates).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
        # trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
        # gpd.GeoDataFrame(geometry=intersecting_edges).plot(ax=plt.gca(), color='green')
        # plt.title('No trias found!')
        # plt.show(block=False)
        # breakpoint()
    #     raise
#         raise
    return out_trias

def get_new_tria_poly_list_from_group_and_mesh_gdf2(group, mesh_gdf) -> List[Polygon]:
    id, group = group
    new_trias = []
    trias_to_drop = group[group["element_type"] == "tria"]
    intersecting_edges = get_intersecting_edges(group) # List[LineString]
    participant_points = get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges) # List[Point]
    convex_hull = MultiPoint(participant_points).convex_hull
    def get_mesh_subset():
        possible_matches_index = list(mesh_gdf.sindex.intersection(convex_hull.bounds))
        joined = gpd.sjoin(
                gpd.GeoDataFrame(geometry=[convex_hull], crs=mesh_gdf.crs),
                mesh_gdf.iloc[possible_matches_index],
                how='inner',
                predicate='intersects'
                )
        return mesh_gdf.iloc[joined.index_right.unique()].drop(index=trias_to_drop.index)
    mesh_subset = get_mesh_subset()

    the_patches = convex_hull.difference(mesh_subset.unary_union)
    if isinstance(the_patches, Polygon):
        the_patches = MultiPolygon([the_patches])
    elif isinstance(the_patches, MultiPolygon):
        pass # hooray!
    else:
        raise NotImplementedError(f"Got {the_patches=}")
    tria_coll = []
    tria_non_conf = []
    for the_patch in the_patches.geoms:
        triangle_candidates = ops.triangulate(the_patch)
        triangle_candidates_gdf = gpd.GeoDataFrame(geometry=triangle_candidates, crs=mesh_gdf.crs)
        triangle_candidates_buffered_gdf = triangle_candidates_gdf.copy()
        triangle_candidates_buffered_gdf.geometry = triangle_candidates_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
        joined = gpd.sjoin(
                triangle_candidates_buffered_gdf,
                mesh_subset,
                how='inner',
                predicate='intersects',
                )
        # reset the joined geometries to non-buffered poly
        joined.geometry = triangle_candidates_gdf.loc[joined.index].geometry

        def is_conforming(row):
            element_left = row.geometry
            element_right = mesh_gdf.loc[row.index_right].geometry
            if element_left.buffer(-np.finfo(np.float32).eps).overlaps(element_right.buffer(-np.finfo(np.float32).eps)):
                return False
            ls_left = LinearRing(row.geometry.exterior.coords)
            # ls_right = LinearRing(mesh_gdf.loc[row.index_right].geometry.exterior.coords)
            ls_right = LinearRing(mesh_subset.loc[row.index_right].geometry.exterior.coords)
            return elements_are_conforming(ls_left, ls_right)

        # is_conforming_bool_series = joined.apply(is_conforming, axis=1)

        the_conforming_ones = joined.apply(is_conforming, axis=1)
        if np.all(the_conforming_ones.values):
            tria_coll.extend(triangle_candidates)
            continue
        # indexes_to_drop = intersection_gdf.index_right.unique()
        new_triangles = triangle_candidates_gdf.loc[joined[the_conforming_ones].index.unique()]
        new_triangles_non_conforming = triangle_candidates_gdf.loc[joined[~the_conforming_ones].index.unique()]
        # always_conforming = the_conforming_ones.groupby(the_conforming_ones.index).all()
        # indices_always_conforming = always_conforming[always_conforming].index
        # candidates_conforming = triangle_candidates_gdf.loc[indices_always_conforming]
        # tria_coll.extend(candidates_conforming.geometry)
        tria_coll.extend(new_triangles.geometry)
        tria_non_conf.extend(new_triangles_non_conforming.geometry)

    logger.debug("check")
    if not trias_to_drop.unary_union.within(ops.unary_union(MultiPolygon(tria_coll).buffer(np.finfo(np.float32).eps))):
        mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
        gpd.GeoDataFrame(geometry=tria_coll).plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
        gpd.GeoDataFrame(geometry=[convex_hull]).plot(ax=plt.gca(), facecolor='none', edgecolor='green')
        plt.title('not fully padded')
        plt.show(block=False)
        breakpoint()
        raise


    return tria_coll

#     indices_non_conforming = is_conforming_bool_series[~is_conforming_bool_series].index.unique()
#     candidates_non_conforming = triangle_candidates_gdf.loc[indices_non_conforming]
#     all_potentially_conforming_indices = is_conforming_bool_series[is_conforming_bool_series].index.unique()
#     indices_conforming = [idx for idx in all_potentially_conforming_indices if idx not in indices_non_conforming]
#     return triangle_candidates_gdf.loc[indices_conforming].geometry.tolist()
    # # joined_conforming_index = joined[is_conforming_bool_series].index.unique()
    # candidates_conforming = triangle_candidates_gdf.loc[indices_conforming]
    # candidates_non_conforming = triangle_candidates_gdf.loc[indices_non_conforming]

    # # verify
    # mesh_subset.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3) # triangles that will be kept for this iteration
    # candidates_conforming.plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
    # candidates_non_conforming.plot(ax=plt.gca(), facecolor='g', edgecolor='g', alpha=0.3)
    # trias_to_drop.plot(ax=plt.gca(), facecolor='r', alpha=0.3)
    # plt.show(block=False)
    # breakpoint()
    # raise





    # def make_delaunay_triangula






    # participant_points = get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges)
    # triangle_candidates = ops.triangulate(participant_points)
    # new_triangles.extend(triangle_candidates)
        # triangle_candidates = manually_triangulate(linear_ring)
        # filter_triangle_candidates_mut(triangle_candidates, mesh_gdf.drop(index=trias_to_drop.index))
        # new_triangles.extend([Polygon(x) for x in triangle_candidates])



def filter_skinny(tria_list: typing.List[Polygon]) -> typing.List[Polygon]:
    buffer_size = np.finfo(np.float32).eps
    def is_skinny(polygon):
        return any(Point(p1).buffer(buffer_size).intersects(Point(p2).buffer(buffer_size))
                   for p1, p2 in zip(polygon.exterior.coords[:-1], polygon.exterior.coords[1:]))
    trias = gpd.GeoDataFrame(geometry=tria_list)
    trias['is_skinny'] = trias['geometry'].apply(is_skinny)
    if not trias['is_skinny'].any():
        return tria_list
    adjacent_pairs = gpd.sjoin(
            trias,
            trias,
            how='inner',
            predicate='intersects'
            )
    adjacent_pairs = adjacent_pairs[adjacent_pairs.index != adjacent_pairs.index_right]

    G = nx.Graph()
    for _, row in adjacent_pairs.iterrows():
        G.add_edge(row.name, row.index_right)
    connected_components = list(nx.connected_components(G))
    grouped_triangles = [component for component in connected_components if any(trias.loc[idx, 'is_skinny'] for idx in component)]

    breakpoint()

def point_on_ring_edge(p, ring):
    ring_coords = ring.coords[:]
    line_segments = [LineString([ring_coords[i], ring_coords[i + 1]]) for i in range(len(ring_coords) - 1)]
    return any(point_on_line_segment(p, segment) for segment in line_segments)



class Quads:

    def __init__(self, gdf: gpd.GeoDataFrame):
        self.quads_gdf = gdf

    def __call__(self, msh_t: jigsaw_msh_t) -> jigsaw_msh_t:
        omsh_t = jigsaw_msh_t()
        omsh_t.mshID = msh_t.mshID
        omsh_t.ndims = msh_t.ndims
        omsh_t.vert2 = msh_t.vert2.copy()
        omsh_t.tria3 = msh_t.tria3.copy()
        omsh_t.crs = msh_t.crs
        self.add_quads_to_msh_t(omsh_t)
        return omsh_t

        # if comm is None:
        #     omsh_t = jigsaw_msh_t()
        #     omsh_t.mshID = msh_t.mshID
        #     omsh_t.ndims = msh_t.ndims
        #     omsh_t.vert2 = msh_t.vert2.copy()
        #     omsh_t.tria3 = msh_t.tria3.copy()
        #     omsh_t.crs = msh_t.crs
        #     self.add_quads_to_msh_t(omsh_t)
        #     return omsh_t
        # else:
        #     self.__mpicall__(comm, msh_t)

    def __mpicall__(self, comm, msh_t):
        if comm.Get_rank() == 0:
            omsh_t = jigsaw_msh_t()
            omsh_t.mshID = msh_t.mshID
            omsh_t.ndims = msh_t.ndims
            omsh_t.vert2 = msh_t.vert2.copy()
            omsh_t.tria3 = msh_t.tria3.copy()
            omsh_t.crs = msh_t.crs
        else:
            omsh_t = None
        self._set_quads_poly_gdf_uu_with_mpi(comm)
        self.add_quads_to_msh_t_mpi(comm, omsh_t)







    # def _get_new_mesh_gdf_unclean_boundaries(self, msh_t):
    #     def get_unique_diff(gdf1, gdf2):
    #         return gdf1.unary_union.difference(gdf2.unary_union)

    #         # return [
    #         #     triangle for hole in get_multipolygon(holes).geoms
    #         #     if is_convex(hole.exterior.coords)
    #         #     for triangle in ops.triangulate(hole)
    #         # ]
    #         # return [
    #         #     triangle
    #         #     for hole in get_multipolygon(holes).geoms
    #         #     for triangle in ops.triangulate(hole)
    #         #     if triangle.within(hole)
    #         # ]

    #     tri_gdf = gpd.GeoDataFrame(
    #         geometry=[Polygon(msh_t.vert2['coord'][idx, :]) for idx in msh_t.tria3['index']],
    #         crs=msh_t.crs
    #     ).to_crs(self.quads_poly_gdf.crs)

    #     intersecting_tri = gpd.sjoin(tri_gdf, self.quads_poly_gdf, how='inner', predicate='intersects')
    #     to_drop_indexes = intersecting_tri.index.unique()

    #     the_diff_uu = get_unique_diff(tri_gdf.iloc[to_drop_indexes], self.quads_poly_gdf)
    #     triangulated_diff = [tri for tri in ops.triangulate(the_diff_uu) if tri.within(the_diff_uu)]

    #     new_mesh_gdf = pd.concat([
    #         tri_gdf.drop(index=to_drop_indexes),
    #         gpd.GeoDataFrame(geometry=triangulated_diff, crs=self.quads_poly_gdf.crs),
    #         self.quads_poly_gdf.to_crs(tri_gdf.crs),
    #     ], ignore_index=True)

    #     def get_multipolygon(poly):
    #         return MultiPolygon([poly]) if isinstance(poly, Polygon) else poly

    #     def get_triangles_from_holes(holes):
    #         return [
    #             triangle
    #             for hole in get_multipolygon(holes).geoms
    #             for triangle in ops.triangulate(hole)
    #             if is_convex(hole.exterior.coords) or
    #             # len(hole.exterior.coords) - 1 == 3 or  # Check if the hole has only 3 nodes (subtracting 1 because the first and last coordinate of a polygon are the same)
    #             triangle.buffer(-2*np.finfo(np.float32).eps).within(hole.buffer(2*np.finfo(np.float32).eps))
    #         ]
    #     holes_to_pad = get_unique_diff(tri_gdf, new_mesh_gdf).difference(self.quads_poly_gdf.unary_union)
    #     triangles_to_add = get_triangles_from_holes(holes_to_pad)
    #     new_mesh_gdf = pd.concat([new_mesh_gdf, gpd.GeoDataFrame(geometry=triangles_to_add, crs=self.quads_poly_gdf.crs)], ignore_index=True)

    #     prev_len = -1  # Set an initial value that will never be equal to the len of holes_to_pad on the first iteration.

    #     while len(holes_to_pad.geoms) != prev_len:
    #         prev_len = len(holes_to_pad.geoms)  # Store the current length to compare after updating
    #         holes_to_pad = get_unique_diff(tri_gdf, new_mesh_gdf).difference(self.quads_poly_gdf.unary_union)
    #         triangles_to_add = get_triangles_from_holes(holes_to_pad)
    #         new_mesh_gdf = pd.concat([new_mesh_gdf, gpd.GeoDataFrame(geometry=triangles_to_add, crs=self.quads_poly_gdf.crs)], ignore_index=True)
    #         holes_to_pad = get_unique_diff(tri_gdf, new_mesh_gdf).difference(self.quads_poly_gdf.unary_union)

    #     triangles_to_add = [Polygon(geom) for poly in holes_to_pad.geoms for geom in manually_triangulate(LinearRing(poly.boundary))]
    #     # new_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3)
    #     new_mesh_gdf = pd.concat([new_mesh_gdf, gpd.GeoDataFrame(geometry=triangles_to_add, crs=self.quads_poly_gdf.crs)], ignore_index=True)
    #     # holes_to_pad = get_unique_diff(tri_gdf, new_mesh_gdf).difference(self.quads_poly_gdf.unary_union)
    #     # gpd.GeoDataFrame(geometry=[holes_to_pad]).plot(ax=plt.gca(), facecolor='none', edgecolor='r', linewidth=0.7)

    #     # # holes_to_pad = get_unique_diff(tri_gdf, new_mesh_gdf).difference(self.quads_poly_gdf.unary_union)
    #     # # gpd.GeoDataFrame(geometry=[holes_to_pad]).plot(ax=plt.gca(), facecolor='none', edgecolor='g')

    #     # plt.title('the first round 1')
    #     # plt.show(block=False)
    #     # breakpoint()
    #     # raise
    #     # # for linear_ring in linear_rings:
    #     # #     if not linear_ring.is_valid:
    #     # #         from shapely import make_valid
    #     # #         logger.debug("linear_ring is not valid.")
    #     # #         logger.debug(f"{explain_validity(linear_ring)=}")
    #     # #         logger.debug(f"{make_valid(linear_ring)=}")
    #     # #         breakpoint()

    #     # def get_mesh_subset(lr):
    #     #     possible_matches_index = list(new_mesh_gdf.sindex.intersection(lr.bounds))
    #     #     return new_mesh_gdf.iloc[possible_matches_index]

    #     # new_triangles = [Polygon(geom) for lr in linear_rings for geom in manually_triangulate(
    #     #     lr,
    #     #     # get_mesh_subset(lr)
    #     #     )]
    #     # new_triangles = gpd.GeoDataFrame(geometry=new_triangles, crs=new_mesh_gdf.crs)
    #     # # # verify
    #     # new_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3, alpha=0.3)
    #     # new_triangles.plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3)
    #     # for x, y, label in zip(new_triangles.geometry.centroid.x, new_triangles.geometry.centroid.y, new_triangles.index):
    #     #     plt.gca().text(x, y, str(label), fontsize=12)
    #     # plt.title('the first round')
    #     # plt.show(block=False)
    #     # breakpoint()
    #     # raise

    #     # new_mesh_gdf = pd.concat([
    #     #     new_mesh_gdf,
    #     #     new_triangles,
    #     # ], ignore_index=True).to_crs(msh_t.crs)
    #     new_mesh_gdf['element_type'] = new_mesh_gdf['quad_group_id'].map(lambda x: 'tria' if np.isnan(x) else 'quad')
    #     new_mesh_gdf.drop(columns=['quad_group_id', 'quad_row_id', 'quad_id',
    #         # 'normal_vector'
    #         ], inplace=True)
    #     return new_mesh_gdf

    @staticmethod
    def _are_within(geometry, original_mesh_gdf_tmpfile):
        original_mesh_gdf = gpd.read_feather(original_mesh_gdf_tmpfile)
        possible_matches_index = list(original_mesh_gdf.sindex.query(geometry, predicate="intersects"))
        possible_matches = original_mesh_gdf.iloc[possible_matches_index]
        return possible_matches[possible_matches.geometry.within(geometry)].index

    def _get_new_msh_t_mpi(self, comm, msh_t):

        if comm.Get_rank() == 0:
            import tempfile
            from time import time
            quads_poly_gdf_uu_tmpfile = tempfile.NamedTemporaryFile(dir=Path.cwd(), suffix=".feather")

            print(f"begin generation of quads_poly_gdf_uu on rank={comm.Get_rank()}", flush=True)
            start = time()
            gpd.GeoDataFrame(geometry=[geom for geom in self.quads_poly_gdf_uu.geoms], crs=self.quads_poly_gdf.crs).to_feather(quads_poly_gdf_uu_tmpfile.name)
            print(f"Took: {time()-start}", flush=True)
        else:
            quads_poly_gdf_uu_tmpfile = None

        # print(f"Creating quads_poly_gdf_uu on rank {comm.Get_rank()}", flush=True)
        # from geomesh.cli.mpi.geom_build import combine_geoms, logger as geom_build_logger
        # geom_build_logger.setLevel(logging.DEBUG)
        # # quads_poly_gdf = comm.bcast(self.quads_poly_gdf, root=0)
        # # quads_poly_gdf_uu = combine_geoms(
        # #         comm,
        # #         self.quads_poly_gdf,
        # #         )
        # print(f"done with combine geoms", flush=True)
        # if comm.Get_rank() != 0:
        #     del self.quads_poly_gdf
        #     del self.quads_poly
        #     del quads_poly_gdf_uu
        # else:

        comm.barrier()

        def get_target_mesh_mpi():
            target_mesh = None
            with MPICommExecutor(comm) as executor:
                if executor is not None:
                    print("make coords", flush=True)
                    # coords = msh_t.vert2['coord'][msh_t.tria3['index'], :].tolist()
                    # print(coords, flush=True)
                    print("make gdf", flush=True)
                    original_mesh_gdf = gpd.GeoDataFrame(
                        # geometry=[Polygon(msh_t.vert2['coord'][idx, :]) for idx in msh_t.tria3['index']],
                        # geometry=list(map(Polygon, coords)),
                        geometry=list(map(Polygon, msh_t.vert2['coord'][msh_t.tria3['index'], :].tolist())),
                        # geometry=list(
                        #     executor.map(
                        #         Polygon,
                        #         # msh_t.vert2['coord'][msh_t.tria3['index'], :].tolist(),
                        #         coords,
                        #         )),
                        # geometry=np.vectorize(Polygon)(msh_t.vert2['coord'][msh_t.tria3['index'], :].tolist()),
                        crs=msh_t.crs
                    )
                    print("transform crs", flush=True)
                    original_mesh_gdf = original_mesh_gdf.to_crs(self.quads_poly_gdf.crs)
                    print("do sjoin for get_target_mesh_mpi", flush=True)
                    joined = gpd.sjoin(
                        original_mesh_gdf,
                        gpd.GeoDataFrame(geometry=[geom for geom in self.quads_poly_gdf_uu.geoms], crs=self.quads_poly_gdf.crs),
                        how='left',
                        predicate='within'
                        )
                    target_mesh = original_mesh_gdf.loc[joined[joined.index_right.isna()].index.unique()]

                    # original_mesh_gdf_tmpfile = tempfile.NamedTemporaryFile(dir=Path.cwd(), prefix='.', suffix=".feather")
                    # original_mesh_gdf.to_feather(original_mesh_gdf_tmpfile.name)

                    # print(f"Begin parallel is_within computation {comm.Get_rank()}", flush=True)
                    # to_remove = []
                    # for indexes in list(executor.starmap(
                    #     self._are_within,
                    #     [(geom, original_mesh_gdf_tmpfile.name) for geom in self.quads_poly_gdf_uu.geoms]
                    #     )):
                    #     to_remove.extend(indexes)
                    # target_mesh = original_mesh_gdf.loc[original_mesh_gdf.index.difference(to_remove)]
                    target_mesh["element_type"] = "tria"
                    print(target_mesh, flush=True)
            return target_mesh

        def handle_quads_fully_within_mpi(target_mesh):
            with MPICommExecutor(comm) as executor:
                if executor is not None:
                    print("start joined", flush=True)
                    joined = gpd.sjoin(
                            self.quads_poly_gdf,
                            target_mesh[target_mesh["element_type"] == "tria"],
                            how='inner',
                            predicate='within'
                            )
                    print("begin making new_triangles", flush=True)
                    new_triangles = []
                    quads_to_keep = []
                    for index_right, quad_group in joined.groupby("index_right"):
                        quad_points = []
                        for quad in quad_group.geometry:
                            quads_to_keep.append(quad)
                            quad_points.extend([Point(x) for x in quad.exterior.coords[:-1]])
                        tria_points = [Point(x) for x in target_mesh.loc[index_right].geometry.exterior.coords[:-1]]
                        new_triangles.append(MultiPoint(quad_points + tria_points))
                    new_triangles = [item for sublist in executor.map(ops.triangulate, new_triangles) for item in sublist]
                    print("done making new_triangles", flush=True)
                    new_triangles = gpd.GeoDataFrame(geometry=new_triangles, crs=target_mesh.crs)
                    new_triangles["element_type"] = "tria"
                    quads_to_keep = gpd.GeoDataFrame(geometry=quads_to_keep, crs=target_mesh.crs)
                    quads_to_keep["element_type"] = "quad"
                    trias_to_drop = target_mesh.loc[joined.index_right.unique()]
                    target_mesh = target_mesh.drop(index=joined.index_right.unique())
                    print("begin making quads_exterior_bds", flush=True)
                    quads_exterior_bds = gpd.GeoDataFrame(
                            geometry=[geom.buffer(np.finfo(np.float32).eps) for geom in self.quads_poly_gdf_uu.geoms],
                            crs=self.quads_poly_gdf.crs,
                            )
                    print("begin making joined from quads_exterior_bds", flush=True)
                    joined = gpd.sjoin(
                            new_triangles,
                            quads_exterior_bds,
                            how='inner',
                            predicate='within',
                            )
                    new_triangles = new_triangles.loc[new_triangles.index.difference(joined.index)]
                    target_mesh = pd.concat([target_mesh, new_triangles, quads_to_keep], ignore_index=True)
                    print("begin making holes_to_pad", flush=True)
                    # nodes, elements = poly_gdf_to_elements(target_mesh)
                    # msh_t = self._jigsaw_msh_t_from_nodes_elements(nodes, elements, crs=target_mesh.crs)
                    # del nodes, elements
                    # target_mesh_geom_mp = utils.get_geom_msh_t_from_msh_t_as_mp(msh_t)
                    # del msh_t
                    # nodes, elements = poly_gdf_to_elements(trias_to_drop)
                    # msh_t = self._jigsaw_msh_t_from_nodes_elements(nodes, elements, crs=target_mesh.crs)
                    # del nodes, elements
                    # trias_to_drop_geom_mp = utils.get_geom_msh_t_from_msh_t_as_mp(msh_t)
                    # del msh_t
                    # holes_to_pad = trias_to_drop_geom_mp.difference(target_mesh_geom_mp)
                    # del target_mesh_geom_mp, trias_to_drop_geom_mp
                    # TODO: Cast holes_to_pad to MultiPolygon
                    holes_to_pad = trias_to_drop.unary_union.difference(target_mesh.unary_union)
                    print("begin manually_triangulate", flush=True)
                    triangles_to_add = []
                    for poly in holes_to_pad.geoms:
                        poly_boundary = poly.boundary
                        if isinstance(poly_boundary, LineString):
                            triangles_to_add.append(LinearRing(poly_boundary))
                            # for geom in manually_triangulate(LinearRing(poly_boundary)):
                            #     triangles_to_add.append(Polygon(geom))
                        elif isinstance(poly_boundary, MultiLineString):
                            triangles_to_add.extend([LinearRing(x) for x in poly_boundary.geoms])
                            # for ls in poly_boundary.geoms:

                                # for geom in manually_triangulate(LinearRing(ls)):
                                #     triangles_to_add.append(Polygon(geom))
                        else:
                            raise NotImplementedError(f"Unreachable: Expected LineString or MultiLineString but got {poly_boundary=}")
                    # new_triangles = [item for sublist in executor.map(ops.triangulate, new_triangles) for item in sublist]
                    triangles_to_add = [item for sublist in executor.map(manually_triangulate, triangles_to_add) for item in sublist]
                    # triangles_to_add = [Polygon(geom) for poly in holes_to_pad.geoms for geom in manually_triangulate(LinearRing(poly.boundary))]
                    triangles_to_add = gpd.GeoDataFrame(geometry=triangles_to_add, crs=target_mesh.crs)
                    triangles_to_add["element_type"] = "tria"
                    target_mesh = pd.concat([target_mesh, triangles_to_add], ignore_index=True)
            return target_mesh

        def handle_triangles_that_intersect_mpi(target_mesh):
            with MPICommExecutor(comm) as executor:
                if executor is not None:
                    joined = gpd.sjoin(
                            target_mesh[target_mesh["element_type"] == "tria"],
                            self.quads_poly_gdf,
                            how='inner',
                            predicate='intersects'
                            )
                    trias_to_drop = target_mesh.loc[joined.index.unique()]

                    the_holes_to_triangulate = trias_to_drop.difference(self.quads_poly_gdf_uu)
                    new_triangles = []
                    for geometry in the_holes_to_triangulate:
                        if isinstance(geometry, Polygon):
                            geometry = MultiPolygon([geometry])
                        if not isinstance(geometry, MultiPolygon):
                            raise ValueError(f"Unreachable: Expected a Polygon or MultiPolygon but got {geometry=}")
                        for poly in geometry.geoms:
                            if len(poly.exterior.coords[:-1]) == 3:
                                new_triangles.append(poly)
                                continue
                            this_triangles = [Polygon(x) for x in manually_triangulate(LinearRing(poly.boundary))]
                            new_triangles.extend(this_triangles)
                            # new_triangles.extend(filter_skinny(this_triangles))
                            # this_trias = [tria for tria in ops.triangulate(poly.boundary) if tria.within(poly.boundary)]
                            # remainder = poly.difference(MultiPolygon(this_trias))
                            # new_triangles.extend(this_trias)
                            # if not remainder.is_empty:
                            #     new_triangles.extend([Polygon(x) for x in manually_triangulate(LinearRing(remainder.boundary))])

                    target_mesh = target_mesh.drop(index=joined.index.unique())
                    new_triangles = gpd.GeoDataFrame(geometry=new_triangles, crs=target_mesh.crs)
                    new_triangles["element_type"] = "tria"
                    # verify
                    # target_mesh.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
                    # new_triangles.plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3, linewidth=0.3)
                    # plt.show(block=False)
                    # breakpoint()
                    # raise
                    target_mesh = pd.concat([target_mesh, new_triangles], ignore_index=True)
            return target_mesh

        def handle_quads_that_intersect_mpi(target_mesh):
            with MPICommExecutor(comm) as executor:
                if executor is not None:
                    eps = np.finfo(np.float32).eps
                    tria_points = []
                    for row in target_mesh[target_mesh["element_type"] == "tria"].itertuples():
                        tria_points.extend(Point(x) for x in row.geometry.exterior.coords[:-1])
                    tria_points = gpd.GeoDataFrame(geometry=tria_points, crs=target_mesh.crs)
                    tria_points_buffered = tria_points.copy()
                    tria_points_buffered.geometry = tria_points.geometry.map(lambda x: x.buffer(eps))
                    joined = gpd.sjoin(
                            tria_points_buffered,
                            self.quads_poly_gdf,
                            how='inner',
                            predicate='intersects'
                            )
                    # joined.geometry = tria_points.loc[joined.index].geometry
                    the_new_trias = []
                    for index_right, group in joined.groupby("index_right"):
                        the_quad_to_modify = LinearRing(self.quads_poly_gdf.loc[index_right].geometry.exterior.coords[:-1])
                        the_points_to_triangulate = [Point(x) for x in the_quad_to_modify.coords[:-1]]
                        the_quad_points_buffered = gpd.GeoDataFrame(geometry=the_points_to_triangulate, crs=target_mesh.crs)
                        the_quad_points_buffered.geometry = the_quad_points_buffered.geometry.map(lambda x: x.buffer(eps))
                        joined2 = gpd.sjoin(
                                tria_points.loc[group.index],
                                the_quad_points_buffered,
                                how='left',
                                predicate='intersects'
                                )
                        if not joined2.index_right.isna().any():  # all the points are quad boundary points
                            continue
                        the_points_to_triangulate.extend([p for p in joined2.geometry if point_on_ring_edge((p.x, p.y), the_quad_to_modify)])
                        geometry = [tria for tria in ops.triangulate(MultiPoint(the_points_to_triangulate))]
                        the_new_trias.extend(geometry)
                    the_new_trias = gpd.GeoDataFrame(geometry=the_new_trias, crs=target_mesh.crs)
                    the_new_trias["element_type"] = "tria"
                    # # verify:
                    target_mesh.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
                    the_new_trias.plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3, linewidth=0.3)
                    plt.show(block=False)
                    breakpoint()
                    raise
                    target_mesh = pd.concat([target_mesh, the_new_trias], ignore_index=True)
            return target_mesh

        def put_the_remaining_quads(target_mesh):
            with MPICommExecutor(comm) as executor:
                if executor is not None:
                    quads_poly_buffered = self.quads_poly_gdf.copy()
                    quads_poly_buffered.geometry = quads_poly_buffered.geometry.map(lambda x: x.buffer(-np.finfo(np.float32).eps))
                    joined = gpd.sjoin(quads_poly_buffered, target_mesh, how='left', predicate='intersects')
                    joined = joined[joined.index_right.isna()]
                    target_mesh = pd.concat([target_mesh, self.quads_poly_gdf.loc[joined.index.unique()]], ignore_index=True)
            return target_mesh

        if comm.Get_rank() == 0:
            print("getting target mesh", flush=True)
        target_mesh = get_target_mesh_mpi()
        if comm.Get_rank() == 0:
            print("got target mesh", flush=True)
        target_mesh = handle_quads_fully_within_mpi(target_mesh)
        target_mesh = handle_triangles_that_intersect_mpi(target_mesh)
        target_mesh = handle_quads_that_intersect_mpi(target_mesh)
        target_mesh = put_the_remaining_quads(target_mesh)
        if comm.Get_rank() == 0:
            nodes, elements = poly_gdf_to_elements(target_mesh)
            msh_t = self._jigsaw_msh_t_from_nodes_elements(nodes, elements, crs=target_mesh.crs)
            geom_msh_t = utils.get_geom_msh_t_from_msh_t_as_mp(msh_t)
        # pass
        return msh_t



    def _get_new_msh_t(self, msh_t):



        def get_target_mesh():
            original_mesh_gdf = gpd.GeoDataFrame(
                geometry=[Polygon(msh_t.vert2['coord'][idx, :]) for idx in msh_t.tria3['index']],
                crs=msh_t.crs
            ).to_crs(self.quads_poly_gdf.crs)
            to_remove = original_mesh_gdf.geometry.map(lambda x: x.within(self.quads_poly_gdf_uu))
            target_mesh = original_mesh_gdf[~to_remove]
            target_mesh["element_type"] = "tria"
            return target_mesh

        def handle_quads_fully_within(target_mesh):
            joined = gpd.sjoin(
                    self.quads_poly_gdf,
                    target_mesh[target_mesh["element_type"] == "tria"],
                    how='inner',
                    predicate='within'
                    )
            new_triangles = []
            quads_to_keep = []
            for index_right, quad_group in joined.groupby("index_right"):
                quad_points = []
                for quad in quad_group.geometry:
                    quads_to_keep.append(quad)
                    quad_points.extend([Point(x) for x in quad.exterior.coords[:-1]])
                tria_points = [Point(x) for x in target_mesh.loc[index_right].geometry.exterior.coords[:-1]]
                new_triangles.extend(ops.triangulate(MultiPoint(quad_points + tria_points)))
            new_triangles = gpd.GeoDataFrame(geometry=new_triangles, crs=target_mesh.crs)
            new_triangles["element_type"] = "tria"
            quads_to_keep = gpd.GeoDataFrame(geometry=quads_to_keep, crs=target_mesh.crs)
            quads_to_keep["element_type"] = "quad"
            trias_to_drop = target_mesh.loc[joined.index_right.unique()]
            target_mesh = target_mesh.drop(index=joined.index_right.unique())
            quads_exterior_bds = gpd.GeoDataFrame(
                    geometry=[geom.buffer(np.finfo(np.float32).eps) for geom in self.quads_poly_gdf_uu.geoms],
                    crs=self.quads_poly_gdf.crs,
                    )
            joined = gpd.sjoin(
                    new_triangles,
                    quads_exterior_bds,
                    how='inner',
                    predicate='within',
                    )
            new_triangles = new_triangles.loc[new_triangles.index.difference(joined.index)]
            target_mesh = pd.concat([target_mesh, new_triangles, quads_to_keep], ignore_index=True)
            holes_to_pad = trias_to_drop.unary_union.difference(target_mesh.unary_union)
            triangles_to_add = [Polygon(geom) for poly in holes_to_pad.geoms for geom in manually_triangulate(LinearRing(poly.boundary))]
            triangles_to_add = gpd.GeoDataFrame(geometry=triangles_to_add, crs=target_mesh.crs)
            triangles_to_add["element_type"] = "tria"
            return pd.concat([target_mesh, triangles_to_add], ignore_index=True)

        def handle_triangles_that_intersect(target_mesh):
            joined = gpd.sjoin(
                    target_mesh[target_mesh["element_type"] == "tria"],
                    self.quads_poly_gdf,
                    how='inner',
                    predicate='intersects'
                    )
            trias_to_drop = target_mesh.loc[joined.index.unique()]

            the_holes_to_triangulate = trias_to_drop.difference(self.quads_poly_gdf_uu)
            new_triangles = []
            for geometry in the_holes_to_triangulate:
                if isinstance(geometry, Polygon):
                    geometry = MultiPolygon([geometry])
                if not isinstance(geometry, MultiPolygon):
                    raise ValueError(f"Unreachable: Expected a Polygon or MultiPolygon but got {geometry=}")
                for poly in geometry.geoms:
                    if len(poly.exterior.coords[:-1]) == 3:
                        new_triangles.append(poly)
                        continue
                    this_triangles = [Polygon(x) for x in manually_triangulate(LinearRing(poly.boundary))]
                    new_triangles.extend(this_triangles)
                    # new_triangles.extend(filter_skinny(this_triangles))
                    # this_trias = [tria for tria in ops.triangulate(poly.boundary) if tria.within(poly.boundary)]
                    # remainder = poly.difference(MultiPolygon(this_trias))
                    # new_triangles.extend(this_trias)
                    # if not remainder.is_empty:
                    #     new_triangles.extend([Polygon(x) for x in manually_triangulate(LinearRing(remainder.boundary))])

            target_mesh = target_mesh.drop(index=joined.index.unique())
            new_triangles = gpd.GeoDataFrame(geometry=new_triangles, crs=target_mesh.crs)
            new_triangles["element_type"] = "tria"
            # verify
            # target_mesh.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
            # new_triangles.plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3, linewidth=0.3)
            # plt.show(block=False)
            # breakpoint()
            # raise
            return pd.concat([target_mesh, new_triangles], ignore_index=True)

        def handle_quads_that_intersect(target_mesh):
            eps = np.finfo(np.float32).eps
            tria_points = []
            for row in target_mesh[target_mesh["element_type"] == "tria"].itertuples():
                tria_points.extend(Point(x) for x in row.geometry.exterior.coords[:-1])
            tria_points = gpd.GeoDataFrame(geometry=tria_points, crs=target_mesh.crs)
            tria_points_buffered = tria_points.copy()
            tria_points_buffered.geometry = tria_points.geometry.map(lambda x: x.buffer(eps))
            joined = gpd.sjoin(
                    tria_points_buffered,
                    self.quads_poly_gdf,
                    how='inner',
                    predicate='intersects'
                    )
            # joined.geometry = tria_points.loc[joined.index].geometry
            the_new_trias = []
            for index_right, group in joined.groupby("index_right"):
                the_quad_to_modify = LinearRing(self.quads_poly_gdf.loc[index_right].geometry.exterior.coords[:-1])
                the_points_to_triangulate = [Point(x) for x in the_quad_to_modify.coords[:-1]]
                the_quad_points_buffered = gpd.GeoDataFrame(geometry=the_points_to_triangulate, crs=target_mesh.crs)
                the_quad_points_buffered.geometry = the_quad_points_buffered.geometry.map(lambda x: x.buffer(eps))
                joined2 = gpd.sjoin(
                        tria_points.loc[group.index],
                        the_quad_points_buffered,
                        how='left',
                        predicate='intersects'
                        )
                if not joined2.index_right.isna().any():  # all the points are quad boundary points
                    continue
                the_points_to_triangulate.extend([p for p in joined2.geometry if point_on_ring_edge((p.x, p.y), the_quad_to_modify)])
                geometry = [tria for tria in ops.triangulate(MultiPoint(the_points_to_triangulate))]
                the_new_trias.extend(geometry)
            the_new_trias = gpd.GeoDataFrame(geometry=the_new_trias, crs=target_mesh.crs)
            the_new_trias["element_type"] = "tria"
            # # verify:
            # target_mesh.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
            # the_new_trias.plot(ax=plt.gca(), facecolor='b', edgecolor='b', alpha=0.3, linewidth=0.3)
            # plt.show(block=False)
            # breakpoint()
            # raise
            return pd.concat([target_mesh, the_new_trias], ignore_index=True)

        def put_the_remaining_quads(target_mesh):
            quads_poly_buffered = self.quads_poly_gdf.copy()
            quads_poly_buffered.geometry = quads_poly_buffered.geometry.map(lambda x: x.buffer(-np.finfo(np.float32).eps))
            joined = gpd.sjoin(quads_poly_buffered, target_mesh, how='left', predicate='intersects')
            joined = joined[joined.index_right.isna()]
            return pd.concat([target_mesh, self.quads_poly_gdf.loc[joined.index.unique()]], ignore_index=True)

        target_mesh = get_target_mesh()
        target_mesh = handle_quads_fully_within(target_mesh)
        target_mesh = handle_triangles_that_intersect(target_mesh)
        target_mesh = handle_quads_that_intersect(target_mesh)
        target_mesh = put_the_remaining_quads(target_mesh)

        nodes, elements = poly_gdf_to_elements(target_mesh)
        msh_t = self._jigsaw_msh_t_from_nodes_elements(nodes, elements, crs=target_mesh.crs)
        geom_msh_t = utils.get_geom_msh_t_from_msh_t_as_mp(msh_t)
        # pass
        return msh_t
        # # new_mesh_gdf = self._get_new_mesh_gdf_unclean_boundaries(msh_t)
        # new_msh_t = self._try_with_new_method(msh_t,
        #         # new_mesh_gdf
        #         )
        # # # verify
        # # new_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3)
        # # # for x, y, label in zip(tria_to_fix.geometry.centroid.x, tria_to_fix.geometry.centroid.y, tria_to_fix.index):
        # #     plt.gca().text(x, y, str(label), fontsize=12)
        # # plt.show(block=False)
        # # breakpoint()
        # # raise
        # # new_msh_t = self._cleanup_overlapping_edges(new_mesh_gdf)
        # # utils.split_bad_quality_quads(new_msh_t)
        # # utils.remove_flat_triangles(new_msh_t)
        # # self._fix_triangles_outside_SCHISM_skewness_tolerance(new_msh_t)
        # return new_msh_t


    #     target_mesh.plot(ax=plt.gca(), facecolor='lightgrey', alpha=0.3)
    #     target_mesh.plot(ax=plt.gca(), facecolor='none', edgecolor='k', linewidth=0.3)
    #     plt.show(block=False)
    #     breakpoint()
    #     raise
    #     # joined.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
    #     # plt.show(block=True)
    #     # raise


    #     # within_tri = gpd.sjoin(
    #     #         tri_gdf,
    #     #         self.quads_poly_gdf_uu, how='inner', predicate='within')
    #     # to_drop_indexes = intersecting_tri.index.unique()



    #     # original_boundary_mp = utils.get_geom_msh_t_from_msh_t_as_mp(msh_t)
    #     # self.quads_poly_gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='r')
    #     # the_resulting_mp = original_boundary_mp.difference(new_mesh_gdf.unary_union)

    #     # gpd.GeoDataFrame(geometry=[the_resulting_mp], crs=new_mesh_gdf.crs).plot(ax=plt.gca(), facecolor='none', edgecolor='r')
    #     # plt.show(block=False)

    #     def get_unique_diff(gdf1, gdf2):
    #         return gdf1.unary_union.difference(gdf2.unary_union)

    #         # return [
    #         #     triangle for hole in get_multipolygon(holes).geoms
    #         #     if is_convex(hole.exterior.coords)
    #         #     for triangle in ops.triangulate(hole)
    #         # ]
    #         # return [
    #         #     triangle
    #         #     for hole in get_multipolygon(holes).geoms
    #         #     for triangle in ops.triangulate(hole)
    #         #     if triangle.within(hole)
    #         # ]

    #     tri_gdf = gpd.GeoDataFrame(
    #         geometry=[Polygon(msh_t.vert2['coord'][idx, :]) for idx in msh_t.tria3['index']],
    #         crs=msh_t.crs
    #     ).to_crs(self.quads_poly_gdf.crs)

    #     intersecting_tri = gpd.sjoin(tri_gdf, self.quads_poly_gdf, how='inner', predicate='intersects')
    #     to_drop_indexes = intersecting_tri.index.unique()

    #     the_diff_uu = get_unique_diff(tri_gdf.iloc[to_drop_indexes], self.quads_poly_gdf)

    # def _cleanup_overlapping_edges(self, mesh_gdf):

    #     def get_intersection_gdf():

    #         def is_conforming(row):
    #             ls_left = LinearRing(row.geometry.exterior.coords)
    #             ls_right = LinearRing(mesh_gdf.loc[row.index_right].geometry.exterior.coords)
    #             return elements_are_conforming(ls_left, ls_right)

    #         tria_elements_buffered = mesh_gdf[mesh_gdf.element_type=='tria']
    #         tria_elements_buffered['geometry'] = tria_elements_buffered.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
    #         quad_elements_buffered = mesh_gdf[mesh_gdf.element_type=='quad']
    #         intersection_gdf = gpd.sjoin(
    #                 quad_elements_buffered,
    #                 tria_elements_buffered,
    #                 how='inner',
    #                 predicate='intersects'
    #                 )
    #         intersection_gdf = intersection_gdf[~intersection_gdf.apply(is_conforming, axis=1)]
    #         return intersection_gdf

    #     def custom_group(intersection_gdf):
    #         quads_to_consider_gdf = mesh_gdf.loc[intersection_gdf.index.unique()]
    #         trias_to_drop_gdf = mesh_gdf.loc[intersection_gdf.index_right.unique()]
    #         trias_to_drop_uu = trias_to_drop_gdf.unary_union
    #         quads_uu = quads_to_consider_gdf.unary_union
    #         full_uu = ops.unary_union([
    #             trias_to_drop_uu.buffer(np.finfo(np.float32).eps),
    #             quads_uu.buffer(np.finfo(np.float32).eps)
    #             ])
    #         if isinstance(full_uu, Polygon):
    #             full_uu = MultiPolygon([full_uu])
    #         elif not isinstance(full_uu, MultiPolygon):
    #             raise ValueError(f"Unreachable: Expected Polygon or MultiPolygon but got {full_uu=}")
    #         groups_envelope_gdf = gpd.GeoDataFrame(geometry=list(full_uu.geoms), crs=intersection_gdf.crs)
    #         groups_envelope_gdf.geometry = groups_envelope_gdf.geometry.buffer(np.finfo(np.float32).eps)
    #         elements_to_consider = pd.concat([quads_to_consider_gdf, trias_to_drop_gdf])
    #         joined = gpd.sjoin(elements_to_consider, groups_envelope_gdf, how='left', op='within')
    #         return joined.groupby('index_right')
    #         breakpoint()
    #         # verify
    #         # for group in



    #         graph = {}
    #         for idx, row in intersection_gdf.iterrows():
    #             graph[idx] = graph.get(idx, []) + [row['index_right']]
    #             graph[row['index_right']] = graph.get(row['index_right'], []) + [idx]
    #         # mesh_gdf.loc[intersection_gdf.index_right.unique()].plot(edgecolor='r')
    #         # intersection_gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='b')
    #         # plt.show(block=False)
    #         # breakpoint()
    #         # # Find connected components
    #         # try:
    #         components = find_connected_components(graph)
    #         # except RecursionError:
    #         #     breakpoint()
    #         #     raise

    #         # Group rows by their component
    #         grouped_rows = []
    #         for component in components:
    #             group = intersection_gdf[intersection_gdf.index.isin(component) | intersection_gdf['index_right'].isin(component)]
    #             grouped_rows.append(group)

    #         return grouped_rows

    #     intersection_gdf = get_intersection_gdf()
    #     # verify
    #     # mesh_gdf.loc[mesh_gdf.index.difference(intersection_gdf.index.union(intersection_gdf.index_right))].plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', linewidth=0.3, alpha=0.3)
    #     # mesh_gdf.loc[intersection_gdf.index.unique()].plot(ax=plt.gca(), facecolor='blue', edgecolor='b', linewidth=0.5, alpha=0.3)
    #     # mesh_gdf.loc[intersection_gdf.index_right.unique()].plot(ax=plt.gca(), facecolor='red', edgecolor='r', linewidth=0.5, alpha=0.3)
    #     # plt.show(block=False)
    #     # breakpoint()
    #     # raise
    #     new_triangles = []
    #     group_ids = []
    #     for group in custom_group(intersection_gdf):
    #         group_ids.append((group[0], group[1].centroid))
    #         new_triangles.extend(get_new_tria_poly_list_from_group_and_mesh_gdf(group, mesh_gdf))
    #         # trias_to_drop = mesh_gdf.loc[group.index_right.unique()]
    #         # intersecting_edges = get_intersecting_edges(group, mesh_gdf)
    #         # participant_points = get_participant_points_from_trias_and_edges(trias_to_drop, intersecting_edges)
    #         # triangle_candidates = ops.triangulate(MultiPoint([Point(x) for x in participant_points]))
    #         # new_triangles.extend(triangle_candidates)
    #         #     triangle_candidates = manually_triangulate(linear_ring)
    #         #     # filter_triangle_candidates_mut(triangle_candidates, mesh_gdf.drop(index=trias_to_drop.index))
    #         #     new_triangles.extend([Polygon(x) for x in triangle_candidates])


    #     # mesh_gdf.drop(intersection_gdf.index_right.unique()).plot(ax=plt.gca(), facecolor='none', edgecolor='lightgrey')
    #     # new_triangles_gdf =gpd.GeoDataFrame(geometry=new_triangles, crs=mesh_gdf.crs)
    #     # new_triangles_gdf.plot(ax=plt.gca(), facecolor='blue', edgecolor='b', alpha=0.3)
    #     # for group_id, centroid in group_ids:
    #     #     plt.gca().text(centroid.x.mean(), centroid.y.mean(), str(group_id), fontsize=12)
    #     # plt.show(block=False)
    #     # breakpoint()
    #     # raise

    #     new_triangles = gpd.GeoDataFrame(geometry=new_triangles, crs=mesh_gdf.crs)
    #     new_triangles_buffered = new_triangles.copy()
    #     new_triangles_buffered.geometry = new_triangles.geometry.apply(lambda x: x.buffer(np.finfo(np.float32).eps))
    #     joined = gpd.sjoin(
    #             new_triangles_buffered,
    #             mesh_gdf.loc[intersection_gdf.index.unique()],  # quads only
    #             how='inner',
    #             predicate='intersects'
    #             )
    #     joined.geometry = new_triangles.loc[joined.index].geometry
    #     def is_conforming(row):
    #         ls_left = LinearRing(row.geometry.exterior.coords)
    #         ls_right = LinearRing(mesh_gdf.loc[row.index_right].geometry.exterior.coords)
    #         return elements_are_conforming(ls_left, ls_right)
    #     the_conforming_ones = joined.apply(is_conforming, axis=1)
    #     indexes_to_drop = intersection_gdf.index_right.unique()
    #     new_triangles_conforming = new_triangles.loc[joined[the_conforming_ones].index.unique()]
    #     new_triangles_non_conforming = new_triangles.loc[joined[~the_conforming_ones].index.unique()]
    #     if len(new_triangles_non_conforming) > 0:
    #         new_triangles_non_conforming.plot(ax=plt.gca(), facecolor='green', edgecolor='green', alpha=0.3) # new bad elements
    #     # mesh_gdf.loc[mesh_gdf.index.difference(indexes_to_drop)].plot(ax=plt.gca(), facecolor='lightgrey', alpha=0.5) # faces of good elements
    #     # mesh_gdf.loc[mesh_gdf.index.difference(indexes_to_drop)].plot(ax=plt.gca(), facecolor='none', edgecolor='k', linewidth=0.3) # edges of good elements
    #     # mesh_gdf.loc[indexes_to_drop].plot(ax=plt.gca(), facecolor='red', edgecolor='red', alpha=0.3) # to drop
    #     # new_triangles_conforming.plot(ax=plt.gca(), facecolor='blue', edgecolor='b', alpha=0.3) # new good elements
    #     # plt.show(block=False)
    #     # breakpoint()
    #     # raise



    #     # mesh_gdf.drop(index=intersection_gdf.index_right.unique()).plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3) # trias that stay
    #     # mesh_gdf.drop(index=mesh_gdf.index.difference(intersection_gdf.index_right.unique())).plot(ax=plt.gca(), facecolor='b', edgecolor='b', linewidth=0.3, alpha=0.3) # trias that will be dropped
    #     # intersection_gdf.plot(ax=plt.gca(), facecolor='magenta', alpha=0.3) # the quads that played some role.
    #     # new_triangles.plot(ax=plt.gca(), facecolor='none', edgecolor='g') # the triangles that were built
    #     # for x, y, label in zip(new_triangles.geometry.centroid.x, new_triangles.geometry.centroid.y, new_triangles.index):
    #     #     plt.gca().text(x, y, str(label), fontsize=12)
    #     # plt.show(block=False)
    #     # breakpoint()
    #     # raise

    #     # indexes_to_drop = list(sorted(set(intersection_gdf.index_right)))
    #     mesh_gdf.drop(indexes_to_drop, inplace=True)
    #     mesh_gdf.reset_index(inplace=True, drop=True)
    #     mesh_gdf = pd.concat([mesh_gdf, new_triangles])

    #     # mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', alpha=0.3)
    #     # plt.show(block=False)
    #     # breakpoint()
    #     # raise


    #     nodes, elements = poly_gdf_to_elements(mesh_gdf)
    #     msh_t = self._jigsaw_msh_t_from_nodes_elements(nodes, elements, crs=mesh_gdf.crs)
    #     geom_msh_t = utils.get_geom_msh_t_from_msh_t_as_mp(msh_t)
    #     # pass
    #     return msh_t

    @staticmethod
    def jigsaw_msh_t_from_nodes_elements(nodes, elements, crs=None) -> jigsaw_msh_t:
        vert2 = np.array([(coord, 0) for coord in nodes], dtype=jigsaw_msh_t.VERT2_t)
        tria3 = np.array([(element, 0) for element in elements if len(element) == 3], dtype=jigsaw_msh_t.TRIA3_t)
        quad4 = np.array([(element, 0) for element in elements if len(element) == 4], dtype=jigsaw_msh_t.QUAD4_t)
        msh_t = jigsaw_msh_t()
        msh_t.mshID = 'euclidean-mesh'
        msh_t.ndims = 2
        msh_t.crs = crs
        msh_t.vert2 = vert2
        msh_t.tria3 = tria3
        msh_t.quad4 = quad4
        return msh_t

    def add_quads_to_msh_t_mpi(self, comm, msh_t):
        new_msh_t = self._get_new_msh_t_mpi(comm, msh_t)
        if comm.Get_rank == 0:
            msh_t.vert2 = new_msh_t.vert2
            msh_t.tria3 = new_msh_t.tria3
            msh_t.quad4 = new_msh_t.quad4

    def add_quads_to_msh_t(self, msh_t):
        # new_msh_t = self._get_new_msh_t(msh_t)
        new_msh_t = self._get_new_msh_t_from_cdt(msh_t)
        msh_t.vert2 = new_msh_t.vert2
        msh_t.tria3 = new_msh_t.tria3
        msh_t.quad4 = new_msh_t.quad4

    @staticmethod
    def _elements_are_conforming_wrapper(coords_left, coords_right):
        linear_ring_left = LinearRing(coords_left)
        linear_ring_right = LinearRing(coords_right)
        return elements_are_conforming(linear_ring_left, linear_ring_right)

    @staticmethod
    def _get_chunk_min_side_length(chunk):
        return chunk.geometry.map(lambda x: np.min(get_side_lengths(Polygon(x))))

    # @staticmethod
    # def _get_min_distance_chunk(coords):

    @staticmethod
    def _get_distance_from_chunk(chunk, quads_poly_gdf_uu):
        return chunk.geometry.map(lambda x: x.distance(quads_poly_gdf_uu))


    def _get_new_msh_t_from_cdt(self, msh_t):

        def get_constrained_triangulation():
            print("begin get_constrained_triangulation()", flush=True)
            quads_poly_gdf = self.quads_poly_gdf.copy()
            bnd_mp = utils.get_geom_msh_t_from_msh_t_as_mp(msh_t)
            joined = gpd.sjoin(
                    quads_poly_gdf.to_crs(msh_t.crs),
                    gpd.GeoDataFrame(geometry=list(bnd_mp.geoms), crs=msh_t.crs),
                    how='left',
                    predicate='within',
                    # predicate='intersects',
                    )
            quads_poly_gdf.drop(index=joined[joined.index_right.isna()].index, inplace=True)
            # with Pool(cpu_count()) as pool:
            #     quads_poly_gdf["min_side_length"] = pd.concat(
            #             pool.map(
            #                 self._get_chunk_min_side_length,
            #                 np.array_split(quads_poly_gdf.to_crs("epsg:6933").geometry, cpu_count())
            #                 )
            #             ).values
            # with Pool(cpu_count()) as pool:
            #     quads_poly_gdf["min_side_length"] =
            #     pd.concat(np.quads_poly_gdf.to_crs("epsg:6933").geometry.map(lambda x: np.min(get_side_lengths(Polygon(x))))
            # quads_poly_gdf.drop(index=quads_poly_gdf[quads_poly_gdf["min_side_length"] < 10.].index, inplace=True)
            # quads_poly_gdf.drop(columns=["min_side_length"], inplace=True)
            # vertices, elements = poly_gdf_to_elements(quads_poly_gdf.to_crs(msh_t.crs))
            from shapely.ops import transform
            from pyproj import Transformer
            del bnd_mp, joined
            def get_msh_t_vertices(quad_vertices, crs):
                # with Pool(cpu_count()) as pool:
                #     result = pool.map(
                #                 self._get_min_distance_chunk,
                #                 np.array_split(msh_t.vert2["coord"], cpu_count()),
                #                 )
                # print("begin get_msh_t_vertices", flush=True)
                # coords_gdf = gpd.GeoDataFrame(geometry=[Point(x) for x in msh_t.vert2["coord"]], crs=msh_t.crs).to_crs("epsg:6933")
                # quads_poly_gdf_uu = transform(Transformer.from_crs(self.quads_poly_gdf.crs, "epsg:6933", always_xy=True).transform, self.quads_poly_gdf_uu)
                # print("begin compouting distances", flush=True)
                # with Pool(cpu_count()) as pool:
                #     distances = pd.concat(
                #             pool.starmap(
                #                 self._get_distance_from_chunk,
                #                 [(chunk, quads_poly_gdf_uu) for chunk in np.array_split(coords_gdf, cpu_count())]
                #                 )
                #             )
                # distance_threshold = 10.

                # # Filter the points based on the distance threshold
                # filtered_points = coords_gdf.geometry[distances > distance_threshold]

                # # Extract the coordinates of the points that meet the criteria
                # return [(point.x, point.y) for point in filtered_points]


                # qvertices, elements = poly_gdf_to_elements(quads_poly_gdf.to_crs("epsg:6933"))
                # distances, indices = KDTree(vertices[:, 0], vertices[:, 1]).query(utils.reprojectmsh_t.vert2["coord"])
                # dixs = np.where(distances < 3.)


                # IN lat/lon flot32 eps is 0.164 mm and float16.eps is ~1.34 meters
                # Using large epsilon (~1 meter) avoids very narrow triangles.
                # eps = 0.25*np.finfo(np.float16).eps if msh_t.crs.is_geographic else 0.25
                # eps = 30.
                # msh_t_vert2_buffered_gdf = gpd.GeoDataFrame(geometry=list(map(lambda x: Point(x).buffer(eps), msh_t.vert2["coord"].tolist())), crs=msh_t.crs)
                msh_t_vert2_buffered_gdf = gpd.GeoDataFrame(geometry=[Point(x) for x in msh_t.vert2["coord"]], crs=msh_t.crs).to_crs("epsg:6933")
                msh_t_vert2_buffered_gdf["geometry"] = msh_t_vert2_buffered_gdf.geometry.map(lambda x: x.buffer(5.))
                joined = gpd.sjoin(
                        msh_t_vert2_buffered_gdf,
                        quads_poly_gdf.to_crs("epsg:6933"),
                        how='left',
                        predicate='intersects',
                        # predicate='within',
                        )
                msh_t_vertices = msh_t.vert2["coord"][joined[joined.index_right.isna()].index.unique(), :]
                transformer = Transformer.from_crs(msh_t.crs, "epsg:6933", always_xy=True)
                msh_t_vertices = np.array(transformer.transform(msh_t_vertices[:, 0], msh_t_vertices[:, 1])).T
                distances, indices = KDTree(quad_vertices).query(msh_t_vertices)
                # breakpoint()
                msh_t_vertices = msh_t_vertices[np.where(distances > 10.)[0], :]
                transformer = Transformer.from_crs("epsg:6933", msh_t.crs, always_xy=True)
                return np.array(transformer.transform(msh_t_vertices[:, 0], msh_t_vertices[:, 1])).T






            vertices, elements = poly_gdf_to_elements(quads_poly_gdf.to_crs("epsg:6933"))
            vertices = np.array(vertices)
            elements = np.array(elements)
            threshold = 1.0
            distances, indices = KDTree(vertices).query(vertices, k=2)
            distances = distances[:, 1]
            indices = indices[:, 1]
            while distances.min() < threshold:
                print("doing iter")
                _idxs = np.where(distances <= threshold)[0]
                node_indices = np.arange(vertices.shape[0])[_idxs]
                close_pairs = indices[_idxs]
                mask = node_indices < close_pairs
                node_indices = node_indices[mask]
                close_pairs = close_pairs[mask]
                mapping = np.arange(vertices.shape[0])
                mapping[close_pairs] = node_indices
                elements = mapping[elements]
                is_collapsed = np.apply_along_axis(lambda x: len(np.unique(x)) != 4, arr=elements, axis=1)
                elements = elements[~is_collapsed]
                vertices, elements = self._cleanup_isolates(vertices, elements)
                distances, indices = KDTree(vertices).query(vertices, k=2)
                distances = distances[:, 1]
                indices = indices[:, 1]
            print("begin get_msh_t_vertices", flush=True)


            filtered_base_mesh_vertices = get_msh_t_vertices(vertices, "epsg:6933")

            transformer = Transformer.from_crs("EPSG:6933", msh_t.crs, always_xy=True)
            quad_vertices = np.array(transformer.transform(vertices[:, 0], vertices[:, 1])).T



            vertices = np.vstack([
                quad_vertices,
                filtered_base_mesh_vertices,
                ])
            del quads_poly_gdf
            # new_vertices = np.array(transformer.transform(new_vertices[:, 0], new_vertices[:, 1])).T
            # tree = KDTree(new_vertices)
            # distances, indices = tree.query(vertices, k=1)  # k=1 for nearest neighbor
            # # breakpoint()
            # # res = tree1.query(
            # vertices =


            # vertices.extend(get_msh_t_vertices())
            t = cdt.Triangulation(
                    cdt.VertexInsertionOrder.AS_PROVIDED,
                    cdt.IntersectingConstraintEdges.RESOLVE,
                    0.
                    )
            print("begin insert vertices", flush=True)
            # print(vertices, flush=True)
            t.insert_vertices([cdt.V2d(*coord) for coord in vertices])
            print("begin insert edges", flush=True)
            t.insert_edges([cdt.Edge(e0, e1) for e0, e1 in elements_to_edges(elements)])
            return t, msh_t.crs


        def get_conforming_tri_mesh_gdf():

            def get_vertices_and_unfiltered_elements():
                t, crs = get_constrained_triangulation()

                # the constrained triangulation has all the triangles we need, plus many others that we
                # need to filter out. Normally, one would use erase_outer_triangles_and_holes() but in
                # our tests, we see that it leads to significant over-filtering.
                # Here we implement a custom filter for cdt.Triangulation.
                print("erase super triangle", flush=True)
                # Begin by erasing the super triangle, since we definitely don't need that one.
                t.erase_super_triangle()
                # t.erase_outer_triangles_and_holes()
                # t.erase_outer_triangles(:set laststatus=2                # we assume operations over members of t are not safe, so we extract the remaining
                # outputs into numpy arrays.
                print("get constrained triangulation vertices", flush=True)
                vertices = np.array([(v.x, v.y) for v in t.vertices])
                print("get constrained triangulation elements", flush=True)
                elements = np.array([tria.vertices for tria in t.triangles])
                # verify
                # nprocs = cpu_count()
                # chunksize = len(msh_t.vert2) // nprocs
                # with Pool(nprocs) as pool:
                #     original_mesh_gdf = gpd.GeoDataFrame(
                #         geometry=list(pool.map(
                #             Polygon,
                #             vertices[elements, :].tolist(),
                #             chunksize
                #             )),
                #         crs=msh_t.crs
                #     )
                # original_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
                # plt.title("this is contrained mesh")
                # plt.show(block=False)
                # breakpoint()
                # raise
                return vertices, elements, crs

            def get_centroid_based_elements():

                def get_area_limited_vertices_elements():
                    vertices, all_elements, crs = get_vertices_and_unfiltered_elements()
                    # print("Eliminate constrained elements with area less than 1 square meter", flush=True)
                    # nprocs = cpu_count()
                    # chunksize = len(vertices) // nprocs
                    # with Pool(nprocs) as pool:
                    #     original_mesh_gdf = gpd.GeoDataFrame(
                    #         geometry=list(pool.map(
                    #             Polygon,
                    #             vertices[all_elements, :].tolist(),
                    #             chunksize
                    #             )),
                    #         crs=crs
                    #     )
                    # original_mesh_gdf = original_mesh_gdf[original_mesh_gdf.to_crs("epsg:6933").geometry.area >= 1.]
                    # # original_mesh_gdf = self._fix_triangles_outside_SCHISM_skewness_tolerance(original_mesh_gdf)
                    # vertices, all_elements = poly_gdf_to_elements(original_mesh_gdf)
                    vertices = np.array(vertices)
                    all_elements = np.array(all_elements)
                    return vertices, all_elements, crs


                def get_centroid_based_element_mask():
                    print("get original_geom_msh_t", flush=True)
                    original_geom_msh_t = utils.get_geom_msh_t_from_msh_t_as_msh_t(msh_t)
                    print("get quads_bnd_as_msh_t", flush=True)
                    quads_bnd_as_msh_t = utils.multipolygon_to_jigsaw_msh_t(self.quads_poly_gdf_uu)
                    print("take the mean", flush=True)
                    centroids = np.mean(vertices[all_elements, :], axis=1)
                    print("inpoly2-1 now", flush=True)
                    centroid_in_base_poly = np.array(inpoly2(centroids, original_geom_msh_t.vert2['coord'], original_geom_msh_t.edge2['index'])[0], dtype=bool)
                    print("inpoly2-2 now", flush=True)
                    centroid_in_quads_poly = np.array(inpoly2(centroids, quads_bnd_as_msh_t.vert2['coord'], quads_bnd_as_msh_t.edge2['index'])[0], dtype=bool)
                    return np.logical_or(centroid_in_base_poly, centroid_in_quads_poly)


                # original_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
                # plt.show(block=False)
                # breakpoint()
                # raise
                # def get_pinched_node_vertex_indices(elements_to_consider):
                #     tri = Triangulation(vertices[:, 0], vertices[:, 1], elements_to_consider)
                #     boundary_edges = tri.neighbors == -1
                #     boundary_vertices = tri.triangles[boundary_edges]
                #     boundary_vertices_flat = boundary_vertices.flatten()
                #     unique, inverse, counts = np.unique(boundary_vertices_flat, return_inverse=True, return_counts=True)
                #     pinched_node_mask = unique[counts > 1]
                #     return pinched_node_mask

                def get_element_indices_with_pinched_nodes(elements_to_consider):
                    print("get element indices with pinched nodes", flush=True)
                    tri = Triangulation(vertices[:, 0], vertices[:, 1], elements_to_consider)
                    boundary_edges = tri.neighbors == -1
                    boundary_vertices = tri.triangles[boundary_edges]
                    boundary_vertices_flat = boundary_vertices.flatten()
                    unique, counts = np.unique(boundary_vertices_flat, return_counts=True)
                    pinched_nodes = unique[counts > 1]
                    vertex_to_elements = {}
                    for idx, element in enumerate(tri.triangles):
                        for vertex in element:
                            if vertex in vertex_to_elements:
                                vertex_to_elements[vertex].add(idx)
                            else:
                                vertex_to_elements[vertex] = {idx}
                    elements_with_pinched_nodes = set()
                    for node in pinched_nodes:
                        elements_with_pinched_nodes.update(vertex_to_elements[node])
                    return list(elements_with_pinched_nodes)

                def separate_elements_by_pinched_nodes(target_elements, pinched_element_indices):
                    print("separate elements by pinched nodes", flush=True)

                    # Use np.take to select elements with pinched nodes
                    elements_with_pinched_nodes = np.take(target_elements, pinched_element_indices, axis=0)

                    # For elements without pinched nodes, use a boolean mask
                    mask = np.ones(len(target_elements), dtype=bool)
                    mask[pinched_element_indices] = False
                    elements_without_pinched_nodes = target_elements[mask]

                    return elements_with_pinched_nodes, elements_without_pinched_nodes

                # def separate_elements_by_pinched_nodes(target_elements, pinched_element_indices):
                #     print("separate elements by pinched nodes")
                #     elements_with_pinched_nodes = []
                #     elements_without_pinched_nodes = []
                #     for idx, element in enumerate(target_elements):
                #         if idx in pinched_element_indices:
                #             elements_with_pinched_nodes.append(element)
                #         else:
                #             elements_without_pinched_nodes.append(element)

                #     return elements_with_pinched_nodes, elements_without_pinched_nodes

                vertices, all_elements, crs = get_area_limited_vertices_elements()
                centroid_based_element_mask = get_centroid_based_element_mask()
                target_elements = all_elements[centroid_based_element_mask, :]
                pinched_element_indices = get_element_indices_with_pinched_nodes(target_elements)
                while len(pinched_element_indices) > 0:
                    _elements_with_pinched_nodes, target_elements = separate_elements_by_pinched_nodes(target_elements, pinched_element_indices)
                    pinched_element_indices = get_element_indices_with_pinched_nodes(target_elements)
                    print(f"{len(pinched_element_indices)} pinched nodes remaining", flush=True)

                # results = []
                # neighbors_table = Triangulation(vertices[:, 0], vertices[:, 1], target_elements).neighbors
                # for i, triangle in enumerate(target_elements):
                #     for j, neighbor_index in enumerate(neighbors_table[i]):
                #         if neighbor_index != -1:
                #             shared_edge = set(triangle) - set(target_elements[neighbor_index])
                #             if len(shared_edge) == 2:
                #                 if tuple(shared_edge) in zip(target_elements[neighbor_index], np.roll(target_elements[neighbor_index], -1)):
                #                     results.append(f"Element {i} and {neighbor_index} have opposite orientation on edge {shared_edge}")
                # if len(results) > 0:
                #     for result in results:
                #         print(result, flush=True)
                #     raise NotImplementedError("Non-conforming elements were found")


                print("done with vertices/element creation", flush=True)
                return vertices, target_elements, crs
            # target_elements = all_elements[get_final_element_mask(), :]
            print("begin getting centroid based elements", flush=True)
            vertices, elements, crs = get_centroid_based_elements()

            # def orient_ccw(indices):
            #     # Get the actual coordinates
            #     points = vertices[indices]

            #     # Calculate the cross product of vectors (p2 - p1) and (p3 - p1)
            #     v1 = points[1] - points[0]
            #     v2 = points[2] - points[0]
            #     cross_product = np.cross(v1, v2)

            #     # If cross product is negative, reverse the order of points
            #     if cross_product < 0:
            #         return indices[::-1]
            #     return indices


#             vertices = np.array(Transformer.from_crs(crs, "epsg:6933", always_xy=True).transform(*vertices.T)).T
#             threshold = 3.
#             distances, indices = KDTree(vertices).query(vertices, k=2)
#             distances = distances[:, 1]
#             indices = indices[:, 1]
#             while distances.min() < threshold:
#                 print("doing iter")
#                 _idxs = np.where(distances <= threshold)[0]
#                 node_indices = np.arange(vertices.shape[0])[_idxs]
#                 close_pairs = indices[_idxs]
#                 mask = node_indices < close_pairs
#                 node_indices = node_indices[mask]
#                 close_pairs = close_pairs[mask]
#                 mapping = np.arange(vertices.shape[0])
#                 mapping[close_pairs] = node_indices
#                 elements = mapping[elements]
#                 is_collapsed = np.apply_along_axis(lambda x: len(np.unique(x)) != 3, arr=elements, axis=1)
#                 elements = elements[~is_collapsed]
#                 elements = np.apply_along_axis(orient_ccw, axis=1, arr=elements)
#                 vertices, elements = self._cleanup_isolates(vertices, elements)
#                 distances, indices = KDTree(vertices).query(vertices, k=2)
#                 distances = distances[:, 1]
#                 indices = indices[:, 1]
#                 print(distances.min())
            # transformer = Transformer.from_crs("EPSG:6933", msh_t.crs, always_xy=True)
            # vertices = np.array(transformer.transform(vertices[:, 0], vertices[:, 1])).T
            # print("begin get_msh_t_vertices", flush=True)
            # vertices = np.vstack([
            #     vertices,
            #     np.array(get_msh_t_vertices())
            #     ])

            # convert into the final_gdf and return
            # vertices = np.array(Transformer.from_crs("epsg:6933", crs, always_xy=True).transform(*vertices.T)).T
            return vertices, elements, crs
            # nprocs = cpu_count()
            # chunksize = len(vertices) // nprocs
            # with Pool(nprocs) as pool:
            #     print("make final gdf", flush=True)
            #     final_gdf = gpd.GeoDataFrame(
            #         geometry=list(pool.map(Polygon, vertices[elements, :].tolist(), chunksize)),
            #         crs=crs,
            #         )
            # verify
            # final_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
            # plt.show(block=False)
            # breakpoint()
            # raise

            # Debug: omit non-conforming filter
            # return final_gdf

            # print("begin filtering out non-conforming", flush=True)
            # # final_gdf_buffered = final_gdf.copy()
            # # final_gdf_buffered.geometry = final_gdf_buffered.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
            # joined = gpd.sjoin(
            #         final_gdf,
            #         final_gdf,
            #         how='inner',
            #         predicate='intersects'
            #         )
            # joined = joined[joined.index != joined.index_right]
            # # del final_gdf_buffered

            # def get_elements_are_conforming_parallel_job_args():
            #     exterior_coords_left = final_gdf.loc[joined.index].geometry.apply(lambda g: g.exterior.coords).tolist()
            #     exterior_coords_right = final_gdf.loc[joined.index_right].geometry.apply(lambda g: g.exterior.coords).tolist()
            #     return zip(exterior_coords_left, exterior_coords_right)

            # with Pool(cpu_count()) as pool:
            #     print("check conformity", flush=True)
            #     is_conforming_bool_list = pool.starmap(
            #             self._elements_are_conforming_wrapper,
            #             get_elements_are_conforming_parallel_job_args()
            #             )
            # joined = joined[~np.array(is_conforming_bool_list)]
            # return final_gdf.drop(index=joined.index.unique())

        def get_final_mesh_gdf():
            print("begin generating conforming tri_mesh_gdf", flush=True)
            tri_mesh_gdf = get_conforming_tri_mesh_gdf()
            return tri_mesh_gdf
            print("begin replacing trias with quads", flush=True)
            joined = gpd.sjoin(
                    gpd.GeoDataFrame(geometry=tri_mesh_gdf.to_crs("epsg:6933").geometry.centroid, crs="epsg:6933"),
                    self.quads_poly_gdf.to_crs("epsg:6933"),
                    how='left',
                    predicate='within',
                    )
            joined = joined.to_crs(tri_mesh_gdf.crs)
            joined.geometry = tri_mesh_gdf.geometry
            grouped = joined.groupby('index_right')
            filtered_groups = {idx: group for idx, group in grouped if len(group) == 2}
            with Pool(cpu_count()) as pool:
                grouped = gpd.GeoDataFrame(
                        geometry=list(pool.map(
                            ops.unary_union,
                            [group.geometry for group in filtered_groups.values()],
                            len(filtered_groups) // cpu_count(),
                            )),
                        crs=tri_mesh_gdf.crs
                        )
            # grouped.geometry = grouped.geometry.map(lambda x: polygon.orient(x, sign=1.))
            # Collect indices of geometries in tri_mesh_gdf to be dropped
            indices_to_drop = [idx for group in filtered_groups.values() for idx in group.index]

            # Drop these indices from tri_mesh_gdf
            tri_mesh_gdf.drop(index=list(set(indices_to_drop)), inplace=True)
            # tri_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
            # grouped.plot(ax=plt.gca(), facecolor='r', edgecolor='r', alpha=0.3, linewidth=0.3)
            # plt.show(block=False)
            # breakpoint()
            # raise
            final_mesh_gdf = pd.concat([tri_mesh_gdf, grouped], ignore_index=True)
            final_mesh_gdf.geometry = final_mesh_gdf.geometry.map(lambda x: polygon.orient(x, sign=1.))
            return final_mesh_gdf
            breakpoint()
            # final_mesh_gdf = pd.c
            # breakpoint()

            # print("gener:ate final_mesh_gdf (the hybrid one)", flush=True)
            # final_mesh_gdf = pd.concat([tri_mesh_gdf, self.quads_poly_gdf.loc[joined[~joined.index_right.isna()].index_right.unique()].to_crs(tri_mesh_gdf.crs)], ignore_index=True)
            # final_mesh_gdf = final_mesh_gdf[final_mesh_gdf.to_crs("epsg:6933").geometry.area >= 1.]
            # breakpoint()
            # def is_conforming(row):
            #     ls_right = LinearRing(final_mesh_gdf.loc[row.index_right].geometry.exterior.coords)
            #     return elements_are_conforming(row.geometry, ls_right)

            # joined = gpd.sjoin(
            #         final_mesh_gdf,
            #         final_mesh_gdf,
            #         how='inner',
            #         predicate='intersects'
            #         )
            # joined = joined[joined.index != joined.index]
            # joined = joined[joined.apply(is_conforming, axis=1)]
            # verification
            # final_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
            # plt.show(block=False)
            # breakpoint()
            return final_mesh_gdf

        def get_output_msh_t():

            print("begin generating final_mesh_gdf", flush=True)
            # final_mesh_gdf = self._fix_triangles_outside_SCHISM_skewness_tolerance(get_final_mesh_gdf())
            # final_mesh_gdf = get_final_mesh_gdf()
            # final_mesh_gdf = get_conforming_tri_mesh_gdf()

            # final_mesh_gdf.to_feather("final_mesh_gdf_unfiltered_for_debug.feather")
            # raise
            # final_mesh_gdf.to_feather("final_mesh_gdf_unfiltered_for_debug.feather")
            # final_mesh_gdf = gpd.read_feather()

            # final_mesh_gdf = self._fix_triangles_outside_SCHISM_skewness_tolerance(final_mesh_gdf)
            # final_mesh_gdf = self._fix_triangles_outside_SCHISM_skewness_tolerance(final_mesh_gdf)
            # verification
            # final_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
            # final_mesh_gdf.to_feather("finalized_mesh.feather")
            # plt.show(block=False)
            # breakpoint()
            # print("begin convering final_mesh_gdf to msh_t", flush=True)
            vertices, elements, crs = get_conforming_tri_mesh_gdf()
            # output_msh_t = self.jigsaw_msh_t_from_nodes_elements(*poly_gdf_to_elements(final_mesh_gdf), crs=final_mesh_gdf.crs)
            output_msh_t = self.jigsaw_msh_t_from_nodes_elements(vertices, elements, crs=crs)

            # del final_mesh_gdf
            print("cleanup pinched nodes", flush=True)
            utils.cleanup_pinched_nodes_iter(output_msh_t)

            # import pickle
            # with open("final_output_msh_t_with_skew_elements.pkl", "wb") as file:
            #     pickle.dump(output_msh_t, file)

            # raise

#             utils.ensure_ccw_triangles(output_msh_t)
            self._fix_triangles_outside_SCHISM_skewness_tolerance_mut_msh_t(output_msh_t)
            # print("split bad quality quads", flush=True)
            # utils.split_bad_quality_quads(output_msh_t)
            # print("swap large angle edges", flush=True)
            # utils.swap_large_angle_edges(output_msh_t, 175.)
            return output_msh_t
        print("begin generating output_msh_t", flush=True)
        return get_output_msh_t()

    @staticmethod
    def _compute_skewness(output_msh_t):
        vert2 = output_msh_t.vert2["coord"]
        transformer = Transformer.from_crs(output_msh_t.crs, "epsg:6933", always_xy=True)
        vert2 = np.array(transformer.transform(vert2[:, 0], vert2[:, 1])).T
        tria3 = output_msh_t.tria3["index"]
        elem_nodes = vert2[tria3]
        edge_dx_dy = np.abs(elem_nodes - np.roll(elem_nodes, shift=-1, axis=1))
        euclidean_distances = np.sqrt(np.sum(np.square(edge_dx_dy), axis=2))

        # Calculate vectors AB and AC for each triangle
        vec_AB = elem_nodes[:, 1, :] - elem_nodes[:, 0, :]
        vec_AC = elem_nodes[:, 2, :] - elem_nodes[:, 0, :]

        # Calculate the area of each triangle using the cross product (in 2D)
        area = 0.5 * np.abs(vec_AB[:, 0] * vec_AC[:, 1] - vec_AB[:, 1] * vec_AC[:, 0])
        equivalent_radius = np.abs(area) / np.pi
        row_wise_max = np.max(euclidean_distances, axis=1)
        skewness = row_wise_max / equivalent_radius
        return skewness

    @classmethod
    def _fix_triangles_outside_SCHISM_skewness_tolerance_mut_msh_t(cls, output_msh_t, tol=60.):

        skewness = cls._compute_skewness(output_msh_t)
        print(f"initial skewness {skewness.max()=}", flush=True)
        while np.any(skewness > tol):
            tria3 = output_msh_t.tria3["index"]
            vert2 = output_msh_t.vert2["coord"]
            mask = skewness > tol
            element_index = np.where(mask)
            participant_node_indexes = np.array(list(set(tria3[element_index].flatten().tolist())))
            matches = np.isin(tria3, participant_node_indexes)
            matching_row_indices = np.any(matches, axis=1)
            matching_rows = np.where(matching_row_indices)[0]
            rows_to_keep = np.where(~matching_row_indices)[0]
            triangles = [Polygon(x) for x in [vert2[tria3[this_index]] for this_index in matching_rows]]
            tria_to_fix = gpd.GeoDataFrame(geometry=triangles, crs=output_msh_t.crs)
            mps_to_triangulate = tria_to_fix.unary_union
            if isinstance(mps_to_triangulate, Polygon):
                mps_to_triangulate = MultiPolygon([mps_to_triangulate])
            new_trias = []
            tree = KDTree(vert2)
            for i, poly in enumerate(mps_to_triangulate.geoms):
                boundary = poly.boundary
                # print(boundary)
                boundary = simplify(boundary, np.finfo(np.float32).eps)

                if isinstance(boundary, LineString):
                    boundary = MultiLineString([boundary])

                if isinstance(boundary, MultiLineString):
                    for this_bnd in boundary.geoms:
                        tria3 = output_msh_t.tria3["index"]
                        vert2 = output_msh_t.vert2["coord"]
                        new_vertices, new_elements = cls._get_new_trias_vert_ele(this_bnd)
                        ii = tree.query(new_vertices)[1]
                        # Directly remap new_elements to vert2 indices
                        renumbered_elements = ii[new_elements]

                        mask = np.apply_along_axis(lambda x: len(np.unique(x)) == 3, arr=renumbered_elements, axis=1)
                        renumbered_elements = renumbered_elements[mask]
                        new_trias.append(renumbered_elements)


#                         # Apply the mapping to new_elements
#                         renumbered_elements = mapping[new_elements]
#                         mask = np.apply_along_axis(lambda x: len(np.unique(x)) == 3, arr=renumbered_elements, axis=1)
#                         renumbered_elements = renumbered_elements[mask]
#                         new_trias.append(renumbered_elements)

                else:
                    raise NotImplementedError(f"{type(boundary)=}\n{boundary=}")

            output_msh_t.tria3 = output_msh_t.tria3.take(rows_to_keep)
            new_trias = np.array(
                    [(e, 0) for e in np.vstack(new_trias)],
                    dtype=jigsaw_msh_t.TRIA3_t
                    )
            output_msh_t.tria3 = np.hstack(
                    [
                        output_msh_t.tria3,
                        new_trias
                    ]
                )

            utils.cleanup_isolates(output_msh_t)
            skewness = cls._compute_skewness(output_msh_t)
            print(f"{skewness.max()=}", flush=True)

        # utils.reproject(output_msh_t, original_crs)


    @staticmethod
    def _cleanup_isolates(vertices, elements) -> Tuple[np.array, np.array]:
        """
        Removes nodes that exist in vertices but are not referenced in elements.
        Returns the filtered vertices and renumbered elements array.
        """
        node_indexes = np.arange(vertices.shape[0])
        used_indexes, inverse = np.unique(elements, return_inverse=True)
        isin = np.isin(node_indexes, used_indexes, assume_unique=True)
        if np.all(isin):
            # nothing to do, return the original as they are
            return vertices, elements
        vert2_idxs = np.where(isin)[0]
        df = pd.DataFrame(index=node_indexes).iloc[vert2_idxs].reset_index()
        mapping = {v: k for k, v in df.to_dict()["index"].items()}
        elements = np.array([mapping[x] for x in used_indexes])[inverse].reshape(
            elements.shape
        )
        vertices = vertices[vert2_idxs]
        return vertices, elements
    @staticmethod
    def _get_buffered_geom(geom, eps=None):
        eps = eps or np.finfo(np.float32).eps
        return geom.buffer(eps)

    @staticmethod
    def _clean_and_renumber(coords, edges):
        # Check if the first and last points are the same
        if coords[0] == coords[-1]:
            # Remove the last point
            coords = coords[:-1]

            # Renumber the edges
            new_edges = []
            for start, end in edges:
                if end == len(coords):  # If it was pointing to the last (now removed) point
                    end = 0  # Point it to the first point instead
                new_edges.append((start, end))
            new_edges = new_edges[:-1]
        else:
            new_edges = edges

        return coords, edges

    @staticmethod
    def _get_new_trias_vert_ele(boundary):
        coords = boundary.coords
        edges = [(i, (i + 1) % len(coords)) for i in range(len(coords))]
        # coords, edges = self._clean_and_renumber(coords, edges)
        t = cdt.Triangulation(
                cdt.VertexInsertionOrder.AS_PROVIDED,
                cdt.IntersectingConstraintEdges.RESOLVE,
                0.
                )
        t.insert_vertices([cdt.V2d(*coord) for coord in coords])
        try:
            # using simplify above sould have fixed this but I don't dare to check
            t.insert_edges([cdt.Edge(e0, e1) for e0, e1 in edges])
        except RuntimeError:
            print(f"{edges=}")
            print(f"{list(coords)=}")
            # breakpoint()
            # raise
            # continue
            return []
        t.erase_outer_triangles_and_holes()
        vertices = np.array([(v.x, v.y) for v in t.vertices])
        elements = np.array([tria.vertices for tria in t.triangles])
        return vertices, elements


    @classmethod
    def _get_new_trias(cls, boundary):
        vertices, elements = cls._get_new_trias_vert_ele(boundary)
        return list(map(Polygon, vertices[elements, :].tolist()))

    @classmethod
    def _fix_triangles_outside_SCHISM_skewness_tolerance(cls, tria3_gdf):
        # SCHISM can tolerate sknewness<=60 (defined as (largest side)/(equivalent radius), where  eq. radius is sqrt(area/pi).

        from time import time
        start = time()
        print("compute initial skewness", flush=True)
        with Pool(cpu_count()) as pool:
            tria3_gdf["skewness"] = pd.concat(
                    pool.map(
                        get_chunk_skewness,
                        np.array_split(tria3_gdf.to_crs('epsg:6933').geometry, cpu_count())
                        )
                    ).values
        # tria3_gdf[tria3_gdf["skewness"] >= 60.].plot(ax=plt.gca(), facecolor='r')
        # tria3_gdf[tria3_gdf["skewness"] >= 60.].geometry.boundary.plot(ax=plt.gca(), edgecolor='k')
        # plt.show()

        print(f"compute initial skewness took: {time()-start}", flush=True)
        current_count = float("inf")
        while np.any(tria3_gdf['skewness'] > 60.):
            tria_with_high_skewness = tria3_gdf[tria3_gdf['skewness'] > 60.].copy().to_crs('epsg:6933')
        #     if len(tria_with_high_skewness) == current_count:
        #         break
            current_count = len(tria_with_high_skewness)
            # print(f"{current_count=}", flush=True)
            # with Pool(cpu_count()) as pool:
            #     tria_with_high_skewness.geometry = pool.starmap(cls._get_buffered_geom, [(geom, 1.) for geom in tria_with_high_skewness.geometry])

            # Perform spatial join to find all triangles that touch the ones with high skewness
            start = time()
            print("tria_to_fix compute", flush=True)
            tria_to_fix = gpd.sjoin(
                tria3_gdf.to_crs('epsg:6933'),
                tria_with_high_skewness,
                how='inner',
                predicate='intersects'
            )
            print(f"tria_to_fix compute: {time()-start}", flush=True)
        #     # Combine indices from both conditions and select rows from original DataFrame
            tria_to_fix = tria3_gdf.loc[
                tria_to_fix.index.union(tria_with_high_skewness.index)
            ]

            tria3_gdf = tria3_gdf.drop(index=tria_to_fix.index.unique())

            start = time()
            mps_to_triangulate = tria_to_fix.unary_union
            print(f"mps to triangulate took: {time()-start}", flush=True)
            if isinstance(mps_to_triangulate, Polygon):
                mps_to_triangulate = MultiPolygon([mps_to_triangulate])
            # breakpoint()
            new_trias = []
            start = time()
            for i, poly in enumerate(mps_to_triangulate.geoms):
                boundary = poly.boundary
                # print(boundary)
                boundary = simplify(boundary, np.finfo(np.float32).eps)
                if isinstance(boundary, LineString):

                    new_trias.extend(cls._get_new_trias(boundary))
                elif isinstance(boundary, MultiLineString):
                    for this_bnd in boundary.geoms:
                        new_trias.extend(cls._get_new_trias(this_bnd))
                else:
                    raise NotImplementedError(f"{type(boundary)=}\n{boundary=}")
                # elif isinstance(boundary, MultiLineString):

                # if hasattr(boundary, "geoms"):
                #     print(type(boundary))
                #     raise
                # coords = max(boundary.geoms, key=lambda bnd: bnd.length).coords
        #     print(f"building new trias took: {time()-start}", flush=True)
        #     new_gdf = gpd.GeoDataFrame(geometry=new_trias, crs=tria3_gdf.crs)
        #     # new_gdf = new_gdf[new_gdf.to_crs("epsg:6933").geometry.area >= 1.]
        #     start = time()
        #     with Pool(cpu_count()) as pool:
        #         new_gdf["skewness"] = pool.map(get_skewness, new_gdf.to_crs("epsg:6933").geometry)
        #     print(f"computing new skewness took: {time()-start}", flush=True)

        #     start = time()
        #     tria3_gdf = pd.concat([tria3_gdf, new_gdf], ignore_index=True)
        #     print(f"concat took: {time()-start}", flush=True)
        # # tria_to_fix = tria3_gdf.loc[
        # #     tria_to_fix.index.union(tria_with_high_skewness.index)
        # # ]

        # # tria3_gdf = tria3_gdf.drop(index=tria_with_high_skewness.index.unique())
        # print("start plus minus buffer")
        # start = time()
        # # tria3_gdf = tria3_gdf[tria3_gdf.to_crs("epsg:6933").geometry.area >= 1.]
        # tria3_gdf_plus_buffered = tria3_gdf.copy()
        # tria3_gdf_minus_buffered = tria3_gdf.copy()
        # with Pool(cpu_count()) as pool:
        #     tria3_gdf_plus_buffered.geometry = pool.starmap(cls._get_buffered_geom, [(geom, 2*np.finfo(np.float32).eps) for geom in tria3_gdf.geometry])
        #     tria3_gdf_minus_buffered.geometry = pool.starmap(cls._get_buffered_geom, [(geom, -2*np.finfo(np.float32).eps) for geom in tria3_gdf.geometry])
        # print(f"plus minus buffer took {time() - start} ", flush=True)
        # print("start final sjoin")
        # start = time()
        # joined = gpd.sjoin(
        #     tria3_gdf_minus_buffered,
        #     tria3_gdf_plus_buffered,
        #     how='inner',
        #     predicate='within'
        # )
        # print(f"final sjoin took: {time()-start}", flush=True)
        # joined = joined[joined.index != joined.index_right]

        # indices_to_drop = joined.index.unique()
        # del tria3_gdf_minus_buffered
        # del tria3_gdf_plus_buffered

        # joined = gpd.sjoin(
        #     tria3_gdf,
        #     tria3_gdf,
        #     how='inner',
        #     predicate='overlaps'
        # )
        # joined = joined[joined.index != joined.index_right].reset_index(drop=False)
        # # Step 1: Find indices with more than one overlap
        # overlap_count = joined.index.value_counts()
        # indices_more_than_one_overlap = overlap_count[overlap_count > 1].index.tolist()

        # # # Step 2: Find the item that's within the other for one-to-one overlaps
        # one_to_one_overlaps = joined[joined.index.value_counts() == 1]
        # indices_to_drop_for_one_to_one = []

        # for idx, row in one_to_one_overlaps.iterrows():
        #     rows = tria3_gdf.loc[[idx]].within(tria3_gdf.loc[[row.index_right]])
        #     if len(rows) > 0:
        #         indices_to_drop_for_one_to_one.extend(rows.index.tolist())
        #     # elif tria3_gdf.loc[row.index_right].within(tria3_gdf.loc[idx]):
        #     else:
        #         indices_to_drop_for_one_to_one.append(row.index_right)

        # # Combine both lists of indices to drop
        # indices_to_drop = set(indices_more_than_one_overlap + indices_to_drop_for_one_to_one)

        # # Drop these indices from the GeoDataFrame
        # tria3_gdf.drop(index=list(indices_to_drop.union(joined.index.unique())), inplace=True)
        # from shapely.ops import orient
        # tria3_gdf['geometry'] = tria3_gdf['geometry'].apply(lambda x: orient(x, sign=1))
        # print("returning from skewness tolerance fix", flush=True)
        # return tria3_gdf
        # tria3_gdf.plot(ax=plt.gca(), facecolor='r', edgecolor='k')
        # plt.show(block=True)
        # print("here")

        # breakpoint()

        # self._remove_invalid_triangles_from_msh_t(msh_t, tria_to_fix)
        # self._append_new_tris_gdf_to_msh_t(self._get_new_tris_gdf(tria_to_fix), msh_t)
        # # remove any remaining tri with high skewness
        # tria3_gdf = utils.get_msh_t_tria3_Polygon_gdf(msh_t)
        # tria3_gdf['skewness'] = tria3_gdf.to_crs('epsg:6933').geometry.map(get_skewness)
        # tria_with_high_skewness = tria3_gdf[tria3_gdf['skewness'] > 60.]
        # self._remove_invalid_triangles_from_msh_t(msh_t, tria_with_high_skewness)
        # self._remove_self_intersections(msh_t)
        # utils.remove_flat_triangles(msh_t)

        # verification
        # import matplotlib.colors
        # plt.gca().tripcolor(
        #         utils.mesh_to_tri(msh_t),
        #         np.ones(msh_t.vert2['coord'].shape[0]),
        #         cmap=matplotlib.colors.ListedColormap("lightgrey"),
        #         edgecolor='k',
        #         lw=0.7,
        #         alpha=0.3,
        #         )
        # tria_to_fix = get_tria_to_fix(msh_t)
        # if len(tria_to_fix) > 0:
        #     tria_to_fix.plot(ax=plt.gca(), facecolor='red', edgecolor='none', alpha=0.3)
        #     for x, y, label in zip(tria_to_fix.geometry.centroid.x, tria_to_fix.geometry.centroid.y, tria_to_fix.index):
        #         plt.gca().text(x, y, str(label), fontsize=12)
        # plt.show(block=False)
        # breakpoint()
        # raise

        # # import contextily as cx
        # # cx.add_basemap(
        # #         plt.gca(),
        # #         source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        # #         crs=msh_t.crs
        # #         )
        # # cx.add_basemap(plt.gca(), source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", crs=msh_t.crs)
        # # partially_intersecting_gdf = self._get_partially_intersecting_tri_gdf(msh_t)
        # # partially_intersecting_gdf.plot(ax=plt.gca(), facecolor='blue', alpha=0.3)
        # plt.show(block=False)
        # breakpoint()
        # -- end verification plot

    @classmethod
    def from_raster(
            cls,
            raster: Union[Path, 'Raster'],
            zmin=None,
            zmax=None,
            raster_opts=None,
            window=None,
            pad_width=None,
            lower_bound=None,
            upper_bound=None,
            **kwargs
            ):
        if zmin is None and zmax is None:
            return cls.from_mp(
                    *get_thalwegs_multipolygon_for_raster(
                        raster,
                        raster_opts=raster_opts,
                        window=window,
                        pad_width=pad_width,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        threshold_size=kwargs.get("threshold_size", None)
                        ),
                    **kwargs
                    )
        return cls.from_mp(
                *get_multipolygon_for_raster(
                    raster,
                    zmin=zmin,
                    zmax=zmax,
                    raster_opts=raster_opts,
                    window=window,
                    pad_width=pad_width,
                    ),
                **kwargs
                )


    # @classmethod
    # def thalwegs_from_raster(
    #     cls,
    #     raster: Union[Path, 'Raster'],
    #     raster_opts=None,
    #     window=None,
    #     pad_width=None,
    #     **kwargs
    #     ):
    #     return cls.from_mp(
    #             *get_thalwegs_multipolygon_for_raster(
    #                 raster,
    #                 raster_opts=raster_opts,
    #                 window=window,
    #                 pad_width=pad_width,
    #                 ),
    #             **kwargs
    #             )

    @classmethod
    def from_mp(cls, mp, mp_crs, **kwargs):
        return cls(generate_quad_gdf_from_mp(mp, mp_crs, **kwargs))

    @classmethod
    def from_banks_file_mpi(cls, comm, fname, rank=0, cleanup=True, **kwargs):
        return cls(
                generate_quads_gdf_from_banks_file_mpi(
                    comm,
                    fname,
                    rank=rank,
                    cleanup=cleanup,
                    **kwargs,
                    )
                    )

    @classmethod
    def from_triplets_mpi(cls, comm, fname, **kwargs):
        return cls(
                generate_quads_gdf_from_triplets_file_mpi(
                    comm,
                    fname,
                    **kwargs,
                    )
                    )
        # # TODO: delete the feathers!
        # # return cls(quads_gdf)

        # breakpoint()

    def plot(self, **kwargs):
        return self.quads_gdf.plot(**kwargs)

    @cached_property
    def quads_poly_gdf(self) -> gpd.GeoDataFrame:
        if self.quads_gdf is not None:  # self.quads_gdf can be None is the class is used as an MPI worker
            quads_poly_gdf = self.quads_gdf.copy()  # This will include all original columns
            quads_poly_gdf['geometry'] = quads_poly_gdf['geometry'].map(Polygon)
            return quads_poly_gdf

    def _set_quads_poly_gdf_uu_with_mpi(self, comm):
        with MPICommExecutor(comm) as executor:
            if executor is not None:
                tmpfile = tempfile.NamedTemporaryFile(dir=Path.cwd(), prefix='.', suffix='.feather')
                # self.quads_poly_gdf["quad_group_id_file_index"] =(
                #     self.quads_poly_gdf["quad_group_id"].astype(str) + "_" + self.quads_poly_gdf["file_index"].astype(str)
                #     )
                # print(self.quads_poly_gdf)
                # print("saving to feather", flush=True)
                # self.quads_poly_gdf.to_feather(tmpfile.name)
                # groupby_keys = self.quads_poly_gdf["file_index"].unique()
                print("do groupby for _set_quads_poly_gdf_uu_with_mpi", flush=True)
                groups = self.quads_poly_gdf.groupby("file_index")
                # self.quads_poly_gdf.drop(columns=["quad_group_id_file_index"], inplace=True)
                # Building a MultiPolygon should suffice bc suppossedly the groups don't touch across group_ids
                print("compute UU in parallel", flush=True)
                quads_poly_gdf_uu = []
                for geometry in list(executor.map(
                        unary_union_wrapper_mpi,
                        # [(key, tmpfile.name) for key in groupby_keys]
                        [group for key, group in groups]
                        )):
                    if geometry.is_empty:
                        continue
                    if isinstance(geometry, MultiPolygon):
                        quads_poly_gdf_uu.extend(geometry.geoms)
                    elif isinstance(geometry, Polygon):
                        quads_poly_gdf_uu.append(geometry)
                    else:
                        raise ValueError(f"Unreachable: Expected Polygon or MultiPolygon but got {geometry=}")
                quads_poly_gdf_uu = MultiPolygon(quads_poly_gdf_uu)
            else:
                quads_poly_gdf_uu = None
        self.quads_poly_gdf_uu = quads_poly_gdf_uu


    @cached_property
    def quads_poly_gdf_uu(self):
        # Note: This assembles union based on quad_group_id since it _assumes_ they're unique and strictly non-overlapping
        with Pool(cpu_count()) as pool:
            quads_poly_gdf_uu = ops.unary_union(
                    pool.map(
                        unary_union_wrapper,
                        [group.geometry for _, group in self.quads_poly_gdf.groupby('quad_group_id')]
                        )
                    )
        pool.join()
        return quads_poly_gdf_uu


def get_side_lengths(polygon):
    num_coords = len(polygon.exterior.coords) - 1
    sides = [Point(polygon.exterior.coords[i]).distance(Point(polygon.exterior.coords[i+1]))
             for i in range(num_coords)]
    # print(sides, flush=True)
    return sides


def get_skewness(geometry):
    sides = get_side_lengths(geometry)
    # Calculate the lengths of the sides
    # Find the largest side length
    # largest_side_length = max(sides)
    # min_side_length = min(sides)
    # if min_side_length < 1.:
    #     return float('inf')
    equivalent_radius = np.sqrt(geometry.area/np.pi)
    if equivalent_radius == 0.:
        return float('inf')
    return max(sides) / equivalent_radius
    # return skewness

def get_chunk_skewness(chunk):
    # output = chunk.copy().to_frame()
    # output.drop(columns="geometry", inplace=True)
    return chunk.geometry.map(lambda x: get_skewness(x))
    # return output

def get_msh_t_poly(msh_t_pickle_path, quads_poly_gdf_uu_tmpfile, tria_index):
    import pickle
    msh_t = pickle.load(open(msh_t_pickle_path, "rb"))
    print(f"Processing {tria_index=} of {len(msh_t.tria3['index'])}", flush=True)
    poly = Polygon(msh_t.vert2['coord'][msh_t.tria3["index"][tria_index], :])
    del msh_t
    quads_poly_gdf_uu = gpd.read_feather(quads_poly_gdf_uu_tmpfile)
    possible_matches_index = list(quads_poly_gdf_uu.sindex.query(poly, predicate="intersects"))
    possible_matches = quads_poly_gdf_uu.iloc[possible_matches_index]
    return poly, possible_matches.geometry.contains(poly).any()

def unary_union_wrapper_mpi(group):
    # print(f"merging {key=}", flush=True)
    polygons = []
    for _, subgroup in group.groupby("quad_group_id"):
        group_uu = subgroup.unary_union
        if isinstance(group_uu, MultiPolygon):
            polygons.extend(group_uu.geoms)
        elif isinstance(group_uu, Polygon):
            polygons.append(group_uu)
        else:
            raise ValueError(f"Expected Polygon or MultiPolygon but got {group_uu=}")
    return MultiPolygon(polygons)

# def unary_union_wrapper_mpi(key, feather_path):
#     print(f"merging {key=}", flush=True)
#     polygons = []
#     for _, group in gpd.read_feather(feather_path).groupby("file_index").get_group(key).groupby("quad_group_id"):
#         group_uu = group.unary_union
#         if isinstance(group_uu, MultiPolygon):
#             polygons.extend(group_uu.geoms)
#         elif isinstance(group_uu, Polygon):
#             polygons.append(group_uu)
#         else:
#             raise ValueError(f"Expected Polygon or MultiPolygon but got {group_uu=}")
    return MultiPolygon(polygons)


def unary_union_wrapper(gdf):
    return gdf.unary_union


def get_quad_group_feather_from_banks(
        banks: MultiLineString,
        banks_crs: CRS,
        quad_group_id,
        cache_directory=None,
        min_cross_section_node_count: int = None,
        max_quad_width: float = None,
        min_quad_width: float = None,
        ):
    cache_directory = cache_directory or Path(os.getenv('GEOMESH_TEMPDIR', os.getcwd() + '/.tmp-geomesh'))
    cache_directory /= 'quad_feather_from_banks_file'
    cache_directory.mkdir(exist_ok=True, parents=True)

    serialized_banks = wkb.dumps(banks)
    banks_hash = hashlib.sha256(
            ''.join([
                f"{serialized_banks}",
                f"{max_quad_width}",
                f"{min_quad_width}",
                f"{quad_group_id}",  # TODO: the hash counts the quad_group_id cause the feather file includes it. =(
                f"{min_cross_section_node_count}",
                ]).encode()
            ).hexdigest()

    # Create the hash-based filename
    outfname = cache_directory / f"{banks_hash}.feather"

    # Check if cached file already exists
    if outfname.exists():
        logger.debug(f"Returning cached file: {outfname}")
        return outfname

    # If cached file does not exist, generate it
    quad_group_gdf = generate_quad_group_from_banks(
            banks,
            banks_crs,
            quad_group_id,
            min_cross_section_node_count=min_cross_section_node_count,
            max_quad_width=max_quad_width,
            min_quad_width=min_quad_width,
            )
    quad_group_gdf.to_feather(outfname)
    logger.debug(f"Saved new cache file: {outfname}")
    return outfname


def func_wrap_triplet(func, pair):
    return func(quad_group_id=pair[0], triplet=pair[1])


def func_wrap_banks(func, pair):
    return func(quad_group_id=pair[0], banks=pair[1])


def read_with_bbox(banks_file: str, bbox: Union[None, Bbox, typing.Dict]) -> gpd.GeoDataFrame:
    if bbox is None:
        return gpd.read_file(banks_file)

    if isinstance(bbox, Bbox):
        xmin, ymin, xmax, ymax = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        return gpd.read_file(banks_file, bbox=(xmin, ymin, xmax, ymax))

    if isinstance(bbox, dict):
        # Retrieve file CRS if it exists
        banks_gdf_crs = gpd.read_file(banks_file, rows=1).crs
        bbox_crs = CRS(bbox.get('crs', banks_gdf_crs))

        # Transform CRS if different
        if bbox_crs != CRS.from_user_input(banks_gdf_crs):
            transformer = Transformer.from_crs(bbox_crs, banks_gdf_crs, always_xy=True)
            xmin, ymin = transformer.transform(bbox['xmin'], bbox['ymin'])
            xmax, ymax = transformer.transform(bbox['xmax'], bbox['ymax'])
        else:
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

        return gpd.read_file(banks_file, bbox=(xmin, ymin, xmax, ymax))


def read_and_concat(file_paths):
    return pd.concat([gpd.read_feather(fp) for fp in file_paths], ignore_index=True)


def balanced_chunking(output_feathers, chunk_size):
    n = len(output_feathers)
    base_size = n // chunk_size  # Base size of each chunk
    remainder = n % chunk_size   # Remaining elements to be distributed

    chunks = []
    start = 0  # Starting index for each chunk

    for i in range(chunk_size):
        end = start + base_size + (1 if i < remainder else 0)  # Add 1 to the size for the first 'remainder' chunks
        chunks.append(output_feathers[start:end])
        start = end  # Update the starting index for the next iteration

    return chunks


def dfs(node, visited, graph):
    if node in visited:
        return []
    visited.add(node)
    component = [node]
    for neighbor in graph.get(node, []):
        component.extend(dfs(neighbor, visited, graph))
    return component



def find_connected_components(graph):
    visited = set()
    components = []
    for node in graph.keys():
        if node not in visited:
            components.append(dfs(node, visited, graph))
    return components


def generate_quads_gdf_from_banks_file_mpi(
        comm,
        banks_file,
        rank=0,
        bbox=None,
        cleanup=True,
        **kwargs
        ) -> List[List[LinearRing]]:

    banks_gdf = read_with_bbox(banks_file, bbox)
    from concurrent.futures import as_completed
    with MPICommExecutor(comm, root=rank) as executor:
        if executor is not None:
            func = partial(
                get_quad_group_feather_from_banks,
                banks_crs=banks_gdf.crs,
                **kwargs
            )

            # Submitting tasks to the executor
            futures = {
                executor.submit(func, quad_group_id=quad_group_id, banks=bank_pair): quad_group_id
                for quad_group_id, bank_pair in zip(banks_gdf.index, banks_gdf.geometry)
            }

            output_feathers = []

            # Retrieving results
            for future in as_completed(futures):
                quad_group_id = futures[future]
                try:
                    output_feather = future.result()
                    output_feathers.append(output_feather)
                except Exception as e:
                    logger.error(f"Exception raised when processing quad_group_id {quad_group_id}: {e}")
                    raise

            read_futures = {executor.submit(gpd.read_feather, fname): fname for fname in output_feathers}

            logger.debug('will wait as completed')
            quads_gdf_list = []

            # Retrieving results for gpd.read_feather
            for future in as_completed(read_futures):
                fname = read_futures[future]
                try:
                    quad_gdf = future.result()
                    quads_gdf_list.append(quad_gdf)
                except Exception as e:
                    logger.error(f"Exception raised when reading feather file {fname}: {e}")
                    raise
            chunk_size = comm.Get_size()
            chunks = balanced_chunking(output_feathers, chunk_size)
            func = partial(
                    read_and_concat,
                    # cleanup=False
                    )
            quads_gdf = pd.concat(list(executor.map(func, chunks)), ignore_index=True)
            # quads_gdf.drop(columns=['normal_vector']).to_file('all_pairs_as_quads_not_cleaned.gpkg')
    comm.barrier()
    raise NotImplementedError
            # if cleanup is True:
            #     cleanup_quads_gdf_mut(quads_gdf)
    # logger.debug('dask takes over...')
    # import dask.dataframe as dd
    # import dask_geopandas as dgpd

    # from dask.distributed import LocalCluster, Client

    # if comm.Get_rank() == 0:
    #     # import socket
    #     # cluster = LocalCluster(n_workers=32, host=socket.gethostname())
    #     # cluster.scale(1)
    #     # client = Client()
    #     logger.debug('calling partial')
    #     quads_gdf = quads_gdf[~quads_gdf.is_empty]
    #     quads_gdf = quads_gdf[quads_gdf.is_valid]
    #     loaded = dgpd.from_geopandas(quads_gdf, npartitions=comm.Get_size())
    #     # logger.debug('calling concat')
    #     # result = dd.concat(loaded)
    #     logger.debug('calling compute now')
    #     computed_gdf = loaded.unary_union.compute(progressbar=True)
    #     # computed_gdf = client.compute(dgpd.from_geopandas(gdf, npartitions=npartitions).dissolve()))
    # comm.barrier()



    # if comm.Get_rank() == 0:
    #     import socket
    #     cluster = LocalCluster(scheduler_file='scheduler.json', host=socket.gethostname(), n_workers=comm.Get_size())
    #     print(f"{cluster=}", flush=True)

    # # print(f"rank {comm.Get_rank()}, will wait for cluster", flush=True)
    # comm.barrier()
    # print(f"rank {comm.Get_rank()}, will start client", flush=True)
    # client = Client(scheduler_file='scheduler.json', processes=True, threads_per_worker=1, n_workers=1)
    # print(f"rank {comm.Get_rank()}, {client.status=}", flush=True)
    # if comm.Get_rank() == 0:
    #     logger.debug('loading from pandas')
    #     loaded = list(map(partial(dgpd.from_pandas, npartitions=comm.Get_size()), quads_gdf_list))
    #     logger.debug('calling concat now')
    #     result = dd.concat(loaded)
    #     logger.debug('calling compute now')
    #     computed_gdf = result.compute(scheduler=cluster, progressbar=True)
    #     print(computed_gdf)
    # comm.barrier()
    # client.close()
    if comm.Get_rank() == 0:
        cluster.close()
        return computed_gdf


def generate_quads_gdf_from_triplets_file_mpi(
        comm,
        triplets_file,
        # max_quad_length: float,
        # min_quad_length=0.,
        # shrinkage_factor=0.9,
        # cross_distance_factor=0.95,
        # min_branch_length=None,
        # threshold_size=None,
        # resample_distance=None,
        # simplify_tolerance=None,
        # interpolation_distance=None,
        # min_ratio=0.1,
        # min_area=np.finfo(np.float64).min,
        # min_cross_section_node_count=4,
        # max_quad_width=None,
        # min_quad_width=None,
        # previous: Quads = None,
        **kwargs
        ) -> List[List[LinearRing]]:
    from functools import partial

    from mpi4py.futures import MPICommExecutor
    triplets_gdf = gpd.read_file(triplets_file)
    with MPICommExecutor(comm) as executor:
        if executor is not None:
            func = partial(
                    get_quad_group_feather_from_triplet,
                    triplet_crs=triplets_gdf.crs,
                    # max_quad_length=max_quad_length,
                    **kwargs
                    )
            # func = partial(func_wrap, func)
            output_feathers = []
            geo_id_pairs = zip(triplets_gdf.index, triplets_gdf.geometry)
            for pair in geo_id_pairs:
                output_feathers.append(func_wrap_triplet(func, pair))  # quad_group_id=pair[0], triplet=pair[1]))
            quads_gdf = pd.concat([gpd.read_feather(fname) for fname in output_feathers], ignore_index=True)
            # print(quads_gdf.dtypes, flush=True)
            print(quads_gdf, flush=True)
            quads_gdf.to_file('/sciclone/pscr/jrcalzada/thesis/runs/tropicald-validations/hindcasts/Harvey2017/test.gpkg')
            # import matplotlib.pyplot as plt
            # quads_gdf.plot(ax=plt.gca())
            # plt.show(block=True)


def create_polygon_from_disjoint_linestrings(line1, line2):
    # Get coordinates of endpoints for each LineString
    coords1 = line1.coords
    coords2 = line2.coords

    start1, end1 = coords1[0], coords1[-1]
    start2, end2 = coords2[0], coords2[-1]

    # Find the closest pair of endpoints between the two LineStrings
    pairs = [(start1, start2), (start1, end2), (end1, start2), (end1, end2)]
    closest_pair = min(pairs, key=lambda pair: LineString(pair).length)

    # Create the connecting segments based on the closest pair
    connect1 = LineString([closest_pair[0], closest_pair[1]])

    # Find the remaining pair for connecting segment
    remaining_pair = [pair for pair in pairs if pair != closest_pair][0]
    connect2 = LineString([remaining_pair[0], remaining_pair[1]])

    # Create the Polygon by combining both LineStrings and connecting segments
    ring_coords = list(line1.coords) + list(connect1.coords)[1:] + list(line2.coords)[::-1] + list(connect2.coords)[1:] + [line1.coords[0]]
    polygon = Polygon(ring_coords)

    return polygon


def unit_vector(vector):
    """ Returns the unit vector of the input vector. """
    return vector / np.linalg.norm(vector)


def compute_quads_from_banks(
        bank_left: LineString,
        bank_right: LineString,
        min_quad_width=None,
        max_quad_width=None,
        min_cross_section_node_count=4,
        ):
    quads_coll = []

    # Verify that both LineStrings have the same number of points
    if len(bank_left.coords) != len(bank_right.coords):
        raise ValueError("Both banks must have the same number of points.")

    for i in range(len(bank_left.coords) - 1):
        p1 = bank_left.coords[i]
        p2 = bank_left.coords[i + 1]
        p3 = bank_right.coords[i + 1]
        p4 = bank_right.coords[i]
        # Calculate vectors u and v
        u = np.array(p2) - np.array(p1)
        v = np.array(p4) - np.array(p3)

        # Normalize to get unit vectors
        u_hat = unit_vector(u)
        v_hat = unit_vector(v)

        # Calculate the bisector unit vector
        b = u_hat + v_hat
        b_hat = unit_vector(b)

        # Store the bisector

        quad = LinearRing([p1, p4, p3, p2, p1])
        quads_coll.append((quad, b_hat))


    return split_base_quads(quads_coll, min_quad_width, max_quad_width, min_cross_section_node_count)


def generate_quad_group_from_banks(
        banks: MultiLineString,
        banks_crs: CRS,
        quad_group_id,
        min_cross_section_node_count=4,
        max_quad_width=None,
        min_quad_width=None,
        ) -> List[LinearRing]:

    if banks is None or banks.is_empty:
        return []

    if not isinstance(banks, MultiLineString) or len(banks.geoms) != 2:
        raise ValueError('Expected the bank to be a MultiLineString with exactly 2 lines.')

    if not banks.is_valid:
        return gpd.GeoDataFrame(
                columns=[
                    'quad_group_id',
                    'quad_id',
                    'quad_row_id',
                    'geometry',
                    # 'normal_vector',
                    ],
                crs=banks_crs,
                )

    if banks_crs.is_geographic:
        centroid = np.array(banks.centroid.coords).flatten()
        local_proj = CRS.from_user_input(
            f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
            )
        banks = ops.transform(Transformer.from_crs(banks_crs, local_proj, always_xy=True).transform, banks)
        transformer = Transformer.from_crs(local_proj, banks_crs, always_xy=True).transform

    bank_left = banks.geoms[0]
    bank_right = banks.geoms[1]

    if len(bank_left.coords) != len(bank_right.coords):
        raise ValueError('Each bank pair must contain exactly the same number of nodes')

    this_quads = compute_quads_from_banks(
            bank_left,
            bank_right,
            min_quad_width=min_quad_width,
            max_quad_width=max_quad_width,
            min_cross_section_node_count=min_cross_section_node_count,
            )
    if len(this_quads) == 0:
        return gpd.GeoDataFrame(
                columns=[
                    'quad_group_id',
                    'quad_id',
                    'quad_row_id',
                    'geometry',
                    # 'normal_vector',
                    ],
                crs=banks_crs,
                )
    data = []
    for i, (this_quad_row, normal_vector) in enumerate(this_quads):
        for j, this_quad in enumerate(this_quad_row):
            if banks_crs.is_geographic:
                this_quad = ops.transform(transformer, this_quad)
            data.append({
                'quad_group_id': quad_group_id,
                'quad_row_id': i,
                'quad_id': j,
                'geometry': this_quad,
                # 'normal_vector': (float(normal_vector[0]), float(normal_vector[1])),
                })

    quads_gdf = gpd.GeoDataFrame(data, crs=banks_crs)
    # verification plot
    # import matplotlib.pyplot as plt
    # if banks_crs.is_geographic:
    #     gpd.GeoDataFrame(geometry=[banks], crs=local_proj).to_crs(banks_crs).plot(ax=plt.gca(), edgecolor='red')
    # else:
    #     gpd.GeoDataFrame(geometry=[banks], crs=banks_crs).plot(ax=plt.gca(), edgecolor='red', alpha=0.5)
    # quads_gdf.plot(ax=plt.gca(), edgecolor='k')
    # import contextily as cx
    # cx.add_basemap(plt.gca(), source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", crs=banks_crs)
    # plt.show(block=True)
    return quads_gdf



def generate_quad_group_from_triplet(
        triplet,
        triplet_crs,
        quad_group_id,
        max_quad_length: float,
        min_quad_length=None,
        shrinkage_factor=0.9,
        cross_distance_factor=0.95,
        min_cross_section_node_count=4,
        max_quad_width=None,
        min_quad_width=None,
        ) -> List[LinearRing]:

    if triplet is None or triplet.is_empty:
        return []

    if not triplet.is_valid:
        return gpd.GeoDataFrame(
                columns=[
                    'quad_group_id',
                    'quad_id',
                    'quad_row_id',
                    'geometry',
                    # 'normal_vector',
                    ],
                crs=triplet_crs,
                )

    if triplet_crs.is_geographic:
        centroid = np.array(triplet.centroid.coords).flatten()
        local_proj = CRS.from_user_input(
            f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
            )
        triplet = ops.transform(Transformer.from_crs(triplet_crs, local_proj, always_xy=True).transform, triplet)
        transformer = Transformer.from_crs(local_proj, triplet_crs, always_xy=True).transform

    bank_left = triplet.geoms[0]
    this_centerline = triplet.geoms[1]
    bank_right = triplet.geoms[2]

    # patch = Polygon(sort_points_clockwise(np.vstack([np.array(bank_left.coords), np.array(bank_right.coords)])))
    patch = create_polygon_from_disjoint_linestrings(bank_left, bank_right)
    this_quads = compute_quads_from_centerline(
            patch,
            this_centerline,
            max_quad_length,
            min_quad_length,
            shrinkage_factor,
            cross_distance_factor,
            min_cross_section_node_count,
            min_quad_width,
            max_quad_width,
            )
    # this_quads = compute_quads_from_banks(
    #         bank_left,
    #         bank_right,
    #         # max_quad_length,
    #         # min_quad_length,
    #         # shrinkage_factor,
    #         # None,  # base_quad
    #         # cross_distance_factor,
    #         min_cross_section_node_count,
    #         min_quad_width,
    #         max_quad_width,
    #         )
    if len(this_quads) == 0:

        return gpd.GeoDataFrame(
                columns=[
                    'quad_group_id',
                    'quad_id',
                    'quad_row_id',
                    'geometry',
                    # 'normal_vector',
                    ],
                crs=triplet_crs,
                )
    data = []
    for i, (this_quad_row, normal_vector) in enumerate(this_quads):
        for j, this_quad in enumerate(this_quad_row):
            if triplet_crs.is_geographic:
                this_quad = ops.transform(transformer, this_quad)
            data.append({
                'quad_group_id': quad_group_id,
                'quad_row_id': i,
                'quad_id': j,
                'geometry': this_quad,
                # 'normal_vector': (float(normal_vector[0]), float(normal_vector[1])),
                })

    quads_gdf = gpd.GeoDataFrame(data, crs=triplet_crs)
    # veriication plot
    import matplotlib.pyplot as plt
    if triplet_crs.is_geographic:
        gpd.GeoDataFrame(geometry=[triplet], crs=local_proj).to_crs(triplet_crs).plot(ax=plt.gca(), edgecolor='red')
    else:
        gpd.GeoDataFrame(geometry=[triplet], crs=triplet_crs).plot(ax=plt.gca(), edgecolor='red', alpha=0.5)
    quads_gdf.plot(ax=plt.gca(), edgecolor='k')
    import contextily as cx
    cx.add_basemap(plt.gca(), source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", crs=triplet_crs)
    plt.show(block=True)
    return quads_gdf


def generate_quad_gdf_from_mp(
        mp,
        mp_crs,
        max_quad_length: float,
        min_quad_length=0.,
        shrinkage_factor=0.9,
        cross_distance_factor=0.95,
        min_branch_length=None,
        # lower_threshold_size=None,
        # upper_threshold_size=None,
        threshold_size: float = None,
        resample_distance=None,
        simplify_tolerance=None,
        interpolation_distance=None,
        min_area_to_length_ratio=0.1,
        min_area=np.finfo(np.float32).eps,
        min_cross_section_node_count=4,
        max_quad_width=None,
        min_quad_width=None,
        min_quads_per_group: Union[int | None] = None,
        previous: Quads = None,
        nprocs=cpu_count(),
        ) -> gpd.GeoDataFrame:
    logger.debug("Begin generate_quad_gdf_from_mp")
    if mp is None or mp.is_empty:
        return gpd.GeoDataFrame(
                columns=[
                    'quad_group_id',
                    'quad_id',
                    'quad_row_id',
                    'geometry',
                    # 'normal_vector',
                    ],
                crs=CRS.from_epsg(4326)
                )
    if isinstance(mp, Polygon):
        mp = MultiPolygon([mp])

    if float(min_cross_section_node_count) < 2:
        raise ValueError('Argument `cross_section_node_count` must be float >= 2.')
    if mp_crs.is_geographic:
        centroid = np.array(mp.centroid.coords).flatten()
        local_azimuthal_projection = CRS.from_user_input(
            f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
            )
        final_patches = list(gpd.GeoDataFrame(geometry=[mp], crs=mp_crs).to_crs(local_azimuthal_projection).iloc[0].geometry.geoms)
        local_crs = local_azimuthal_projection
    else:
        final_patches = [list(mp.geoms)]
        local_crs = mp_crs
    # verify
    # gpd.GeoDataFrame(geometry=final_patches, crs=local_crs).plot(ax=plt.gca())
    # plt.show(block=True)
    # raise

    if simplify_tolerance is not None:
        final_patches = [patch.simplify(
            tolerance=float(simplify_tolerance),
            preserve_topology=True
            ) for patch in final_patches]

    if resample_distance is not None:
        final_patches = [resample_polygon(patch, resample_distance) for patch in final_patches]

    # verify before area and ratio filter
    # print(len(final_patches), min_area, min_area_to_length_ratio, simplify_tolerance)
    # gpd.GeoDataFrame(geometry=final_patches, crs=local_crs).plot(ax=plt.gca(), facecolor='none')
    # import contextily as cx
    # cx.add_basemap(ax=plt.gca(), crs=local_crs)
    # plt.show(block=True)
    # raise NotImplementedError('verified before area and ratio filter')

    from shapely import make_valid
    new_final_patches = []
    for patch in final_patches:
        if not patch.is_valid:
            result = make_valid(patch)
            if isinstance(result, Polygon):
                new_final_patches.append(result)
            elif isinstance(result, MultiPolygon):
                new_final_patches.extend(result.geoms)
            else:
                # print(result)
                raise NotImplementedError(type(result))
        else:
            new_final_patches.append(patch)
    final_patches = new_final_patches
    final_patches = [
            patch for patch in final_patches
            if not patch.is_empty
            and patch.area > min_area
            and (patch.length / patch.area) < min_area_to_length_ratio
        ]

    # verify
    # gpd.GeoDataFrame(geometry=final_patches, crs=local_crs).plot(ax=plt.gca(), facecolor='none')
    # import contextily as cx
    # cx.add_basemap(ax=plt.gca(), crs=local_crs)
    # plt.show(block=True)
    # raise


    # if upper_threshold_size is not None:
    #     final_patches = ops.unary_union(final_patches)
    #     buffered_geom = final_patches.buffer(-upper_threshold_size).buffer(upper_threshold_size)
    #     if not buffered_geom.is_empty:
    #         final_patches = final_patches.difference(
    #             gpd.GeoDataFrame([{'geometry': buffered_geom}], crs=local_crs).unary_union
    #                 )
    #     if isinstance(final_patches, MultiPolygon):
    #         final_patches = [polygon for polygon in final_patches.geoms]
    #     elif isinstance(final_patches, Polygon):
    #         final_patches = [final_patches]

    # if lower_threshold_size is not None:
    #     final_patches = ops.unary_union(final_patches)
    #     buffered_geom = final_patches.buffer(-lower_threshold_size).buffer(lower_threshold_size)
    #     if not buffered_geom.is_empty:
    #         final_patches = final_patches.difference(
    #             gpd.GeoDataFrame([{'geometry': buffered_geom}], crs=local_crs).unary_union
    #                 )
    #     if isinstance(final_patches, MultiPolygon):
    #         final_patches = [polygon for polygon in final_patches.geoms]
    #     elif isinstance(final_patches, Polygon):
    #         final_patches = [final_patches]

    if threshold_size is not None:
        final_patches = ops.unary_union(final_patches)
        buffered_geom = final_patches.buffer(-threshold_size).buffer(threshold_size)
        if not buffered_geom.is_empty:
            final_patches = final_patches.difference(
                gpd.GeoDataFrame([{'geometry': buffered_geom}], crs=local_crs).unary_union
                    )
        if isinstance(final_patches, MultiPolygon):
            final_patches = [polygon for polygon in final_patches.geoms]
        elif isinstance(final_patches, Polygon):
            final_patches = [final_patches]

        elif isinstance(final_patches, GeometryCollection):
            final_patches = [patch for patch in final_patches.geoms if isinstance(patch, Polygon) and not patch.is_empty]
        else:
            raise NotImplementedError(f"Unreachable: Unexpected {type(final_patches)=}")
    # print(final_patches)
    job_args = []
    for patch_id, this_patch in enumerate(final_patches):
        job_args.append((this_patch, {'interpolation_distance': interpolation_distance}))
    if resample_distance is not None:
        interpolation_distance = interpolation_distance or 0.5*resample_distance
    from time import time
    logger.info(f'launching pool to get centerlines with {nprocs=}')
    start = time()
    with Pool(nprocs) as pool:
        centerlines: List[MultiLineString] = pool.starmap(get_centerlines, job_args)
    logger.debug(f"computing centerlines took: {time()-start}")

    centerlines: List[LineString] = [line for multilinestring in centerlines for line in multilinestring.geoms]

    # if lower_threshold_size or upper_threshold_size:
    #     with Pool(nprocs) as pool:
    #         is_within = pool.map(MultiPolygon(final_patches).contains, centerlines)
    #     pool.join()
    #     centerlines = [line for line, result in zip(centerlines, is_within) if result is True]
    # verify
    # gpd.GeoDataFrame(geometry=final_patches, crs=local_crs).plot(ax=plt.gca(), facecolor='none')
    # gpd.GeoDataFrame(geometry=centerlines, crs=local_crs).plot(cmap='tab20', ax=plt.gca())
    # plt.show(block=True)
    # raise
    centerlines = filter_linestrings(
            centerlines,
            min_branch_length=min_branch_length or 3.*max_quad_length,
            )

    # gpd.GeoDataFrame(geometry=centerlines).plot(ax=plt.gca(), cmap='tab20')
    # gpd.GeoDataFrame(geometry=final_patches, crs=local_crs).plot(ax=plt.gca(), facecolor='none')
    # plt.show(block=False)
    # breakpoint()
    # raise
    # new_centerlines.extend(filter_linestrings(list(MultiLineString(centerlines).difference(MultiLineString(new_centerlines)).geoms)))
    # residuals = MultiLineString(new_centerlines).difference(centerlines)
    # # verify
    # gpd.GeoDataFrame(geometry=final_patches, crs=local_crs).plot(ax=plt.gca(), facecolor='none')
    # gpd.GeoDataFrame(geometry=new_centerlines, crs=local_crs).plot(ax=plt.gca(), color=[np.random.rand(3,) for _ in range(len(new_centerlines))])
    # # gpd.GeoDataFrame(geometry=residuals, crs=local_crs).plot(ax=plt.gca(), color=[np.random.rand(3,) for _ in range(len(residuals))])
    # plt.show(block=False)
    # breakpoint()
    # raise
    if isinstance(centerlines, GeometryCollection) and centerlines.is_empty:
        return gpd.GeoDataFrame(
                columns=[
                    'quad_group_id',
                    'quad_id',
                    'quad_row_id',
                    'geometry',
                    # 'normal_vector',
                ],
                crs=CRS.from_epsg(4326)
                )

    if isinstance(centerlines, LineString):
        centerlines = [centerlines]

    centerlines_and_patches = match_centerlines_to_patches(centerlines, final_patches, local_azimuthal_projection)

    job_args = []
    for group_id, (this_centerline, this_patch) in enumerate(centerlines_and_patches):
        # verification plot
        # this_color = np.random.rand(3,)
        # gpd.GeoDataFrame(
        #         [{'geometry': this_centerline}],
        #         crs=local_azimuthal_projection
        #     ).plot(
        #             ax=plt.gca(),
        #             color=this_color,
        #             )
        job_args.append((
            this_patch,
            this_centerline,
            group_id,
            max_quad_length,
            min_cross_section_node_count,
            min_quad_length,
            min_quad_width,
            max_quad_width,
            shrinkage_factor,
            cross_distance_factor
            ))

    # print(f"{len(job_args)=}")

    logger.info(f'launching pool to get quad groups on {nprocs=}')
    start = time()
    # with Pool(nprocs) as pool:
    #     quads_gdf = pool.starmap(get_quad_group_data, job_args)
    quads_gdf = list(map(lambda args: get_quad_group_data(*args), job_args))
    quads_gdf = [item for sublist in quads_gdf for item in sublist if len(item) > 0]
    logger.debug(f"computing quad_groups took: {time()-start}")
    if len(quads_gdf) == 0:
        return gpd.GeoDataFrame(
                columns=[
                    'quad_group_id',
                    'quad_id',
                    'quad_row_id',
                    'geometry',
                    # 'normal_vector',
                ],
                crs=CRS.from_epsg(4326)
                )
    quads_gdf = gpd.GeoDataFrame(quads_gdf, crs=local_azimuthal_projection)
    quads_gdf = quads_gdf[~quads_gdf.geometry.is_empty]
    quads_gdf = quads_gdf[quads_gdf.geometry.is_valid]
    quads_gdf["area"] = quads_gdf.geometry.map(Polygon).area
    quads_gdf = quads_gdf[quads_gdf["area"] >= 2]
    quads_gdf.drop(columns=["area"], inplace=True)


    def final_skewness_is_acceptable(geometry) -> bool:
        trias = ops.triangulate(geometry)
        return np.all([get_skewness(tria) < 60. for tria in trias])


    quads_gdf["skewness_is_within_range"] = quads_gdf.geometry.map(final_skewness_is_acceptable)
    quads_gdf = quads_gdf[quads_gdf["skewness_is_within_range"] == True]
    quads_gdf.drop(columns=["skewness_is_within_range"], inplace=True)


    # make sure we keep only the unique
    quads_gdf.drop_duplicates(subset='geometry', keep="first", inplace=True)
    quads_gdf = quads_gdf.reset_index(drop=True)

    # verify
    # import contextily as cx
    # quads_gdf.plot(ax=plt.gca())
    # cx.add_basemap(plt.gca(), source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", crs=quads_gdf.crs)
    # plt.show(block=False)
    # breakpoint()
    # raise

    # make quad group_id sequential
    # Step 1: Get unique values and sort them
    unique_ids = sorted(quads_gdf['quad_group_id'].unique())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    quads_gdf['quad_group_id'] = quads_gdf['quad_group_id'].map(id_mapping)
    def ensure_ccw(ring):
        geom = Polygon(ring)
        oriented_polygon = polygon.orient(geom, sign=-1.0)
        return oriented_polygon.exterior

    # Apply the orientation correction to all rows
    quads_gdf['geometry'] = quads_gdf['geometry'].apply(ensure_ccw)

    start = time()
    logger.info("Begin cleanup of overlapping quads")
    cleanup_quads_gdf_mut(quads_gdf)
    # logger.info(f"Cleanup of overlapping quads took {timedelta(seconds=time()-start)}.")
    if previous is not None:
        previous_gdf = previous.quads_gdf.copy()
        quads_gdf['quad_group_id'] += previous_gdf['quad_group_id'].max() + 1
        quads_gdf = pd.concat([quads_gdf, previous_gdf.to_crs(quads_gdf.crs)], ignore_index=True)
        cleanup_quads_gdf_mut(quads_gdf)

    if min_quads_per_group:
        counts = quads_gdf.groupby('quad_group_id').size()
        quads_gdf = quads_gdf[quads_gdf['quad_group_id'].isin(counts[counts >= min_quads_per_group].index)]
        # renumber the id's so that they are continuous
        # Get the unique remaining group ids
        unique_ids = quads_gdf['quad_group_id'].unique()

        # Sort them and generate a new consecutive range
        unique_ids.sort()
        new_ids = range(1, len(unique_ids)+1)

        # Map old ids to new ids in a dictionary
        id_map = dict(zip(unique_ids, new_ids))

        # Replace old ids with new ids
        quads_gdf['quad_group_id'] = quads_gdf['quad_group_id'].map(id_map)


    # verify
    # import contextily as cx
    # quads_gdf.plot(ax=plt.gca())
    # cx.add_basemap(plt.gca(), source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", crs=quads_gdf.crs)
    # plt.show(block=False)
    # breakpoint()
    # raise

    return quads_gdf.to_crs(CRS.from_epsg(4326))



def check_conforming(polygon_left, polygon_right) -> bool:
    from shapely import equals_exact

    # from shapely.geometry import Point
    # polygon_right = mesh_gdf.loc[row.index_right].geometry
    # quad = row.geometry
    polygon_right_to_polygon_left_intersection = polygon_right.intersection(polygon_left)
    if isinstance(polygon_right_to_polygon_left_intersection, Point):
        if any(Point(polygon_left_point).equals(polygon_right_to_polygon_left_intersection)
                for polygon_left_point in polygon_left.exterior.coords[:-1]):
            cnt = 0
            for polygon_right_point in polygon_right.exterior.coords[:-1]:
                if Point(polygon_right_point).buffer(np.finfo(np.float32).eps).intersects(polygon_left):
                    cnt += 1
            if cnt == 1:
                return True
    elif isinstance(polygon_right_to_polygon_left_intersection, LineString):
        mod = len(polygon_left.exterior.coords) - 1
        polygon_left_edges = [
            LineString([polygon_left.exterior.coords[i], polygon_left.exterior.coords[(i+1)%mod]])
            for i in range(4)
        ]
        eps = 1.e-8
        if any(equals_exact(polygon_right_to_polygon_left_intersection, polygon_left_edge, tolerance=eps)
                or  equals_exact(polygon_right_to_polygon_left_intersection.reverse(), polygon_left_edge, tolerance=eps)
                for polygon_left_edge in polygon_left_edges):
            return True
    elif isinstance(polygon_right_to_polygon_left_intersection, Polygon):
        if not polygon_right.buffer(np.finfo(np.float32).eps).intersects(polygon_left.buffer(np.finfo(np.float32).eps)):
            return True
    return False


def cleanup_touches_with_eps_tolerance_mut(quads_gdf):
    original_quads_gdf = quads_gdf.copy()
    quads_gdf['geometry'] = quads_gdf['geometry'].apply(lambda x: Polygon(x).buffer(np.finfo(np.float16).eps))

    def get_intersecting_pairs(gdf):
        # Use 'intersects' predicate and filter pairs with different 'quad_group_id'
        intersects = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
        return intersects[(intersects.index != intersects.index_right) &
                          (intersects['quad_group_id_left'] != intersects['quad_group_id_right'])]

    intersecting_pairs = get_intersecting_pairs(quads_gdf)

    intersect_dict = {}
    for idx, row in intersecting_pairs.iterrows():
        intersect_dict.setdefault(idx, set()).add(row['index_right'])
        intersect_dict.setdefault(row['index_right'], set()).add(idx)

    while intersect_dict:
        # Get the largest intersecting polygon by directly looking up its geometry in the quads_gdf
        max_area_idx = max(intersect_dict.keys(), key=lambda x: quads_gdf.at[x, 'geometry'].area)

        # Drop the largest intersecting polygon
        original_quads_gdf.drop(max_area_idx, inplace=True)
        quads_gdf.drop(max_area_idx, inplace=True)

        # Remove its intersections from the dictionary
        for idx in intersect_dict[max_area_idx]:
            intersect_dict[idx].discard(max_area_idx)
            if not intersect_dict[idx]:  # if no more intersections for this polygon
                del intersect_dict[idx]
        del intersect_dict[max_area_idx]

    # quads_gdf['geometry'] = quads_gdf['geometry'].apply(lambda x: LinearRing(x.exterior.coords))
    quads_gdf['geometry'] = original_quads_gdf['geometry']


def cleanup_overlapping_pairs_mut(quads_gdf):
    quads_gdf['geometry'] = quads_gdf['geometry'].apply(lambda x: Polygon(x))

    def get_overlapping_pairs(gdf):
        overlaps = gpd.sjoin(gdf, gdf, how='inner', predicate='overlaps')
        return overlaps[overlaps.index != overlaps.index_right]

    overlapping_pairs = get_overlapping_pairs(quads_gdf)

    overlap_dict = {}
    for idx, row in overlapping_pairs.iterrows():
        overlap_dict.setdefault(idx, set()).add(row['index_right'])
        overlap_dict.setdefault(row['index_right'], set()).add(idx)

    while overlap_dict:
        # Get the largest overlapping polygon by directly looking up its geometry in the quads_gdf
        max_area_idx = max(overlap_dict.keys(), key=lambda x: quads_gdf.at[x, 'geometry'].area)

        # Drop the largest overlapping polygon
        quads_gdf.drop(max_area_idx, inplace=True)

        # Remove its overlaps from the dictionary
        for idx in overlap_dict[max_area_idx]:
            overlap_dict[idx].discard(max_area_idx)
            if not overlap_dict[idx]:  # if no more overlaps for this polygon
                del overlap_dict[idx]
        del overlap_dict[max_area_idx]

    quads_gdf['geometry'] = quads_gdf['geometry'].apply(lambda x: LinearRing(x.exterior.coords))

def elements_to_edges(elements):
    from ordered_set import OrderedSet
    side_sets = OrderedSet()
    temp_set = set()
    def reversed_side(side):
        return (side[1], side[0])
    for row in elements:
        row_length = len(row)
        sides_to_add = []
        for i in range(row_length):
            sides_to_add.append((row[i], row[(i + 1) % row_length]))
        for side in sides_to_add:
            if side not in temp_set and reversed_side(side) not in temp_set:
                side_sets.add(side)
                temp_set.add(side)
                temp_set.add(reversed_side(side))
    unique_sides = np.array(list(side_sets))
    return unique_sides

def cleanup_quads_gdf_mut(quads_gdf):
    from time import time
    start = time()
    logger.info("Start cleanup of overlapping pairs.")
    cleanup_overlapping_pairs_mut(quads_gdf)
    logger.debug(f"cleanup overlapping pairs took: {time()-start}")
    logger.info("Start cleanup of slightly touching pairs.")
    start = time()
    cleanup_touches_with_eps_tolerance_mut(quads_gdf)
    logger.debug(f"cleanup of slightly touching pairs took: {time()-start}")
    # logger.info("Start cleanup touches")
    # logger.debug(f"cleanup touches with tolerance took: {time()-start}")
    # verify
    # quads_gdf.plot(ax=plt.gca())
    # plt.show(block=False)
    # breakpoint()

    # tria_elements_buffered = mesh_gdf[mesh_gdf.element_type=='tria']
    # tria_elements_buffered['geometry'] = tria_elements_buffered.geometry.map(lambda x: x.buffer(np.finfo(np.float16).eps))
    # quad_elements = mesh_gdf[mesh_gdf.element_type=='quad']
    # original_geometries = quad_elements.geometry
    # quad_elements['geometry'] = quad_elements.geometry.map(lambda x: x.buffer(np.finfo(np.float16).eps))
    # def get_intersection_gdf():
    #     quads_buffered = quads_gdf.copy()
    #     eps = np.finfo(np.float32).eps
    #     quads_buffered['geometry'] = quads_gdf.geometry.map(lambda x: Polygon(x).buffer(eps))
    #     intersection_gdf = gpd.sjoin(
    #             quads_buffered,
    #             quads_buffered,
    #             how='inner',
    #             predicate='intersects'
    #             )
    #     del quads_buffered
    #     intersection_gdf = intersection_gdf[intersection_gdf.index != intersection_gdf["index_right"]]


    #     # Condition for quad_row_id and quad_group_id equality
    #     cond1 = ~((intersection_gdf['quad_row_id_left'] == intersection_gdf['quad_row_id_right']) &
    #               (intersection_gdf['quad_group_id_left'] == intersection_gdf['quad_group_id_right']))

    #     # Condition for quad_group_id equality and quad_row_id being off by one
    #     cond2 = ~((intersection_gdf['quad_group_id_left'] == intersection_gdf['quad_group_id_right']) &
    #               (intersection_gdf['quad_row_id_left'] - intersection_gdf['quad_row_id_right']).abs() == 1)

    #     # Apply both conditions
    #     intersection_gdf = intersection_gdf[cond1 & cond2]
    #     intersection_gdf['geometry'] = quads_gdf.loc[intersection_gdf.index].geometry.map(Polygon)
    #     intersection_gdf['area'] = intersection_gdf.to_crs("epsg:6933").area
    #     intersection_gdf['geometry'] = quads_gdf.loc[intersection_gdf.index].geometry
    #     return intersection_gdf

    # def custom_group(df):
    #     # Building the graph
    #     graph = {}
    #     for idx, row in df.iterrows():
    #         graph[idx] = graph.get(idx, []) + [row['quad_id_left']]
    #         graph[row['quad_id_left']] = graph.get(row['quad_id_left'], []) + [idx]

    #     # Find connected components
    #     components = find_connected_components(graph)

    #     # Filter rows by their component keeping only the one with the smallest area
    #     result_df = pd.DataFrame()
    #     for component in components:
    #         group = df[df.index.isin(component)]
    #         min_area_row = group[group['area'] == group['area'].min()]
    #         result_df = pd.concat([result_df, min_area_row])

    #     return result_df

    # intersection_gdf = get_intersection_gdf()

    # breakpoint()

def test_quadgen_on_messy_tile():
    from appdirs import user_data_dir

    from geomesh import Raster
    rootdir = user_data_dir('geomesh')
    raster = Raster(
            f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/'
            'raster2/elevation/NCEI_ninth_Topobathy_2014_8483/'
            "LA_MS/ncei19_n30x25_w090x25_2020v1.tif"
            )
    # raster.resampling_factor = 0.2
    quads = Quads.from_raster(
                            raster,
                            # min_quad_length=10.,
                            max_quad_length=500.,
                            resample_distance=100.,
                            # simplify_tolerance=True,
                            # lower_threshold_size=1.,
                            # upper_threshold_size=1.,
                            # upper_threshold_size=1500.,
                            # lower_threshold_size=1500.,
                            threshold_size=1500.,
                            max_quad_width=500.,
                            # zmin=-30.,
                            # zmax=0.,
                            # zmin=0., zmax=10.,
                            # upper_threshold_size=500.,
                            # threshold_size=500.,
                            # min_quad_width=10.,
                            )
    quads.plot(ax=plt.gca(), facecolor='none')
    import contextily as cx
    cx.add_basemap(
            plt.gca(),
            source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            crs=quads.quads_gdf.crs
            )
    plt.show(block=True)
def test_quadgen_for_Boqueron():
    from appdirs import user_data_dir

    from geomesh import Raster
    rootdir = user_data_dir('geomesh')
    raster = Raster(
            f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/'
            'raster2/elevation/NCEI_ninth_Topobathy_2014_8483/'
            "prusvi/ncei19_n18x25_w067x25_2019v1.tif"
            )
    raster.resampling_factor = 0.2
    quads = Quads.from_raster(
                            raster,
                            # min_quad_length=10.,
                            max_quad_length=500.,
                            resample_distance=100.,
                            # simplify_tolerance=True,
                            max_quad_width=500.,
                            zmin=0.,
                            zmax=10.,
                            # upper_threshold_size=500.,
                            # threshold_size=500.,
                            # min_quad_width=10.,
                            )
    quads.plot(ax=plt.gca(), facecolor='none')
    import contextily as cx
    cx.add_basemap(
            plt.gca(),
            source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            crs=quads.quads_gdf.crs
            )
    plt.show(block=True)


def test_quadgen_for_tile_that_takes_too_long_and_needs_to_be_debugged():
    from appdirs import user_data_dir

    from geomesh import Geom, Hfun, JigsawDriver, Raster
    rootdir = user_data_dir('geomesh')
    raster = Raster(
            f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/'
            'raster2/elevation/NCEI_ninth_Topobathy_2014_8483/'
            # 'FL/ncei19_n29X75_w081X75_2018v1.tif'
            # 'northeast_sandy/ncei19_n40x75_w073x75_2015v1.tif'
            # 'FL/ncei19_n25x25_w081x00_2022v2.tif'
            'FL/ncei19_n25x50_w080x50_2016v1.tif'
            )
    # raster.make_plot(show=True)
    raster.resampling_factor = 0.2
    print('start quadgen')
    quads = Quads.from_raster(
                            raster,
                            # min_quad_length=10.,
                            max_quad_length=500.,
                            resample_distance=100.,
                            max_quad_width=500.,
                            threshold_size=1500.,
                            zmin=0,
                            zmax=10.,
                            # threshold_size=500.,
                            # min_quad_width=10.,
                            )
    print('done quadgen')

def test_quadgen_for_Harlem_River():
    import pickle
    from pathlib import Path

    import matplotlib.pyplot as plt
    from appdirs import user_data_dir

    from geomesh import Geom, Hfun, JigsawDriver, Raster

    rootdir = user_data_dir('geomesh')
    raster = Raster(
            f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/'
            'raster2/elevation/NCEI_ninth_Topobathy_2014_8483/'
            # "northeast_sandy/ncei19_n39x75_w075x75_2014v1.tif"
            'northeast_sandy/ncei19_n41x00_w074x00_2015v1.tif',
            )
    raster.resampling_factor = 0.2
    Path("the_quads.pkl").unlink(missing_ok=True)
    if Path("the_quads.pkl").is_file():
        quads = pickle.load(open("the_quads.pkl", "rb"))
    else:
        quads = Quads.from_raster(
                                raster,
                                # min_quad_length=10.,
                                max_quad_length=500.,
                                resample_distance=100.,
                                max_quad_width=500.,
                                threshold_size=1500.,
                                # threshold_size=500.,
                                # min_quad_width=10.,
                                )
        quads = Quads.from_raster(
                                raster,
                                threshold_size=1500.,
                                # min_quad_length=10.,
                                max_quad_length=500.,
                                resample_distance=100.,
                                # zmin=0.,
                                zmax=0.,
                                max_quad_width=500.,
                                # min_quad_width=10.,
                                previous=quads,
                                )
        quads = Quads.from_raster(
                                raster,
                                threshold_size=1500.,
                                # min_quad_length=10.,
                                max_quad_length=500.,
                                resample_distance=100.,
                                zmin=0.,
                                zmax=10.,
                                max_quad_width=500.,
                                # min_quad_width=10.,
                                previous=quads,
                                )
        pickle.dump(quads, open("the_quads.pkl", "wb"))
    # verification
    # quads.plot(ax=plt.gca(), facecolor='none')
    # plt.show(block=True)
    # raise
    raster.resampling_factor = 0.2
    # Path("the_old_msh_t.pkl").unlink(missing_ok=True)
    if Path("the_old_msh_t.pkl").is_file():
        old_msh_t = pickle.load(open("the_old_msh_t.pkl", "rb"))
    else:
        geom = Geom(
                raster,
                zmax=10.,
                )
        hfun = Hfun(
                raster,
                nprocs=cpu_count(),
                verbosity=1.,
                )
        hfun.add_gradient_delimiter(hmin=100., hmax=500.)
        # hfun.add_contour(
        #         0.,
        #         target_size=100.,
        #         expansion_rate=0.007
        #         )
        # hfun.add_contour(
        #         10.,
        #         target_size=100.,
        #         expansion_rate=0.007
        #         )
        # hfun.add_constant_value(
        #         value=500.,
        #         # lower_bound=0.
        #         )
        driver = JigsawDriver(
                geom=geom,
                hfun=hfun,
                verbosity=1,
                # sieve=True,
                # finalize=False,
                )
        # driver.opts.geom_feat = True
        old_msh_t = driver.msh_t()
        pickle.dump(old_msh_t, open("the_old_msh_t.pkl", "wb"))
    # raise NotImplementedError("ready")
    quads.quads_gdf["area"] = quads.quads_gdf.to_crs("epsg:6933").geometry.area
    quads.quads_gdf = quads.quads_gdf[quads.quads_gdf["area"] < 1]
    quads.quads_gdf.drop(columns=["area"], inplace=True)
    # new_msh_t = old_msh_t
    new_msh_t = quads(old_msh_t)
    # quads_poly_gdf_uu = quads.quads_poly_gdf_uu
    # quads_poly_uu_gdf = gpd.GeoDataFrame(
    #         geometry=[geom for geom in quads_poly_gdf_uu.geoms],
    #         crs=quads.quads_gdf.crs
    #         )
    # with Pool(cpu_count()) as pool:
    #     original_mesh_gdf = gpd.GeoDataFrame(
    #         geometry=list(pool.map(Polygon, old_msh_t.vert2['coord'][old_msh_t.tria3['index'], :].tolist())),
    #         crs=old_msh_t.crs
    #     ).to_crs(quads.quads_poly_gdf.crs)
    # print("inner join")
    # joined = gpd.sjoin(
    #         original_mesh_gdf,
    #         quads_poly_uu_gdf,
    #         how='left',
    #         predicate='within',
    #         )
    # target_mesh = original_mesh_gdf.loc[joined[joined.index_right.isna()].index.unique()]
    # nodes, elements = poly_gdf_to_elements(target_mesh)
    # target_msh_t = Quads.jigsaw_msh_t_from_nodes_elements(nodes, elements, crs=target_mesh.crs)
    # boundary_mp = target_mesh.unary_union
    # if isinstance(boundary_mp, Polygon):
    #     boundary_mp = MultiPolygon([boundary_mp])
    # # boundary_mp = utils.get_geom_msh_t_from_msh_t_as_mp(target_msh_t)
    # boundary_edges_msh_t = utils.multipolygon_to_jigsaw_msh_t(boundary_mp)
    # nodes_gdf = gpd.GeoDataFrame(geometry=[Point(x) for x in nodes])
    # nodes_gdf_buffered = nodes_gdf.copy()
    # nodes_gdf_buffered.geometry = nodes_gdf.geometry.map(lambda x: x.buffer(np.finfo(np.float32).eps))
    # joined = gpd.sjoin(
    #         nodes_gdf_buffered,
    #         gpd.GeoDataFrame(geometry=[boundary_mp], crs=target_mesh.crs),
    #         how='left',
    #         predicate='within',
    #         )
    # nodes_gdf_indices = joined[~joined.index_right.isna()].index.unique()
    # t = cdt.Triangulation(
    #         cdt.VertexInsertionOrder.AS_PROVIDED,
    #         cdt.IntersectingConstraintEdges.RESOLVE,
    #         0.
    #         )
    # vertices = [cdt.V2d(*coord) for coord in boundary_edges_msh_t.vert2["coord"]]
    # nodes, elements = poly_gdf_to_elements(quads.quads_poly_gdf)
    # quads_msh_t = Quads.jigsaw_msh_t_from_nodes_elements(nodes, elements, crs=quads.quads_poly_gdf.crs)
    # vertices.exte
    # vertices.extend([cdt.V2d(*point.coords[0]) for point in nodes_gdf.loc[nodes_gdf_indices].geometry])
    # t.insert_vertices(vertices)
    # t.insert_edges([cdt.Edge(*edge) for edge in boundary_edges_msh_t.edge2["index"]])

    # t.erase_outer_triangles_and_holes()

    # print("make final vertices")
    # vertices = np.array([(v.x, v.y) for v in t.vertices])
    # print("make final elements")
    # elements = np.array([tria.vertices for tria in t.triangles])
    # print("make final gdf")
    # # with Pool(cpu_count()) as pool:
    # final_mesh_gdf = gpd.GeoDataFrame(
    #     geometry=list(map(Polygon, vertices[elements, :].tolist())),
    #     crs=target_mesh.crs
    #     )
    # final_mesh_gdf.plot(ax=plt.gca(), facecolor='lightgrey', edgecolor='k', alpha=0.3, linewidth=0.3)
    # plt.show(block=False)
    # breakpoint()
    raster.resampling_factor = None
    new_msh_t.value = np.full((new_msh_t.vert2['coord'].shape[0], 1), np.nan)
    from time import time
    logger.debug('interpolating raster to mesh...')
    start = time()
    idxs, values = interpolate_raster_to_mesh(
            new_msh_t,
            raster,
            )
    new_msh_t.value[idxs] = values.reshape((values.size, 1)).astype(jigsaw_msh_t.REALS_t)
    logger.debug(f'Done interpolating raster to mesh, took: {time()-start}...')
    if np.all(np.isnan(new_msh_t.value)):
        raise ValueError('All values are NaN!')
    from scipy.interpolate import griddata
    if np.any(np.isnan(new_msh_t.value)):
        value = new_msh_t.value.flatten()
        non_nan_idxs = np.where(~np.isnan(value))[0]
        nan_idxs = np.where(np.isnan(value))[0]
        value[nan_idxs] = griddata(
                new_msh_t.vert2['coord'][non_nan_idxs, :],
                value[non_nan_idxs],
                new_msh_t.vert2['coord'][nan_idxs, :],
                method='nearest'
                )
        new_msh_t.value = value.reshape((value.size, 1)).astype(jigsaw_msh_t.REALS_t)
    from geomesh import Mesh
    the_mesh = Mesh(new_msh_t)
    # print("writting output file")
    # the_mesh.write('test_output_no_bnd.grd', overwrite=True)
    # the_mesh.make_plot(ax=plt.gca(), elements=True)
    # plt.show(block=True)
    # pickle.dump(the_mesh, open("the_quad_mesh.pkl", "wb"))

    # raise NotImplementedError("Ready for auto_bndgen")
    # exit()
    # import pickle
    # the_mesh = pickle.load(open("the_quad_mesh.pkl", "rb"))
    # the_mesh.make_plot(ax=plt.gca(), elements=True)
    # plt.show(block=True)
    # raise
    # the_mesh.wireframe(ax=plt.gca())
    # plt.show(block=False)
    # breakpoint()
    # raise
    the_mesh.boundaries.auto_generate(
            # min_open_bound_length=10000.
            )
    the_mesh.boundaries.open.plot(ax=plt.gca(), color='b')
    # Add text labels to the centroid of each LineString
    for idx, row in the_mesh.boundaries.open.iterrows():
        centroid = row['geometry'].centroid
        plt.gca().text(centroid.x, centroid.y, f"{idx+1=}", color='red')
    the_mesh.boundaries.land.plot(ax=plt.gca(), color='g')
    the_mesh.boundaries.interior.plot(ax=plt.gca(), color='r')
    ax = plt.gca()
    the_mesh.make_plot(ax=ax, elements=True)
    # quads.quads_gdf.plot(ax=ax, color='magenta')
    # the_mesh.wireframe(ax=plt.gca())
    logger.debug('begin making mesh triplot')
    plt.title(f'node count: {len(new_msh_t.vert2["coord"])}')
    plt.gca().axis('scaled')
    plt.show(block=False)
    the_mesh.write('test_with_schism/test_output.grd', overwrite=True)
    print('done writting test file')
    breakpoint()

def interpolate_raster_to_mesh(
        msh_t,
        raster,
        ):

    from scipy.interpolate import (NearestNDInterpolator,
                                   RegularGridInterpolator)
    coords = np.array(msh_t.vert2['coord'])
    coords_crs = msh_t.crs
    idxs = []
    values = []
    for xi, yi, zi in raster:
        zi = zi[0, :]
        vert2_idxs = np.where(
            np.logical_and(
                np.logical_and(np.min(xi) <= coords[:, 0], np.max(xi) >= coords[:, 0]),
                np.logical_and(np.min(yi) <= coords[:, 1], np.max(yi) >= coords[:, 1]),
            )
        )[0]
        coords = coords[vert2_idxs, :]
        if not raster.crs.equals(coords_crs):
            transformer = Transformer.from_crs(coords_crs, raster.crs, always_xy=True)
            coords[:, 0], coords[:, 1] = transformer.transform(coords[:, 0], coords[:, 1])
        _values = RegularGridInterpolator(
                (xi, yi),
                zi.T.astype(np.float64),
                'linear',
                bounds_error=False,
                fill_value=np.nan
                )(coords)
        nan_idxs = np.where(np.isnan(_values))
        non_nan_idxs = np.where(~np.isnan(_values))
        # start = time()
        _values[nan_idxs] = NearestNDInterpolator(
                # xyzo[non_nan_idxs],
                coords[non_nan_idxs],
                _values[non_nan_idxs],
                )(coords[nan_idxs, :])
        idxs.append(vert2_idxs)
        values.append(_values)
    if len(idxs) == 0:
        return (np.array([]), np.array([]))
    values = np.hstack(values)
    idxs, values = np.hstack(idxs), values
    values = values.reshape((values.size, 1))
    return idxs, values

def test_thalweg_detection():
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    import scipy.ndimage
    from appdirs import user_data_dir

    rootdir = user_data_dir('geomesh')
    raster_path = f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/' \
            'raster2/elevation/NCEI_ninth_Topobathy_2014_8483/' \
            "northeast_sandy/ncei19_n39x75_w075x75_2014v1.tif"
            # "chesapeake_bay/ncei19_n39x25_w076x50_2019v1.tif"



    from rasterio.transform import array_bounds
    with rasterio.open(raster_path) as src:
        bathymetry = src.read(1)
        window = src.window(*src.bounds)
        crs = src.crs
        transform = src.transform
        height = int(window.height)
        width = int(window.width)
        x0, y0, x1, y1 = array_bounds(
            height,
            width,
            rasterio.windows.transform(window, src.transform)
        )
        xvals = np.linspace(x0, x1, int(window.height))
        yvals = np.linspace(y1, y0, int(window.width))
    if crs.is_geographic:
        logger.debug(
            "CRS is geographic, transforming points to local projection."
        )
        local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m "\
            f"+lat_0={np.median(yvals)} +lon_0={np.median(xvals)}"
        local_crs = CRS.from_user_input(local_azimuthal_projection)
        transformer = Transformer.from_crs(crs, local_crs, always_xy=True)
        geographic_to_local = transformer.transform
        x0, x1 = np.min(xvals), np.max(xvals)
        y0, y1 = np.min(yvals), np.max(yvals)
        (x0, x1), (y0, y1) = geographic_to_local([x0, x1], [y0, y1])
        dx = np.diff(np.linspace(x0, x1, int(window.width)))[0]
        dy = np.diff(np.linspace(y0, y1, int(window.height)))[0]
    else:
        dx = np.diff(xvals)[0]
        dy = np.diff(yvals)[0]
        # crs = local_crs

    from scipy.ndimage import binary_erosion
    logger.debug("Loading bathymetry values from raster.")
    # bathymetry = self.raster.get_values(band=1, window=window)
    # bathymetry = self.raster.get_values(band=1, window=window)
    bathymetry[bathymetry > 0.] = np.nan
    dzdx, dzdy = np.gradient(bathymetry, dx, dy)
    mask = np.logical_and(np.abs(dzdx) < 0.1, np.abs(dzdy) < 0.1)
    bathy = bathymetry.copy()
    non_saddle_values = bathymetry.copy()
    non_saddle_values[mask] = np.nan
    bathy[~mask] = np.nan
    bathy[bathy > np.nanmean(non_saddle_values)] = np.nan
    binary_image = bathymetry.copy()
    binary_image[~np.isnan(bathy)] = 1
    binary_image[np.isnan(bathy)] = -1
    plt.contourf(xvals, yvals, binary_image, levels=[0, 1])
    # plt.imshow(bathymetry, cmap='viridis', origin='lower', interpolation='nearest')
    # plt.scatter(saddle_points[1], saddle_points[0], color='red', marker='x', label='Saddle Points')
    # gpd.GeoDataFrame(geometry=gpd.points_from_xy(saddle_x, saddle_y), crs=crs).plot(ax=plt.gca())
    # plt.colorbar(label='Bathymetry')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Bathymetry and Saddle Points')
    # plt.legend()
    plt.show()
    raise


    with warnings.catch_warnings():
        # in case self._src.values is a masked array
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dh = np.sqrt(dx ** 2 + dy ** 2)
    dh = np.ma.masked_equal(dh, 0.0)

    from skimage.morphology import skeletonize
    threshold_value = 0.05

    # Apply thresholding
    binary_image = dh > threshold_value

    # Apply binary erosion to further refine the delineation
    eroded_image = binary_erosion(binary_image)


    # Skeletonize the binary image to get the centerline
    from scipy.spatial import Voronoi, voronoi_plot_2d
    skeleton = skeletonize(eroded_image)
    skeleton_iy, skeleton_ix = np.nonzero(skeleton)[1], np.nonzero(skeleton)[0]
    skeleton_x, skeleton_y = rasterio.transform.xy(transform, skeleton_ix, skeleton_iy)
    points = np.column_stack((skeleton_x, skeleton_y))
    # desired_distance = .0001
    # from simplification.cutil import simplify_coords
# Downsample the points using the simplify_coords function
    # points = simplify_coords(points, desired_distance)
    # vor = Voronoi(points)
    # edges = vor.ridge_vertices
    # lines = []
    # for edge in edges:
    #     p1_x, p1_y = vor.vertices[edge[0]]
    #     p2_x, p2_y = vor.vertices[edge[1]]
    #         # Convert point coordinates to pixel coordinates
    #     p1_ix, p1_iy = ~transform * (p1_x, p1_y)
    #     p2_ix, p2_iy = ~transform * (p2_x, p2_y)

    #     # Check if the point is within the bounds of the raster
    #     if 0 <= p1_ix < width and 0 <= p1_iy < height:
    #         if 0 <= p2_ix < width and 0 <= p2_iy < height:
    #             p1_z = bathymetry[int(p1_iy), int(p1_ix)]
    #             p2_z = bathymetry[int(p1_iy), int(p1_ix)]
    #             if np.any([np.isclose(p1_z, 0), np.isclose(p2_z, 0.)]):
    #                 continue
    #             custom_atol = 1e-5
    #             custom_rtol = 1e-5
    #             if np.isclose(p1_z, p2_z, atol=custom_atol, rtol=custom_rtol):
    #                 line = LineString([(p1_x, p1_y), (p2_x, p2_y)])

    #                 lines.append(line)

    #         # try:
    #         #     p1_in_skel = skeleton[int(p1_iy), int(p1_ix)] == 0
    #         #     p2_in_skel = skeleton[int(p2_iy), int(p2_ix)] == 0
    #         # except IndexError:
    #         #     continue
    #         # if p1_in_skel and p2_in_skel:
    #         #     p1_z = bathymetry[p1_ix, p1_iy]
    #         #     p2_z = bathymetry[p1_ix, p1_iy]
    #         #     p1_x, p1_y = rasterio.transform.xy(transform, p1_ix, p1_iy)
    #         #     p2_x, p2_y = rasterio.transform.xy(transform, p2_ix, p2_iy)
    #         #     line = LineString([(p1_x, p1_y), (p2_x, p2_y)])
    #         #     lines.append(line)
    # plt.gca().imshow(skeleton, cmap='gray', extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()])
    plt.gca().imshow(bathymetry, cmap='gray', extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()])
    # gpd.GeoDataFrame(geometry=lines, crs=crs).plot(ax=plt.gca())
    gpd.GeoDataFrame(geometry=[Point(x) for x in points], crs=crs).plot(ax=plt.gca())
    plt.show()
    raise

# # Dilate the thalweg region to create polygons representing the thalwegs
#     dilated_thalwegs = []

#     for region in vor.regions:
#         if not -1 in region and len(region) > 0:
#             # Extract the vertices of the Voronoi region
#             vertices = [vor.vertices[i] for i in region]

#             # Create a Polygon from the vertices
#             thalweg_polygon = Polygon(vertices)

#             # Dilate the thalweg polygon (replace this with your dilation logic)
#             dilated_thalweg = thalweg_polygon.buffer(1.0)

#             dilated_thalwegs.append(dilated_thalweg)

# # Identify edges not intersecting with dilated thalwegs
#     thalweg_edges = [edge for edge in voronoi_edges if not any(edge.intersects(dilated_thalweg) for dilated_thalweg in dilated_thalwegs)]

# Visualize the results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(bathymetry, cmap='viridis')
    plt.title('Bathymetry')

    plt.subplot(1, 3, 2)
    plt.plot(skeleton_x, skeleton_y, 'ro', label='Skeleton Points')
    plt.title('Skeleton Points')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.imshow(bathymetry, cmap='viridis')
    plt.plot(skeleton_x, skeleton_y, 'ro', label='Skeleton Points')
    plt.title('Voronoi Diagram')
    voronoi_plot_2d(vor, show_vertices=False, line_colors='k', line_width=2, line_alpha=0.6)

    for edge in thalweg_edges:
        plt.plot(*edge.xy, color='r', linewidth=2)

    plt.legend()

    plt.show()




#     # Dilate the thalweg region to create polygons representing the thalwegs
#     dilated_thalwegs = [Polygon(zip(*np.transpose(region))) for region in vor.regions if not -1 in region and len(region) > 0]

# # Collect all Voronoi edges
#     voronoi_edges = []

#     for region in vor.regions:
#         if not -1 in region and len(region) > 0:
#             # Extract the vertices of the Voronoi region
#             vertices = [vor.vertices[i] for i in region]

#             # Create LineStrings from consecutive pairs of vertices
#             edges = [LineString([vertices[i], vertices[i + 1]]) for i in range(len(vertices) - 1)]

#             voronoi_edges.extend(edges)

# # Identify edges not intersecting with dilated thalwegs
#     thalweg_edges = [edge for edge in voronoi_edges if not any(edge.intersects(dilated_thalweg) for dilated_thalweg in dilated_thalwegs)]

# # Visualize the results
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 3, 1)
#     plt.imshow(bathymetry, cmap='viridis')
#     plt.title('Bathymetry')

#     plt.subplot(1, 3, 2)
#     plt.plot(skeleton_x, skeleton_y, 'ro', label='Skeleton Points')
#     plt.title('Skeleton Points')
#     plt.legend()

#     plt.subplot(1, 3, 3)
#     plt.imshow(bathymetry, cmap='viridis')
#     plt.plot(skeleton_x, skeleton_y, 'ro', label='Skeleton Points')
#     plt.title('Voronoi Diagram')
#     voronoi_plot_2d(vor, show_vertices=False, line_colors='k', line_width=2, line_alpha=0.6)

#     for edge in thalweg_edges:
#         plt.plot(*edge.xy, color='r', linewidth=2)

#     plt.legend()

#     plt.show()
# # Visualize the results
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 3, 1)
#     plt.imshow(bathymetry, cmap='viridis')
#     plt.title('Bathymetry')

#     plt.subplot(1, 3, 2)
#     plt.plot(skeleton_x, skeleton_y, 'ro', label='Skeleton Points')
#     plt.title('Skeleton Points')
#     plt.legend()

# # Compute the Voronoi diagram
#     points = np.column_stack((skeleton_x, skeleton_y))
#     vor = Voronoi(points)

# # Visualize the Voronoi diagram
#     plt.subplot(1, 3, 3)
#     plt.imshow(bathymetry, cmap='viridis')
#     plt.plot(skeleton_x, skeleton_y, 'ro', label='Skeleton Points')
#     plt.title('Voronoi Diagram')
#     voronoi_plot_2d(vor, show_vertices=False, line_colors='k', line_width=2, line_alpha=0.6)
#     plt.legend()

#     plt.show()

    # # Visualize the results
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 3, 1)
    # plt.imshow(bathymetry, cmap='viridis')
    # plt.title('Bathymetry')

    # plt.subplot(1, 3, 2)
    # plt.imshow(dh, cmap='viridis')
    # plt.title('Gradient Magnitude')

    # plt.subplot(1, 3, 3)
    # plt.imshow(eroded_image, cmap='gray')
    # plt.plot(np.nonzero(skeleton)[1], np.nonzero(skeleton)[0], 'r.', markersize=2, label='Centerline')
    # plt.title('Delineated Thalwegs with Centerline')
    # plt.legend()

    # plt.show()


# # Distance transform to assign each pixel a distance value from the background
#     distance_transform = distance_transform_edt(eroded_image)

# # Find the pixels with maximum distance values (likely the centerline)
#     # centerline = np.where(distance_transform == distance_transform.max())

# # Visualize the results
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 3, 1)
#     plt.imshow(bathymetry, cmap='viridis')
#     plt.title('Bathymetry')

#     plt.subplot(1, 3, 2)
#     plt.imshow(dh, cmap='viridis')
#     plt.title('Gradient Magnitude')

#     plt.subplot(1, 3, 3)
#     plt.imshow(eroded_image, cmap='gray')
#     plt.plot(np.nonzero(skeleton)[1], np.nonzero(skeleton)[0], 'r.', markersize=2, label='Centerline')
#     plt.title('Delineated Thalwegs with Centerline')
#     plt.legend()

# #     plt.subplot(1, 3, 3)
# #     plt.imshow(eroded_image, cmap='gray')
# #     plt.scatter(centerline[1], centerline[0], color='red', marker='x', label='Centerline')
# #     plt.title('Delineated Thalwegs with Centerline')
# #     plt.legend()

#     plt.show()


# # Visualize the results
#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 3, 1)
#     plt.imshow(bathymetry, cmap='viridis')
#     plt.title('Bathymetry')

#     plt.subplot(1, 3, 2)
#     plt.imshow(dh, cmap='viridis')
#     plt.title('Gradient Magnitude')

#     plt.subplot(1, 3, 3)
#     plt.imshow(eroded_image, cmap='gray')
#     plt.title('Delineated Thalwegs')

#     plt.show()

    # plt.imshow(bathymetry)
    # plt.gca().invert_yaxis()
    # plt.title(f"{x=}, {y=}")
    # plt.show()
    # # # raise
    # print(idx_xmin,
    # # x = xvals[idx_xmin[0]]
    # y = yvals[idx_ymin[0]]
    # print(x, y)
    # raise


    # from skimage.measure import label, find_contours
    # dh = (dh - dh.min()) / (dh.max() - dh.min())

    # # labels = label(dh)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(bathymetry, cmap='gray')
    # ax[0].set_title('DEM')
    # # ax[1].imshow(dh, cmap='jet')
    # ax[1].contour(dh, levels=[1.])
    # # ax[1].set_title('Thalwegs')
    # # plt.imshow(dh, cmap='grey')
    # ax[1].invert_yaxis()
    # plt.show(block=False)
    # breakpoint()
    # raise
    # # mask = np.zeros_like(dh)
    # # print(labels)
    # # breakpoint()
    # for i in range(1, labels.max() + 1):
    #     mask_i = (labels == i)
    #     contours = find_contours(labels[mask_i], 0.5)
    #     for contour in contours:
    #         x, y = np.where(np.logical_and(mask_i, np.logical_or(contour >= contour.min(), contour <= contour.max())))
    #         thalweg_x = np.mean(x)
    #         thalweg_y = np.mean(y)
    #         plt.plot(thalweg_x, thalweg_y, 'ro')

# # Visualize thalwegs
    # plt.imshow(dh, cmap='gray')
    # plt.gca().invert_yaxis()
    # plt.show(block=False)
    # breakpoint()
    # raise



    # # logger.debug("Loading hfun_values.")
    # # multiplier = 1./3.
    # # values = np.abs((multiplier) * (bathymetry / dh))
    # # bathymetry[bathymetry > 0] = np.nan
    # # values[bathymetry > 0] = np.nan
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(bathymetry, cmap='gray')
    # ax[0].set_title('DEM')
    # ax[1].imshow(dh, cmap='jet')
    # ax[1].set_title('Thalwegs')
    # breakpoint()
    # raise
    # # plt.show(block=False)


    # from pysheds.grid import Grid

    # grid = Grid.from_raster(raster_path)
    # dem = grid.read_raster(raster_path)
    # pit_filled_dem = grid.fill_pits(dem)

    # # Fill depressions in DEM
    # flooded_dem = grid.fill_depressions(pit_filled_dem)

    # # Resolve flats in DEM
    # inflated_dem = grid.resolve_flats(flooded_dem)
    # dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # # Compute flow directions
    # # -------------------------------------
    # fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    # acc = grid.accumulation(fdir, dirmap=dirmap)
    # # Specify pour point
    # # x, y = -97.294, 32.737
    # # x, y = -76.294, 32.737

    # # Snap pour point to high accumulation cell
    # x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))

    # # Delineate the catchment
    # catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap,
    #                        xytype='coordinate')

# # Crop and plot the catchment
# # ---------------------------
# # Clip the bounding box to the catchment
    # grid.clip_to(catch)
    # # clipped_catch = grid.view(catch)
    # branches = grid.extract_river_network(fdir, acc > 50, dirmap=dirmap)
    # import seaborn as sns
    # # sns.set_palette('husl')
    # fig, ax = plt.subplots(figsize=(8.5,6.5))

    # # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # # ax.pcolor(xvals, yvals, bathymetry, cmap='gray')
    # # ax[0].set_title('DEM')
    # # dx = (xvals[1]-xvals[0])/2.
    # # dy = (yvals[1]-yvals[0])/2.
    # # extent = [xvals[0]-dx, xvals[-1]+dx, yvals[0]-dy, yvals[-1]+dy]
    # ax.imshow(bathymetry, cmap='gray', extent=[grid.bbox[0], grid.bbox[2], grid.bbox[1], grid.bbox[3]])
    # # ax[1].set_title('Thalwegs')
    # ax.set_xlim(grid.bbox[0], grid.bbox[2])
    # ax.set_ylim(grid.bbox[1], grid.bbox[3])
    # ax.set_aspect('equal')

    # for branch in branches['features']:
    #     line = np.asarray(branch['geometry']['coordinates'])
    #     ax.plot(line[:, 0], line[:, 1])

    # _ = plt.title('D8 channels', size=14)
    # plt.show(block=False)
    # breakpoint()

    # Load DEM
    # image = bathymetry
    # from scipy import ndimage as ndi

    # distance = ndi.distance_transform_edt(image)

    # from skimage.feature import peak_local_max

    # max_coords = peak_local_max(distance, labels=image,

    #                             footprint=np.ones((3, 3)))

    # local_maxima = np.zeros_like(image, dtype=bool)

    # local_maxima[tuple(max_coords.T)] = True

    # from skimage.segmentation import watershed
    # markers = ndi.label(local_maxima)[0]
    # labels = watershed(-distance, markers, mask=image)

    # fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    # ax = axes.ravel()

    # ax[0].imshow(image, cmap=plt.cm.gray)
    # ax[0].set_title('Overlapping objects')
    # ax[1].imshow(-distance, cmap=plt.cm.gray)
    # ax[1].set_title('Distances')
    # ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    # ax[2].set_title('Separated objects')

    # for a in ax:
    #     a.set_axis_off()

    # fig.tight_layout()
    # plt.show()


    # dx = np.diff(xvals)[0]
    # dy = np.diff(yvals)[0]
    # logger.debug("Loading bathymetry values from raster.")
    # # bathymetry = self.raster.get_values(band=1, window=window)
    # # bathymetry = self.raster.get_values(band=1, window=window)
    # dx, dy = np.gradient(bathymetry, dx, dy)
    # with warnings.catch_warnings():
    #     # in case self._src.values is a masked array
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    #     dh = np.sqrt(dx ** 2 + dy ** 2)
    # dh = np.ma.masked_equal(dh, 0.0)
    # logger.debug("Loading hfun_values.")
    # multiplier = 1./3.
    # values = np.abs((multiplier) * (bathymetry / dh))
    # bathymetry[bathymetry > 0] = np.nan
    # values[bathymetry > 0] = np.nan
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(bathymetry, cmap='gray')
    # ax[0].set_title('DEM')
    # ax[1].imshow(values, cmap='gray')
    # ax[1].set_title('Thalwegs')
    # plt.show()
#     from scipy.ndimage import morphological_gradient as gradient_wrap

#     from skimage.measure import label, find_contours, labeled_image
#     from skimage.segmentation import watershed

# # Invert slope to create a blob image for watershed segmentation
#     slope_inverted = 1 - slope

# # Apply watershed to segment catchment basins
#     labels = label(slope_inverted)
#     mask = np.zeros_like(bathymetry)
#     for i in range(1, labels.max() + 1):
#         mask_i = (labels == i)
#         mask_i[mask_i] = bathymetry[mask_i]
#         contours = find_contours(slope_inverted[mask_i], 0.5)
#         for contour in contours:
#             x, y = np.where(np.logical_and(mask_i, np.logical_or(contour >= contour.min(), contour <= contour.max())))
#             thalweg_x = np.mean(x)
#             thalweg_y = np.mean(y)
#             plt.plot(thalweg_x, thalweg_y, 'ro')

# # Visualize thalwegs
#     plt.imshow(bathymetry, cmap='gray')
#     plt.gca().invert_yaxis()
#     plt.show()
    # Filter land areas
    # mask = dem == float('nan')
    # dem[dem > 0] = np.nan
    # dem_filtered = dem

    # # Calculate slope
    # slope = scipy.ndimage.sobel(dem_filtered, mode='wrap')
    # slope[np.isnan(slope)] = np.finfo(np.float64).max
    # slope = np.abs(slope)
    # # Find local minima
    # thalwegs, min_val = scipy.ndimage.label(slope == slope.min())
    # thalwegs = thalwegs.astype(float)
    # thalwegs[thalwegs == 0] = np.nan
    # breakpoint()
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(dem_filtered, cmap='gray')
    # ax[0].set_title('DEM')
    # ax[1].imshow(thalwegs)
    # ax[1].set_title('Thalwegs')
    # plt.show()

    # Connect neighboring thalwegs
    # thalwegs_connected = np.zeros_like(thalwegs)
    # for i in range(1, thalwegs.max()):
    #     thalwegs_connected[thalwegs == i] = i - 1

    # # Visualize
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(dem_filtered, cmap='gray')
    # ax[0].set_title('DEM')
    # ax[1].imshow(thalwegs_connected, cmap='gray')
    # ax[1].set_title('Thalwegs')
    # plt.show()
