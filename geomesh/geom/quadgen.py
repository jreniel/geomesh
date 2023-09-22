from functools import cached_property, partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Union
import hashlib
import logging
import math
import os
import typing

from centerline import exceptions
from centerline.geometry import Centerline
from inpoly import inpoly2
from jigsawpy import jigsaw_msh_t
from matplotlib.tri import Triangulation
from matplotlib.transforms import Bbox
from mpi4py.futures import MPICommExecutor
from numpy.linalg import norm
from pyproj import CRS, Transformer
from scipy.spatial import KDTree
from shapely import ops
from shapely import wkb
from shapely.validation import explain_validity
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import LinearRing
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import polygon
from shapely.geometry import box
import centerline.exceptions
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial
import shapely.errors

from geomesh import utils

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
    from shapely.geometry import LineString
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


# def build_quadrilaterals(
#     initial_point,
#     downstream_point,
#     next_point,
#     bounding_ring,
#     cross_section_node_count,
#     previous_base_quad=None,
#     cross_distance_factor=0.75,
# ) -> List[LinearRing]:
#     # cross_distance_factor = 1.0
#     p0 = np.array(initial_point.coords).flatten()
#     p1 = np.array(downstream_point.coords).flatten()
#     p2 = np.array(next_point.coords).flatten()
#     normal_vector = compute_bisector_or_perpendicular(p0, p1, p2)
#     if previous_base_quad is not None:
#         coords = list(previous_base_quad.coords)
#         p0_left = None
#         for i in range(len(coords) - 1):
#             edge = LineString([coords[i], coords[i+1]])
#             if edge.intersects(initial_point.buffer(np.finfo(np.float32).eps)):
#                 p0_left = Point(coords[i])
#                 p0_right = Point(coords[i+1])
#                 break
#         if p0_left is None:
#             distance_p0 = cross_distance_factor * compute_distances(p0, bounding_ring)
#             p0_left, p0_right = compute_new_points(p0, distance_p0, normal_vector)
#     else:
#         distance_p0 = cross_distance_factor * compute_distances(p0, bounding_ring)
#         p0_left, p0_right = compute_new_points(p0, distance_p0, normal_vector)
#     distance_p1 = cross_distance_factor * compute_distances(p1, bounding_ring)
#     p1_left, p1_right = compute_new_points(p1, distance_p1, normal_vector)

#     # # Check if p1_left or p1_right are within previous_base_quad
#     if previous_base_quad is not None:
#         if Polygon(previous_base_quad).contains(p1_left) or Polygon(previous_base_quad).contains(p1_right):
#             # Recompute normal_vector as the perpendicular
#             normal_vector = compute_perpendicular(p0, p1, p2)
#             # Recompute p1_left and p1_right
#             p1_left, p1_right = compute_new_points(p1, distance_p1, normal_vector)

#     points = np.array([np.array(p.coords)[0] for p in [p0_left, p0_right, p1_left, p1_right]])
#     this_quad = [Point(p) for p in sort_points_clockwise(points)]
#     # this_quad = [Point(p) for p in points]
#     this_base_quad = LinearRing([np.array(p.coords[0]) for p in this_quad])
#     this_quads = subdivide_quadrilateral(normal_vector, this_base_quad, cross_section_node_count-1)
#     return this_quads, this_base_quad
#     # return [this_base_quad], this_base_quad


# def build_last_quad(
#         initial_point,
#         downstream_point,
#         bounding_ring,
#         cross_section_node_count,
#         previous_base_quad,
#         cross_distance_factor
#         ):
#     p0 = np.array(initial_point.coords).flatten()
#     p1 = np.array(downstream_point.coords).flatten()
#     normal_vector = compute_last_perpendicular(p0, p1)
#     distance_p1 = cross_distance_factor * compute_distances(p1, bounding_ring)
#     if previous_base_quad is not None:
#         coords = previous_base_quad.coords
#         distances = [np.linalg.norm(np.array(coord) - p1) for coord in coords]
#         idx = np.argsort(distances)
#         p0_left = Point(coords[int(idx[0])])
#         if np.array_equal(p0_left, Point(coords[int(idx[1])])):
#             p0_right = Point(coords[int(idx[2])])
#         else:
#             p0_right = Point(coords[int(idx[1])])
#         p1_left, p1_right = compute_new_points(p1, distance_p1, normal_vector)
#         # if previous_base_quad.contains(p1_left) or previous_base_quad.contains(p1_right):
#         #     # Recompute normal_vector as the perpendicular
#         #     normal_vector = compute_last_perpendicular(p0, p1)
#         #     # Recompute p1_left and p1_right
#         #     p1_left, p1_right = compute_new_points(p1, distance_p1, normal_vector)
#     else:
#         # just build a quad from the start and end points
#         distance_p0 = cross_distance_factor * compute_distances(p0, bounding_ring)
#         p0_left, p0_right = compute_new_points(p0, distance_p0, normal_vector)
#         p1_left, p1_right = compute_new_points(p1, distance_p1, normal_vector)
#     points = [p0_left, p0_right, p1_left, p1_right]
#     if np.any([p.is_empty for p in points]):
#         return []
#     points = np.array([np.array(p.coords)[0] for p in points])

#     this_quad = [Point(p) for p in sort_points_clockwise(points)]

#     this_quads = subdivide_quadrilateral(
#             normal_vector,
#             LinearRing([np.array(p.coords[0]) for p in this_quad]),
#             cross_section_node_count-1
#             )
#     return this_quads


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
    p0 = Point(centerline.coords[0])
    p1 = centerline.interpolate(max_quad_length)
    if p1 == Point(centerline.coords[-1]):
        p2 = p1
    else:
        distance_along_line = centerline.project(p1)
        p2 = centerline.interpolate(distance_along_line + max_quad_length)
    quads_coll = []
    current_quad_length = max_quad_length
    base_quad = None
    previous_tail_points = None
    while p1 != p2:
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(build_base_quad)
        # new_base_quad, new_tail_points = lp_wrapper(p0, p1, p2, bounding_ring, base_quad, cross_distance_factor)
        # lp.print_stats()
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
        p1 = centerline.interpolate(distance_along_line + current_quad_length)
        if p1 == Point(centerline.coords[-1]):
            p2 = p1
        else:
            distance_along_line = centerline.project(p1)
            p2 = centerline.interpolate(distance_along_line + current_quad_length)

    if base_quad is not None:
        if p0 != p1 and p1 == p2:
            # most common condition.
            logger.debug(f'Condition A-1: {type(base_quad)=} {p0=} {p1=} {p2=} p0!=p1==p2 so we built last quad')
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
                logger.debug(f'Condition A-1: {type(base_quad)=} {p0=} {p1=} {p2=} {max_quad_length=} >= {d=} >= {min_quad_length=} is False so passed')
                pass
                # raise NotImplementedError(' A-1 Presumbaly unreachable: {p0.distance(p1)}')
        else:
            # rare, but it happens (all points are presumably different)
            logger.debug(f'Condition A-2: {type(base_quad)=} {p0=} {p1=} {p2=} All must be different so we added two quad rows')
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
                raise NotImplementedError(f'Condition A-2: Presumably unreachable: {d01=} {d12=} {d02=} {min_quad_length=} {max_quad_length=}')

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
            logger.debug(f'Condition B-1: {type(base_quad)=} {p0=} {p1=} {p2=} all equal and base_quad is None so we passed')
            pass

        elif p0 != p1 and p1 == p2:
            # This condition is the second most common
            logger.debug(f'Condition B-2-I: {type(base_quad)=} {p0=} {p1=} {p2=} (base_quad is None and p0!=p1==p2) we built last quad...')
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
            logger.debug(f'Condition B-2-II: {type(base_quad)=} {p0=} {p1=} {p2=} (base_quad is None and p0!=p1==p2 IS NOT TRUE) we passed...')
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
        else:
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


def get_multipolygon_for_raster(raster, zmin=None, zmax=None, raster_opts=None, window=None, pad_width=20):
    from geomesh import Geom, Raster
    from geomesh.cli.raster_opts import get_raster_from_opts
    if zmax is None:
        zmax = 0 if zmin is None else np.finfo(np.float64).max
    if not isinstance(raster, Raster):
        raster = get_raster_from_opts(raster, raster_opts, window)

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
                'normal_vector': (normal_vector[0], normal_vector[1]),
                })
    # verify
    # gpd.GeoDataFrame(quad_groups).plot(ax=plt.gca())
    # gpd.GeoDataFrame(geometry=[patch]).plot(ax=plt.gca(), facecolor='none', edgecolor='blue')
    # plt.show(block=True)
    return quad_groups


def poly_gdf_to_elements(poly_gdf):
    node_mappings = {}
    connectivity_table = []

    for row in poly_gdf.itertuples():
        # element: LinearRing = row.geometry.exterior
        element = polygon.orient(row.geometry).exterior
        element_indices = []
        # for i in range(len(element.exterior.coords) - 1):
        for i in range(len(element.coords) - 1):
            p = element.coords[i]
            if p not in node_mappings:
                node_mappings[p] = len(node_mappings)
            element_indices.append(node_mappings[p])
        connectivity_table.append(element_indices)

    return list(node_mappings.keys()), connectivity_table


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


def manually_triangulate(linear_ring):

    def angle(a, b, c):
        """Returns the angle at vertex b in degrees."""
        if a == b or b == c or a == c:
            return 0.0

        ba = ((a[0] - b[0]), (a[1] - b[1]))
        bc = ((c[0] - b[0]), (c[1] - b[1]))

        cosine_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2))

        # Clip the cosine_angle to the valid range to avoid math domain error
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle = math.acos(cosine_angle)

        return math.degrees(angle)

    def find_ears(linear_ring):
        ears = []
        n = len(linear_ring.coords)
        for i in range(n-1):  # Adjusted the loop range to avoid adding the duplicate first point in the last triangle
            triangle = [linear_ring.coords[i], linear_ring.coords[(i + 1) % n], linear_ring.coords[(i + 2) % n]]
            if len(set(triangle)) < 3:  # Check that all vertices are distinct
                continue
            # mid_point = Point((triangle[0][0] + triangle[2][0]) / 2, (triangle[0][1] + triangle[2][1]) / 2)
            if Polygon(linear_ring).contains(Polygon(triangle)):
                ears.append(triangle)
        return ears

    triangles = []
    coords_list = list(linear_ring.coords)[:-1]  # Excluding the duplicate first point

    while len(coords_list) > 3:
        ears = find_ears(linear_ring)
        if ears:
            # Find the ear with the minimum smallest angle
            best_ear = min(ears, key=lambda ear: min(angle(*ear), angle(ear[1], ear[2], ear[0]), angle(ear[2], ear[0], ear[1])))
            triangles.append(LinearRing(best_ear))
            idx = coords_list.index(best_ear[1])
            coords_list.pop(idx)
            linear_ring = LinearRing(coords_list)
        else:
            break

    if len(coords_list) == 3:
        triangles.append(LinearRing(coords_list))

    return triangles


class Quads:

    def __init__(self, gdf: gpd.GeoDataFrame):
        self.quads_gdf = gdf

    def __call__(self, msh_t: jigsaw_msh_t) -> jigsaw_msh_t:
        # properly, we should make a jigsaw_msh_t a Copy type
        # currently, if the input msh_t doesn't have any of the properties, it will crash
        # but then again, without this properties there would be nothing to do.
        omsh_t = jigsaw_msh_t()
        omsh_t.mshID = msh_t.mshID
        omsh_t.ndims = msh_t.ndims
        omsh_t.vert2 = msh_t.vert2.copy()
        omsh_t.tria3 = msh_t.tria3.copy()
        omsh_t.crs = msh_t.crs
        self.add_quads_to_msh_t(omsh_t)
        return omsh_t


    def _get_new_tris_gdf_unclean(self, msh_t):
        def get_unique_diff(gdf1, gdf2):
            return gdf1.unary_union.difference(gdf2.unary_union)

        def get_multipolygon(poly):
            return MultiPolygon([poly]) if isinstance(poly, Polygon) else poly

        def get_triangles_from_holes(holes):
            return [
                triangle for hole in get_multipolygon(holes).geoms
                if is_convex(hole.exterior.coords)
                for triangle in ops.triangulate(hole)
            ]

        tri_gdf = gpd.GeoDataFrame(
            geometry=[Polygon(msh_t.vert2['coord'][idx, :]) for idx in msh_t.tria3['index']],
            crs=msh_t.crs
        ).to_crs(self.quads_poly_gdf.crs)

        intersecting_tri = gpd.sjoin(tri_gdf, self.quads_poly_gdf, how='inner', predicate='intersects')
        to_drop_indexes = intersecting_tri.index.unique()

        the_diff_uu = get_unique_diff(tri_gdf.iloc[to_drop_indexes], self.quads_poly_gdf)
        triangulated_diff = [tri for tri in ops.triangulate(the_diff_uu) if tri.within(the_diff_uu)]

        new_tri_gdf = pd.concat([
            tri_gdf.drop(index=to_drop_indexes),
            gpd.GeoDataFrame(geometry=triangulated_diff, crs=self.quads_poly_gdf.crs),
            self.quads_poly_gdf.to_crs(tri_gdf.crs),
        ], ignore_index=True)

        holes_to_pad = get_unique_diff(tri_gdf, new_tri_gdf).difference(self.quads_poly_gdf.unary_union)
        triangles_to_add = get_triangles_from_holes(holes_to_pad)
        new_tri_gdf = pd.concat([new_tri_gdf, gpd.GeoDataFrame(geometry=triangles_to_add, crs=self.quads_poly_gdf.crs)], ignore_index=True)

        holes_to_pad = get_unique_diff(tri_gdf, new_tri_gdf).difference(self.quads_poly_gdf.unary_union)
        linear_rings = [LinearRing(hole.exterior) for hole in get_multipolygon(holes_to_pad).geoms]
        new_triangles = [Polygon(geom) for lr in linear_rings for geom in manually_triangulate(lr)]

        return pd.concat([
            new_tri_gdf,
            gpd.GeoDataFrame(geometry=new_triangles, crs=new_tri_gdf.crs)
        ], ignore_index=True).to_crs(msh_t.crs)

    def _get_new_msh_t(self, msh_t):



        def cleanup_overlapping_edges(nodes, elements):
            vert2 = np.array(nodes)
            tria3 = np.array([element for element in elements if len(element) == 3])
            quad4 = np.array([element for element in elements if len(element) == 4])
            tria3_edges = np.hstack((tria3[:, [0, 1]], tria3[:, [1, 2]], tria3[:, [2, 0]])).reshape(-1, 2)
            quad4_edges = np.hstack((quad4[:, [0, 1]], quad4[:, [1, 2]], quad4[:, [2, 3]], quad4[:, [3, 0]])).reshape(-1, 2)
            all_edges = np.vstack([tria3_edges, quad4_edges])
            sorted_edges = np.sort(all_edges, axis=1)
            unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
            boundary_edges = unique_edges[counts == 1]
            boundary_rings = ops.linemerge(MultiLineString([LineString(x) for x in vert2[boundary_edges]]))
            if isinstance(boundary_rings, LineString):
                boundary_rings = MultiLineString(boundary_rings)

            this_gdf = gpd.GeoDataFrame(geometry=[Polygon(ls) for ls in boundary_rings.geoms], crs=msh_t.crs)
            this_gdf['is_valid'] = this_gdf.geometry.map(lambda geom: geom.is_valid)
            invalid_polys = this_gdf[~this_gdf['is_valid']]
            invalid_polys['reason'] = invalid_polys.geometry.map(lambda geom: explain_validity(geom))
            reasons = invalid_polys.reason.values
            import re
            coordinate_tuples = np.array([
                tuple(map(float, re.findall(r"[-+]?[0-9]*\.?[0-9]+", reason)))
                for reason in reasons
            ])
            tria3_edges_vert2 = np.array(vert2[tria3_edges])
            tria3_edges_vert2 = np.vstack([tria3_edges_vert2[:, :, 0], tria3_edges_vert2[:, :, 1]])

            # Reshape your arrays to have shape (n, 1, 2) and (1, m, 2)
            # tria3_edges_vert2 = tria3_edges_vert2.reshape(tria3_edges_vert2.shape[0], 1, -1)
            # coordinate_tuples = coordinate_tuples.reshape(1, coordinate_tuples.shape[0], -1)

            # Find common rows
            # common_rows = np.any(np.all(tria3_edges_vert2 == coordinate_tuples, axis=2), axis=1)

            # Get the indices of the common rows
            # common_row_indices = np.where(common_rows)[0]

            # Creating views for structured arrays
            a = tria3_edges_vert2
            b = coordinate_tuples
            a_rows = a.view([('', a.dtype)] * a.shape[1])
            b_rows = b.view([('', b.dtype)] * b.shape[1])

            # Finding the common rows
            common_rows = np.intersect1d(a_rows, b_rows)

            # Finding the indices of the common rows in the original arrays
            common_indices_in_a = np.where(np.isin(a_rows, common_rows))[0]
            common_indices_in_b = np.where(np.isin(b_rows, common_rows))[0]

            breakpoint()
            quad4_edges_vert2 = np.array(vert2[quad4_edges])

            # Reshape coordinate_tuples to allow broadcasting
            # coordinate_tuples_expanded = coordinate_tuples[:, np.newaxis, np.newaxis, :]

            # # Identify the rows of tria3_edges_vert2 where each coordinate in coordinate_tuples is found
            # matches = np.any(np.all(tria3_edges_vert2[:, :, np.newaxis, :] == coordinate_tuples_expanded, axis=3), axis=1)

            # # Find the indices of the matching rows
            # matching_indices = np.nonzero(matches)

            breakpoint()

            gpd.GeoDataFrame(geometry=[ls for ls in boundary_rings.geoms], crs=msh_t.crs).plot(
                    facecolor='none',
                    cmap='tab20',
                    ax=plt.gca(),
                    )
            plt.show(block=False)
            breakpoint()
            raise
            return nodes, elements
        nodes, elements = poly_gdf_to_elements(self._get_new_tris_gdf_unclean(msh_t))
        nodes, elements = cleanup_overlapping_edges(nodes, elements)
        vert2 = np.array([(coord, 0) for coord in nodes], dtype=jigsaw_msh_t.VERT2_t)
        tria3 = np.array([(element, 0) for element in elements if len(element) == 3], dtype=jigsaw_msh_t.TRIA3_t)
        quad4 = np.array([(element, 0) for element in elements if len(element) == 4], dtype=jigsaw_msh_t.QUAD4_t)
        crs = msh_t.crs
        msh_t = jigsaw_msh_t()
        msh_t.mshID = 'euclidean-mesh'
        msh_t.ndims = 2
        msh_t.crs = crs
        msh_t.vert2 = vert2
        msh_t.tria3 = tria3
        msh_t.quad4 = quad4
        return msh_t

    def add_quads_to_msh_t(self, msh_t):
        new_msh_t = self._get_new_msh_t(msh_t)
        msh_t.vert2 = new_msh_t.vert2
        msh_t.tria3 = new_msh_t.tria3
        msh_t.quad4 = new_msh_t.quad4
        utils.split_bad_quality_quads(msh_t)
        # utils.remove_flat_triangles(msh_t)
        # utils.cleanup_pinched_nodes_iter(msh_t)
        # utils.cleanup_isolates(msh_t)

    def _fix_triangles_outside_SCHISM_skewness_tolerance(self, msh_t):
        # SCHISM can tolerate sknewness<=60 (defined as (largest side)/(equivalent radius), where  eq. radius is sqrt(area/pi).

        def get_skewness(geometry):
            # Calculate the lengths of the sides
            sides = [Point(geometry.exterior.coords[i]).distance(Point(geometry.exterior.coords[i-1]))
                     for i in range(len(geometry.exterior.coords)-1)]
            # Find the largest side length
            largest_side_length = max(sides)
            equivalent_radius = np.sqrt(geometry.area/np.pi)
            if equivalent_radius == 0.:
                return float('inf')
            skewness = largest_side_length / equivalent_radius
            return skewness

        def get_tria_to_fix(msh_t):
            tria3_gdf = utils.get_msh_t_tria3_Polygon_gdf(msh_t)

            tria3_gdf['skewness'] = tria3_gdf.to_crs('epsg:6933').geometry.map(get_skewness)

            tria_with_high_skewness = tria3_gdf[tria3_gdf['skewness'] > 60.]

            # Perform spatial join to find all triangles that touch the ones with high skewness
            tria_to_fix = gpd.sjoin(
                tria3_gdf,
                tria_with_high_skewness,
                how='inner',  # Use inner join to get only the matching rows
                predicate='touches'
            )

            # Combine indices from both conditions and select rows from original DataFrame
            result = tria3_gdf.loc[
                tria_to_fix.index.union(tria_with_high_skewness.index)
            ]

            return result

        utils.remove_flat_triangles(msh_t)
        tria_to_fix = get_tria_to_fix(msh_t)

        self._remove_invalid_triangles_from_msh_t(msh_t, tria_to_fix)
        self._append_new_tris_gdf_to_msh_t(self._get_new_tris_gdf(tria_to_fix), msh_t)
        # remove any remaining tri with high skewness
        tria3_gdf = utils.get_msh_t_tria3_Polygon_gdf(msh_t)
        tria3_gdf['skewness'] = tria3_gdf.to_crs('epsg:6933').geometry.map(get_skewness)
        tria_with_high_skewness = tria3_gdf[tria3_gdf['skewness'] > 60.]
        self._remove_invalid_triangles_from_msh_t(msh_t, tria_with_high_skewness)
        self._remove_self_intersections(msh_t)
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
            pad_width=20,
            **kwargs
            ):
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
        quads_poly_gdf = self.quads_gdf.copy()  # This will include all original columns
        quads_poly_gdf['geometry'] = quads_poly_gdf['geometry'].map(Polygon)
        return quads_poly_gdf

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
    from mpi4py.futures import MPICommExecutor
    from functools import partial
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
            quads_gdf.drop(columns=['normal_vector']).to_file('/sciclone/pscr/jrcalzada/thesis/runs/tropicald-validations/hindcasts/Harvey2017/test.gpkg')
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
                    'normal_vector',
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
                    'normal_vector',
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
                'normal_vector': (float(normal_vector[0]), float(normal_vector[1])),
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
                    'normal_vector',
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
                    'normal_vector',
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
                'normal_vector': (float(normal_vector[0]), float(normal_vector[1])),
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
        # threshold_size=None,
        lower_threshold_size=None,
        upper_threshold_size=None,
        resample_distance=None,
        simplify_tolerance=None,
        interpolation_distance=None,
        min_area_to_length_ratio=0.1,
        min_area=np.finfo(np.float64).min,
        min_cross_section_node_count=4,
        max_quad_width=None,
        min_quad_width=None,
        previous: Quads = None,
        ) -> gpd.GeoDataFrame:

    if mp is None or mp.is_empty:
        return gpd.GeoDataFrame(
                columns=[
                    'quad_group_id',
                    'quad_id',
                    'quad_row_id',
                    'geometry',
                    'normal_vector',
                    ],
                crs=CRS.from_epsg(4326)
                )

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

    final_patches = [
            patch for patch in final_patches
            if patch.is_valid and not patch.is_empty
            and patch.area > min_area
            and (patch.length / patch.area) < min_area_to_length_ratio
        ]

    # verify
    # gpd.GeoDataFrame(geometry=final_patches, crs=local_crs).plot(ax=plt.gca(), facecolor='none')
    # plt.show(block=True)
    # raise

    job_args = []
    for patch_id, this_patch in enumerate(final_patches):
        job_args.append((this_patch, {'interpolation_distance': interpolation_distance}))
    if resample_distance is not None:
        interpolation_distance = interpolation_distance or 0.5*resample_distance
    from time import time
    print('launching pool to get centerlines', flush=True)
    start = time()
    with Pool(cpu_count()) as pool:
        centerlines: List[MultiLineString] = pool.starmap(get_centerlines, job_args)
    # centerlines: List[MultiLineString] = list(map(lambda args: get_centerlines(*args), job_args))
    print(f"computing centerlines took: {time()-start}", flush=True)
    # expand centerlines into a flat list of LineString
    centerlines: List[LineString] = [line for multilinestring in centerlines for line in multilinestring.geoms]

    if upper_threshold_size is not None:
        final_patches = ops.unary_union(final_patches)
        buffered_geom = final_patches.buffer(-upper_threshold_size).buffer(upper_threshold_size)
        if not buffered_geom.is_empty:
            final_patches = final_patches.difference(
                gpd.GeoDataFrame([{'geometry': buffered_geom}], crs=local_crs).unary_union
                    )
        if isinstance(final_patches, MultiPolygon):
            final_patches = [polygon for polygon in final_patches.geoms]
        elif isinstance(final_patches, Polygon):
            final_patches = [final_patches]

    if lower_threshold_size is not None:
        final_patches = ops.unary_union(final_patches)
        buffered_geom = final_patches.buffer(-lower_threshold_size).buffer(lower_threshold_size)
        if not buffered_geom.is_empty:
            final_patches = final_patches.difference(
                gpd.GeoDataFrame([{'geometry': buffered_geom}], crs=local_crs).unary_union
                    )
        if isinstance(final_patches, MultiPolygon):
            final_patches = [polygon for polygon in final_patches.geoms]
        elif isinstance(final_patches, Polygon):
            final_patches = [final_patches]

    if lower_threshold_size or upper_threshold_size:
        with Pool(cpu_count()) as pool:
            is_within = pool.map(MultiPolygon(final_patches).contains, centerlines)
        pool.join()
        centerlines = [line for line, result in zip(centerlines, is_within) if result is True]
    # verify
    # gpd.GeoDataFrame(geometry=final_patches, crs=local_crs).plot(ax=plt.gca(), facecolor='none')
    # gpd.GeoDataFrame(geometry=centerlines, crs=local_crs).plot(cmap='tab20', ax=plt.gca())
    # plt.show(block=True)
    # raise
    centerlines = filter_linestrings(
            centerlines,
            min_branch_length=min_branch_length or 3.*max_quad_length,
            )
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
                    'normal_vector',
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

    print('launching pool to get quad groups', flush=True)
    start = time()
    with Pool(cpu_count()) as pool:
        quads_gdf = pool.starmap(get_quad_group_data, job_args)
    pool.join()
    # quads_gdf = list(map(lambda args: get_quad_group_data(*args), job_args))
    quads_gdf = [item for sublist in quads_gdf for item in sublist if len(item) > 0]
    print(f"computing quad_groups took: {time()-start}", flush=True)
    if len(quads_gdf) == 0:
        return gpd.GeoDataFrame(
                columns=[
                    'quad_group_id',
                    'quad_id',
                    'quad_row_id',
                    'geometry',
                    'normal_vector',
                ],
                crs=CRS.from_epsg(4326)
                )
    quads_gdf = gpd.GeoDataFrame(quads_gdf, crs=local_azimuthal_projection)
    quads_gdf = quads_gdf[~quads_gdf.geometry.is_empty]
    quads_gdf = quads_gdf[quads_gdf.geometry.is_valid]

    # verify
    # import contextily as cx
    # quads_gdf.plot(ax=plt.gca())
    # cx.add_basemap(plt.gca(), source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", crs=quads_gdf.crs)
    # plt.show(block=False)
    # breakpoint()
    # raise


    cleanup_quads_gdf_mut(quads_gdf)
    if previous is not None:
        previous_gdf = previous.quads_gdf.copy()
        quads_gdf['quad_group_id'] += previous_gdf['quad_group_id'].max() + 1
        quads_gdf = pd.concat([quads_gdf, previous_gdf.to_crs(quads_gdf.crs)], ignore_index=True)
        cleanup_quads_gdf_mut(quads_gdf)

    # verify
    # import contextily as cx
    # quads_gdf.plot(ax=plt.gca())
    # cx.add_basemap(plt.gca(), source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", crs=quads_gdf.crs)
    # plt.show(block=False)
    # breakpoint()
    # raise

    return quads_gdf.to_crs(CRS.from_epsg(4326))




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


def cleanup_quads_gdf_mut(quads_gdf):
    cleanup_overlapping_pairs_mut(quads_gdf)
    cleanup_touches_with_eps_tolerance_mut(quads_gdf)


def test_quadgen_for_Harlem_River():
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
    #                         # upper_threshold_size=500.,
    #                         # threshold_size=500.,
    #                         # min_quad_width=10.,
    #                         )
    # quads = Quads.from_raster(
    #                         raster,
    #                         # min_quad_length=10.,
    #                         max_quad_length=500.,
    #                         resample_distance=100.,
    #                         zmin=0.,
    #                         zmax=10.,
    #                         max_quad_width=500.,
    #                         # min_quad_width=10.,
    #                         previous=quads,
    #                         )
    # pickle.dump(quads, open("the_quads.pkl", "wb"))
    quads = pickle.load(open("the_quads.pkl", "rb"))
    # verification
    # quads.plot(ax=plt.gca(), facecolor='none')
    # plt.show(block=True)
    # raise
    # raster.resampling_factor = 0.2
    # geom = Geom(
    #         raster,
    #         zmax=10.,
    #         )
    # hfun = Hfun(
    #         raster,
    #         nprocs=cpu_count(),
    #         verbosity=1.,
    #         )
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
    # driver = JigsawDriver(
    #         geom=geom,
    #         hfun=hfun,
    #         verbosity=1,
    #         # sieve_area=True,
    #         # finalize=False,
    #         )
    # driver.opts.geom_feat = True
    # old_msh_t = driver.msh_t()
    # pickle.dump(old_msh_t, open("the_old_msh_t.pkl", "wb"))
    old_msh_t = pickle.load(open("the_old_msh_t.pkl", "rb"))
    # raise NotImplementedError("ready")
    # new_msh_t = old_msh_t
    new_msh_t = quads(old_msh_t)

    # utils.split_quad4_to_tria3(new_msh_t)

    from geomesh.cli.mpi.hgrid_build import interpolate_raster_to_mesh
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
    the_mesh.write('test_output_no_bnd.grd', overwrite=True)
    the_mesh.make_plot(ax=plt.gca(), elements=True)
    plt.show(block=True)
    # pickle.dump(the_mesh, open("the_quad_mesh.pkl", "wb"))
    
    # raise NotImplementedError("Ready for auto_bndgen")
    # exit()
    # import pickle
    # the_mesh = pickle.load(open("the_quad_mesh.pkl", "rb"))
    # the_mesh.make_plot(ax=plt.gca(), elements=True)
    # plt.show(block=True)
    # raise
    the_mesh.boundaries.auto_generate(
            min_open_bound_length=10000.
            )
    the_mesh.boundaries.open.plot(ax=plt.gca(), color='b')
    the_mesh.boundaries.land.plot(ax=plt.gca(), color='g')
    the_mesh.boundaries.interior.plot(ax=plt.gca(), color='r')
    the_mesh.triplot(ax=plt.gca())
    the_mesh.quadplot(ax=plt.gca())
    plt.show(block=False)
    breakpoint()
    raise
    the_mesh.write('test_with_schism/test_output.grd', overwrite=True)
    the_mesh.make_plot(ax=plt.gca(), elements=True)
    logger.debug('begin making mesh triplot')
    plt.title(f'node count: {len(new_msh_t.vert2["coord"])}')
    plt.gca().axis('scaled')
    plt.show(block=True)
