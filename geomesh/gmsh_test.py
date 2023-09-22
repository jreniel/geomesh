#!/usr/bin/env python3

import pygmsh
# from shapely.geometry import MultiPolygon


# from geomesh.geom.raster import resample_polygon
# import numpy as np
# from pyproj import CRS


# def add_multipolygon_to_gmsh_model(model_name, multipolygon, src_crs):
#     gmsh.initialize()
#     gmsh.model.add(model_name)

#     for i, polygon in enumerate(multipolygon.geoms):
#         print(f"begin adding polygon {i+1}")
#         centroid = np.array(gdf.centroid.coords).flatten()
#         local_azimuthal_projection = CRS.from_user_input(
#             f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
#             )
        
#         polygon = gpd.GeoDataFrame(geometry=[polygon]).to_crs(local_azimuthal_projection).iloc[0].geometry

#         polygon = resample_polygon(polygon, 100.)

#         exterior_coords = list(polygon.exterior.coords)
#         interior_coords = [list(ring.coords) for ring in polygon.interiors]

#         print('add exteriors')
#         # Add exterior
#         exterior_points = [gmsh.model.geo.addPoint(*coord, 0) for coord in exterior_coords]
#         exterior_lines = [gmsh.model.geo.addLine(exterior_points[i - 1], exterior_points[i]) for i in range(1, len(exterior_points))]
#         exterior_curve_loop = gmsh.model.geo.addCurveLoop(exterior_lines)

#         print('add interiors')
#         # Add interiors
#         interior_curve_loops = []
#         for coords in interior_coords:
#             interior_points = [gmsh.model.geo.addPoint(*coord, 0) for coord in coords]
#             interior_lines = [gmsh.model.geo.addLine(interior_points[i - 1], interior_points[i]) for i in range(1, len(interior_points))]
#             interior_curve_loop = gmsh.model.geo.addCurveLoop(interior_lines)
#             interior_curve_loops.append(interior_curve_loop)

#         # Create surface
#         gmsh.model.geo.addPlaneSurface([exterior_curve_loop] + interior_curve_loops, i)
#     print('calling synchronize')
#     gmsh.model.geo.synchronize()
#     print('calling mesh generate')
#     gmsh.model.mesh.generate(2)

#     node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
#     element_types, element_tags, element_connectivity = gmsh.model.mesh.getElements()

#     # Cleanup
#     gmsh.finalize()
    
#     # Reshape the node coordinates and element connectivity to form a more convenient data structure
#     node_coords = node_coords.reshape((-1, 3))
#     element_connectivity = element_connectivity[0].reshape((-1, 3))
    
#     return node_tags, node_coords, element_connectivity

# Test the function
# multipolygon = MultiPolygon([...])  # Replace this with your MultiPolygon

import pickle
import geopandas as gpd
import matplotlib.pyplot as plt
thepath = "/sciclone/home/jrcalzada/pscr/tropicald-validations/hindcasts/Harvey2017/.cache/geom_build/32574492c09c89385263821e70cc65f5a0de0fe8c4730b6a134d86ec373102a6.pkl"
obj = pickle.load(open(thepath, "rb"))
gdf = gpd.GeoDataFrame(geometry=[obj.get_multipolygon()], crs=obj.crs)
with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(
        [
            [0.0, 0.0],
            [1.0, -0.2],
            [1.1, 1.2],
            [0.1, 0.7],
        ],
        mesh_size=0.1,
    )
    mesh = geom.generate_mesh()

print('enter gmsh')
# node_tags, node_coords, element_connectivity = add_multipolygon_to_gmsh_model("multipolygon_model", gdf.iloc[0].geometry, gdf.crs)
# print(f"{node_tags=}")
# print(f"{node_coords=}")
# print(f"{element_connectivity=}")

