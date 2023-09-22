#!/usr/bin/env python3

import CGAL
import numpy as np
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2

from geomesh import Geom, Hfun, Raster
import contextily as cx
import matplotlib.pyplot as plt
from appdirs import user_data_dir
from multiprocessing import cpu_count


def main():

    rootdir = user_data_dir('geomesh')
    raster = Raster(
            f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/ncei19_n41x00_w074x00_2015v1.tif',
            resampling_factor=0.2,
            )
    geom = Geom(
            raster,
            zmax=20.,
            )
    geom.generate_quads(
            resample_distance=100.,
            )
    geom.generate_quads(
            resample_distance=100.,
            zmin=0.,
            zmax=20.
            )

    geom_mp = geom.get_multipolygon(quad_holes=False)
    centroid = np.array(geom_mp.centroid.coords).flatten()
    local_azimuthal_projection = CRS.from_user_input(
        f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
        )
    geom_mp = gpd.GeoDataFrame([{'geometry': geom_mp}], crs=geom.crs).to_crs(local_azimuthal_projection).iloc[0].geometry
    geom_mp = geom._remove_interiors_for_quads(geom_mp, geom._quads_gdf)
    geom_mp = geom.resample_multipolygon(geom_mp, 100.)
    boundary_gdf = gpd.GeoDataFrame([{'geometry': geom_mp}], crs=local_azimuthal_projection)
    quads_gdf = geom._quads_gdf.to_crs(local_azimuthal_projection)

    import triangle

    def add_polygon(polygon, points, segments):
        # Add the points of the polygon to the points list and get their indices
        start_idx = len(points)
        polygon_points_idx = list(range(start_idx, start_idx + len(polygon.exterior.coords) - 1))
        points.extend(polygon.exterior.coords[:-1])

        # Create segments from the point indices and add them to the segments list
        polygon_segments = [(polygon_points_idx[i], polygon_points_idx[i+1]) for i in range(len(polygon_points_idx)-1)]
        segments.extend(polygon_segments)

    # Initialize lists for points and segments
    points = []
    segments = []

    # Extract the MultiPolygon
    multipolygon = boundary_gdf.geometry.values[0]

    # Loop through each Polygon in the MultiPolygon and add its points and segments
    for boundary in multipolygon.geoms:
        add_polygon(boundary, points, segments)

    # Loop through each row in quads_gdf and add its points and segments
    for _, row in quads_gdf.iterrows():
        add_polygon(row.geometry, points, segments)

    # Create a dict to hold the points and segments
    input_dict = {'vertices': np.array(points), 'segments': np.array(segments)}

    print('will triangulate')
    # Create a constrained Delaunay triangulation
    triangulation = triangle.triangulate(input_dict)

    print('will plot')
    # Plot the triangulation
    triangle.plot.plot(plt.axes(), **triangulation)
    plt.show(block=True)


    # Create a Constrained_Delaunay_triangulation_2 object
    # triangulation = Constrained_Delaunay_triangulation_2()

    # # Extract the MultiPolygon
    # multipolygon = boundary_gdf.geometry.values[0]

    # # Loop through each Polygon in the MultiPolygon

    # # Loop through each Polygon in the MultiPolygon
    # for boundary in multipolygon.geoms:
    #     # Convert the boundary to CGAL points
    #     boundary_cgal_points = [Point_2(*coord) for coord in list(boundary.exterior.coords)]
        
    #     # Add each edge of the polygon as a constraint
    #     for i in range(len(boundary_cgal_points) - 1):
    #         triangulation.insert_constraint(boundary_cgal_points[i], boundary_cgal_points[i + 1])

    # # Convert each quadrilateral to CGAL points and insert them as constraints into the triangulation
    # for _, row in quads_gdf.iterrows():
    #     quad = row.geometry
    #     quad_cgal_points = [Point_2(*coord) for coord in list(quad.exterior.coords)]
    #     for i in range(len(quad_cgal_points) - 1):
    #         triangulation.insert_constraint(quad_cgal_points[i], quad_cgal_points[i + 1])


    # # List to store the coordinates of the points (vertices)
    # vertices = []

    # # List to store the indices of the points forming each triangle
    # triangles = []

    # # Dictionary to map points to their indices
    # point_indices = {}

    # # Index counter
    # idx = 0

    # # Iterate over the vertices of the triangulation
    # for vertex in triangulation.finite_vertices():
    #     # Get the point corresponding to the vertex
    #     point = vertex.point()
        
    #     # Add the coordinates of the point to the vertices list
    #     vertices.append([point.x(), point.y()])
        
    #     # Add the point and its index to the point_indices dictionary
    #     point_indices[(point.x(), point.y())] = idx
        
    #     # Increment the index counter
    #     idx += 1

    # # Iterate over the finite faces (triangles) of the triangulation
    # for face in triangulation.finite_faces():
    #     # Get the vertices of the triangle
    #     triangle_points = [face.vertex(i).point() for i in range(3)]
        
    #     # Get the indices of the vertices
    #     triangle_indices = [point_indices[(point.x(), point.y())] for point in triangle_points]
        
    #     # Add the indices to the triangles list
    #     triangles.append(triangle_indices)

    # # Convert the vertices and triangles lists to numpy arrays
    # vertices = np.array(vertices)
    # triangles = np.array(triangles)


    # from matplotlib.tri import Triangulation

    # # Create a Triangulation object
    # triang = Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    # # Plot the triangulation
    # plt.figure()
    # plt.gca().set_aspect('equal')
    # plt.triplot(triang, 'go-', lw=1.0)
    # plt.show(block=True)
    # exit()
    # cx.add_basemap(
    #     plt.gca(),
    #     source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    #     crs=geom.crs,
    # )
    # plt.show(block=True)
    # from matplotlib.tri import Triangulation
    # tri = Triangulation()
    # exit()
    # # geom_msh_t = geom.msh_t()
    hfun = Hfun(
            raster,
            # geom=geom,
            nprocs=cpu_count(),
            hmin=100.,
            verbosity=1.
            )
    # print('adding contour')
    # # hfun.add_contour(0., target_size=100., expansion_rate=0.07)
    # # hfun.add_contour(20., target_size=100., expansion_rate=0.07)
    hfun.add_narrow_channel_anti_aliasing(
            zmax=0.,
            )
    hfun.add_narrow_channel_anti_aliasing(
            zmin=0.,
            zmax=20.
            )
    hfun.add_narrow_channel_anti_aliasing(
            zmin=20.,
            )
    hfun._add_quad_sizes(geom._quads_gdf)
    # # for xvals, yvals, hfun_vals in hfun:
    # #     # plt.contourf(xvals, yvals, hfun_vals, levels=256, cmap='jet')
    # #     plt.contour(xvals, yvals, hfun_vals, levels=256, cmap='jet')
    # #     plt.show(block=True)
    # # exit()
    # # msh_t = hfun.msh_t()
    # # hfun.add_narrow_channel_anti_aliasing(
    # #         resample_distance=100.,
    # #         zmin=0.,
    # #         zmax=20.
    # #         )
    # # hfun.add_quad_sizes(geom)
    # print('enter jigsaw driver')
    driver = JigsawDriver(geom, hfun, verbosity=1)
    msh_t = driver.msh_t()

    # # utils.triplot(msh_t)
    # # plt.gca().axis('scaled')
    # # plt.show(block=True)
    # # exit()

    # ----------- Start try to remove triangles and put the quads
    from geopandas.tools import sjoin
    polygons = [Polygon([msh_t.vert2['coord'][i] for i in tria]) for tria in msh_t.tria3['index']]
    gdf = gpd.GeoDataFrame(index=[i for i in range(len(msh_t.tria3['index']))], geometry=polygons, crs=msh_t.crs)
    geom.generate_quads(
            resample_distance=100.,
            )
    geom.generate_quads(
            resample_distance=100.,
            zmin=0.,
            zmax=20.,
            )
    quads_gdf = geom._quads_gdf
    intersecting_polygons = sjoin(gdf.to_crs(CRS.from_epsg(4326)), quads_gdf, how="inner", predicate="intersects")
    intersecting_indices = intersecting_polygons.index
    gdf = gdf.drop(intersecting_indices)

    # Convert intersecting_indices to numpy array
    intersecting_indices = np.array(intersecting_indices)

    # Find indices of intersecting polygons in the msh_t.tria3['index']
    tria3_mask = np.isin(np.arange(msh_t.tria3['index'].shape[0]), intersecting_indices)

    # Remove intersecting indices from tria3_index
    tria3_index = msh_t.tria3['index'][~tria3_mask, :]

    # Find unique indices and their inverse mapping
    used_indexes, inverse = np.unique(tria3_index, return_inverse=True)

    # Get all node indexes
    node_indexes = np.arange(msh_t.vert2['coord'].shape[0])

    # Find which nodes are still in use
    isin = np.isin(node_indexes, used_indexes)

    # Get indexes of nodes that are still in use
    vert2_idxs = np.where(isin)[0]
    import pandas as pd
    # Create a mapping from old indices to new indices
    df = pd.DataFrame(index=node_indexes).iloc[vert2_idxs].reset_index()
    mapping = {v: k for k, v in df.to_dict()['index'].items()}

    # Apply the mapping to reindex tria3_index
    tria3_index = np.array([mapping[x] for x in used_indexes])[inverse].reshape(tria3_index.shape)

    # Update msh_t.vert2
    msh_t.vert2 = msh_t.vert2.take(vert2_idxs, axis=0)

    # Update tria3_IDtag
    tria3_IDtag = msh_t.tria3['IDtag'][~tria3_mask]

    # If there are values, update msh_t.value
    if len(msh_t.value) > 0:
        msh_t.value = msh_t.value.take(vert2_idxs)

    # Finally, update msh_t.tria3
    msh_t.tria3 = np.array(
        [(tuple(indices), tria3_IDtag[i])
         for i, indices in enumerate(tria3_index)],
        dtype=jigsaw_msh_t.TRIA3_t)


    # from scipy.spatial import Delaunay
    # import pandas as pd
    # Get nodes from gdf and quads_gdf
    # boundary_nodes = np.array([list(shape.centroid.coords)[0] for shape in gdf.geometry])
    # interior_nodes = np.array([list(shape.centroid.coords)[0] for shape in quads_gdf.geometry])

    # Concatenate nodes
    # all_nodes = np.concatenate((boundary_nodes, interior_nodes))

    # # Perform triangulation
    # tri = Delaunay(all_nodes)
    # from matplotlib.tri import Triangulation
    # tri = Delaunay(all_nodes)


    # # Create a list of polygons
    # new_tria3 = [Polygon([all_nodes[i] for i in triangle]) for triangle in tri.simplices]

    # # Create a new GeoDataFrame
    # new_gdf = gpd.GeoDataFrame(index=pd.RangeIndex(start=0, stop=len(new_tria3), step=1), geometry=new_tria3)
    
    # from shapely.geometry import Point
    # from scipy.spatial import cKDTree
    # import pandas as pd

    # # Get vertices of removed triangles
    # removed_vertices = [Point(vert) for triangle in intersecting_polygons.geometry for vert in triangle.exterior.coords[:-1]]

    # # Get centroids of quads_gdf
    # interior_points = [Point(shape.centroid.coords[0]) for shape in quads_gdf.geometry]

    # # Create a KDTree for quick nearest neighbor search
    # tree = cKDTree([point.coords[0] for point in interior_points])

    # # Initialize list of new triangles
    # new_tria3 = []

    # # For each vertex of the removed triangles...
    # for vertex in removed_vertices:
    #     # Get two nearest points from quads_gdf
    #     dists, indices = tree.query(vertex.coords[0], k=2)

    #     # Create two new triangles for each vertex
    #     for index in indices:
    #         # Get nearest point as a Shapely Point
    #         nearest_point = interior_points[index]

    #         # Create new triangle
    #         new_tria3.append(Polygon([vertex.coords[0], nearest_point.coords[0], interior_points[(index+1)%len(interior_points)].coords[0]]))

    # # Add new triangles to gdf
    # gdf = pd.concat([gdf, gpd.GeoDataFrame(geometry=new_tria3)], ignore_index=True)


    # gdf.plot(ax=plt.gca(), facecolor='none', linewidth=0.5)
    # # quads_gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='red')

    # plt.show(block=True)

    # utils.triplot(msh_t)
    # plt.gca().axis('scaled')
    # plt.show(block=True)
    # exit()
    # ----------- END try to remove triangles and put the quads
    utils.triplot(msh_t)
    plt.gca().axis('scaled')
    plt.show(block=True)

    from geomesh.cmd.mpi.hgrid_build import interpolate_raster_to_mesh
    raster.resampling_factor = None
    msh_t.value = np.full((msh_t.vert2['coord'].shape[0], 1), np.nan)
    idxs, values = interpolate_raster_to_mesh(
            msh_t,
            raster,
            use_aa=True,
            threshold_size=100.
            )
    msh_t.value[idxs] = values
    if np.all(np.isnan(msh_t.value)):
        raise ValueError('All values are NaN!')
    from scipy.interpolate import griddata
    if np.any(np.isnan(msh_t.value)):
        value = msh_t.value.flatten()
        non_nan_idxs = np.where(~np.isnan(value))[0]
        nan_idxs = np.where(np.isnan(value))[0]
        value[nan_idxs] = griddata(
                msh_t.vert2['coord'][non_nan_idxs, :],
                value[non_nan_idxs],
                msh_t.vert2['coord'][nan_idxs, :],
                method='nearest'
                )
        msh_t.value = value.reshape((value.size, 1)).astype(jigsaw_msh_t.REALS_t)
    from geomesh import Mesh
    ax = Mesh(msh_t).make_plot()
    utils.triplot(msh_t, axes=ax)
    # quads_gdf.plot(ax=ax, facecolor='none', edgecolor='red')
    plt.title(f'node count: {len(msh_t.vert2["coord"])}')
    plt.gca().axis('scaled')
    plt.show(block=True)


if __name__ == "__main__":
    main()
