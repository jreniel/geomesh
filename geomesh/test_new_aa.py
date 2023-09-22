from geomesh import Geom, Hfun, Raster, JigsawDriver
import contextily as cx
import matplotlib.pyplot as plt
from appdirs import user_data_dir
import geopandas as gpd


def test_new_aa_idea():
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
            ltc_ratio=5.,
            )
    geom.generate_quads(
            resample_distance=100.,
            zmin=0.,
            zmax=20.,
            ltc_ratio=5.,
            )

    geom_mp = geom.get_multipolygon(quad_holes=False)
    centroid = np.array(geom_mp.centroid.coords).flatten()
    local_azimuthal_projection = CRS.from_user_input(
        f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
        )
    geom_mp = gpd.GeoDataFrame([{'geometry': geom_mp}], crs=geom.crs).to_crs(local_azimuthal_projection).iloc[0].geometry
    geom_mp = geom._remove_interiors_for_quads(geom_mp, geom._quads_gdf)
    # geom_mp = geom.resample_multipolygon(geom_mp, 100.)
    geom_mp = gpd.GeoDataFrame([{'geometry': geom_mp}], crs=local_azimuthal_projection).to_crs(geom.crs).iloc[0].geometry
    gpd.GeoDataFrame(geometry=[geom_mp]).plot(facecolor='none', edgecolor='k', ax=plt.gca())
    geom._quads_gdf.plot(facecolor='none', edgecolor='r', ax=plt.gca())
    cx.add_basemap(
        plt.gca(),
        source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        crs=geom.crs,
    )
    plt.show(block=True)
    exit()
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
    # hfun.add_contour(0., target_size=100., expansion_rate=0.07)
    # hfun.add_contour(20., target_size=100., expansion_rate=0.07)
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
    # # for xvals, yvals, hfun_vals in hfun:
    # #     # plt.contourf(xvals, yvals, hfun_vals, levels=256, cmap='jet')
    # #     plt.contour(xvals, yvals, hfun_vals, levels=256, cmap='jet')
    # #     plt.show(block=True)
    # # exit()
    # msh_t = hfun.msh_t()
    # # hfun.add_narrow_channel_anti_aliasing(
    # #         resample_distance=100.,
    # #         zmin=0.,
    # #         zmax=20.
    # #         )
    # # hfun.add_quad_sizes(geom)
    # print('enter jigsaw driver')
    # driver = JigsawDriver(geom, hfun, verbosity=1)
    # msh_t = driver.msh_t()

    # # utils.triplot(msh_t)
    # # plt.gca().axis('scaled')
    # # plt.show(block=True)
    # # exit()

    # ----------- Start try to remove triangles and put the quads
    # from geopandas.tools import sjoin
    # polygons = [Polygon([msh_t.vert2['coord'][i] for i in tria]) for tria in msh_t.tria3['index']]
    # gdf = gpd.GeoDataFrame(index=[i for i in range(len(msh_t.tria3['index']))], geometry=polygons, crs=msh_t.crs)
    geom.generate_quads(
            zmax=0.,
            resample_distance=100.,
            ltc_ratio=5,
            )
    # geom.generate_quads(
    #         resample_distance=100.,
    #         zmin=0.,
    #         zmax=20.,
    #         )
    # quads_gdf = geom._quads_gdf



#     intersecting_polygons = sjoin(gdf.to_crs(CRS.from_epsg(4326)), quads_gdf, how="inner", predicate="intersects")
#     intersecting_indices = intersecting_polygons.index
#     gdf = gdf.drop(intersecting_indices)

#     # Convert intersecting_indices to numpy array
#     intersecting_indices = np.array(intersecting_indices)

#     # Find indices of intersecting polygons in the msh_t.tria3['index']
#     tria3_mask = np.isin(np.arange(msh_t.tria3['index'].shape[0]), intersecting_indices)

#     # Remove intersecting indices from tria3_index
#     tria3_index = msh_t.tria3['index'][~tria3_mask, :]

#     # Find unique indices and their inverse mapping
#     used_indexes, inverse = np.unique(tria3_index, return_inverse=True)

#     # Get all node indexes
#     node_indexes = np.arange(msh_t.vert2['coord'].shape[0])

#     # Find which nodes are still in use
#     isin = np.isin(node_indexes, used_indexes)

#     # Get indexes of nodes that are still in use
#     vert2_idxs = np.where(isin)[0]
#     # Create a mapping from old indices to new indices
#     df = pd.DataFrame(index=node_indexes).iloc[vert2_idxs].reset_index()
#     mapping = {v: k for k, v in df.to_dict()['index'].items()}

#     # Apply the mapping to reindex tria3_index
#     tria3_index = np.array([mapping[x] for x in used_indexes])[inverse].reshape(tria3_index.shape)

#     # Update msh_t.vert2
#     msh_t.vert2 = msh_t.vert2.take(vert2_idxs, axis=0)

#     # Update tria3_IDtag
#     # tria3_IDtag = msh_t.tria3['IDtag'][~tria3_mask]

#     # If there are values, update msh_t.value
#     if len(msh_t.value) > 0:
#         msh_t.value = msh_t.value.take(vert2_idxs)

#     msh_t.tria3 = np.array([], dtype=jigsaw_msh_t.TRIA3_t)

#     # convert the tria3 to edge2
#     def triangulation_to_edges(triangulation):
#         edges = set()
#         for triangle in triangulation:
#             # sort vertices in ascending order to ensure uniqueness of edges
#             edges.add(tuple(sorted([triangle[0], triangle[1]])))
#             edges.add(tuple(sorted([triangle[0], triangle[2]])))
#             edges.add(tuple(sorted([triangle[1], triangle[2]])))
#         return np.array(list(edges))

#     msh_t.edge2 = np.array([(edge, 0) for edge in triangulation_to_edges(tria3_index)], dtype=jigsaw_msh_t.EDGE2_t)


    

    # from shapely.geometry import LineString

    # # Assuming `edges` and `coord` (vertices coordinates) are defined already
    # lines = [LineString([msh_t.vert2['coord'][edge[0]], msh_t.vert2['coord'][edge[1]]]) for edge in msh_t.edge2['index']]
    # gdf = gpd.GeoDataFrame(geometry=lines, crs=msh_t.crs)

    # # Now we can plot the GeoDataFrame
    # gdf.plot()
    # plt.show(block=True)
    # exit()

    # # Finally, update msh_t.tria3
    # msh_t.tria3 = np.array(
    #     [(tuple(indices), tria3_IDtag[i])
    #      for i, indices in enumerate(tria3_index)],
    #     dtype=jigsaw_msh_t.TRIA3_t)

    # msh_t.edge2 = np.array([], dtype=jigsaw_msh_t.EDGE2_t)
    



    # # Get the original number of vertices in msh_t
    # num_original_vertices = msh_t.vert2.shape[0]

    # # Append quad_msh_t.vert2 to msh_t.vert2
    # msh_t.vert2 = np.append(msh_t.vert2, quad_msh_t.vert2, axis=0)

    # # Offset indices in quad_msh_t.edge2['index']
    # quad_msh_t.edge2['index'] += num_original_vertices

    # # Now you can assign msh_t.edge2 = quad_msh_t.edge2
    # msh_t.edge2 = quad_msh_t.edge2

    # hfun._add_quad_sizes(geom._quads_gdf)

    # geom_mp = geom.get_multipolygon(quad_holes=False)
    # centroid = np.array(geom_mp.centroid.coords).flatten()
    # local_azimuthal_projection = CRS.from_user_input(
    #     f"+proj=aeqd +R=6371000 +units=m +lat_0={centroid[1]} +lon_0={centroid[0]}"
    #     )
    # geom_mp = gpd.GeoDataFrame([{'geometry': geom_mp}], crs=geom.crs).to_crs(local_azimuthal_projection).iloc[0].geometry
    # geom_mp = geom._remove_interiors_for_quads(geom_mp, geom._quads_gdf)
    # geom_mp = geom.resample_multipolygon(geom_mp, 100.)
    # boundary_gdf = gpd.GeoDataFrame([{'geometry': geom_mp}], crs=local_azimuthal_projection)
    # quads_gdf = geom._quads_gdf.copy()
    # quad_msh_t = geom._get_initial_msh_t()

    # quad_nodes, quad_edges = quad_gdf_to_edge2_t(geom._quads_gdf, split=True)
    # quad_vert2 = np.array([(node, -1) for node in quad_nodes], dtype=jigsaw_msh_t.VERT2_t)
    # quad_edge2 = np.array([(edge, -1) for edge in quad_edges], dtype=jigsaw_msh_t.EDGE2_t)
    
    # init_msh_t = jigsaw_msh_t()
    # init_msh_t.vert2 = quad_vert2
    # init_msh_t.edge2 = quad_edge2
    # init_msh_t.crs = geom._quads_gdf.crs
    # msh_t.vert2 = np.append(msh_t.vert2, quad_vert2, axis=0)
    # quad_edge2['index'] += num_original_vertices
    # msh_t.edge2 = np.append(msh_t.edge2, quad_edge2, axis=0)
    # print(msh_t.edge2)
    init_msh_t = geom._get_initial_msh_t(quads_only=False)
    geom._quads_gdf = geom._quads_gdf.drop(geom._quads_gdf.index)

    driver = JigsawDriver(
            geom,
            hfun,
            # initial_mesh=msh_t,
            initial_mesh=init_msh_t,
            verbosity=1
            )
    # driver.opts.
    msh_t = driver.msh_t()

    #     init_msh_t.value = np.append(init_msh_t.value, window_mesh.value)
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

    from geomesh.cli.mpi.hgrid_build import interpolate_raster_to_mesh
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
    test_new_aa_idea()
    
