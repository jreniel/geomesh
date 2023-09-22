from geomesh import Geom, Hfun, Raster
import contextily as cx
import matplotlib.pyplot as plt
from appdirs import user_data_dir
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np


from compas_cgal.triangulation import constrained_delaunay_triangulation

def quad_gdf_to_edge2_t(quad_gdf, split: bool = False):
    node_mappings = {}
    edges = []
    for row in quad_gdf.itertuples():
        quad = row.geometry
        for i in range(4):
            p0 = quad.exterior.coords[i]
            p1 = quad.exterior.coords[i+1]
            if p0 not in node_mappings:
                node_mappings[p0] = len(node_mappings)
            if p1 not in node_mappings:
                node_mappings[p1] = len(node_mappings)
            edges.append((node_mappings[p0], node_mappings[p1]))
        if split:
            p0 = quad.exterior.coords[0]
            p1 = quad.exterior.coords[2]
            edges.append((node_mappings[p0], node_mappings[p1]))
    nodes = np.array(list(node_mappings.keys()))
    edges = np.array(edges)
    return nodes, edges


def main():

    rootdir = user_data_dir('geomesh')
    raster = Raster(
            f'{rootdir}/raster_cache/chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/ncei19_n41x00_w074x00_2015v1.tif',
            resampling_factor=0.2,
            )
    geom = Geom(
            raster,
            zmax=10.,
            )
    geom.generate_quads(
            resample_distance=100.,
            )
    geom.generate_quads(
            resample_distance=100.,
            zmin=0.,
            zmax=10.,
            )
    # quad_nodes, quad_edges = quad_gdf_to_edge2_t(geom._quads_gdf, split=True)

    coord, tria3_index = constrained_delaunay_triangulation(boundary, points=None, holes=None, curves=None)

if __name__ == '__main__':
    main()

