import argparse
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS

from geomesh.raster import Raster
from geomesh.geom import RasterGeom
from geomesh.geom.shapely import MultiPolygonGeom

from geomesh.cmd.raster_opts import add_raster_args

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

def parse_args():
    parser = argparse.ArgumentParser()
    add_raster_args(parser)
    parser.add_argument('--zmin', type=float)
    parser.add_argument('--zmax', type=float)
    parser.add_argument('--no-unary-union', action='store_true', default=False)
    parser.add_argument('--nprocs', type=int)
    # parser.add_argument('--chunk-size', '--chunk_size', type=int)
    parser.add_argument('--dst-crs', '--dst_crs', type=CRS.from_user_input)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--to-file', '--to_file', type=Path)
    parser.add_argument('--to-feather', '--to_feather', type=Path)
    # parser.add_argument('--to-pickle',  type=lambda x: Path(x))
    return parser.parse_args()


def main():
    args = parse_args()
    geom = RasterGeom(
        raster=args.raster,
        zmin=args.zmin,
        zmax=args.zmax
    )
    if args.dst_crs is not None:
        dst_crs = args.dst_crs
    else:
        dst_crs = geom.crs
    mp = geom.get_multipolygon(
            zmin=args.zmin,
            zmax=args.zmax,
            dst_crs=dst_crs,
            nprocs=args.nprocs,
            unary_union=False if args.no_unary_union else True
        )
    if args.no_unary_union:
        gdf = gpd.GeoDataFrame([{'geometry': geometry} for geometry in mp], crs=dst_crs)
    else:
        gdf = gpd.GeoDataFrame([{'geometry': mp}], crs=dst_crs)

    if args.to_file:
        gdf.to_file(args.to_file)
    if args.to_feather:
        gdf.to_feather(args.to_feather)
        
    if args.show:
        if args.no_unary_union:
            gdf.plot(facecolor='none')
        else:
            for polygon in mp.geoms:
                plt.plot(*polygon.exterior.xy, color="k")
                for interior in polygon.interiors:
                    plt.plot(*interior.xy, color="r")
        plt.show(block=True)

if __name__ == "__main__":
    main()
