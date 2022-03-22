import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS

# from geomesh.raster import Raster
from geomesh.geom import Geom
from geomesh.mesh import Mesh

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('feature')
    # parser.add_argument('--zmin', type=float)
    # parser.add_argument('--zmax', type=float)
    # parser.add_argument('--nprocs', type=int)
    parser.add_argument('--crs', '--feat-crs', type=lambda x: CRS.from_user_input(x))
    parser.add_argument('--dst-crs', '--dst_crs', type=lambda x: CRS.from_user_input(x))
    parser.add_argument('--show', action='store_true')
    # outputs = parser.add_subparsers(dest='outputs')
    # to_file = outputs.add_parser('to_file')
    parser.add_argument('--to-file', '--to_file', type=lambda x: Path(x))
    parser.add_argument('--to-feather', '--to_feather', type=lambda x: Path(x))
    # outputs.add_argument('--to-file')
    # outputs.add_argument('--to-file')
    # outputs.add_argument('--to-file')

    return parser.parse_args()


def get_feature_from_args(args):
    feature = None
    try:
        feature = Mesh.open(
            args.feature,
            crs=args.crs
        )
    except:
        pass

    if feature is None:
        raise TypeError(f'Could not open file: {args.feature}')
    
    return feature


def main():
    args = parse_args()

    geom = Geom(
        get_feature_from_args(args),
        # zmin=args.zmin,
        # zmax=args.zmax
    )
    mp = geom.get_multipolygon(
        # zmin=args.zmin,
        # zmax=args.zmax,
        # dst_crs=args.dst_crs,
        # nprocs=args.nprocs,
    )

    if args.to_file or args.to_feather:
        gdf = gpd.GeoDataFrame([{'geometry': mp}], crs=args.dst_crs)
        if args.to_file:
            gdf.to_file(args.to_file)
        if args.to_feather:
            gdf.to_feather(args.to_feather)

    if args.show:
        for polygon in mp.geoms:
            plt.plot(*polygon.exterior.xy, color="k")
            for interior in polygon.interiors:
                plt.plot(*interior.xy, color="r")
        plt.show(block=True)


    
        


if __name__ == "__main__":
    main()

# class GeomCli:

#     args: argparse.Namespace

#     def __init__(self, args: argparse.Namespace):
#         self.args = args

#     def main(self):
#         if self.args.geom_actions == 'build':
#             GeomBuildCli(self.args).main()

