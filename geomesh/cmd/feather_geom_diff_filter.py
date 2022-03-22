import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import geopandas as gpd
import pandas as pd

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')


def parse_args():
    parser = argparse.ArgumentParser(description='Performs b.difference(a) in-place.')
    parser.add_argument('a', type=lambda x: Path(x))
    parser.add_argument('b', nargs='*')
    parser.add_argument('--nprocs', default=cpu_count(), type=int)
    return parser.parse_args()

def get_difference(a, b):
    return b.difference(a)

def main():
    args = parse_args()
    filepaths = [args.a]
    filepaths.extend([Path(path) for path in args.b])
    # print(filepaths)
    gdfs = []
    for fpath in filepaths:
        gdfs.append(gpd.read_feather(fpath))
    gdf = pd.concat(gdfs)
    geoms = [geom.geometry for geom in gdf.itertuples()]
    a = geoms.pop(0)
    job_args = []
    for b in geoms:
        job_args.append([a, b])

    with Pool(processes=args.nprocs) as pool:
        new_geoms = pool.starmap(
            get_difference,
            job_args,
        )
    pool.join()
    # In-place replace
    for i, outpath in enumerate(args.b):
        gpd.GeoDataFrame([{'geometry': new_geoms[i]}], crs=gdf.crs).to_feather(outpath)

if __name__ == "__main__":
    main()