#!/usr/bin/env python
import argparse
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('feather_path', type=Path)
    parser.add_argument('outdir', type=Path)
    parser.add_argument('--nprocs', type=int, default=1)
    return parser.parse_args()

def flatten_filepath_index(this_row, outname):
    gpd.GeoDataFrame([{'geometry': this_row.geometry}]).to_feather(outname)

    

def main():
    args = parser_args()
    this_gdf = gpd.read_feather(args.feather_path) 
    outnames = [args.outdir / (f'{row.Index}_' + f'{args.feather_path.name}') for row in this_gdf.itertuples()]
    for i, outname in enumerate(outnames):
        flatten_filepath_index(this_gdf.iloc[i], outname)

if __name__ == "__main__":
    main()