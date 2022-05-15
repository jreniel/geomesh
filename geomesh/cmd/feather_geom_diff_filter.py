#!/usr/bin/env python
import argparse
from genericpath import exists
from pathlib import Path
from multiprocessing import Pool
from typing import List

import geopandas as gpd
# import pandas as pd
import numpy as np
# import pyarrow
from shapely.geometry import box, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, split

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
# def check_intersects(this_geom, other_feather):
#     other_geom = gpd.read_feather(other_feather).iloc[0].geometry
#     if this_geom.intersection(box(*other_geom.bounds)).intersects(other_geom):
#         return other_geom

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('a', type=Path)
    parser.add_argument('b', nargs='*')
    parser.add_argument('--nprocs', type=int, default=1)
    parser.add_argument('--output-filename', '-o', type=Path, required=True)
    return parser.parse_args()

def get_total_bounds(feather_path):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        return gpd.read_feather(feather_path).total_bounds   # this gets caught and handled as an exception

def get_geom_difference(this, others):
    if len(others) == 0:
        return
    for other in others:
        this = this.difference(other)
    return this

def main():
    args = parse_args()
    this_gdf = gpd.read_feather(args.a)
    with Pool(args.nprocs) as pool:
        other_bboxes = pool.map(get_total_bounds, args.b)
    pool.join()
    other_bboxes = [box(*bbox) for bbox in other_bboxes if not np.isnan(bbox).all()]
    this_gdf = gpd.read_feather(args.a)
    if len(other_bboxes) > 0:
        r_index = gpd.GeoDataFrame([{'geometry': bbox} for bbox in other_bboxes]).sindex
        job_args = []
        for row in this_gdf.itertuples():
            if row.geometry is None:
                job_args.append([
                    row.geometry,
                    [],
                ])
            else:
                job_args.append([
                    row.geometry,
                    [other_bboxes[index] for index in r_index.intersection(row.geometry.bounds)]
                ])
        with Pool(args.nprocs) as pool:
            new_pieces = pool.starmap(get_geom_difference, job_args)
        pool.join()
        for i, new_geom in  enumerate(new_pieces):
            if new_geom is None:
                continue
            this_gdf.at[i, 'geometry'] = new_geom
    this_gdf.to_feather(args.output_filename)
    

if __name__ == '__main__':
    main()


