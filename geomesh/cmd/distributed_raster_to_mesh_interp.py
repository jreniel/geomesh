import argparse
from pathlib import Path
import pickle

import numpy as np
from scipy.interpolate import RectBivariateSpline
from pyproj import Transformer

from geomesh.cmd.raster_opts import add_raster_args
from geomesh.cmd.build import RasterToMeshInterpResult

# class RasterToMeshInterpResult:
#     raster_path: Path
#     raster_opts: dict
#     pkl_mesh_path: Path
#     values: np.ndarray
#     indexes: np.ndarray
    

def interpolate_raster_to_mesh(mesh, raster):
    coords = np.array(mesh.coord)
    idxs = []
    values = []
    for window in raster.iter_windows():
        if not raster.crs.equals(mesh.crs):
            transformer = Transformer.from_crs(mesh.crs, raster.crs, always_xy=True)
            coords[:, 0], coords[:, 1] = transformer.transform(
                coords[:, 0], coords[:, 1]
            )
        xi = raster.get_x(window)
        yi = raster.get_y(window)
        zi = raster.get_values(window=window)
        f = RectBivariateSpline(
            xi,
            np.flip(yi),
            np.flipud(zi).T,
            kx=3,
            ky=3,
            s=0,
            # bbox=[min(x), max(x), min(y), max(y)]  # ??
        )
        _idxs = np.where(
            np.logical_and(
                np.logical_and(np.min(xi) <= coords[:, 0], np.max(xi) >= coords[:, 0]),
                np.logical_and(np.min(yi) <= coords[:, 1], np.max(yi) >= coords[:, 1]),
            )
        )[0]
        _values = f.ev(coords[_idxs, 0], coords[_idxs, 1])

        idxs.append(_idxs)
        values.append(_values)

    return (np.hstack(idxs), np.hstack(values))


def parse_args():
    parser = argparse.ArgumentParser()
    add_raster_args(parser)
    parser.add_argument('pkl_mesh_path', type=lambda x: Path(x))
    parser.add_argument('--output-filename', '-o', required=True, type=lambda x: Path(x))
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.pkl_mesh_path, 'rb') as f:
        mesh = pickle.load(f)
    # TODO: Memory usage can be saved if the indexing is moved out here and the mesh reference is dropped (meaning to subsample the mesh)
    indexes, values = interpolate_raster_to_mesh(mesh, args.raster)
    result = RasterToMeshInterpResult()
    result.raster_path = args.raster.path
    result.raster_opts = args.raster_opts
    result.pkl_mesh_path = args.pkl_mesh_path
    result.values = values
    result.indexes = indexes
    with open(args.output_filename, 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    main()