import argparse
import pickle
from multiprocessing import Pool, cpu_count
import logging
from pathlib import Path
from turtle import distance

from jigsawpy import savemsh, jigsaw_msh_t
import numpy as np
from pyproj import CRS, Transformer

from geomesh.mesh import Mesh
from geomesh.hfun.mesh import MeshHfun

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

logger = logging.getLogger(__name__)

wgs84 = CRS.from_user_input("+proj=longlat +datum=WGS84 +no_defs")

def get_node_size(origin, other_nodes, crs):
        
    if crs.is_geographic:
        # print('node transform')
        aeqd = CRS.from_user_input(
            "+proj=aeqd +R=6371000 +units=m " f"+lat_0={origin[1]} +lon_0={origin[0]}"
        )
        wgs84_to_aeqd = Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
        # aeqd_to_wgs84 = Transformer.from_crs(aeqd, wgs84, always_xy=True).transform
        # center = Point(float(lon), float(lat))
        # point_transformed = ops.transform(wgs84_to_aeqd, center)
        # return ops.transform(aeqd_to_wgs84, point_transformed.buffer(radius))
        # origin = wgs84_to_aeqd(*origin)
        # for i, other_node in enumerate(other_nodes):
        #     other_nodes[i] = wgs84_to_aeqd(*other_node)
        origin = wgs84_to_aeqd(*origin)
        other_nodes = wgs84_to_aeqd(other_nodes[:, 0], other_nodes[:, 1])
       
    mean = np.nan
    for other_node in other_nodes:
        distance = np.sqrt((origin[0]-other_node[0])**2 + (origin[1]-other_node[1])**2)
        mean = np.nanmean([mean, distance])
    return mean


def parse_args():
    parser = argparse.ArgumentParser()
    # add_raster_args(parser)
    parser.add_argument('mesh', help='URI (path or url).')      
    # parser.add_argument('--raster-opts', help='JSON raster config.', action=RasterOptsAction)
    parser.add_argument('--hmin', type=float)
    parser.add_argument('--hmax', type=float)
    parser.add_argument('--nprocs', type=int, default=cpu_count())
    class QuadsAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string):
            # data = json.loads(values)
            # if not isinstance(data, dict):
            #     raise ValueError('--quads must be a json dictionary.') #  containing at least a `level` key.')
            # quads = data['quads']
            # if isinstance(level, list):
            #     level = list(map(float, level))
            # else:
            #     level = float(level)
            # data.update({'level': level})
            # if 'expansion_rate' in data:
            #     data.update({'expansion_rate': float(data['expansion_rate'])})
            # if 'target_size' in data:
            #     data.update({'target_size': float(data['target_size'])})
            # if 'nprocs' in data:
            #     data.update({'nprocs': float(data['nprocs'])})
            raise NotImplementedError('--quads')
            getattr(namespace, self.dest, values)
    parser.add_argument('--quads', action=QuadsAction, default='split', choices=['split']) # 'freeze'
    # parser.add_argument('--chunk-size', '--chunk_size', type=int)
    parser.add_argument('--dst-crs', '--dst_crs', type=lambda x: CRS.from_user_input(x))

    # class ContourAction(argparse.Action):
    #     def __call__(self, parser, namespace, values, option_string):
    #         data = json.loads(values)
    #         if not isinstance(data, dict):
    #             raise ValueError('--contour must be a json dictionary containing at least a `level` key.')
    #         if not 'level' in data:
    #             raise ValueError('--contour must contain `level` key.')
    #         level = data['level']
    #         if isinstance(level, list):
    #             level = list(map(float, level))
    #         else:
    #             level = float(level)
    #         data.update({'level': level})
    #         if 'expansion_rate' in data:
    #             data.update({'expansion_rate': float(data['expansion_rate'])})
    #         if 'target_size' in data:
    #             data.update({'target_size': float(data['target_size'])})
    #         if 'nprocs' in data:
    #             data.update({'nprocs': float(data['nprocs'])})
    #         getattr(namespace, self.dest).append(data)

    # parser.add_argument('--contour', action=ContourAction, default=[], help="JSON formatted configuration")
    # parser.add_argument('--verbosity', choices=[0, 1, 2], default=0, type=int)
    # parser.add_argument('--log-level', choices=[0, 1, 2], default=0, type=int)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--crs', type=lambda x: CRS.from_user_input(x))
    parser.add_argument('--to-msh', '--to_msh', type=lambda x: Path(x))
    parser.add_argument('--to-pickle', '--to_pickle', '--to-pkl', '--pkl', type=lambda x: Path(x))
    # parser.add_argument('--to-feather', '--to_feather', type=lambda x: Path(x))
    return parser.parse_args()


def main():
    args = parse_args()
    # logging.getLogger('geomesh').setLevel(logging.DEBUG)
    mesh = Mesh.open(args.mesh, crs=args.crs)
    msh_t = mesh.msh_t
    if args.quads == 'split':
        msh_t = jigsaw_msh_t()
        msh_t.mshID = "euclidean-mesh"
        msh_t.ndims = +2
        msh_t.vert2 = mesh.vert2
        msh_t.tria3 = np.array([(index, 0) for index in mesh.elements.triangulation().triangles], dtype=jigsaw_msh_t.TRIA3_t)
        msh_t.value = mesh.value
        msh_t.crs = mesh.crs
    # replace topobathy values with average side side
    # size = []
    job_args = []
    coords = np.array(mesh.nodes.coords())

    for i, (x, y) in enumerate(coords):
        job_args.append([
            (x, y),
            coords[mesh.nodes.get_indexes_around_index(i), :],
            mesh.crs,
        ])
    with Pool(processes=args.nprocs) as pool:
        sizes = pool.starmap(get_node_size, job_args)
    pool.join()
    msh_t.value = np.array(np.array(sizes).reshape(len(sizes),1), dtype=jigsaw_msh_t.REALS_t)
    # hfun = MeshHfun(
    #     msh_t,
        # hmin=args.hmin,
        # hmax=args.hmax,
        # verbosity=args.verbosity,
    # )
    # for contour_kwargs in args.contour:
    #     # I think we can use setdefault instead.
    #     contour_kwargs.update({'nprocs': contour_kwargs.get('nprocs', args.nprocs)})
    #     hfun.add_contour(**contour_kwargs)
    # hfun.tricontourf(levels=256, show=True)
    # exit()
    # msh_t = hfun.msh_t()
        
    if args.to_msh:
        savemsh(f'{args.to_msh.resolve()}', msh_t)
        
    if args.to_pickle:
        with open(args.to_pickle, 'wb') as fh:
            pickle.dump(msh_t, fh)
        # savemsh(f'{args.to_msh.resolve()}', msh_t)

    if args.show:
        Mesh(msh_t).tricontourf(show=True)
    #     for polygon in mp.geoms:
    #         plt.plot(*polygon.exterior.xy, color="k")
    #         for interior in polygon.interiors:
    #             plt.plot(*interior.xy, color="r")
    #     plt.show(block=True)

if __name__ == "__main__":
    main()
