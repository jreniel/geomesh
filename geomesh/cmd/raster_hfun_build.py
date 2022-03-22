import argparse
import pickle
import json
import logging
from pathlib import Path

from jigsawpy import savemsh
from pyproj import CRS

from geomesh.raster import Raster
from geomesh.mesh import Mesh
from geomesh.hfun.raster import RasterHfun

from geomesh.cmd.raster_opts import add_raster_args

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    add_raster_args(parser)
    # parser.add_argument('raster', help='URI (path or url).')      
    # parser.add_argument('--raster-opts', help='JSON raster config.', action=RasterOptsAction)
    parser.add_argument('--hmin', type=float)
    parser.add_argument('--hmax', type=float)
    parser.add_argument('--nprocs', type=int)
    # parser.add_argument('--chunk-size', '--chunk_size', type=int)
    parser.add_argument('--dst-crs', '--dst_crs', type=lambda x: CRS.from_user_input(x))

    class ContourAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string):
            data = json.loads(values)
            if not isinstance(data, dict):
                raise ValueError('--contour must be a json dictionary containing at least a `level` key.')
            if not 'level' in data:
                raise ValueError('--contour must contain `level` key.')
            level = data['level']
            if isinstance(level, list):
                level = list(map(float, level))
            else:
                level = float(level)
            data.update({'level': level})
            if 'expansion_rate' in data:
                data.update({'expansion_rate': float(data['expansion_rate'])})
            if 'target_size' in data:
                data.update({'target_size': float(data['target_size'])})
            if 'nprocs' in data:
                data.update({'nprocs': float(data['nprocs'])})
            getattr(namespace, self.dest).append(data)

    parser.add_argument('--contour', action=ContourAction, default=[], help="JSON formatted configuration")
    parser.add_argument('--verbosity', choices=[0, 1, 2], default=0, type=int)
    # parser.add_argument('--log-level', choices=[0, 1, 2], default=0, type=int)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--to-msh', '--to_msh', type=lambda x: Path(x))
    parser.add_argument('--to-pickle', '--to_pickle', '--to-pkl', '--pkl', type=lambda x: Path(x))
    # parser.add_argument('--to-feather', '--to_feather', type=lambda x: Path(x))
    return parser.parse_args()


def main():
    args = parse_args()
    # logging.getLogger('geomesh').setLevel(logging.DEBUG)
    hfun = RasterHfun(
        raster=args.raster,
        hmin=args.hmin,
        hmax=args.hmax,
        verbosity=args.verbosity,
    )
    for contour_kwargs in args.contour:
        # I think we can use setdefault instead.
        contour_kwargs.update({'nprocs': contour_kwargs.get('nprocs', args.nprocs)})
        hfun.add_contour(**contour_kwargs)
        
    msh_t = hfun.msh_t()
    
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
