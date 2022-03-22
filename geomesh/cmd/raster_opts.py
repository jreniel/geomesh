import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import box

from geomesh.raster import Raster

class RasterOptsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        raster_opts = json.loads(values)
        if raster_opts is None:
            setattr(namespace, self.dest, {})
            return
        if 'clip' in raster_opts:
            if 'bbox' in raster_opts['clip']:
                raster = self.get_raster(raster_opts.get('crs'))
                raster_opts['clip'] = box(
                    raster_opts['clip']['bbox'].get('xmin', np.min(raster.x)),
                    raster_opts['clip']['bbox'].get('ymin', np.min(raster.y)),
                    raster_opts['clip']['bbox'].get('xmax', np.max(raster.x)),
                    raster_opts['clip']['bbox'].get('ymax', np.max(raster.y))
                )
        setattr(namespace, self.dest, raster_opts)
        
    def get_raster(self, crs=None):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('raster', type=lambda x: Path(x))
        known_args, _ = parser.parse_known_args()
        return Raster(known_args.raster, crs=crs)
    
class RasterAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('raster', type=lambda x: Path(x))
        parser.add_argument('--raster-opts', '--raster_opts', action=RasterOptsAction, default={})
        args, _ = parser.parse_known_args()
        raster = Raster(
            args.raster,
            crs=args.raster_opts.get('crs'),
            chunk_size=args.raster_opts.get('chunk_size'),
            overlap=args.raster_opts.get('overlap', 2),
            )
        if 'clip' in args.raster_opts:
            raster.clip(args.raster_opts['clip'])
        if 'resample' in args.raster_opts:
            resample = args.raster_opts['resample']
            if not isinstance(resample, dict):
                raise ValueError('resample argument must be a dict with req key sacling_factor ')
            raster.resample(
                args.raster_opts['resample']['scaling_factor'],
                resampling_method=args.raster_opts['resample'].get('resampling_method')
            )
        setattr(namespace, self.dest, raster)


def add_raster_args(parser):
    parser.add_argument('raster', help='URI (path or url).', action=RasterAction)
    parser.add_argument('--raster-opts', '--raster_opts',
                        # action=RasterOptsAction
                        )
    
def append_cmd_opts(cmd, opts):
    raster_opts = {}
    if 'clip' in opts:
        raster_opts.update({'clip': opts['clip']})
        
    if 'chunk_size' in opts:
        raster_opts.update({'chunk_size': opts['chunk_size']})

    if 'overlap' in opts:
        raster_opts.update({'overlap': opts['overlap']})
        
    if 'gaussian_filter' in opts:
        raster_opts.update({'gaussian_filter': opts['gaussian_filter']})

    if 'resample' in opts:
        raster_opts.update({'resample': opts['resample']})

    if 'fill_nodata' in opts:
        raster_opts.update({'fill_nodata': opts['fill_nodata']})
        
    # TODO: There are additional more raster_opts to consider !!!!!!!!!!
    # fill_nodata
    # resample
    if len(raster_opts) > 0:
        cmd.append(f"--raster-opts='{json.dumps(raster_opts)}'")
