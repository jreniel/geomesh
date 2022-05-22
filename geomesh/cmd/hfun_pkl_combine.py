#! /usr/bin/env python
import argparse
import gc
import logging
from multiprocessing import Pool
import numpy as np
import pickle
from time import time

from shapely.geometry import box

from jigsawpy import jigsaw_msh_t

from geomesh.hfun.mesh import MeshHfun

logger = logging.getLogger(__name__)

def find_exact_elements_to_discard(bbox, possible_elements_to_discard):
    exact_elements_to_discard = set()
    for row in possible_elements_to_discard.itertuples():
        if row.geometry.within(bbox):
            exact_elements_to_discard.add(row.Index)
    return list(exact_elements_to_discard)


def combine_hfun_collection(hfun_collection, nprocs):
    areas = [mp.area for mp in[hfun.mesh.hull.multipolygon() for hfun in hfun_collection]]
    base_hfun = hfun_collection.pop(np.where(areas == np.max(areas))[0][0])
    base_hfun_crs = base_hfun.crs
    logger.info('Generating base hfun geodataframe...')
    start = time()
    elements = base_hfun.mesh.elements.geodataframe()

    logger.info(f'geodataframe generation took {time()-start}.')

    logger.info('Generating base hfun rtree index...')
    start = time()
    elements_r_index = elements.sindex
    logger.info(f'base_hfun rtree index gen took {time()-start}.')

    logger.info('Using r-tree indexing to find possible elements to discard...')
    start = time()
    possible_elements_to_discard = set()
    for hfun in hfun_collection:
        for index in list(elements_r_index.intersection(hfun.mesh.get_bbox(crs=base_hfun.crs).extents)):
            possible_elements_to_discard.add(index)
    del elements_r_index
    gc.collect()
    possible_elements_to_discard = elements.iloc[list(
        possible_elements_to_discard)]
    # print(possible_elements_to_discard)
    # exit()
    logger.info(
        f'Found possible elements to discard in {time()-start} seconds.')

    logger.info('Finding exact elements to discard...')
    start = time()
    with Pool(processes=nprocs) as pool:
        result = pool.starmap(
                find_exact_elements_to_discard,
                [(box(*hfun.mesh.get_bbox(crs=base_hfun.crs).extents),
                  possible_elements_to_discard)
                 for hfun in hfun_collection]
            )
    pool.join()
    # print([item for sublist in result for item in sublist])
    # exit()
    # breakpoint()
    to_keep = elements.loc[elements.index.difference([item for sublist in result for item in sublist])].index
    # print(to_keep)
    logger.info(
        f'Found exact elements to discard in {time()-start} seconds.')
    del elements
    gc.collect()

    base_hfun_msh_t = base_hfun.msh_t()
    final_tria = base_hfun_msh_t.tria3['index'][np.array(to_keep), :]
    
    del to_keep
    gc.collect()

    lookup_table = {index: i for i, index
                    in enumerate(sorted(np.unique(final_tria.flatten())))}
    hfun = jigsaw_msh_t()
    hfun.mshID = 'euclidean-mesh'
    hfun.ndims = 2

    coord = [base_hfun_msh_t.vert2['coord'][list(lookup_table.keys()), :]]
    index = [np.array([list(map(lambda x: lookup_table[x], element))
                      for element in final_tria])]
    # print(lookup_table)
    # print(index)
    # exit()

    value = [base_hfun_msh_t.value[list(lookup_table.keys())]]
    offset = coord[-1].shape[0]
    for _hfun in hfun_collection:
        _msh_t = _hfun.msh_t()
        index.append(_msh_t.tria3['index'] + offset)
        coord.append(_msh_t.vert2['coord'])
        value.append(_msh_t.value.flatten())
        offset += _msh_t.vert2['coord'].shape[0]
    hfun.vert2 = np.array([(coord, 0) for coord in np.vstack(coord)],
                          dtype=jigsaw_msh_t.VERT2_t)
    # breakpoint()
    hfun.tria3 = np.array([(index, 0) for index in np.vstack(index)],
                          dtype=jigsaw_msh_t.TRIA3_t)
    hfun.value = np.array(np.hstack(value).T, dtype=jigsaw_msh_t.REALS_t)
    hfun.crs = base_hfun_crs
    return hfun

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hfun_pkl_paths', nargs='*')
    parser.add_argument('--nprocs', '-np', type=int, required=True)
    parser.add_argument('--to-pickle', '-o', required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    hfun_collection = []
    for hfun_pkl_path in args.hfun_pkl_paths:
        with open(hfun_pkl_path, 'rb') as fh:
            hfun_collection.append(MeshHfun(pickle.load(fh)))
    hfun_msh_t = combine_hfun_collection(hfun_collection, args.nprocs)
        
    if args.to_pickle:
        with open(args.to_pickle, 'wb') as fh:
            pickle.dump(hfun_msh_t, fh)
    
    

if __name__ == '__main__':
    main()



# import gc
# import logging
# from multiprocessing import Pool, cpu_count
# import pathlib
# from time import time

# from jigsawpy import jigsaw_msh_t
# import numpy as np

# from geomesh import Mesh


# _logger = logging.getLogger(__name__)


# class Sms2dmHfunCollector:

#     def __init__(self):
#         collection = []
#         _logger.info('Reading 2dm hfun files...')
#         start = time()
#         for path in pathlib.Path('.').glob('*.2dm'):
#             collection.append(Mesh.open(path, crs='EPSG:4326'))
#         _logger.info(f'Reading 2dm hfun files took {time()-start}.')
#         self.collection = collection

#     def __iter__(self):
#         for hfun in self.collection:
#             yield hfun



# def get_hfun_composite(nproc):

#     hfun_collection = Sms2dmHfunCollector()

#     _logger.info('Figuring out which one is the base hfun...')
#     start = time()
#     areas = [mp.area for mp in
#              [mesh.hull.multipolygon() for mesh in hfun_collection]]
#     base_hfun = hfun_collection.collection.pop(
#         np.where(areas == np.max(areas))[0][0])
#     _logger.info(f'Found base hfun in {time()-start} seconds.')

#     _logger.info('Generating base hfun geodataframe...')
#     start = time()
#     elements = base_hfun.elements.geodataframe()
#     _logger.info(f'geodataframe generation took {time()-start}.')

#     _logger.info('Generating base hfun rtree index...')
#     start = time()
#     elements_r_index = elements.sindex
#     _logger.info(f'base_hfun rtree index gen took {time()-start}.')

#     _logger.info(
#         'Using r-tree indexing to find possible elements to discard...')
#     start = time()
#     possible_elements_to_discard = set()
#     for hfun in hfun_collection:
#         bounds = hfun.get_bbox(crs=base_hfun.crs).bounds
#         for index in list(elements_r_index.intersection(bounds)):
#             possible_elements_to_discard.add(index)
#     del elements_r_index
#     gc.collect()
#     possible_elements_to_discard = elements.iloc[list(
#         possible_elements_to_discard)]

#     _logger.info(
#         f'Found possible elements to discard in {time()-start} seconds.')

#     _logger.info('Finding exact elements to discard...')
#     start = time()
#     with Pool(processes=nproc) as pool:
#         result = pool.starmap(
#                 find_exact_elements_to_discard,
#                 [(hfun.get_bbox(crs=base_hfun.crs),
#                   possible_elements_to_discard)
#                  for hfun in hfun_collection]
#             )
#     to_keep = elements.loc[elements.index.difference(
#         [item for sublist in result for item in sublist])].index
#     _logger.info(
#         f'Found exact elements to discard in {time()-start} seconds.')
#     del elements
#     gc.collect()

#     final_tria = base_hfun.tria3['index'][to_keep, :]
#     del to_keep
#     gc.collect()

#     lookup_table = {index: i for i, index
#                     in enumerate(sorted(np.unique(final_tria.flatten())))}
#     hfun = jigsaw_msh_t()
#     hfun.mshID = 'euclidean-mesh'
#     hfun.ndims = 2

#     coord = [base_hfun.coord[list(lookup_table.keys()), :]]
#     index = [np.array([list(map(lambda x: lookup_table[x], element))
#                       for element in final_tria])]
#     value = [base_hfun.value[list(lookup_table.keys()), :]]
#     offset = coord[-1].shape[0]
#     for _hfun in hfun_collection:
#         index.append(_hfun.tria3['index'] + offset)
#         coord.append(_hfun.coord)
#         value.append(_hfun.value)
#         offset += _hfun.coord.shape[0]

#     hfun.vert2 = np.array([(coord, 0) for coord in np.vstack(coord)],
#                           dtype=jigsaw_msh_t.VERT2_t)
#     hfun.tria3 = np.array([(index, 0) for index in np.vstack(index)],
#                           dtype=jigsaw_msh_t.TRIA3_t)
#     hfun.value = np.array(np.vstack(value), dtype=jigsaw_msh_t.REALS_t)
#     return hfun
