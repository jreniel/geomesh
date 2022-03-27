from typing import Union

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

class Boundaries:
    def __init__(self, hgrid, boundaries: Union[dict, None] = None):
        if boundaries is None:
            boundaries = {}
        ocean_boundaries = []
        land_boundaries = []
        interior_boundaries = []
        for ibtype, bnds in boundaries.items():
            # print(bnds)
            if ibtype is None:
                for id, data in bnds.items():
                    indexes = list(
                        map(hgrid.nodes.get_index_by_id, data["indexes"])
                    )
                    ocean_boundaries.append(
                        {
                            "id": str(id + 1),  # hacking it
                            "index_id": data["indexes"],
                            "indexes": indexes,
                            "geometry": LineString(hgrid.vertices[indexes]),
                        }
                    )

            elif str(ibtype).endswith("1"):
                for id, data in bnds.items():
                    indexes = list(
                        map(hgrid.nodes.get_index_by_id, data["indexes"])
                    )
                    interior_boundaries.append(
                        {
                            "id": str(id + 1),
                            "ibtype": ibtype,
                            "index_id": data["indexes"],
                            "indexes": indexes,
                            "geometry": LineString(hgrid.vertices[indexes]),
                        }
                    )
            else:
                for id, data in bnds.items():
                    _indexes = np.array(data["indexes"])
                    if _indexes.ndim > 1:
                        # ndim > 1 implies we're dealing with an ADCIRC
                        # mesh that includes boundary pairs, such as weir
                        new_indexes = []
                        for i, line in enumerate(_indexes.T):
                            if i % 2 != 0:
                                new_indexes.extend(np.flip(line))
                            else:
                                new_indexes.extend(line)
                        _indexes = np.array(new_indexes).flatten()
                    else:
                        _indexes = _indexes.flatten()
                    indexes = list(map(hgrid.nodes.get_index_by_id, _indexes))

                    land_boundaries.append(
                        {
                            "id": str(id + 1),
                            "ibtype": ibtype,
                            "index_id": data["indexes"],
                            "indexes": indexes,
                            "geometry": LineString(hgrid.vertices[indexes]),
                        }
                    )

        self.open = gpd.GeoDataFrame(
            ocean_boundaries, crs=hgrid.crs if len(ocean_boundaries) > 0 else None
        )
        self.land = gpd.GeoDataFrame(
            land_boundaries, crs=hgrid.crs if len(land_boundaries) > 0 else None
        )
        self.interior = gpd.GeoDataFrame(
            interior_boundaries, crs=hgrid.crs if len(interior_boundaries) > 0 else None
        )
        self.hgrid = hgrid
        self.data = boundaries
        # print(self.data)
        
    @property
    def gdf(self):
        # readonly
        if not hasattr(self, '_gdf'):
            self._gdf = pd.concat([self.open, self.land, self.interior])
        return self._gdf

    def auto_generate(
        self,
        threshold=0.,
        land_ibtype=0,
        interior_ibtype=1,
        min_open_bound_length=2,
    ):
        self.open = None
        self.land = None
        self.interior = None
        self.data = {}

        values = self.hgrid.values
        if np.any(np.isnan(values)):
            raise Exception("Mesh contains invalid values. Raster values must"
                            "be interpolated to the mesh before generating "
                            "boundaries.")
        ocean_boundaries = []
        land_boundaries = []
        interior_boundaries = []
        cnt = 0
        for rings in self.hgrid.hull.rings.sorted().values():
            ring = rings['exterior']
            edge_tag = np.full(ring.shape, 0)
            edge_tag[np.where(values[ring[:, 0]] <= threshold)[0], 0] = -1
            edge_tag[np.where(values[ring[:, 1]] <= threshold)[0], 1] = -1
            edge_tag[np.where(values[ring[:, 0]] > threshold)[0], 0] = 1
            edge_tag[np.where(values[ring[:, 1]] > threshold)[0], 1] = 1

            # sort boundary edges
            ocean_boundary = list()
            land_boundary = list()

            for i, (e0, e1) in enumerate(edge_tag):

                if np.all(np.asarray((e0, e1)) == -1):
                    ocean_boundary.append(tuple(ring[i, :]))

                elif np.any(np.asarray((e0, e1)) == 1):
                    land_boundary.append(tuple(ring[i, :]))
                # else:
                #     if edge_tag[i-1][-1] == e0 == 1:
                #         land_boundary.append(tuple(ring[i, :]))
                #     elif edge_tag[i-1][-1] == e0 == -1:
                #         ocean_boundary.append(tuple(ring[i, :]))
                # else:
                    # print(edge_tag[i-1], edge_tag[i])

            for bnd in self.hgrid.edges_to_rings(ocean_boundary):
                if len(bnd) < min_open_bound_length:
                    land_boundary.extend(np.fliplr(bnd))
                    continue
                e0, e1 = [list(t) for t in zip(*bnd)]
                if len(e0) == 1:
                    e0 = [*e0, *e1]
                
                
                indexes = np.hstack([e0, e1[-1]])
                index_id = list(map(self.hgrid.nodes.get_id_by_index, indexes))
                coords = self.hgrid.coord[indexes]
                ocean_boundaries.append(
                            {
                                "id": str(cnt + 1),  # hacking it
                                "index_id": index_id,
                                "indexes": indexes,
                                "geometry": LineString(coords),
                            }
                        )
                # print(ocean_boundaries[-1])
                cnt += 1
 
            for bnd in self.hgrid.edges_to_rings(land_boundary):
                e0, e1 = [list(t) for t in zip(*bnd)]
                indexes = np.hstack([e0, e1[-1]])
                index_id = list(map(self.hgrid.nodes.get_id_by_index, indexes))
                coords = self.hgrid.coord[indexes]
                land_boundaries.append(
                            {
                                "id": str(cnt + 1),
                                "ibtype": land_ibtype,
                                "index_id": index_id,
                                "indexes": indexes,
                                "geometry": LineString(coords),
                            }
                        )
                cnt += 1

            for interior in rings['interiors']:
                e0, e1 = [list(t) for t in zip(*interior)]
                # if self.hgrid.signed_polygon_area(self.hgrid.coord[e0, :]) < 0:
                #     e0 = list(reversed(e0))
                #     e1 = list(reversed(e1))
                indexes = np.hstack([e0, e1[-1]])
                index_id = list(map(self.hgrid.nodes.get_id_by_index, indexes))
                # index_id.append(e0[0])
                coords = self.hgrid.coord[indexes]
                interior_boundaries.append(
                            {
                                "id": str(cnt + 1),
                                "ibtype": interior_ibtype,
                                "index_id": index_id,
                                "indexes": indexes,
                                "geometry": LineString(coords),
                            }
                        )
                cnt += 1
        self.open = gpd.GeoDataFrame(
            ocean_boundaries, crs=self.hgrid.crs if len(ocean_boundaries) > 0 else None
        )
        self.land = gpd.GeoDataFrame(
            land_boundaries, crs=self.hgrid.crs if len(land_boundaries) > 0 else None
        )
        self.interior = gpd.GeoDataFrame(
            interior_boundaries, crs=self.hgrid.crs if len(interior_boundaries) > 0 else None
        )
        ocn_bnd = {}
        for i, bnd in enumerate(ocean_boundaries):
            ocn_bnd.update({str(i+1): {'indexes': ocean_boundaries[i]['index_id']}})
        lnd_bnd = {}
        for i, bnd in enumerate(land_boundaries):
            lnd_bnd.update({str(i+1): {'indexes': land_boundaries[i]['index_id']}})
        int_bnd = {}
        for i, bnd in enumerate(interior_boundaries):
            int_bnd.update({str(i+1): {'indexes': interior_boundaries[i]['index_id']}})
        self.data = {
            None: ocn_bnd,
            land_ibtype: lnd_bnd,
            interior_ibtype: int_bnd
        }

        # self.hgrid.boundaries = self
        