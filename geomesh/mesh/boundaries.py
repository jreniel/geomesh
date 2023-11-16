from typing import Union

import numpy as np
from scipy.spatial import KDTree
import geopandas as gpd
import pandas as pd
from shapely.ops import linemerge
from shapely.geometry import LineString, MultiLineString


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
                            "btype": "open",
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
                            "btype": "interior"
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
                            "btype": "land",
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

    def __call__(self):
        return self.data

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
        min_open_bound_length=4,
    ):
        self.open = None
        self.land = None
        self.interior = None
        self.data = {}

        values = self.hgrid.values.flatten()
        if np.any(np.isnan(values)):
            raise Exception("Mesh contains invalid values. Raster values must"
                            "be interpolated to the mesh before generating "
                            "boundaries.")
        ocean_boundaries = []
        land_boundaries = []
        interior_boundaries = []
        # cnt = 0
        tree = KDTree(self.hgrid.coord)
        import matplotlib.pyplot as plt
        self.hgrid.hull.rings().plot(ax=plt.gca())
        plt.show(block=False)
        breakpoint()
        for rings in self.hgrid.hull.rings.sorted().values():
            ring = rings['exterior']
            edge_tag = np.full(ring.shape, 0)
            edge_tag[np.where(values[ring[:, 0]] <= threshold)[0], 0] = -1
            edge_tag[np.where(values[ring[:, 1]] <= threshold)[0], 1] = -1
            edge_tag[np.where(values[ring[:, 0]] > threshold)[0], 0] = 1
            edge_tag[np.where(values[ring[:, 1]] > threshold)[0], 1] = 1
            ocean_edges = list()
            land_edges = list()
            from shapely.geometry import Point
            for i, (t0, t1) in enumerate(edge_tag):
                e0, e1 = ring[i, :]
                if np.all(np.asarray((t0, t1)) == -1):
                    ocean_edges.append(LineString([Point(self.hgrid.coord[e0, :]), Point(self.hgrid.coord[e1, :])]))
                elif np.any(np.asarray((t0, t1)) == 1):
                    land_edges.append(LineString([Point(self.hgrid.coord[e0, :]), Point(self.hgrid.coord[e1, :])]))
            ocean_linestrings = linemerge(ocean_edges)
            if isinstance(ocean_linestrings, LineString):
                ocean_linestrings = MultiLineString([ocean_linestrings])
            for ocean_linestring in ocean_linestrings.geoms:
                coords = np.array(ocean_linestring.coords)


                # Note: If the LineString is a closed loop, SCHISM won't allow it (why not? it should be possible!)
                # but we are going to fake it by forcing one edge to be land_edge and the rest remain as ocean_boundaries
                # Maybe the user wants a circular open boundary!
                if coords[0].tolist() == coords[-1].tolist():
                    print("Detected a looped ocean_linestring:", ocean_linestring)
                    # Create land edge using the first two coordinates                                                   
                    print(f"{coords=}")
                    # land_edge_coords = [coords[-2], coords[0]]
                    land_edge_coords = [coords[0], coords[-2]]
                    land_edges.append(LineString([Point(x, y) for x, y in land_edge_coords]))
                    print("Created land edge from loop:", land_edges[-1])
                    print(f"{land_edge_coords=}")
                    # Create ocean linestring using all but the last coordinate (to avoid duplication with the first)    
                    # coords = np.concatenate([coords[:1], coords[2:-1]])
                    coords = coords[:-1]
                    print(f"{coords=}")
                    ocean_linestring = LineString(coords)
                    print("Created open boundary from loop:", ocean_linestring)

                if coords.shape[0] < min_open_bound_length:
                    land_edges.append(
                            # np.fliplr(bnd)
                            LineString([Point(x, y) for x, y in np.flipud(coords)])
                            )
                    continue
                _, ii = tree.query(coords)
                index_id = list(map(self.hgrid.nodes.get_id_by_index, ii))
                ocean_boundaries.append(
                            {
                                # "id": len(ocean_boundaries)+1,  # hacking it
                                "index_id": index_id,
                                "indexes": ii,
                                "geometry": ocean_linestring,
                                "btype": 'ocean',
                            }
                        )
                # cnt += 1

            land_linestrings = linemerge(land_edges)
            if isinstance(land_linestrings, LineString):
                land_linestrings = MultiLineString([land_linestrings])
            for land_linestring in land_linestrings.geoms:
                _, ii = tree.query(land_linestring.coords)
                index_id = list(map(self.hgrid.nodes.get_id_by_index, ii))
                land_boundaries.append(
                            {
                                # "id": str(cnt + 1),
                                "ibtype": land_ibtype,
                                "index_id": index_id,
                                "indexes": ii,
                                "geometry": land_linestring,
                                "btype": 'land',
                            }
                        )
                # cnt += 1

            for interior in rings['interiors']:
                indexes, _1 = [list(t) for t in zip(*interior)]
                index_id = list(map(self.hgrid.nodes.get_id_by_index, indexes))
                coords = self.hgrid.coord[indexes]
                interior_boundaries.append(
                            {
                                # "id": str(cnt + 1),
                                "ibtype": interior_ibtype,
                                "index_id": index_id,
                                "indexes": indexes,
                                "geometry": LineString(coords),
                                'btype': 'interior',
                            }
                        )
                # cnt += 1

        # to overcome out-of-memory array allocation bug in fortran side
        if len(interior_boundaries) > 0:
            land_boundaries = self._optimize_fortran_boundary_allocation_array(land_boundaries, interior_boundaries)
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
            ocn_bnd[str(i+1)] = {'indexes': bnd['index_id']}
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

    def _optimize_fortran_boundary_allocation_array(self, land_boundaries, interior_boundaries):
        """
        Q: Why is this function necessary?
        A: Fortran will allocate the boundary array as:
                allocate(ilnd_global(nland_global,mnlnd_global),stat=stat)
           where ilnd_global is len(land_boundaries) + len(interior_boundaries) and
           mnlnd_global is largest boundary length of all land and interior boundaries.
           At the limits you may have a single very large continuous land boundary, and multiple interior
           boundaries that are just a mere fraction of the single land boundary. This will cause the
           ilnd_global array to become unnecessarily large, to the point where it may run into memory allocation
           issues (not enough memory). To overcome this problem, we make sure that each of the land boundaries
           are not larger than the largest interior boundary.
        Note: ocean_boundaries are only used to make sure the boundary id's are continuous.
        """
        mnlnd_global_optimal = np.max([len(intb["indexes"]) for intb in interior_boundaries])
        new_land_boundaries = []
        _any_adjusted = False
        for land_boundary in land_boundaries:
            land_bound_length = len(land_boundary['indexes'])
            if land_bound_length >= mnlnd_global_optimal:
                _any_adjusted = True
                # boundary needs splitting
                chunks = int(np.ceil(land_bound_length / mnlnd_global_optimal))
                split_indexes = np.array_split(land_boundary['indexes'], chunks)
                split_index_ids = np.array_split(land_boundary['index_id'], chunks)
                for i, _split_indexes in enumerate(split_indexes):
                    if i > 0:
                        _split_indexes = np.insert(split_indexes[i], 0, split_indexes[i-1][-1])
                        _split_index_ids = np.insert(split_index_ids[i], 0, split_index_ids[i-1][-1])
                    else:
                        _split_index_ids = split_index_ids[i]
                    # _split_index_ids = split_index_ids[i]
                    land_linestring = LineString(self.hgrid.coord[_split_indexes, :])
                    new_land_boundaries.append({
                                    # "id": str(cnt + 1),
                                    "ibtype": land_boundary['ibtype'],
                                    "index_id": _split_index_ids,
                                    "indexes": _split_indexes,
                                    "geometry": land_linestring,
                                    "btype": 'land',
                                })
                    # cnt += 1
            else:
                new_land_boundaries.append(land_boundary)

        return new_land_boundaries
