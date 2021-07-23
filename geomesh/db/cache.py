import logging
from typing import Union

import geoalchemy2
from pyproj import CRS
from sqlalchemy.orm.session import Session

from geomesh import Geom, db, Raster


logger = logging.getLogger(__name__)

EPSG_4326 = CRS.from_epsg(4326)


class GeomCollection:
    def __init__(self, session: Session):
        """
        :param session: A session built with :func:`geomesh.db.orm.get_session`
        """
        self.session = session

    def add(self, geom: Geom) -> None:

        if not geom.crs.equals(CRS.from_epsg(4326)):
            logger.info(f"Transforming from {geom.crs} to EPSG:4326 for database storage.")
            geom = Geom(geom.get_multipolygon(crs=CRS.from_epsg(4326)))

        self.session.add(
            db.orm.GeomCollection(
                geom=geoalchemy2.shape.from_shape(geom.multipolygon),
                id=geom.md5,
                )
            )
        self.session.commit()

    def get(self, id: str):
        query = self.session.query(db.orm.Geom).get(id)
        if query is None:
            return
        geom = Geom(
            geoalchemy2.shape.to_shape(query.geom),
            crs=EPSG_4326
        )
        original_crs = CRS.from_user_input(query.original_crs)
        if original_crs.equals(EPSG_4326) is not True:
            geom = Geom(
                    geom.get_multipolygon(crs=original_crs),
                    crs=original_crs
                )
        return geom


class GeomCache:
    """
    API for caching and retrieving geom objects from databases.
    """

    def __init__(self, session: Session):
        """
        :param session: A session built with :func:`geomesh.db.orm.get_session`
        """
        self.session = session
        self.collection = GeomCollection(session)

    def add(self, geom: Geom) -> None:

        if not geom.crs.equals(CRS.from_epsg(4326)):
            logger.info(f"Transforming from {geom.crs} to EPSG:4326 for database storage.")
            geom = Geom(geom.get_multipolygon(crs=CRS.from_epsg(4326)))

        self.session.add(
            db.orm.Geom(
                geom=geoalchemy2.shape.from_shape(geom.multipolygon),
                id=id
                )
            )
        self.session.commit()

    def get(self, id: str, table: db.orm.Base) -> Union[Geom, None]:
        query = self.session.query(table).get(id)
        if query is None:
            logger.info(f"Geom with id {id} not found in database.")
            return
        geom = Geom(
            geoalchemy2.shape.to_shape(query.geom),
            crs=EPSG_4326
        )
        original_crs = CRS.from_user_input(query.original_crs)
        if original_crs.equals(EPSG_4326) is not True:
            geom = Geom(
                    geom.get_multipolygon(crs=original_crs),
                    crs=original_crs
                )
        return geom


# class RasterCache:

#     def __init__(self, session: Session):
#         """
#         :param session: A session built with :func:`geomesh.db.orm.get_session`
#         """
#         self.session = session

#     def add(self, raster: Raster) -> None:
#         raise NotImplementedError('RasterCache.add')

#     def get(self, id: str) -> Raster:
#         raise NotImplementedError('RasterCache.get')
#         query = self.session.query(db.orm.Raster).get(id)


class Cache:

    def __init__(self, session: Session):
        self.session = session
        self.geom = GeomCache(session)
        # self.raster = RasterCache(session)
