# import hashlib
import logging
import pathlib
# from typing import List, TYPE_CHECKING


from geoalchemy2 import Geometry, Raster as _Raster
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.event import listen
from sqlalchemy.sql import select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import Column, String


Base = declarative_base()


logger = logging.getLogger(__name__)


class Geom(Base):
    __tablename__ = 'geom'
    id = Column(String, primary_key=True, nullable=False)
    tag = Column(String)
    geom = Column(
        Geometry(
            geometry_type='MULTIPOLYGON',
            management=True
            ),
        nullable=False
        )


class GeomCollection(Base):
    __tablename__ = "geom_collection"
    id = Column(String, primary_key=True, nullable=False)
    geom = Column(
        Geometry(
            geometry_type='MULTIPOLYGON',
            management=True
            ),
        nullable=False
        )


class Hfun(Base):
    __tablename__ = 'hfun'
    id = Column(String, primary_key=True, nullable=False)
    hfun = Column(
        Geometry(
            geometry_type='MULTIPOLYGON',
            management=True,
            dimension=3,
            ),
        nullable=False
        )


class HfunCollection(Base):
    __tablename__ = 'hfun_collection'
    id = Column(String, primary_key=True, nullable=False)


# class Raster(Base):
#     __tablename__ = "raster"
#     raster = Column(_Raster, nullable=True)
#     uri = Column(String, nullable=True)
#     id = Column(String, primary_key=True)
#     geom = relationship("Geom")


# class TileIndexRasters(Base):
#     __tablename__ = 'tile_index'
#     geom = Column(
#         Geometry(
#             'POLYGON',
#             management=True,
#             srid=4326
#             ),
#         nullable=False)
#     raster = Column(_Raster(srid=4326, spatial_index=False))
#     url = Column(String, primary_key=True, nullable=False)
#     name = Column(String, nullable=False)
#     md5 = Column(String, nullable=False)


def spatialite_session(path, echo=False):

    def engine(path, echo=False):
        path = pathlib.Path(path)
        _new_db = not path.is_file()
        engine = create_engine(f'sqlite:///{str(path)}', echo=echo)

        def load_spatialite(dbapi_conn, connection_record):
            dbapi_conn.enable_load_extension(True)
            dbapi_conn.load_extension('mod_spatialite')

        listen(engine, 'connect', load_spatialite)
        if _new_db:
            logger.info(f'No spatialite db found. Initializing cache database to {path}')
            conn = engine.connect()
            conn.execute(select([func.InitSpatialMetaData()]))
            conn.close()
            Geom.__table__.create(engine)
            GeomCollection.__table__.create(engine)
            Hfun.__table__.create(engine)
            HfunCollection.__table__.create(engine)
            # Raster.__table__.create(engine)

        return engine

    return sessionmaker(bind=engine(path, echo))


def postgis_session():
    raise NotImplementedError


def get_session(path, echo=False, dbtype='spatialite'):
    assert dbtype in ['spatialite']
    return spatialite_session(path, echo)
