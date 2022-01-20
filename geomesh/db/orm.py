# import hashlib
import logging
import pathlib
# from typing import List, TYPE_CHECKING

import appdirs

from geoalchemy2 import Geometry
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


class Raster(Base):
    __tablename__ = "raster"
    id = Column(String, primary_key=True)
    geometry = Column(
        Geometry(
            'POLYGON',
            management=True,
            srid=4326
        ),
        nullable=False
    )
    uri = Column(String, nullable=True)


def spatialite_session(path=None, echo=False):

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
            Raster.__table__.create(engine)

        return engine
    if path is None:
        path = pathlib.Path(appdirs.user_cache_dir('geomesh'))
    else:
        path = pathlib.Path(path)
    return sessionmaker(bind=engine(path, echo))
