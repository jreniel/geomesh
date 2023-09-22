import logging
from pathlib import Path

# from geoalchemy2 import Geometry
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.event import listen
from sqlalchemy.sql import select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
# from sqlalchemy import Column, String


Base = declarative_base()


logger = logging.getLogger(__name__)


# class Geom(Base):
#     __tablename__ = 'geom'
#     id = Column(String, primary_key=True, nullable=False)
#     tag = Column(String)
#     geom = Column(
#         Geometry(
#             geometry_type='MULTIPOLYGON',
#             management=True
#             ),
#         nullable=False
#         )


# class GeomCollection(Base):
#     __tablename__ = "geom_collection"
#     id = Column(String, primary_key=True, nullable=False)
#     geom = Column(
#         Geometry(
#             geometry_type='MULTIPOLYGON',
#             management=True
#             ),
#         nullable=False
#         )


# class Hfun(Base):
#     __tablename__ = 'hfun'
#     id = Column(String, primary_key=True, nullable=False)
#     hfun = Column(
#         Geometry(
#             geometry_type='MULTIPOLYGON',
#             management=True,
#             dimension=3,
#             ),
#         nullable=False
#         )


# class HfunCollection(Base):
#     __tablename__ = 'hfun_collection'
#     id = Column(String, primary_key=True, nullable=False)


# class Raster(Base):
#     __tablename__ = "raster"
#     id = Column(String, primary_key=True)
#     geometry = Column(
#         Geometry(
#             'POLYGON',
#             management=True,
#             srid=4326
#         ),
#         nullable=False
#     )
#     uri = Column(String, nullable=True)


def init_spatialite(engine, path):
    """Initializes a Spatialite database."""
    _new_db = False
    if not Path(path).is_file():
        _new_db = True

    def load_spatialite(dbapi_conn, connection_record):
        dbapi_conn.enable_load_extension(True)
        dbapi_conn.load_extension('mod_spatialite')

    listen(engine, 'connect', load_spatialite)
    if _new_db:
        conn = engine.connect()
        conn.execute(select([func.InitSpatialMetaData()]))
        conn.close()
        # Geom.__table__.create(engine)
        # GeomCollection.__table__.create(engine)
        # Hfun.__table__.create(engine)
        # HfunCollection.__table__.create(engine)
        # Raster.__table__.create(engine)


def init_postgis(engine):
    """Initializes a PostGIS database."""
    # You would add here the necessary steps to initialize your PostGIS database.
    raise NotImplementedError("PostGIS initialization not implemented")


def create_session(database_url, echo=False):
    engine = create_engine(database_url, echo=echo)

    # Based on the type of database, call different init functions
    if 'sqlite' in database_url:
        init_spatialite(engine, database_url.replace('sqlite:///', ''))
    elif 'postgresql' in database_url:
        init_postgis(engine)

    return sessionmaker(bind=engine)




# def spatialite_session(path=None, echo=False):

#     def engine(path, echo=False):
#         path = pathlib.Path(path)
#         _new_db = False
#         if not path.is_file():
#             path.parent.mkdir(exist_ok=True, parents=True)
#             _new_db = True

#         engine = create_engine(f'sqlite:///{str(path)}', echo=echo)

#         def load_spatialite(dbapi_conn, connection_record):
#             dbapi_conn.enable_load_extension(True)
#             dbapi_conn.load_extension('mod_spatialite')

#         listen(engine, 'connect', load_spatialite)
#         if _new_db:
#             logger.info(f'No spatialite db found. Initializing cache database to {path}')
#             conn = engine.connect()
#             conn.execute(select([func.InitSpatialMetaData()]))
#             conn.close()
#             Geom.__table__.create(engine)
#             GeomCollection.__table__.create(engine)
#             Hfun.__table__.create(engine)
#             HfunCollection.__table__.create(engine)
#             Raster.__table__.create(engine)

#         return engine

#     if path is None:
#         path = pathlib.Path(appdirs.user_cache_dir('geomesh')) / 'cache.sqlite'
#     else:
#         path = pathlib.Path(path)

#     return sessionmaker(bind=engine(path, echo))



