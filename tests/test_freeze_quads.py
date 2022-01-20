from geomesh import Geom, Hfun, Mesh
from geomesh.geom.mesh import MeshGeom
from geomesh.raster import RasterTileIndex


def test_freeze_quads(verbosity=0):

    mesh = Mesh.open('https://raw.githubusercontent.com/geomesh/test-data/main/NWM/hgrid.ll', crs='epsg:4326')

    geom = Geom(mesh)
    assert isinstance(geom, MeshGeom)
    geom.freeze_quads = True

    tile_index = RasterTileIndex(
        'https://chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/tileindex_NCEI_ninth_Topobathy_2014.zip',
        bbox=mesh.get_bbox().bounds,
        )

    hfun = Hfun(tile_index)

    hfun.make_plot(show=True)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        force=True,
    )
    logging.captureWarnings(True)

    log_level = logging.DEBUG

    from geomesh.geom.mesh import logger as geom_mesh_logger
    geom_mesh_logger.setLevel(log_level)

    # from geomesh.raster.raster import logger as raster_logger

    # raster_logger.setLevel(log_level)

    # from geomesh.hfun.raster import logger as hfun_raster_logger

    # hfun_raster_logger.setLevel(log_level)

    # from geomesh.geom.raster import logger as geom_raster_logger

    # geom_raster_logger.setLevel(log_level)

    # from geomesh.utils import logger as utils_logger

    # utils_logger.setLevel(log_level)

    # from geomesh.driver import logger as jigsaw_driver_logger

    # jigsaw_driver_logger.setLevel(log_level)

    test_freeze_quads(verbosity=1)
