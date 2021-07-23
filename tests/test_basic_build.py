import os

import matplotlib.pyplot as plt

from geomesh import Geom, Hfun, Raster, JigsawDriver
from geomesh.geom.raster import RasterGeom
from geomesh.hfun.raster import RasterHfun
from geomesh.mesh.base import BaseMesh


def test_basic_build(verbosity=0):

    url = "https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/ncei19_n41x00_w074x00_2015v1.tif"

    raster = Raster(
        url,
        # chunk_size=1500
    )
    raster.resample(0.125)
    assert isinstance(raster, Raster)

    geom = Geom(raster, zmax=15.0)
    assert isinstance(geom, RasterGeom)

    hfun = Hfun(raster, verbosity=verbosity)
    assert isinstance(hfun, RasterHfun)

    hfun.add_gradient_delimiter(hmin=30.0, hmax=250.0)
    hfun.add_contour(level=0.0, target_size=30.0, expansion_rate=0.01)

    driver = JigsawDriver(geom=geom, hfun=hfun, verbosity=verbosity)
    assert isinstance(driver, JigsawDriver)

    mesh = driver.run()
    assert isinstance(mesh, BaseMesh)

    mesh.interpolate(raster)
    mesh.make_plot()
    plt.show(block=True if os.getenv("GITHUB_ACTIONS") is None else False)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        force=True,
    )
    logging.captureWarnings(True)

    log_level = logging.DEBUG

    from geomesh.raster import logger as raster_logger

    raster_logger.setLevel(log_level)

    from geomesh.hfun.raster import logger as hfun_raster_logger

    hfun_raster_logger.setLevel(log_level)

    from geomesh.geom.raster import logger as geom_raster_logger

    geom_raster_logger.setLevel(log_level)

    from geomesh.utils import logger as utils_logger

    utils_logger.setLevel(log_level)

    from geomesh.driver import logger as jigsaw_driver_logger

    jigsaw_driver_logger.setLevel(log_level)

    test_basic_build(verbosity=1)
