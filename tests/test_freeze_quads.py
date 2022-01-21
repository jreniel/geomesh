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
    logging.getLogger('geomesh').setLevel(logging.DEBUG)
    test_freeze_quads(verbosity=1)
