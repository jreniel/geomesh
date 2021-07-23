from importlib import util
import os
import pathlib
import platform
import sys

import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 10000

from geomesh.geom.geom import Geom
from geomesh.hfun.hfun import Hfun
from geomesh.raster import Raster
from geomesh.driver import JigsawDriver
from geomesh.mesh import Mesh


try:
    import jigsawpy  # type: ignore[import]  # noqa: F401
except OSError as e:
    pkg = util.find_spec("jigsawpy")
    libjigsaw = {
        "Windows": "jigsaw.dll",
        "Linux": "libjigsaw.so",
        "Darwin": "libjigsaw.dylib",
    }[platform.system()]
    tgt_libpath = pathlib.Path(pkg.origin).parent / "_lib" / libjigsaw  # type: ignore[union-attr]
    pyenv = pathlib.Path("/".join(sys.executable.split("/")[:-2]))
    src_libpath = pyenv / "lib" / libjigsaw
    if not src_libpath.is_file():
        raise e
    else:
        os.symlink(src_libpath, tgt_libpath)

if util.find_spec("colored_traceback") is not None:
    import colored_traceback  # type: ignore[import]

    colored_traceback.add_hook(always=True)

# tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/geomesh'))+'/'
# os.makedirs(tmpdir, exist_ok=True)


__all__ = [
    "Geom",
    "Hfun",
    "Raster",
    "Mesh",
    "JigsawDriver",
]
