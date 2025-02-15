from importlib import util
import os
import pathlib
import platform
import sys

import matplotlib as mpl

from .geom.geom import Geom
from .hfun.hfun import Hfun
from .raster import Raster
from .driver import JigsawDriver
from .mesh import Mesh


mpl.rcParams["agg.path.chunksize"] = 10000

# try:
#     import jigsawpy  # type: ignore[import]  # noqa: F401
# except OSError as e:
#     pkg = util.find_spec("jigsawpy")
#     libjigsaw = {
#         "Windows": "jigsaw.dll",
#         "Linux": "libjigsaw.so",
#         "Darwin": "libjigsaw.dylib",
#     }[platform.system()]
#     tgt_libpath = pathlib.Path(pkg.origin).parent / "_lib" / libjigsaw  # type: ignore[union-attr]
#     pyenv = pathlib.Path("/".join(sys.executable.split("/")[:-2]))
#     src_libpath = pyenv / "lib" / libjigsaw
#     if not src_libpath.is_file():
#         raise e
#     else:
#         os.symlink(src_libpath, tgt_libpath)

if util.find_spec("colored_traceback") is not None:
    import colored_traceback  # type: ignore[import]
    colored_traceback.add_hook(always=True)

# tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/geomesh'))+'/'
# os.makedirs(tmpdir, exist_ok=True)

# warnings.filterwarnings(
#     "ignore",
#     ".*will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry."
#     )



__all__ = [
    "Geom",
    "Hfun",
    "Raster",
    "Mesh",
    "JigsawDriver",
]
