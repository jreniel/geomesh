#! /usr/bin/env python
import setuptools
import subprocess
from importlib import util
import setuptools.command.build_py
import distutils.cmd
import distutils.util
import shutil
import platform
from multiprocessing import cpu_count
import sysconfig
from pathlib import Path
import sys
import os
import tempfile

import pathlib


# if pathlib.Path('build').exists():
#     shutil.rmtree('build')
# if pathlib.Path('dist').exists():
#     shutil.rmtree('dist')

PARENT = Path(__file__).parent.absolute()
PYENV_PREFIX = Path("/".join(sys.executable.split("/")[:-2]))
SYSLIB = {"Windows": "jigsaw.dll", "Linux": "libjigsaw.so", "Darwin": "libjigsaw.dylib"}

if "install_jigsaw" not in sys.argv:
    if "develop" not in sys.argv:
        if "install" in sys.argv:
            libsaw = PYENV_PREFIX / "lib" / SYSLIB[platform.system()]
            if not libsaw.is_file():
                subprocess.check_call([sys.executable, "setup.py", "install_jigsaw"])

if util.find_spec("setuptools_cythonize") is None:
    subprocess.check_call([sys.executable, "-m", "pip", 'install', 'setuptools-cythonize'])

from setuptools_cythonize import get_cmdclass


class InstallJigsawCommand(distutils.cmd.Command):
    """Custom build command."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.announce("Loading JIGSAW from GitHub", level=3)
        tmpdir = tempfile.TemporaryDirectory()
        subprocess.check_call(
            [
                "git",
                "clone",
                "https://github.com/dengwirda/jigsaw",
                f"{tmpdir.name}/jigsaw",
            ]
        )
        # install jigsawpy first
        cwd = os.getcwd()
        os.chdir(f"{tmpdir.name}/jigsaw")
        self.announce("INSTALLING JIGSAW", level=3)
        # subprocess.check_call(["python", "setup.py", "install"])
        # then install jigsaw
        self.announce(
            "INSTALLING JIGSAW LIBRARY AND BINARIES FROM "
            "https://github.com/dengwirda/jigsaw",
            level=3,
        )
        # os.chdir("external/jigsaw")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        gcc, cpp = self.get_gcc_version()
        subprocess.check_call(
            [
                "cmake",
                "..",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_INSTALL_PREFIX={PYENV_PREFIX}",
                f"-DCMAKE_C_COMPILER={os.getenv('CC', 'gcc')}",
                f"-DCMAKE_CXX_COMPILER={os.getenv('CXX', 'cpp')}",
            ]
        )
        subprocess.check_call(["make", f"-j{cpu_count()}", "install"])
        libsaw_prefix = list(PYENV_PREFIX.glob("**/*jigsawpy*")).pop() / "_lib"
        os.makedirs(libsaw_prefix, exist_ok=True)
        envlib = PYENV_PREFIX / "lib" / SYSLIB[platform.system()]
        os.symlink(envlib, libsaw_prefix / envlib.name)
        os.chdir(cwd)

    @staticmethod
    def get_gcc_version():
        """
        return: major, minor, patch
        """
        cpp = shutil.which(os.getenv("CXX", "cpp"))
        major, minor, patch = (
            subprocess.check_output([cpp, "--version"])
            .decode("utf-8")
            .split("\n")[0]
            .split()[-1]
            .split(".")
        )
        current_version = float(f"{major}.{minor}")
        if current_version < 7.0:
            raise Exception(
                "JIGSAW requires GCC version 7 or later, got "
                f"{major}.{minor}.{patch} from {cpp}"
            )
        return shutil.which(os.getenv("CC", "gcc")), cpp


conf = setuptools.config.read_configuration(PARENT / "setup.cfg")
meta = conf["metadata"]
doc_requires = [
    "sphinx",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "nbsphinx",
]
test_requires = [
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "pytest-socket",
    "pytest-xdist",
]
exclude = ["examples", "docs"]

if "tests" not in sys.argv:
    exclude.append("tests")


def get_files(input):
    for fd, subfds, fns in os.walk(input):
        for fn in fns:
            yield os.path.join(fd, fn)


for file in pathlib.Path("geomesh").glob("**/_*"):
    if not str(file).endswith("__.py"):
        if "__pycache__" in str(file.resolve):
            continue
        if file.is_dir():
            exclude.extend(list(get_files(file)))
        else:
            exclude.append(str(file))

cmdclass = get_cmdclass()
cmdclass.update({"install_jigsaw": InstallJigsawCommand})


class build_py(cmdclass["build_py"]):
    def find_package_modules(self, package, package_dir):
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = []
        for (pkg, mod, filepath) in modules:
            if os.path.exists(filepath.replace(".py", ext_suffix)):
                continue
            if mod.startswith("_") and not filepath.endswith("__.py"):
                continue
            filtered_modules.append(
                (
                    pkg,
                    mod,
                    filepath,
                )
            )

        return filtered_modules


cmdclass.update({"build_py": build_py})

setuptools.setup(
    name=meta["name"],
    version=meta["version"],
    author=meta["author"],
    author_email=meta["author_email"],
    description=meta["description"],
    long_description=meta["long_description"],
    long_description_content_type="text/markdown",
    url=meta["url"],
    packages=setuptools.find_packages(exclude=exclude),
    cmdclass=cmdclass,
    options={
        "build_ext": {"parallel": cpu_count()},
        "build_py": {"exclude_cythonize": []},
    },
    python_requires=">=3.7",
    # setup_requires=['wheel', 'numpy'],
    install_requires=[
        "jigsawpy @ git+https://github.com/dengwirda/jigsaw-python@master",
        "matplotlib",
        "netCDF4",
        "scipy>=1.7.1",
        "pyproj>=3.0",
        "fiona",
        "rasterio",
        "tqdm",
        # "pysheds",
        "colored_traceback",
        "requests",
        "shapely",
        "geoalchemy2",
        "utm",
        "geopandas",
        "pyyaml",
        "appdirs",
        "pylas",
        "lazrs",
    ],
    extras_require={
        "doc": doc_requires,
        "tests": test_requires,
        "all": [*doc_requires, *test_requires],
    },
    entry_points={
        "console_scripts": [
            "geomesh=geomesh.__main__:main",
            "geomesh-api=geomesh.api.__main__:main",
        ]
    },
)
