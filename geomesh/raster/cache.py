import logging
import pathlib

import appdirs
import requests

logger = logging.getLogger(__name__)


class RasterCache:

    def __init__(self, path=None):
        if path is None:
            self._cache_parent_path = pathlib.Path(appdirs.user_cache_dir('geomesh/raster'))
        else:
            self._cache_parent_path = pathlib.Path(path)
        self._container = {}

    def fetch(self, uri: str):
        raster = self.get(uri)
        if raster is None:
            self.add(uri)
            raster = self.get(uri)
        return raster

    def get(self, uri: str):
        return self._container.get(uri, None)

    def add(self, uri: str, raise_on_missing=False):
        filepath = self._cache_parent_path / uri.split('://')[-1]
        if filepath.exists() is True:
            if raise_on_missing is True:
                raise IOError(f'{uri} already in cache.')
        else:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(uri, stream=True)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                logger.debug(f"Downloading raster data from {uri}...")
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        self._container.setdefault(uri, filepath)

    def tile_exists(self, location: str):
        return (self._cache_parent_path / location).exists()

    @property
    def path(self):
        return self._cache_parent_path
