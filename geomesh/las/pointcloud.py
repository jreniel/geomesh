import pylas


class PointCloud:
    def __init__(self, path):
        self._path = path

    @property
    def path(self):
        return self._path

    @property
    def _path(self):
        return self.__path

    @_path.setter
    def _path(self, path: Union[str, os.PathLike]):
        if pathlib.Path(path).exists() is False:
            try:
                r = requests.get(path, stream=True)
                r.raise_for_status()
                tmpfile = tempfile.NamedTemporaryFile()
                with open(tmpfile.name, "wb") as f:
                    logger.debug(f"Downloading point cloud data from {path}...")
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                self._tmpfile = tmpfile
                self.__path = pathlib.Path(tmpfile.name)
            except requests.exceptions.MissingSchema:
                raise Exception(
                    f"Given path {path} is neither a valid URL nor a system file."
                )
        else:
            self.__path = pathlib.Path(path)
