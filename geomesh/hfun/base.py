from abc import ABC, abstractmethod

from jigsawpy import jigsaw_msh_t  # type: ignore[import]


class BaseHfun(ABC):

    @abstractmethod
    def msh_t(self, dst_crs=None) -> jigsaw_msh_t:
        '''Abstract method to generate hfun object. Must contain a dst_crs kwarg'''

    @property
    @abstractmethod
    def crs(self):
        '''Coordinate reference system information.'''

    @property
    @abstractmethod
    def values(self):
        '''Values of hfun.'''
