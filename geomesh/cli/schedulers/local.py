from .base import BaseCluster

class LocalCluster(BaseCluster):
    def __init__(self):
        super().__init__(semaphore=1)
