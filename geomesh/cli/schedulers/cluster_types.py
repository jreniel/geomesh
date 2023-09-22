from enum import Enum

from .slurm import SLURMCluster
from .local import LocalCluster
from .mpi import MPICluster


class ClusterTypes(Enum):

    # Hoping to be able to support them all through a relatively unified interface.
    # Currently only supporting SLURM
    # https://jobqueue.dask.org/en/latest/api.html
    # HTCONDOR = HTCondorCluster
    # LSF = LSFCluster
    # MOAB = MoabCluster
    # OAR = OARCluster
    # PBS = PBSCluster
    # SGE = SGECluster
    MPI = MPICluster
    LOCAL = LocalCluster
    SLURM = SLURMCluster
