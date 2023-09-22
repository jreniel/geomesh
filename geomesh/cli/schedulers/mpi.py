import asyncio
from typing import List
import logging

import subprocess

from .base import BaseCluster


logger = logging.getLogger(__name__)

class MPICluster(BaseCluster):

    def __init__(
            self,
            ntasks: int = None,
            # cpus_per_task: int = None,
            launcher=None,
            # **kwargs
            ):
        super().__init__(semaphore=1)
        self.ntasks = ntasks
        # self.cpus_per_task = cpus_per_task
        self.launcher = launcher
        self.jobs = []

    def submit(
            self,
            command,
            ntasks=None,
            # cpus_per_task=None,
            launcher=None
            ):
        _cmd = [launcher or self.launcher]
        ntasks = ntasks or self.ntasks
        if ntasks is not None:
            _cmd.extend(['-n', str(ntasks)])
        # cpus_per_task = self.cpus_per_task or cpus_per_task
        # if cpus_per_task is not None:
        #     _cmd.extend(['-c', str(cpus_per_task)])
        if isinstance(command, str):
            _cmd.append(command)
        else:
            _cmd.extend(command)
        logger.info(f'Launching command: {" ".join(_cmd)}')
        # import os
        # try:
        #     p = subprocess.run(_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ)
        # except subprocess.CalledProcessError as e:
        #     error_msg = f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.\n"
        #     error_msg += f"Output: {e.output}\n"
        #     error_msg += f"Error: {e.stderr}"
        #     raise Exception(error_msg)
        # if asyncio.get_event_loop().is_running():
        #     dummy_future = asyncio.Future()
        #     dummy_future.set_result(p)  # or set_result(None) if you don't care about the result
        #     return dummy_future
        # return p

        return super().submit(_cmd)

    @property
    def launcher(self):
        return self._launcher

    @launcher.setter
    def launcher(self, launcher):
        if launcher is None:
            launcher = 'mpiexec'
        assert launcher in ['mpiexec', 'mpirun']
        self._launcher = launcher
