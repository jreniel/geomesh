from datetime import datetime, timedelta
from pathlib import Path
from typing import Union
from time import sleep
import asyncio
import logging
import os
import subprocess
import tempfile
import uuid

from psutil import cpu_count
import numpy as np

from .base import BaseCluster

logger = logging.getLogger(__name__)


class SLURMCluster(BaseCluster):

    def __init__(
            self,
            account: str = None,
            walltime: timedelta = None,
            ntasks: int = None,
            min_ntasks: int = None,
            max_ntasks: int = None,
            cpus_per_task: int = None,
            minnodes: int = None,
            maxnodes: int = None,
            partition: str = None,
            cwd: Union[os.PathLike, str] = None,
            job_name: str = None,
            mail_type: str = None,
            mail_user: str = None,
            job_filename: str = None,
            log_filename: str = None,
            begin: datetime = None,
            semaphore: Union[int, asyncio.Semaphore] = None,
            use_srun: bool = True,
            ):
        super().__init__(semaphore=semaphore)
        self.account = account
        self.walltime = walltime
        self.ntasks = ntasks
        self.min_ntasks = min_ntasks
        self.max_ntasks = max_ntasks
        self.cpus_per_task = cpus_per_task
        self.minnodes = minnodes
        self.maxnodes = maxnodes
        self.partition = partition
        self.cwd = cwd
        self.job_name = job_name
        self.mail_type = mail_type
        self.mail_user = mail_user
        self.job_filename = job_filename
        self.log_filename = log_filename
        self.begin = begin
        self.use_srun = bool(use_srun)

    def job_script(self, command):
        return f"""#!/bin/bash
        #SBATCH --job-name=geom-build
        #SBATCH --partition=partition_name
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=1
        #SBATCH --time=00:10:00
        #SBATCH --mem=1gb
        #SBATCH --output=/home/username/geom-build.log
        #SBATCH --mail-type=ALL
        #SBATCH --mail-user=email@domain.com
        {command}
        """.replace('        ', '')

    def submit(self, command, **kwargs):
        # super will send the command to the shell using asyncio.to_thread, and will return a Future
        # that can be used to check the job's status. This is regardless of whether the command is
        # of the sbatch or srun variety.
        command = self._get_blocking_submit_command(command, **kwargs)
        return super().submit(command, **kwargs)

    def write_cmd_to_job_script(self, cmd, cache_directory, **kwargs):
        job_script = self.job_script().split('\n')
        script_string = '\n'.join([
            *job_script[:-2],
            ' '.join(cmd)
        ])
        job_script_filename = cache_directory / f'.{uuid.uuid4().hex}.sh'
        with open(job_script_filename, 'w') as f:
            f.write(script_string)
        return job_script_filename

    # async def wait(self):
    #     await asyncio.wait([self.check_job_status(job) for job in self.jobs])

    async def check_job_status(self, job_id):
        while True:
            process = await asyncio.create_subprocess_exec('squeue', '-j', job_id, stdout=subprocess.PIPE)
            output = await process.communicate()
            if job_id.encode() not in output[0]:
                break
            await asyncio.sleep(5)

    # async def check_job_status(self, job_id):
    #     while True:
    #         output = subprocess.check_output(['sacct', '-j', job_id, '-X', '-n', '-o', 'State'])
    #         job_state = output.decode().strip()
    #         if job_state in ['CANCELLED', 'COMPLETED', 'FAILED']:
    #             break
    #         await asyncio.sleep(5)

    @property
    def walltime(self):
        if isinstance(self._walltime, timedelta):
            hours, remainder = divmod(self._walltime, timedelta(hours=1))
            minutes, remainder = divmod(remainder, timedelta(minutes=1))
            seconds = round(remainder / timedelta(seconds=1))
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        return self._walltime

    @walltime.setter
    def walltime(self, walltime: Union[timedelta, str, None]):
        assert isinstance(walltime, (timedelta, str, type(None)))
        self._walltime = walltime

    @property
    def ntasks(self):
        return self._ntasks

    @ntasks.setter
    def ntasks(self, ntasks: Union[int, None]):
        err_msg = f"Argument ntasks must be int > 0 or None, got {ntasks}."
        if not isinstance(ntasks, (int, type(None))):
            raise ValueError(err_msg)
            if isinstance(ntasks, int):
                assert ntasks >= 1, err_msg
        self._ntasks = ntasks

    @property
    def min_ntasks(self):
        return self._min_ntasks

    @min_ntasks.setter
    def min_ntasks(self, min_ntasks):
        if min_ntasks is not None:
            _err = f'Argument `min_ntasks` must be int > 0 or None, not {min_ntasks}.'
            if not isinstance(min_ntasks, int):
                raise ValueError(_err)
            if not min_ntasks > 0:
                raise ValueError(_err)
        # else:
        #     min_ntasks = cpu_count(logical=False)
        self._min_ntasks = min_ntasks

    @property
    def max_ntasks(self):
        return self._max_ntasks

    @max_ntasks.setter
    def max_ntasks(self, max_ntasks):
        if max_ntasks is not None:
            _err = f'Argument `max_ntasks` must be int > 0 or None, not {max_ntasks}.'
            if not isinstance(max_ntasks, int):
                raise ValueError(_err)
            if not max_ntasks > 0:
                raise ValueError(_err)
        # else:
        #     max_ntasks = cpu_count(logical=False)
            if not max_ntasks >= self.min_ntasks:
                raise ValueError(f'Argument `max_ntasks` must be >= `min_ntasks`.')
        self._max_ntasks = max_ntasks

    @property
    def job_filename(self):
        return self._job_filename

    @job_filename.setter
    def job_filename(self, job_filename):
        if job_filename is None:
            job_filename = Path(tempfile.NamedTemporaryFile().name)
        self._job_filename = job_filename

    def _get_blocking_submit_command(self, cmd, **kwargs):
        if self.use_srun:
            return self._get_srun_command(cmd, **kwargs)
        else:
            return self._get_sbatch_command(cmd, **kwargs)

    def _get_cli_slurm_opts(self, **kwargs):
        # it's currently not exhaustive, see https://slurm.schedmd.com/pdfs/summary.pdf
        # and https://slurm.schedmd.com/sbatch.html for more options

        _cmd = []

        # ntasks
        ntasks = kwargs.get('ntasks') or self.ntasks
        min_ntasks = kwargs.get('min_ntasks') or self.min_ntasks
        max_ntasks = kwargs.get('max_ntasks') or self.max_ntasks
        cpus_per_task = kwargs.get('cpus_per_task') or self.cpus_per_task
        if min_ntasks is not None or max_ntasks is not None:
            ntasks = self._get_adaptive_ntasks(min_ntasks, max_ntasks)
            if cpus_per_task is not None:
                min_ntasks = int(min_ntasks / cpus_per_task)
                max_ntasks = int(max_ntasks / cpus_per_task)
                ntasks = int(ntasks / cpus_per_task)
            _cmd.append(f"-n {ntasks:d}")
        else:
            _cmd.append(f'-n {ntasks:d}')
        # else:
        #     _cmd.append(f"{self._get_adaptive_ntasks():d}")

        # account
        account = kwargs.get('account', self.account)
        if account is not None:
            _cmd.append(f'--account={account}')

        # job_name
        job_name = kwargs.get('job_name', self.job_name)
        if job_name is not None:
            _cmd.append(f'--job-name={job_name}')

        # partition
        partition = kwargs.get('partition', self.partition)
        if partition is not None:
            _cmd.append(f'--partition={partition}')

        # walltime
        walltime = kwargs.get('walltime', self.walltime)
        if walltime is not None:
            _cmd.append(f'--time={walltime}')

        # cpus_per_task
        cpus_per_task = kwargs.get('cpus_per_task', self.cpus_per_task)
        if cpus_per_task is not None:
            _cmd.append(f'--cpus-per-task={cpus_per_task:d}')

        # mail_type
        mail_type = kwargs.get('mail_type', self.mail_type)
        if mail_type is not None:
            _cmd.append(f'--mail-type={mail_type}')

        # mail_user
        mail_user = kwargs.get('mail_user', self.mail_user)
        if mail_user is not None:
            _cmd.append(f'--mail-user={mail_user}')

        # log_filename
        log_filename = kwargs.get('log_filename', self.log_filename)
        if log_filename is not None:
            _cmd.append(f'--log-filename={log_filename}')

        # begin
        begin = kwargs.get('begin', self.begin)
        if begin is not None:
            _cmd.append(f'--begin={begin}')
        return _cmd

    def _get_srun_command(self, cmd, **kwargs):
        srun_cmd = ['srun']
        srun_cmd.extend(self._get_cli_slurm_opts(**kwargs))
        if isinstance(cmd, str):
            srun_cmd.append(cmd)
        else:
            srun_cmd.extend(cmd)
        return srun_cmd

    def _get_sbatch_command(self, cmd, cache_directory=None, **kwargs):
        sbatch_cmd = [self.job_cls.submit_command, '--wait']
        self._append_slurm_opts_to_cmd(sbatch_cmd, **kwargs)
        sbatch_cmd.append(f'{self._write_job_script(cmd, cache_directory).resolve()}')
        return sbatch_cmd

    def _get_idle_core_count(self):
        sinfo_data = subprocess.run(['sinfo', '-N', '-l'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        # Split the data into lines
        lines = sinfo_data.split('\n')[1:]
        
        # Parse the header to get the index of each field
        header = lines[0].split()
        state_idx = header.index('STATE')
        cpus_idx = header.index('CPUS')

        # Initialize a counter
        idle_cores = 0

        # Parse each line (excluding the header)
        for line in lines[1:]:
            # Skip empty lines
            if not line.strip():
                continue

            # Split the line into fields
            fields = line.split()

            # Check if the node is idle
            if fields[state_idx].lower() == 'idle':
                # Add the number of cores to the counter
                idle_cores += int(fields[cpus_idx])
        return idle_cores

    def _get_adaptive_ntasks(self, min_ntasks, max_ntasks):
        idle_core_count = self._get_idle_core_count()

        if min_ntasks <= idle_core_count <= max_ntasks:
            return idle_core_count
        # if self.ntasks is None and self.cpus_per_task is not None:
        #     return self.cpus_per_task
        # cpus_per_task = self.cpus_per_task or 1
        # if self.ntasks is not None:
        #     return self.ntasks
        logger.info(f'Cluster is busy, waiting for more cores to become available {min_ntasks=} {max_ntasks=} {idle_core_count=} {self.ntasks=} {self.cpus_per_task=}...')
        while not (min_ntasks <= idle_core_count):
            sleep(5)
            idle_core_count = self._get_idle_core_count()
        return np.min([idle_core_count, max_ntasks])


