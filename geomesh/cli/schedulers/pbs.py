import asyncio
import subprocess

from .base import BaseCluster


class PBSCluster(BaseCluster):
    def __init__(self, pbs_file='pbs_script.sh'):
        super().__init__()
        self.pbs_file = pbs_file

    def job_script(self, command):
        return """#!/bin/bash
        #PBS -N geom-build
        #PBS -l nodes=1:ppn=1
        #PBS -l walltime=00:10:00
        #PBS -l mem=1gb
        #PBS -q short
        #PBS -j oe
        #PBS -o /home/username/geom-build.log
        #PBS -m abe
        #PBS -M
        {command}
        """.replace('        ', '')

    async def submit(self, command):
        with open(self.pbs_file, 'w') as f:
            f.write(self.job_script(command))
        output = subprocess.check_output(['qsub', self.pbs_file])
        job_id = output.decode().strip().split('.')[0]
        self.jobs.append(job_id)

    async def wait(self):
        await asyncio.wait([self.check_job_status(job) for job in self.jobs])

    async def check_job_status(self, job_id):
        while True:
            output = subprocess.check_output(['qstat', job_id])
            if job_id.encode() not in output:
                break
            await asyncio.sleep(5)


