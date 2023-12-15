from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Union
import asyncio
import logging
import os
import signal
import subprocess
import sys

import pexpect
from asyncio import Future
from typing import List
import uuid

logger = logging.getLogger(__name__)

class BaseCluster(ABC):

    def __init__(self, semaphore=None):
        self.jobs: List[Future] = []
        self.semaphore = semaphore

    def __enter__(self):
        return self

    async def __aenter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.wait()

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._wait()


    # async def check_job_status(self, process):
    #     # Return a coroutine object that can be awaited
    #     return_code = process.returncode
    #     while return_code is None:
    #         await asyncio.sleep(0.1)
    #         return_code = process.returncode
    #     return return_code

    def _submit(self, command: List[str], **kwargs):
        task = asyncio.get_event_loop().create_task(launch_task(command, self.semaphore, **kwargs))
        self.jobs.append(task)
        return task
        # return partial(launch_pexpect, blocking_submit_command)

            # return asyncio.get_event_loop().create_task(launch_task(blocking_submit_command, self.semaphore))

        # await self.semaphore.acquire()
        # try:
            # return asyncio.create_task(launch_task(command, self.semaphore))
            # # # Run the synchronous code in a separate thread using asyncio.to_thread
            # # process = await asyncio.to_thread(subprocess.run, command, shell=True)
            # # coro = self.check_job_status(process)
            # # self.jobs.append(coro)
            # # return coro
        # # except KeyboardInterrupt:
            # # process.send_signal(signal.SIGINT)
            # # process.wait()

        # finally:
            # self.semaphore.release()

    def submit(self, command: List[str], **kwargs):
        if asyncio.get_event_loop().is_running():
            return self._submit(command, **kwargs)
        else:
            return asyncio.get_event_loop().run_until_complete(self._submit(command, **kwargs))

    async def _wait(self):
        if len(self.jobs) > 0:
            await asyncio.gather(*self.jobs)

    def wait(self):
        if asyncio.get_event_loop().is_running():
            yield from self._wait()
        else:
            asyncio.run(self._wait())

    @property
    def semaphore(self):
        return self._semaphore

    @semaphore.setter
    def semaphore(self, semaphore):
        if semaphore is None:
            semaphore = asyncio.Semaphore(100)
        if isinstance(semaphore, (float, int)):
            semaphore = asyncio.Semaphore(semaphore)
        if not isinstance(semaphore, asyncio.Semaphore):
            raise ValueError(f'Argument `semaphore` must be int, None or asyncio.Semaphore, not {semaphore}.')
        self._semaphore = semaphore



async def launch_task(
        cmd,
        semaphore=None,
        expect_pattern=None,
        index_actions=None,
        **kwargs
        ):
    semaphore = asyncio.Semaphore(1) if semaphore is None else semaphore
    if not isinstance(cmd, str):
        cmd = " ".join(cmd)
    expect_pattern = expect_pattern or [pexpect.EOF, '(core dumped)']

    def default_EOF_action(p):
        if p.exitstatus != 0:
            raise Exception(f'Failed command:\n{cmd}\n{p.before}')

    def default_core_dumped_action(p):
        default_EOF_action(p)

    index_actions = index_actions or 2*[default_EOF_action]

    spawn_kwargs = {
        'args': kwargs.get('args', []),
        'timeout': kwargs.get('timeout', None),
        'maxread': kwargs.get('maxread', 2000),
        'searchwindowsize': kwargs.get('searchwindowsize', None),
        'logfile': kwargs.get('logfile', None),
        'cwd': kwargs.get('cwd', None),
        'env': kwargs.get('env', os.environ),
        'ignore_sighup': kwargs.get('ignore_sighup', False),
        'echo': kwargs.get('echo', True),
        'preexec_fn': kwargs.get('preexec_fn', None),
        'encoding': kwargs.get('encoding', 'utf-8'),
        'codec_errors': kwargs.get('codec_errors', 'strict'),
        'dimensions': kwargs.get('dimensions', None),
        'use_poll': kwargs.get('use_poll', False)
    }

    async with semaphore:
        logger.info(f'Running command:\n{cmd}')
        with pexpect.spawn(
                cmd,
                **spawn_kwargs
                ) as p:
            p.logfile_read = sys.stdout
            index = await p.expect(expect_pattern, async_=True)
            if index_actions:
                index_actions[index](p)
    return p
