import asyncio
from enum import Enum
from functools import cached_property, lru_cache
import logging
import os
import random

import cachetools.func
import pexpect
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# original_sigint_handler = signal.getsignal(signal.SIGINT)

class NodeState(Enum):
    ALLOCATED = 'alloc'
    IDLE = 'idle'
    MIXED = 'mix'
    COMPLETING = 'comp'
    
    
class SlurmLoadBalancer:
    
    def __init__(self, tasks, global_tasks):
        
        self._tasks = tasks
        self._global_tasks = global_tasks
        self._loop = asyncio.get_event_loop()
        
    def __enter__(self):
        self._load_balancing_task = self._loop.create_task(self._exec_load_balancing())
     
    
    def __exit__(self, type, value, traceback):
        self._load_balancing_task.cancel()
    
    async def _exec_load_balancing(self):
        await asyncio.sleep(0)
        while True:
            for task_data in self._tasks:
                switch_break = False
                if task_data['task'].done():
                    self._tasks.pop(self._tasks.index(task_data))
                    await asyncio.sleep(0)
                    continue
                job_status = self._get_job_state_by_id(task_data['job_id'])
                if job_status == 'PENDING':
                    reason = self._get_job_reason(task_data['job_id'])
                    # this should switch jobs whose allocation has been taken by another process
                    if 'ReqNodeNotAvail' in reason:
                        for node_name in random.sample(self.node_names, len(self.node_names)):
                            if task_data['target_node'] == node_name:
                                continue
                            if self._get_node_state(node_name) == NodeState.IDLE:
                                await self._switch_job_to_node_another_node(task_data, node_name)
                                switch_break = True
                                break
                    elif 'Dependency' in reason:
                        for node_name in random.sample(self.node_names, len(self.node_names)):
                            if task_data['target_node'] == node_name:
                                continue
                            if self._get_node_state(node_name) == NodeState.IDLE:
                                print(f'Could potentially switch {task_data["job_id"]} to {node_name}')
                                switch_break = True
                                break
                    if switch_break:
                        await asyncio.sleep(0)
                        break
                await asyncio.sleep(0)                 
            await asyncio.sleep(0)

    def _get_job_state_by_id(self, job_id):
        with pexpect.spawn(' '.join([
                'squeue',
                '-O',
                'state',
                '-j',
                f'{job_id}'
            ]),
                encoding='utf-8',
                timeout=None,
                # cwd=output_directory
        ) as p:
            p.expect(pexpect.EOF)
        return p.before.split('\n')[1].strip()

        
    def _get_job_reason(self, job_id):
        with pexpect.spawn(
            ' '.join([
                'squeue',
                '-o',
                r'%r',
                '-j',
                f'{job_id}',
            ]),
            encoding='utf-8',
            timeout=None,
        ) as p:
            p.expect(pexpect.EOF)
        return p.before.split('\n')[1]
    
    def _get_node_state(self, node_name):
        with pexpect.spawn(' '.join([
                'sinfo',
                '-n',
                f'{node_name}'
            ]),
                encoding='utf-8',
                timeout=None,
                # cwd=output_directory
        ) as p:
            p.expect(pexpect.EOF)
        return NodeState(p.before.split('\n')[1].split()[4])
        
        
    async def _switch_job_to_node_another_node(self, task_data, new_node):

        print(f'Switching job {task_data["job_id"]} from {task_data["target_node"]} to {new_node}')
        
        new_name = task_data['job_name'].split('_target:')[0] + f'_target:{new_node}'

        srun_cmd = task_data['srun_cmd']
        srun_cmd.replace(task_data['job_name'], new_name).replace(task_data['target_node'], new_node)
        
        srun_task = self.loop.create_task(self._await_pexpect(srun_cmd)) #, callback))
        new_job_id = await self._get_job_id_by_job_name(new_name, task_data)           
        # safer to submit, update task_data, then cancel job

        # old id debug
        old_job_id = task_data['job_id']
        print(f'old id debug: scontrol show job {task_data["job_id"]}')

        task_data.update({
            'job_name': new_name,
            'target_node': new_node,
            'task': srun_task,
            'job_id': new_job_id,
            'srun_cmd': srun_cmd
        })
        cmd = [
            'scancel',
            f'{old_job_id}',
        ]
        with pexpect.spawn(
                ' '.join(cmd),
                encoding='utf-8',
                timeout=None,
        ) as p:
            p.expect(pexpect.EOF)
        # new id debug
        print(f'new id debug: scontrol show job {task_data["job_id"]}')

        # output of cmd
        print(p.before)
        

    
    
    

class SlurmManager:
    
    def __init__(self, max_tasks_per_node: int = 1, exclude=None, semaphore_value=float('inf')):
        self._tasks = []
        self._max_tasks_per_node = MaxTasksPerNode(max_tasks_per_node, exclude)
        # self._job_ids_by_name_name = {}
        self.loop = asyncio.get_event_loop()
        self._global_tasks = []
        #https://stackoverflow.com/a/52391791/7432462
        self.semaphore = asyncio.Semaphore(semaphore_value)
        # for signame in ('SIGINT', 'SIGTERM'):
        #     self.loop.add_signal_handler(
        #         getattr(signal, signame),
        #         self._close_srun_tasks
        #     )
        # signal.signal(signal.SIGINT, self._close_srun_tasks)
        # self.loop.add_signal_handler(signal.SIGINT, self._close_srun_tasks)
        
    # def __delete__(self):
    #     signal.signal(signal.SIGINT, original_sigint_handler)
        
    def __enter__(self):
        # start load balancer async def _update_load_balancing(self, task_data):
        # self._load_balancer = SlurmLoadBalancer(self._tasks, self._global_tasks)
        return self

    def __exit__(self, type, value, traceback):
        with SlurmLoadBalancer(self._tasks, self._global_tasks):
            self.results = self.loop.run_until_complete(asyncio.gather(*self._global_tasks))
            
    def srun(self, cmd, cpus_per_task=None, **kwargs):
        
        if isinstance(cmd, str):
            cmd = [cmd]
        
        srun_cmd = ['srun']
        
        if cpus_per_task is not None:
            srun_cmd.append(f'--cpus-per-task={int(cpus_per_task)}')
  
        job_name, target_node =  self._get_initial_job_name_target()
        
        srun_cmd.append(f'--job-name={job_name}')
        srun_cmd.append('--dependency=singleton')
        
        # trying to make it work under sbatch%
        # if os.environ['SLURM_JOB_NODELIST'] is None:
        #     srun_cmd.append(f'--nodelist={target_node}')
        srun_cmd.append(f'--nodelist={target_node}')
        srun_cmd.extend(cmd)
        
        self._global_tasks.append(self.loop.create_task(
            self._enqueue_srun_cmd(srun_cmd, job_name, target_node)
            ))
        
    def _get_initial_job_name_target(self):
        node_data = self._get_current_node_state()
        all_allocated = all(list(map(lambda x: x == NodeState.ALLOCATED, node_data.values())))
        if all_allocated:
            return self._max_tasks_per_node.next()
        cjob_name, ctarget_node = self._max_tasks_per_node.next()
        while node_data[ctarget_node] == NodeState.ALLOCATED:
            cjob_name, ctarget_node = self._max_tasks_per_node.next()
        return cjob_name, ctarget_node

    @cachetools.func.ttl_cache(ttl=5)
    def _get_current_node_state(self):
        node_data = {}
        with pexpect.spawn(
                'sinfo -N',
                encoding='utf-8',
                timeout=None,
        ) as p:
            p.expect(pexpect.EOF)
        for i, line in enumerate(p.before.split('\n')):
            if i in [0]:
                continue
            line = line.split()
            if len(line) > 0:
                node_data.update({
                    line[0].strip(): NodeState(line[-1].strip())
                })
        return node_data

            
    # def _close_srun_tasks(self):
    #     for task in self._tasks:
    #         self._close_srun_task(task['job_id'])
        
    # def _close_srun_task(self, job_id):
    #     with pexpect.spawn(
    #         f'scancel {job_id}',
    #         encoding='utf-8',
    #         timeout=None,
    #         # cwd=output_directory
    #     ) as p:
    #         p.expect(pexpect.EOF)
    
    # def __delete__(self):
    #     tasks = []
    #     for task in self._tasks:
    #         tasks.append(self.loop.create_task(self._close_srun_task(task['job_id'])))
    #     self.loop.run_until_complete(asyncio.gather(*tasks))
        

    async def _enqueue_srun_cmd(self, srun_cmd, job_name, target_node):
        async with self.semaphore:
            srun_task = self.loop.create_task(self._await_pexpect(srun_cmd)) #, callback))
            await asyncio.sleep(0)
            job_id = await self._get_job_id_by_job_name(srun_task, job_name)
            task_data = {
                'job_name': job_name,
                'target_node': target_node,
                'task': srun_task,
                'job_id': job_id,
                'srun_cmd': ' '.join(srun_cmd),
            }
            self._tasks.append(task_data)
            # await self._update_load_balancing(task_data)
            # task_data_index = len(self._tasks)
            
            while not task_data['task'].done():
                await asyncio.sleep(0)

            return await task_data['task']
    
    # async def _update_node_state():
    #     while True:
            
        
            
    async def _get_job_id_by_job_name(self, srun_task, job_name):
        cmd = [
            'sacct',
            '-n',
            '-X',
            '--format',
            'jobid',
            '--name',
            f'{job_name}'
        ]
        async def _await_get_job_id_fail_wrapper():
            with pexpect.spawn(
                ' '.join(cmd),
                encoding='utf-8',
                timeout=None,
            ) as p:
                await p.expect(pexpect.EOF, async_=True)
            used_ids = set()
            for task in self._tasks:
                used_ids.add(task['job_id'])
            possible_ids = set()
            for possible_id in p.before.split():
                possible_ids.add(possible_id)
            job_id = possible_ids - used_ids
            return job_id
        job_id = await _await_get_job_id_fail_wrapper()
        # cnt = 0
        while len(job_id) != 1:
            job_id = await _await_get_job_id_fail_wrapper()
            if srun_task.done():
                try:
                    return list(job_id).pop()
                except:
                    return
            # cnt+=1
            # if cnt == 100:
            #     print(f'warning: have retried to get id of job_name: {job_name} more that 100 times.')
        return list(job_id).pop()
        
    def _get_cmd_job_name(self, cmd):
        return cmd.split('--job-name=')[-1].split()[0]
        
    def _get_cmd_target_node(self, cmd):
        return self._get_cmd_job_name(cmd).split('_target:')[1]
            
        
    async def _await_pexpect(self, cmd):
        if not isinstance(cmd, str):
            cmd = ' '.join(list(cmd))
        # async with self.semaphore:
        with pexpect.spawn(
            cmd,
            encoding='utf-8',
            timeout=None,       
        ) as p:
            await p.expect(pexpect.EOF, async_=True)
        if p.exitstatus != 0:
            raise Exception(f'Failed command:\n{cmd}\n{p.before}')   
        return p   
        
    def _get_node_jobs(self, node_name):
        node_pending_jobs = []
        for task in self._tasks:
            if task['target_node'] == node_name:
                node_pending_jobs.append(task['job_id'])
        return node_pending_jobs
        
    # async def _update_load_balancing(self, task_data):
    #     # print()
    #     # break_loop = False

        # job_status = self._get_job_state_by_id(task_data['job_id'])
        # if job_status == 'PENDING':
        #     reason = self._get_job_reason(task_data['job_id'])
        #     # this should switch jobs whose allocation has been taken by another process
        #     if 'ReqNodeNotAvail' in reason:
        #         for node_name in random.sample(self.node_names, len(self.node_names)):
        #             if task_data['target_node'] == node_name:
        #                 continue
        #             if self._get_node_state(node_name) == NodeState.IDLE:
        #                 await self._switch_job_to_node_another_node(task_data, node_name)
        #                 break
    #         # this should switch jobs 
    #         # elif 'Dependency' in reason:
    #         # this 
    #         #     if 
    #         #     for node_name in random.sample(self.node_names, len(self.node_names)):
    #         #         if task_data['target_node'] == node_name:
    #         #             continue
    #         #         if self._get_node_state(node_name) == NodeState.IDLE:
    #         #             await self._switch_job_to_node_another_node(task_data, node_name)
    #         #             break
                
                
                    
                    
            
    #         # if 
    #                 # node_pending_jobs = self._get_node_jobs(node_name)
    #             # node_state = self._get_node_state(node_name) 
    #         # print(task_data['job_id'])  # , task_data['job_name'], node_name, node_pending_jobs, node_state, job_status, print(node_state == NodeState.IDLE))
    #         # logger.info(f'{node_name} has {len(node_pending_jobs)} jobs and it\'s state is: {node_state}')
            
    #         # if node_state == NodeState.IDLE and \
    #         #     len(node_pending_jobs) < self._max_tasks_per_node._max_tasks and \
    #         #         job_status == 'PENDING':
    #         #     print(f'Would be switching {task_data} to {node_name}')
    #         #     # await asyncio.sleep(5)
    #         #     break
    #     await asyncio.sleep(0)

    

        # out = subprocess.check_output([
        #     'squeue',
        #     '-O',
        #     'state',
        #     '-j',
        #     f'{job_id}'
        # ], close_fds=True, bufsize=-1, universal_newlines=True)
        # return out.split('\n')[1]

             
       
    @property
    def node_names(self):
        return self._max_tasks_per_node.nodelist.names


class NodeNames:
    
    def __init__(self, exclude=None,
                #  exclude_allocated=False
                 ):
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = exclude.split(',')
        else:
            exclude = []
        with pexpect.spawn(
                'sinfo -N',
                encoding='utf-8',
                timeout=None,
                # cwd=output_directory
        ) as p:
            p.expect(pexpect.EOF)
        for line in p.before.split('\n'):
            line = line.split()
            if len(line) == 0:
                continue
            status = line[-1].strip('*')
            if status not in ["STATE", "down", 'drain', "drained", "draining", "fail", "failing", "future", "maint", "perfctrs", "planned", "power_down", "power_up", "reserved", "resv", "unknown"]:
                # if status == "alloc" and exclude_allocated is True:
                #     continue
                name = line[0]
                if name not in exclude:
                    self.names.append(name)
        self._cnt = 0
        
    def __iter__(self):
        return self

    def next(self):
        return next(self)

    def __next__(self):
        node_name = self.names[self._cnt]
        if node_name is None:
            return
        self._cnt += 1
        if self._cnt == len(self.names):
            self._cnt = 0
        return node_name
        
    @cached_property
    def names(self):
        return []

class MaxTasksPerNode:

    def __init__(self, max_tasks=None, exclude=None,
                #  exclude_allocated=False
                 ):
        self._cnt = 0
        if max_tasks is None:
            self.names = [None]
        else:
            assert isinstance(max_tasks, int), f'max_tasks must be an int>0 or None but got {max_tasks}'
            assert max_tasks>0, f'max_tasks must be an int>0 or None but got {max_tasks}'
            self._max_tasks = max_tasks
            self.names = [uuid.uuid4().hex for i in range(max_tasks)]
        self.nodelist = NodeNames(exclude
                                #   , exclude_allocated
                                  )

    def __iter__(self):
        return self

    def __next__(self):
        job_name = self.names[self._cnt]
        if job_name is None:
            return
        self._cnt += 1
        if self._cnt == self._max_tasks:
            self._cnt = 0
        nodename = self.nodelist.next()
        job_name += f'_target:{nodename}'
        return (job_name, nodename)

    def next(self):
        return next(self)
