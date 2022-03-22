import logging
from multiprocessing import cpu_count
from typing import Union

from .modules import ModulesConfig


logger = logging.getLogger(__name__)

class ServerConfig:

    def __init__(
            self,
            nprocs: int = -1,
            modules: Union[ModulesConfig, dict] = None,
    ):
        self.nprocs = nprocs
        self.modules = modules
        
    @property
    def nprocs(self):
        return self._nprocs
    
    @nprocs.setter
    def nprocs(self, nprocs: int):
        
        nprocs = cpu_count() if nprocs in [-1, None] else int(nprocs)

        if not nprocs > 0:
            raise ValueError("Argument `nprocs` must be -1 or >= 1.")
        
        self._nprocs = nprocs

    @property
    def modules(self):
        return self._modules
    
    @modules.setter
    def modules(self, modules: Union[None, str, dict]):
        if modules is None:
            pass
        elif isinstance(modules, str):
            modules = ModulesConfig(modules)
        elif isinstance(modules, dict):
            modules = ModulesConfig(**modules)
        elif isinstance(modules, ModulesConfig):
            self._modules = modules
        else:
            raise ValueError(
                f'Argument modules must be of type {ModulesConfig}, '
                f'dict, or None, not type {type(modules)}.'
            )
        self._modules = modules


    # def __call__(self, driver, tail_mirror: bool = False):
    #     self.run(driver, tail_mirror)

    # def run(self, driver, tail_mirror=False):
    #     if platform.system() == 'Windows':
    #         raise ValueError('Running on Windows is currently not supported.')
    #     else:
    #         if self.slurm is not None:
    #             self.slurm(self)
    #         else:
    #             self._posix_exec(driver, tail_mirror)
    
    # def _posix_exec(self, driver, tail_mirror: bool = False):
    #     env = self.env.copy()
    #     if self.modules is not None:
    #         if self.modules.modulepath is not None:
    #             env.update({'MODULEPATH': self.modules.modulepath})
    #     child = pexpect.spawn(
    #         self.shell,
    #         timeout=self.timeout,
    #         cwd=driver.outdir,
    #         env=env,
    #         encoding='utf-8'
    #     )
    #     # child.logfile = sys.stdout
    #     if self.modules is not None:
    #         if self.modules.moduleinit is not None:
    #             child.sendline(f'source {self.modules.moduleinit}')
    #         child.sendline(f'module load {self.modules.loads}')

    #     child.sendline(f'mpiexec -n {self.nprocs} {self.binary} ; exit')
        
    #     def tail_mirrorfile():
    #         for line in sh.tail("-f", driver.outdir / 'outputs/mirror.out', _iter=True):
    #             print(line, end='')



    #     if tail_mirror is True:
    #         mirror_tail = Process(target=tail_mirrorfile)
    #         mirror_tail.start()

    #     def sigint_mirrorfile(*args):
    #         if tail_mirror is True:
    #             mirror_tail.terminate()
    #         raise KeyboardInterrupt
        
    #     if tail_mirror is True:
    #         signal.signal(signal.SIGINT, sigint_mirrorfile)
    
    #     index = child.expect([
    #         pexpect.EOF,
    #         'command not found',
    #         pexpect.TIMEOUT,
    #         'The following module(s) are unknown',
    #     ])

    #     if index != 0: # Processes did not exit with OK status
    #         child.close()
    #         raise Exception(' '.join([child.before, child.after]))
    #     if tail_mirror is True:
    #         mirror_tail.terminate()
    #         signal.signal(signal.SIGINT, signal.default_int_handler)

    #     child.close()
    #     if child.exitstatus != 0:
    #         raise Exception(child.before)
        

    # @cached_property
    # def shell(self) -> str:
    #     return 'bash'
        
    # @cached_property
    # def env(self) -> dict:
    #     return {}
    
    # @cached_property
    # def timeout(self) -> Union[float, None]:
    #     return None
    
