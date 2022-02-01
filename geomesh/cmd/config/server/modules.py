from typing import Union, List


class ModulesConfig:
    
    def __init__(
            self,
            load: Union[str, List[str]],
            modulepath: str = None,
            moduleinit: str = None
    ):
        self.loads = load if isinstance(load, str) else ' '.join(load)
        self.modulepath = modulepath
        self.moduleinit = moduleinit
