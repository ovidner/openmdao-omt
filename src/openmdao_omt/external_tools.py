from dataclasses import dataclass


@dataclass
class VarMap:
    name: str
    ext_name: str = None
    shape: tuple = (1,)

    def __post_init__(self):
        if not self.ext_name:
            self.ext_name = self.name
