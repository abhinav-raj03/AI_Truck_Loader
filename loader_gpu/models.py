from dataclasses import dataclass, field

@dataclass
class Item:
    id: int
    L: float; W: float; H: float
    weight: float
    fragile: int
    stack_limit: int
    can_rotate: int
    drop_order: int
    vol: float = field(init=False)
    def __post_init__(self): self.vol = self.L*self.W*self.H

@dataclass
class Placement:
    id: int
    x: float; y: float; z: float
    L: float; W: float; H: float
    weight: float; drop_order: int
    fragile: int; stack_limit: int
