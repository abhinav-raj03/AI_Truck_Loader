from dataclasses import dataclass

@dataclass
class TruckSpec:
    name: str = "Tata LPT 712"
    L: float = 5.218
    W: float = 1.962
    H: float = 1.812
    payload_kg: float = 3800.0

@dataclass
class Flags:
    max_payload: bool = True
    axle_balance: bool = True
    axle_front_min: float = 0.40
    axle_front_max: float = 0.60
    stacking_fragile: bool = True
    delivery_sequence: bool = True
    orientation_allowed: bool = True

MAX_LAYERS: int = 3
SUPPORT_RATIO: float = 0.75
GRID_STEP: float = 0.10  # 10 cm grid

ALPHA_VOL: int = 10
BETA_WT: int = 1

GA_POP: int = 64
GA_GEN: int = 20
