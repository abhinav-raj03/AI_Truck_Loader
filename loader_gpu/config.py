from dataclasses import dataclass

@dataclass
# class TruckSpec:
#     name: str = "Tata LPT 712"
#     L: float = 5.218
#     W: float = 1.962
#     H: float = 1.812
#     payload_kg: float = 3800.0

class TruckSpec:
    name: str = "20 ft Eicher Box"
    L: float = 6.32   # meters
    W: float = 2.15   # meters (two lanes ≈ 1.065 m each minus small margin)
    H: float = 2.25   # meters
    payload_kg: float = 8500.0

@dataclass
class Flags:
    max_payload: bool = True
    axle_balance: bool = False
    stacking_fragile: bool = True
    delivery_sequence: bool = True
    orientation_allowed: bool = True

MAX_LAYERS: int = 8
SUPPORT_RATIO_CARTON = 0.18
SUPPORT_RATIO_STANDARD = 0.28
SUPPORT_RATIO_HEAVY = 0.38
GRID_STEP: float = 0.05

# Minimum fraction of footprint that must be supported by items beneath (to avoid hanging)
SUPPORT_MIN_FRACTION: float = 0.35

ALPHA_VOL: int = 10
BETA_WT: int = 1

GA_POP: int = 64
GA_GEN: int = 20
