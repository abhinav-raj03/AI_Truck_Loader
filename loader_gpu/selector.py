from ortools.sat.python import cp_model
from .config import ALPHA_VOL, BETA_WT

def select_subset(items, truck, max_keep=180):
    scored = []
    lane_w = truck.W / 2.0 - 0.01

    for it in items:
        # footprint score: prefer items that fit lane width nicely
        footprint_eff = min(it.W, lane_w) * min(it.L, truck.L)

        # stacking score: prefer items that have moderate height
        height_eff = 1.0 / (1.0 + abs(it.H - 0.45))  # peak around 0.45m

        # weight penalty to avoid overweight bottom-layer dominance
        weight_eff = 1.0 / (1.0 + it.weight / 40.0)

        # final score (tuned from real load planning heuristics)
        score = (
            (it.L * it.W * it.H) * 1.0 +   # volume
            footprint_eff * 0.8 +
            height_eff * 0.6 +
            weight_eff * 0.4 -
            it.fragile * 0.3
        )

        scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)

    # keep top N best candidates
    return [x[1] for x in scored[:max_keep]]
