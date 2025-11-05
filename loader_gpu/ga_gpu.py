import torch, random
from typing import List
from .models import Item
from .config import ALPHA_VOL, BETA_WT

def device_auto():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def items_to_tensors(items: List[Item], device):
    vol = torch.tensor([it.vol for it in items], dtype=torch.float32, device=device)
    wt  = torch.tensor([it.weight for it in items], dtype=torch.float32, device=device)
    drop= torch.tensor([it.drop_order for it in items], dtype=torch.float32, device=device)
    score = ALPHA_VOL*vol + BETA_WT*(wt/1000.0) + 0.01*drop
    return vol, wt, drop, score

@torch.no_grad()
def evaluate_population(order_idx_pop: torch.Tensor, vol, wt, cap_vol, cap_wt):
    P, N = order_idx_pop.shape
    vol_ord = vol[order_idx_pop]  # (P,N)
    wt_ord  = wt[order_idx_pop]   # (P,N)
    cvol = torch.cumsum(vol_ord, dim=1)
    cwt  = torch.cumsum(wt_ord,  dim=1)
    mask = (cvol <= cap_vol) & (cwt <= cap_wt)
    util_vol = torch.sum(vol_ord * mask, dim=1) / cap_vol
    util_wt  = torch.sum(wt_ord  * mask, dim=1) / cap_wt
    score = 0.9*util_vol + 0.1*util_wt
    return score, util_vol, util_wt

def ga_reorder(items: List[Item], truck, population=64, generations=20, seed=1234):
    dev = device_auto()
    rnd = random.Random(seed)
    N = len(items)
    if N < 4: return items
    vol, wt, drop, base = items_to_tensors(items, dev)
    cap_vol = torch.tensor(truck.L*truck.W*truck.H, dtype=torch.float32, device=dev)
    cap_wt  = torch.tensor(truck.payload_kg, dtype=torch.float32, device=dev)

    base_idx = torch.argsort(-base)
    pop = [base_idx.clone()]
    for _ in range(population-1):
        idx = base_idx.clone()
        for _ in range(max(1, N//20)):
            i = rnd.randrange(N); j = rnd.randrange(N)
            idx[i], idx[j] = idx[j], idx[i]
        pop.append(idx)
    pop = torch.stack(pop, dim=0)

    best_idx = pop[0]; best_score = -1e9
    for _ in range(generations):
        scores, _, _ = evaluate_population(pop, vol, wt, cap_vol, cap_wt)
        topk = torch.topk(scores, k=max(2, population//5))
        elites = pop[topk.indices]
        if float(topk.values[0]) > best_score:
            best_score = float(topk.values[0]); best_idx = elites[0].clone()
        new_pop = [elites[i % elites.shape[0]].clone() for i in range(elites.shape[0])]
        while len(new_pop) < population:
            p1 = elites[rnd.randrange(elites.shape[0])].clone()
            p2 = elites[rnd.randrange(elites.shape[0])].clone()
            # --- SAFE ORDER CROSSOVER (OX) ---
            a, b = sorted([rnd.randrange(N), rnd.randrange(N)])

            child = [-1] * N
            seg = p1[a:b]
            child[a:b] = seg

            used = set(seg)
            fill_values = [g for g in p2 if g not in used]

            pos = b
            for g in fill_values:
                if pos >= N:
                    pos = 0
                child[pos] = g
                pos += 1

            child = torch.tensor(child, device=dev, dtype=torch.long)

            if rnd.random() < 0.2:
                i, j = rnd.randrange(N), rnd.randrange(N)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        pop = torch.stack(new_pop, dim=0)
    return [items[int(i)] for i in best_idx.tolist()]
