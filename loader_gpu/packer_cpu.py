import numpy as np
from typing import List
from .models import Item, Placement
from .config import MAX_LAYERS, SUPPORT_RATIO, GRID_STEP

def _support_ok(x,y,z,L,W, placed: List[Placement]):
    if abs(z) < 1e-9: return True
    need = L*W*SUPPORT_RATIO; acc = 0.0
    for q in placed:
        top = q.z + q.H
        if abs(top - z) > 1e-6: continue
        ox, oy = max(x, q.x), max(y, q.y)
        ex, ey = min(x+L, q.x+q.L), min(y+W, q.y+q.W)
        if ex>ox and ey>oy:
            acc += (ex-ox)*(ey-oy)
            if acc + 1e-9 >= need: return True
    return acc + 1e-9 >= need

def _collides(x,y,z,L,W,H, placed: List[Placement]):
    for p in placed:
        if (x < p.x + p.L and x + L > p.x and
            y < p.y + p.W and y + W > p.y and
            z < p.z + p.H and z + H > p.z):
            return True
    return False

def pack(truck, flags, items: List[Item]):
    items_sorted = sorted(items, key=lambda i: (
        -i.drop_order, -i.stack_limit, -(i.L*i.W), -i.H
    ))
    placed: List[Placement] = []
    total_weight = 0.0
    surfaces = [{"z":0.0, "layer":1}]

    for it in items_sorted:
        if flags.max_payload and total_weight + it.weight > truck.payload_kg + 1e-6:
            continue
        orientations = [(it.L, it.W)]
        if flags.orientation_allowed and it.can_rotate:
            orientations.append((it.W, it.L))
        placed_flag = False
        for surf in sorted(surfaces, key=lambda s:(s["layer"], s["z"])):
            if surf["layer"] > MAX_LAYERS: continue
            for (L,W) in orientations:
                if L>truck.L or W>truck.W or it.H > truck.H - surf["z"]: continue
                xs = np.arange(0, truck.L - L + 1e-9, GRID_STEP)
                ys = np.arange(0, truck.W - W + 1e-9, GRID_STEP)
                for x in xs:
                    for y in ys:
                        if _collides(x,y,surf["z"],L,W,it.H, placed): continue
                        if not _support_ok(x,y,surf["z"],L,W, placed): continue
                        if surf["z"] > 0:
                            ok=True
                            for q in placed:
                                if abs(q.z+q.H - surf["z"]) < 1e-6:
                                    ox, oy = max(x,q.x), max(y,q.y)
                                    ex, ey = min(x+L,q.x+q.L), min(y+W,q.y+q.W)
                                    if ex>ox and ey>oy:
                                        if q.fragile==1: ok=False; break
                            if not ok: continue
                        p = Placement(it.id, x,y, surf["z"], L,W,it.H, it.weight, it.drop_order, it.fragile, it.stack_limit)
                        placed.append(p)
                        total_weight += it.weight
                        surfaces.append({"z": p.z + p.H, "layer": surf["layer"]+1})
                        placed_flag = True; break
                    if placed_flag: break
                if placed_flag: break
            if placed_flag: break
    return placed, total_weight
