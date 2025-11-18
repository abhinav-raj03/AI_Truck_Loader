from typing import List
from .models import Item, Placement
from .config import (
    MAX_LAYERS,
    SUPPORT_RATIO_CARTON,
    SUPPORT_RATIO_STANDARD,
    SUPPORT_RATIO_HEAVY,
    SUPPORT_MIN_FRACTION,
)

EPS = 1e-9

def _orientations(it: Item):
    return [(it.L, it.W), (it.W, it.L)] if it.can_rotate else [(it.L, it.W)]

def pack(truck, flags, items: List[Item]):
    lane_w = (truck.W / 2.0) - 0.01
    # dynamic max layers: bounded by config and by smallest item height to avoid too many tiny layers
    min_item_h = min((it.H for it in items), default=0.1)
    max_layers = max(1, min(int(MAX_LAYERS), max(1, int(truck.H / max(min_item_h, 0.05)))))

    # Prefer smaller heights first so packer can form multiple thin layers.
    # Tie-break by larger footprint to fill area within each thin layer.
    items_sorted = sorted(
        items,
        key=lambda i: (i.H, -(i.L * i.W), -i.weight, -i.stack_limit)
    )

    placed: List[Placement] = []
    total_weight = 0.0
    z = 0.0
    layers_done = 0

    def can_add_weight(w):
        return (not flags.max_payload) or (total_weight + w <= truck.payload_kg + EPS)

    while layers_done < max_layers and z + EPS < truck.H:
        # For this layer, try several candidate layer heights and pick the one
        # that yields the most area packed (more area -> better chance to stack above).
        remaining_items = [it for it in items_sorted if not any(p.id == it.id for p in placed) and z + it.H <= truck.H + EPS]
        if not remaining_items:
            break

        # Collect candidate heights: most frequent heights + quantiles
        height_counts = {}
        for it in remaining_items:
            h = round(it.H, 6)
            height_counts[h] = height_counts.get(h, 0) + 1
        # sort heights by frequency then numeric
        freq_sorted_heights = sorted(height_counts.keys(), key=lambda h: (-height_counts[h], h))
        candidates = []
        # take top frequent heights
        for h in freq_sorted_heights[:6]:
            candidates.append(h)
        # add quantile-based heights from remaining set (include small quantiles to favor thin layers)
        hvals = sorted(set([it.H for it in remaining_items]))
        if hvals:
            import math
            for q in (0.1, 0.25, 0.5, 0.75, 1.0):
                idx = min(len(hvals)-1, max(0, int(math.floor(q * len(hvals))) ))
                candidates.append(hvals[idx])
            # include the minimal item height explicitly
            candidates.append(hvals[0])

        # deduplicate and clamp candidates to remaining height
        candidates = sorted(list({min(c, truck.H - z) for c in candidates if c > EPS}))[:10]
        if not candidates:
            break

        # helper to simulate packing for a candidate height (does not modify global placed)
        def merge_free_rects(rects):
            # simple merge: if two rects share edge and have same extent in the other axis, merge into one
            changed = True
            rects = [r.copy() for r in rects]
            while changed:
                changed = False
                out = []
                used = [False]*len(rects)
                for i,a in enumerate(rects):
                    if used[i]: continue
                    merged = a.copy()
                    for j,b in enumerate(rects):
                        if i==j or used[j]: continue
                        # merge horizontally if y and W equal and x adjacent
                        if abs(merged['y'] - b['y']) < EPS and abs(merged['W'] - b['W']) < EPS:
                            if abs(merged['x'] + merged['L'] - b['x']) < EPS:
                                merged['L'] += b['L']; used[j]=True; changed=True
                            elif abs(b['x'] + b['L'] - merged['x']) < EPS:
                                merged['L'] += b['L']; merged['x'] = b['x']; used[j]=True; changed=True
                        # merge vertically if x and L equal and y adjacent
                        if abs(merged['x'] - b['x']) < EPS and abs(merged['L'] - b['L']) < EPS:
                            if abs(merged['y'] + merged['W'] - b['y']) < EPS:
                                merged['W'] += b['W']; used[j]=True; changed=True
                            elif abs(b['y'] + b['W'] - merged['y']) < EPS:
                                merged['W'] += b['W']; merged['y'] = b['y']; used[j]=True; changed=True
                    out.append(merged); used[i]=True
                rects = out
            return rects

        def simulate_layer(candidate_h):
            sim_free = [{"x": 0.0, "y": 0.0, "L": truck.L, "W": truck.W}]
            sim_placed = []
            sim_area = 0.0
            sim_layer_h = 0.0
            # iterate through remaining items in the same sorted order
            for it in remaining_items:
                if it.H > candidate_h + EPS:
                    continue
                # same checks as real packer
                if not can_add_weight(it.weight):
                    continue
                orientations = _orientations(it) if getattr(flags, 'orientation_allowed', True) else [(it.L, it.W)]
                for (L, W) in orientations:
                    if L > truck.L + EPS or W > truck.W + EPS:
                        continue
                    # find best-fit free rect
                    chosen_idx = None
                    best_left = None
                    for ri, r in enumerate(sim_free):
                        if L <= r["L"] + EPS and W <= r["W"] + EPS:
                            left = (r["L"] * r["W"]) - (L * W)
                            if best_left is None or left < best_left:
                                best_left = left
                                chosen_idx = ri
                    if chosen_idx is None:
                        continue
                    r = sim_free.pop(chosen_idx)
                    x0, y0 = r["x"], r["y"]
                    x1, y1 = x0 + L, y0 + W
                    # stacking checks against actual placed (previous layers)
                    if z > 0:
                        base_need_ratio = (SUPPORT_RATIO_CARTON if it.weight < 18 else SUPPORT_RATIO_STANDARD if it.weight < 70 else SUPPORT_RATIO_HEAVY)
                        # be slightly stricter on required fraction to avoid hanging items
                        required_fraction = max(base_need_ratio, SUPPORT_MIN_FRACTION, 0.30)
                        support_need = L * W * required_fraction
                        got = 0.0
                        for q in placed:
                            # accept support from items whose top is very near the target z (tolerance)
                            if abs((q.z + q.H) - z) > 0.05:
                                continue
                            ox, oy = max(x0, q.x), max(y0, q.y)
                            ex, ey = min(x1, q.x + q.L), min(y1, q.y + q.W)
                            if ex > ox and ey > oy:
                                got += (ex - ox) * (ey - oy)
                                if got + EPS >= support_need:
                                    break
                        if got + EPS < support_need:
                            # fallback: if the center point of the footprint is supported by some item below,
                            # allow only when there is at least a minimal contact area (e.g. 25% of required)
                            cx = (x0 + x1) / 2.0
                            cy = (y0 + y1) / 2.0
                            center_supported = False
                            for q in placed:
                                if abs((q.z + q.H) - z) > EPS:
                                    continue
                                if (q.x - EPS) <= cx < (q.x + q.L + EPS) and (q.y - EPS) <= cy < (q.y + q.W + EPS):
                                    center_supported = True
                                    break
                            if not center_supported or got < support_need * 0.25:
                                sim_free.append(r)
                                continue
                    # accept placement in simulation
                    sim_placed.append((it, x0, y0, candidate_h, L, W))
                    sim_area += L * W
                    sim_layer_h = max(sim_layer_h, it.H)
                    # split rects - ensure top rect uses the original rect's L (not the placed L)
                    xr = {"x": x1, "y": y0, "L": r["L"] - L, "W": r["W"]}
                    yt = {"x": x0, "y": y1, "L": r["L"], "W": r["W"] - W}
                    if xr["L"] > EPS and xr["W"] > EPS:
                        sim_free.append(xr)
                    if yt["L"] > EPS and yt["W"] > EPS:
                        sim_free.append(yt)
                    # cleanup contained rects
                    cleaned = []
                    for a in sim_free:
                        contained = False
                        for b in sim_free:
                            if a is b:
                                continue
                            if (a["x"] + a["L"] <= b["x"] + b["L"] + EPS and
                                a["y"] + a["W"] <= b["y"] + b["W"] + EPS and
                                a["x"] >= b["x"] - EPS and
                                a["y"] >= b["y"] - EPS):
                                contained = True
                                break
                        if not contained:
                            cleaned.append(a)
                    sim_free = cleaned
                    break
            # compute largest free rect area and number of fragments
            merged = merge_free_rects(sim_free)
            largest_free = 0.0
            for r in merged:
                largest_free = max(largest_free, r['L'] * r['W'])
            # also try alternate ordering (small->big) to capture different filling patterns
            # run a second pass with remaining_items reversed and pick the better of the two
            sim2_free = [{"x": 0.0, "y": 0.0, "L": truck.L, "W": truck.W}]
            sim2_placed = []
            sim2_area = 0.0
            sim2_layer_h = 0.0
            for it in reversed(remaining_items):
                if it.H > candidate_h + EPS:
                    continue
                if not can_add_weight(it.weight):
                    continue
                orientations = _orientations(it) if getattr(flags, 'orientation_allowed', True) else [(it.L, it.W)]
                for (L, W) in orientations:
                    if L > truck.L + EPS or W > truck.W + EPS:
                        continue
                    chosen_idx = None
                    best_left = None
                    for ri, r in enumerate(sim2_free):
                        if L <= r["L"] + EPS and W <= r["W"] + EPS:
                            left = (r["L"] * r["W"]) - (L * W)
                            if best_left is None or left < best_left:
                                best_left = left
                                chosen_idx = ri
                    if chosen_idx is None:
                        continue
                    r = sim2_free.pop(chosen_idx)
                    x0, y0 = r["x"], r["y"]
                    x1, y1 = x0 + L, y0 + W
                    if z > 0:
                        base_need_ratio = (SUPPORT_RATIO_CARTON if it.weight < 18 else SUPPORT_RATIO_STANDARD if it.weight < 70 else SUPPORT_RATIO_HEAVY)
                        required_fraction = max(base_need_ratio, SUPPORT_MIN_FRACTION, 0.30)
                        support_need = L * W * required_fraction
                        got = 0.0
                        for q in placed:
                            if abs((q.z + q.H) - z) > 0.05:
                                continue
                            ox, oy = max(x0, q.x), max(y0, q.y)
                            ex, ey = min(x1, q.x + q.L), min(y1, q.y + q.W)
                            if ex > ox and ey > oy:
                                got += (ex - ox) * (ey - oy)
                                if got + EPS >= support_need:
                                    break
                        if got + EPS < support_need:
                            cx = (x0 + x1) / 2.0
                            cy = (y0 + y1) / 2.0
                            center_supported = False
                            for q in placed:
                                if abs((q.z + q.H) - z) > EPS:
                                    continue
                                if (q.x - EPS) <= cx < (q.x + q.L + EPS) and (q.y - EPS) <= cy < (q.y + q.W + EPS):
                                    center_supported = True
                                    break
                            if not center_supported or got < support_need * 0.25:
                                sim2_free.append(r)
                                continue
                    sim2_placed.append((it, x0, y0, candidate_h, L, W))
                    sim2_area += L * W
                    sim2_layer_h = max(sim2_layer_h, it.H)
                    xr = {"x": x1, "y": y0, "L": r["L"] - L, "W": r["W"]}
                    yt = {"x": x0, "y": y1, "L": r["L"], "W": r["W"] - W}
                    if xr["L"] > EPS and xr["W"] > EPS:
                        sim2_free.append(xr)
                    if yt["L"] > EPS and yt["W"] > EPS:
                        sim2_free.append(yt)
                    # cleanup contained rects
                    cleaned = []
                    for a in sim2_free:
                        contained = False
                        for b in sim2_free:
                            if a is b:
                                continue
                            if (a["x"] + a["L"] <= b["x"] + b["L"] + EPS and
                                a["y"] + a["W"] <= b["y"] + b["W"] + EPS and
                                a["x"] >= b["x"] - EPS and
                                a["y"] >= b["y"] - EPS):
                                contained = True
                                break
                        if not contained:
                            cleaned.append(a)
                    sim2_free = cleaned
                    break
            merged2 = merge_free_rects(sim2_free)
            largest2 = 0.0
            for r in merged2:
                largest2 = max(largest2, r['L'] * r['W'])
            # pick the better simulation by score (area density + free-area bonus)
            floor_area = truck.L * truck.W
            score1 = (sim_area / max(sim_layer_h, 1e-6)) + 0.5 * (largest_free / floor_area) - 0.05 * (len(sim_free))
            score2 = (sim2_area / max(sim2_layer_h, 1e-6)) + 0.5 * (largest2 / floor_area) - 0.05 * (len(sim2_free))
            if score2 > score1:
                return sim2_area, sim2_placed, sim2_layer_h, sim2_free, largest2, len(sim2_free)
            return sim_area, sim_placed, sim_layer_h, sim_free, largest_free, len(sim_free)

        # Evaluate candidates and pick the one with maximum area density (area / height) to encourage thin layers.
        best_candidate = None
        best_score = -1.0
        best_sim = None
        floor_area = truck.L * truck.W
        for c in candidates:
            area, sim_placed, sim_layer_h, sim_free, largest_free, num_free = simulate_layer(c)
            if sim_layer_h <= EPS:
                continue
            density = area / sim_layer_h
            # score: density + bonus for largest_free area, penalty for fragmentation
            score = density + 0.5 * (largest_free / floor_area) - 0.05 * (num_free)
            # prefer higher score; tie-breaker larger area
            if score > best_score + EPS or (abs(score - best_score) <= EPS and (best_sim is None or area > best_sim[0])):
                best_score = score; best_candidate = c; best_sim = (area, sim_placed, sim_layer_h, sim_free, largest_free, num_free)

        # If nothing placed in any candidate, break
        if best_sim is None or best_sim[0] <= EPS:
            break

        # Commit best_sim placements to actual placed list and update free_rects accordingly
        _, sim_placed, sim_layer_h, sim_free, largest_free, num_free = best_sim
        # reuse the sim_free as the starting free rects for next iteration (converted to real list)
        free_rects = merge_free_rects(sim_free)
        layer_h_max = sim_layer_h
        placed_this_layer = 0
        for it, x0, y0, ch, L, W in sim_placed:
            p = Placement(it.id, x0, y0, z, L, W, it.H, it.weight, it.drop_order, it.fragile, it.stack_limit)
            placed.append(p)
            total_weight += it.weight
            placed_this_layer += 1

        print(f"[LAYER] z={z:.2f}m  placed={placed_this_layer}  layer_h={layer_h_max:.2f}m")
        if placed_this_layer == 0 or layer_h_max <= EPS:
            break

        z += layer_h_max
        layers_done += 1

    # allow fallback shelf-based packing below when utilization is poor
    # (do NOT return here)

    # Fallback shelf-based packing: if we ended up with <=1 layer or very low utilization,
    # try fixed shelf heights to force multi-layer packing and pick the best result.
    vol_used = sum(p.L * p.W * p.H for p in placed)
    vol_total = truck.L * truck.W * truck.H
    vol_util = vol_used / (vol_total + EPS)

    # condition to try shelf fallback: only one layer OR utilization low (<20%)
    if len({round(p.z,6) for p in placed}) <= 1 or vol_util < 0.20:
        def shelf_pack_with_height(shelf_h):
            sim_placed = []
            sim_weight = 0.0
            z0 = 0.0
            shelves = int(max(1, min(int(truck.H // shelf_h), int(MAX_LAYERS))))
            for s in range(shelves):
                free_rects = [{"x":0.0, "y":0.0, "L":truck.L, "W":truck.W}]
                for it in items_sorted:
                    if any(p.id == it.id for p in sim_placed): continue
                    if it.H > shelf_h + EPS: continue
                    if z0 + it.H > truck.H + EPS: continue
                    if not ((not flags.max_payload) or (sim_weight + it.weight <= truck.payload_kg + EPS)): continue
                    orientations = _orientations(it) if getattr(flags, 'orientation_allowed', True) else [(it.L, it.W)]
                    for (L,W) in orientations:
                        if L > truck.L + EPS or W > truck.W + EPS: continue
                        # best-fit
                        chosen_idx=None; best_left=None
                        for ri,r in enumerate(free_rects):
                            if L <= r['L'] + EPS and W <= r['W'] + EPS:
                                left = (r['L']*r['W']) - (L*W)
                                if best_left is None or left < best_left: best_left=left; chosen_idx=ri
                        if chosen_idx is None: continue
                        r = free_rects.pop(chosen_idx)
                        x0,y0 = r['x'], r['y']; x1,y1 = x0+L, y0+W
                        # stacking constraints: shelf above ground must be supported by previous shelf placements
                        if s > 0:
                            required_fraction = max(SUPPORT_RATIO_CARTON if it.weight < 18 else SUPPORT_RATIO_STANDARD if it.weight < 70 else SUPPORT_RATIO_HEAVY, SUPPORT_MIN_FRACTION, 0.30)
                            support_need = L*W*required_fraction
                            got = 0.0
                            # consider sim_placed items exactly at z0
                            for q in sim_placed:
                                if abs((q.z + q.H) - z0) > EPS: continue
                                ox,oy = max(x0,q.x), max(y0,q.y); ex,ey = min(x1, q.x+q.L), min(y1, q.y+q.W)
                                if ex>ox and ey>oy: got += (ex-ox)*(ey-oy);
                            if got + EPS < support_need:
                                # center fallback
                                cx=(x0+x1)/2.0; cy=(y0+y1)/2.0; center_supported=False
                                for q in sim_placed:
                                    if abs((q.z + q.H) - z0) > EPS: continue
                                    if (q.x - EPS) <= cx < (q.x + q.L + EPS) and (q.y - EPS) <= cy < (q.y + q.W + EPS): center_supported=True; break
                                if not center_supported or got < support_need * 0.20:
                                    free_rects.append(r); continue
                        # accept
                        p = Placement(it.id, x0, y0, z0, L, W, it.H, it.weight, it.drop_order, it.fragile, it.stack_limit)
                        sim_placed.append(p)
                        sim_weight += it.weight
                        # split
                        xr={"x":x1, "y":y0, "L":r['L']-L, "W":r['W']}
                        yt={"x":x0, "y":y1, "L":r['L'], "W":r['W']-W}
                        if xr['L']>EPS and xr['W']>EPS: free_rects.append(xr)
                        if yt['L']>EPS and yt['W']>EPS: free_rects.append(yt)
                        break
                z0 += shelf_h
            vol_used_s = sum(p.L*p.W*p.H for p in sim_placed)
            return sim_placed, sim_weight, vol_used_s

        # candidate shelf heights (meters): try thin shelves first
        candidate_shelves = [0.25, 0.30, 0.35, 0.40, 0.50]
        # include minimal and median item heights
        try:
            hs = sorted(set([it.H for it in items_sorted]))
            if hs:
                candidate_shelves += [max(0.05, hs[0]), hs[max(0, len(hs)//2)]]
        except Exception:
            pass

        best_shelf_res = (placed, total_weight, vol_used)
        best_shelf_util = vol_util
        for sh in candidate_shelves:
            sim_placed, sim_weight, vol_used_s = shelf_pack_with_height(sh)
            util = vol_used_s / (vol_total + EPS)
            if util > best_shelf_util + 1e-6:
                best_shelf_util = util
                best_shelf_res = (sim_placed, sim_weight, vol_used_s)

        # if shelf strategy improved utilization, return its result
        if best_shelf_util > vol_util + 1e-6:
            sim_placed, sim_weight, vol_used_s = best_shelf_res
            return sim_placed, sim_weight
    return placed, total_weight
