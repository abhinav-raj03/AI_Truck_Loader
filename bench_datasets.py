"""
Save as `bench_datasets.py` at project root and run:
    python3 bench_datasets.py

This script:
- Scans CSV files in the repository root.
- For each CSV it loads items using loader_gpu.main_gpu.load_items_csv.
- Tries a small grid of lane percentiles, candidate pool sizes and GA settings.
- Runs packing (with OR-Tools preselect if available, then optional GA).
- Keeps and writes the best result (layout + report + plot) per dataset into `bench_results/`.

Run notes:
- This is intentionally conservative in search space to finish reasonably fast. You can expand
  `LANE_PCTS`, `CAND_SIZES`, and `GA_CHOICES` to search more combos.
"""
import os, glob
from itertools import product
from importlib import import_module

mg = import_module("loader_gpu.main_gpu")
cfg = import_module("loader_gpu.config")
sel = import_module("loader_gpu.selector")
ga_mod = import_module("loader_gpu.ga_gpu")
pack_mod = import_module("loader_gpu.packer_cpu")
utils = import_module("loader_gpu.utils")

load_items_csv = mg.load_items_csv
select_subset = sel.select_subset
ga_reorder = ga_mod.ga_reorder
pack = pack_mod.pack
save_layout_csv = utils.save_layout_csv
save_report_json = utils.save_report_json
try:
    draw3d = utils.draw3d
except Exception:
    draw3d = lambda *a, **k: None

ROOT = os.path.dirname(__file__)
CSV_FILES = [p for p in glob.glob(os.path.join(ROOT, "*.csv"))]
# ignore output files we might have generated and very small CSVs later
IGNORE_PREFIXES = {"packed_layout", "bench_results", "report", "plot3d"}

# Conservative parameter grid for a quick benchmark. Expand if you want longer runs.
LANE_PCTS = [60, 70, 80]
CAND_SIZES = [180, 260, None]   # None -> use all eligible
GA_CHOICES = [(0, 0), (48, 18)]  # (population, generations)  (0,0) -> no GA

os.makedirs("bench_results", exist_ok=True)

import numpy as np

for csv_path in CSV_FILES:
    base = os.path.basename(csv_path).rsplit('.',1)[0]
    if any(base.startswith(pref) for pref in IGNORE_PREFIXES):
        print(f"Skipping generated file: {csv_path}")
        continue
    name = os.path.basename(csv_path).rsplit(".", 1)[0]
    print(f"\n=== Dataset: {name} ({csv_path}) ===")
    try:
        items = load_items_csv(csv_path)
    except Exception as e:
        print(f"Skipping {csv_path}: failed to load as items CSV ({e})")
        continue
    print(f"Loaded {len(items)} items from {name}")
    if len(items) < 10:
        print(f"Skipping {name} because it has too few items ({len(items)}) to benchmark.")
        continue
    truck = cfg.TruckSpec()

    widths = np.array([min(i.L, i.W) if i.can_rotate else i.W for i in items])
    # guard against empty or nan-only arrays
    if widths.size == 0 or np.isnan(widths).all():
        print(f"No valid width measurements for {name}, skipping.")
        continue

    best = {"vol_util": -1, "placed": [], "cfg": None}

    for lane_pct, cand_size, ga_cfg in product(LANE_PCTS, CAND_SIZES, GA_CHOICES):
        try:
            lane = float(np.percentile(widths, lane_pct))
        except Exception:
            lane = float(np.nanpercentile(widths, min(lane_pct, 100)))
        lane = min(lane, truck.W / 1.95)
        MIN_H = 0.0

        def fits_lane(it):
            dims = [(it.L, it.W), (it.W, it.L)] if it.can_rotate else [(it.L, it.W)]
            return any((w <= lane and l <= truck.L and it.H <= truck.H) for l, w in dims)

        eligible = [i for i in items if i.H >= MIN_H and fits_lane(i)]
        eligible = sorted(eligible, key=lambda i: (
            -i.drop_order, -i.H, -(i.L * i.W), -i.stack_limit
        ))

        if len(eligible) < 150:
            eligible = sorted(items, key=lambda i: -(i.L * i.W))[:150]

        cand = eligible if cand_size is None else eligible[:cand_size]

        try:
            chosen = select_subset(cand, truck)
        except Exception:
            chosen = cand

        order = chosen[:]
        pop, gen = ga_cfg
        if pop and len(order) > 4:
            try:
                order = ga_reorder(order, truck, population=pop, generations=gen)
            except Exception:
                order = chosen[:]

        placed, total_w = pack(truck, cfg.Flags(), order)

        vol_used = sum(p.L * p.W * p.H for p in placed)
        vol_total = truck.L * truck.W * truck.H
        vol_util = 100.0 * vol_used / vol_total

        if vol_util > best['vol_util']:
            best['vol_util'] = vol_util
            best['placed'] = placed
            best['cfg'] = {"lane_pct": lane_pct, "cand_size": cand_size, "ga": ga_cfg, "preselected": len(chosen)}
            out_prefix = os.path.join("bench_results", f"{name}_best")
            save_layout_csv(placed, out_prefix + "_packed_layout.csv")
            save_report_json({
                "placed_items": len(placed),
                "volume_utilization_pct": round(vol_util, 2),
                "weight_utilization_pct": round(100.0 * total_w / truck.payload_kg, 2),
                "config": best['cfg'],
            }, out_prefix + "_report.json")
            try:
                draw3d(placed, truck, out_prefix + "_plot3d.png",
                       title=f"{name} best: {vol_util:.2f}% (cfg={best['cfg']})")
            except Exception:
                pass

        print(f"cfg lane={lane_pct} cand={cand_size or 'ALL'} ga={ga_cfg} -> vol={vol_util:.2f}% placed={len(placed)}")

    print(f"BEST for {name}: vol={best['vol_util']:.2f}% cfg={best['cfg']}")

print('\nDone. Results saved under bench_results/.')
