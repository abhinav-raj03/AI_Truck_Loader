import argparse, pandas as pd
from .config import TruckSpec, Flags, GA_POP, GA_GEN
from .models import Item
from .selector import select_subset
from .packer_cpu import pack
from .ga_gpu import ga_reorder
from .utils import save_layout_csv, save_report_json, draw3d

def load_items_csv(path):
    df = pd.read_csv(path)
    items = []
    for _, r in df.iterrows():
        items.append(Item(
            id=int(r["id"]),
            L=float(r["length_mm"])/1000.0, W=float(r["width_mm"])/1000.0, H=float(r["height_mm"])/1000.0,
            weight=float(r["weight_kg"]),
            fragile=int(r["fragile"]), stack_limit=int(r["stack_limit"]),
            can_rotate=int(r["can_rotate"]), drop_order=int(r["drop_order"]),
        ))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--use_ortools", type=int, default=1)
    ap.add_argument("--use_ga", type=int, default=1)
    ap.add_argument("--ga_generations", type=int, default=GA_GEN)
    ap.add_argument("--ga_population", type=int, default=GA_POP)
    ap.add_argument("--prefilter_small", type=int, default=180)
    ap.add_argument("--prefilter_large", type=int, default=40)
    args = ap.parse_args()

    truck = TruckSpec(); flags = Flags()
    items = load_items_csv(args.items)

    small = [i for i in items if (i.L*i.W) <= 1.0]
    large = [i for i in items if (i.L*i.W) > 1.0]
    small = sorted(small, key=lambda i: (-i.stack_limit, -i.drop_order, -i.vol))
    large = sorted(large, key=lambda i: -i.vol)
    cand = small[:args.prefilter_small] + large[:args.prefilter_large]
    print(f"[INFO] Candidate pool: {len(cand)} (small={min(len(small), args.prefilter_small)}, large={min(len(large), args.prefilter_large)})")

    chosen = select_subset(cand, truck) if args.use_ortools else cand
    print(f"[INFO] Preselected (OR-Tools): {len(chosen)} items")

    order = chosen[:]
    if args.use_ga and len(order) > 4:
        order = ga_reorder(order, truck, population=args.ga_population, generations=args.ga_generations)
        print("[INFO] GA (GPU) reordering done.")

    placed, total_w = pack(truck, flags, order)

    vol_used = sum(p.L*p.W*p.H for p in placed)
    vol_total = truck.L*truck.W*truck.H
    vol_util = 100.0*vol_used/vol_total
    wt_util = 100.0*total_w/truck.payload_kg

    save_layout_csv(placed, "packed_layout.csv")
    save_report_json({
        "placed_items": len(placed),
        "volume_utilization_pct": round(vol_util,1),
        "weight_utilization_pct": round(wt_util,1)
    }, "report.json")
    draw3d(placed, truck, "plot3d.png", title=f"Fill: {vol_util:.1f}% (Vol), {wt_util:.1f}% (Wt)")
    print(f"Placed: {len(placed)} | Vol Util: {vol_util:.1f}% | Wt Util: {wt_util:.1f}%")
    print("Wrote: packed_layout.csv, report.json, plot3d.png")

if __name__ == "__main__":
    main()
