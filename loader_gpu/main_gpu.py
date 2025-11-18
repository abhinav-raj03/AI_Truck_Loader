import argparse, pandas as pd
from .config import TruckSpec, Flags, GA_POP, GA_GEN
from .models import Item
from .selector import select_subset
from .packer_cpu import pack
from .ga_gpu import ga_reorder
from .utils import save_layout_csv, save_report_json, draw3d

def load_items_csv(path):
    import pandas as pd
    from .models import Item

    df = pd.read_csv(path)
    # Accept either *_m or *_mm headers; prefer meters if present.
    def get_dim(row, m, mm):
        if m in row and not pd.isna(row[m]):
            return float(row[m])
        return float(row[mm]) / 1000.0

    items = []
    for _, r in df.iterrows():
        L = float(r["L_m"]) if "L_m" in df.columns else get_dim(r, "length_m", "length_mm")
        W = float(r["W_m"]) if "W_m" in df.columns else get_dim(r, "width_m", "width_mm")
        H = float(r["H_m"]) if "H_m" in df.columns else get_dim(r, "height_m", "height_mm")
        items.append(Item(
            id=str(r.get("id", _)),
            L=L, W=W, H=H,
            weight=float(r.get("weight_kg", 20.0)),
            fragile=int(r.get("fragile", 0)),
            stack_limit=int(r.get("stack_limit", 2)),
            can_rotate=int(r.get("can_rotate", 1)),
            drop_order=int(r.get("drop_order", 1)),
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

    # === ADAPTIVE LANE-AWARE PREFILTER (v4 recommended) ===
    import numpy as np

    # Measure item widths and compute a "good lane width"
    widths = np.array([min(i.L, i.W) if i.can_rotate else i.W for i in items])
    lane = np.percentile(widths, 70)  # allow 70% of items to fit per lane (wider)
    lane = min(lane, truck.W / 1.95)  # never exceed ~half-truck width

    MIN_H = 0.0  # allow smaller items to be considered

    def fits_lane(it):
        dims = [(it.L, it.W), (it.W, it.L)] if it.can_rotate else [(it.L, it.W)]
        return any((w <= lane and l <= truck.L and it.H <= truck.H) for l, w in dims)

    eligible = [i for i in items if i.H >= MIN_H and fits_lane(i)]

    # Sort to form stable columns and maintain drop-sequence
    eligible = sorted(eligible, key=lambda i: (
        -i.drop_order,
        -i.H,
        -(i.L * i.W),
        -i.stack_limit
    ))

    # Guarantee sufficient pool size (this is the key!)
    if len(eligible) < 150:
        eligible = sorted(items, key=lambda i: -(i.L * i.W))[:150]

    cand = eligible[:260]
    print(f"[INFO] Candidate pool (adaptive): {len(cand)}")

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
