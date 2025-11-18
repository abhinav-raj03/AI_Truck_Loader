"""
Debug runner: load items from a CSV and run the packer directly (no GA/ORTools).
Prints detailed information about layers, placements, and returns a CSV for inspection.
Run:
    python3 run_pack_debug.py auto_optimized_truckC_500.csv
"""
import sys, csv
from loader_gpu.main_gpu import load_items_csv
from loader_gpu.config import TruckSpec, Flags
from loader_gpu.packer_cpu import pack

if len(sys.argv) < 2:
    print('Usage: python3 run_pack_debug.py <items.csv>')
    sys.exit(1)

path = sys.argv[1]
items = load_items_csv(path)
print(f'Loaded {len(items)} items from {path}')
truck = TruckSpec(); flags = Flags()
try:
    placed, total_w = pack(truck, flags, items)
except Exception as e:
    import traceback, sys
    print('Packer raised exception:', e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(2)
print(f'Pack returned {len(placed)} placements, total weight {total_w:.2f} kg')
vol_used = sum(p.L*p.W*p.H for p in placed)
vol_total = truck.L*truck.W*truck.H
print(f'Volume used {vol_used:.3f} m3 of {vol_total:.3f} ({100.0*vol_used/vol_total:.2f}%)')

# layer breakdown
layers = {}
for p in placed:
    layers.setdefault(round(p.z,6), []).append(p)
print('Layers:', sorted(layers.keys()))
for z in sorted(layers.keys()):
    ls = layers[z]
    print(f' z={z:.3f} m: {len(ls)} items; layer height sample: {max(p.H for p in ls):.3f} m')

# write debug CSV
out = path.rsplit('.',1)[0] + '_debug_packed.csv'
keys = ["id","x","y","z","L","W","H","weight","drop_order","fragile","stack_limit"]
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for p in placed:
        row = {k: getattr(p,k) for k in keys}
        # convert id to int/str safely
        if isinstance(row['id'], float): row['id'] = int(row['id'])
        w.writerow(row)
print('Wrote debug CSV to', out)
