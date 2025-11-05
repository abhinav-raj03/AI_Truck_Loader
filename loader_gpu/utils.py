import csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Style 1: logistics gradient by drop_order
STOP_COLORS = {
    1: (0.1,0.3,0.9,0.65),  # blue
    2: (0.1,0.7,0.2,0.65),  # green
    3: (0.95,0.85,0.15,0.65), # yellow
    4: (0.95,0.55,0.15,0.65), # orange
    5: (0.9,0.1,0.1,0.65),   # red
}

def save_layout_csv(placements, path):
    if not placements: return
    keys = ["id","x","y","z","L","W","H","weight","drop_order","fragile","stack_limit"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for p in placements: w.writerow({k:getattr(p,k) for k in keys})

def save_report_json(rep, path):
    with open(path, "w") as f: json.dump(rep, f, indent=2)

def draw3d(placements, truck, out_png, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(0, truck.L); ax.set_ylim(0, truck.W); ax.set_zlim(0, truck.H)
    for p in placements:
        color = STOP_COLORS.get(int(p.drop_order), (0.5,0.5,0.5,0.5))
        x,y,z = p.x, p.y, p.z; L,W,H=p.L,p.W,p.H
        X=[x,x+L,x+L,x,x,x+L,x+L,x]; Y=[y,y,y+W,y+W,y,y,y+W,y+W]; Z=[z,z,z,z,z+H,z+H,z+H,z+H]
        faces=[[(X[0],Y[0],Z[0]),(X[1],Y[1],Z[1]),(X[2],Y[2],Z[2]),(X[3],Y[3],Z[3])],
               [(X[4],Y[4],Z[4]),(X[5],Y[5],Z[5]),(X[6],Y[6],Z[6]),(X[7],Y[7],Z[7])],
               [(X[0],Y[0],Z[0]),(X[1],Y[1],Z[1]),(X[5],Y[5],Z[5]),(X[4],Y[4],Z[4])],
               [(X[2],Y[2],Z[2]),(X[3],Y[3],Z[3]),(X[7],Y[7],Z[7]),(X[6],Y[6],Z[6])],
               [(X[1],Y[1],Z[1]),(X[2],Y[2],Z[2]),(X[6],Y[6],Z[6]),(X[5],Y[5],Z[5])],
               [(X[4],Y[4],Z[4]),(X[7],Y[7],Z[7]),(X[3],Y[3],Z[3]),(X[0],Y[0],Z[0])]]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=[color]*6, edgecolors='k', linewidths=0.2))
    ax.set_xlabel("L (m)"); ax.set_ylabel("W (m)"); ax.set_zlabel("H (m)")
    if title: ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
