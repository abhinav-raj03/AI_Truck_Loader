# AI Truck Loader v3 (GPU, Color by Drop-Order)

**Goal:** visually full truck with high fill factor.  
- OR-Tools preselector (CPU, hard 1.8s cap)  
- GA reorder on **GPU (PyTorch)**  
- 3D stacking packer (CPU) with **3 layers** + **≥75% support**  
- **Color by drop order**: 1=blue, 2=green, 3=yellow, 4=orange, 5=red

## Install
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python -m loader_gpu.main_gpu   --items realistic_mix_dataset_2000.csv   --use_ortools 1   --use_ga 1   --ga_population 64   --ga_generations 20   --prefilter_small 180   --prefilter_large 40
```
Outputs: `packed_layout.csv`, `report.json`, `plot3d.png`
