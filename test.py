import pandas as pd
import numpy as np

# Number of items
num_items = 500

np.random.seed(42)

# Generate realistic box dimensions
lengths = np.random.randint(250, 1200, num_items)   # mm
widths  = np.random.randint(200, 1000, num_items)   # mm
heights = np.random.randint(150, 800, num_items)    # mm

# Estimate volume and derive weight based on density
volumes = lengths * widths * heights / 1e9          # m³
density = np.random.uniform(150, 450, num_items)    # kg/m³ typical for packed cartons
weights = np.round(volumes * density, 2)
weights = np.clip(weights, 2, 60)                   # keep within practical truck carton range

# Build DataFrame
df = pd.DataFrame({
    "item_id": [f"ITEM_{i+1:04d}" for i in range(num_items)],
    "length_mm": lengths,
    "width_mm": widths,
    "height_mm": heights,
    "weight_kg": weights
})

# Save file
df.to_csv("realistic_truckload_dataset_500.csv", index=False)

print("✅ File created: realistic_truckload_dataset_500.csv")
print(df.head(10))
