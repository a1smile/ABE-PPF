import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

files = {
    "Baseline": "stanford_baseline_batch.json",
    "w/o Pose": "stanford_no_pose_pipeline_fixed_batch.json",
    "w/o Robust Vote": "stanford_no_robust_vote_batch.json",
    "w/o RSMRQ": "stanford_no_rsmrq_batch.json",
    "Ours": "stanford_retrieval_cached_batch.json",
}

OUT_PDF = "heatmap_final.pdf"
OUT_SVG = "heatmap_final.svg"

def load_metric(path, key):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "results" in raw:
        records = raw["results"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Unsupported JSON format")

    values = []
    for x in records:
        if "metrics" in x:
            val = x["metrics"].get(key, None)
        else:
            val = x.get(key.lower(), None)

        if val is not None:
            try:
                values.append(float(val))
            except:
                pass

    return np.array(values)


# Load data
add_mean = []
add_s_mean = []
rot_mean = []
time_mean = []

for name, path in files.items():
    if not os.path.exists(path):
        print(f"[WARN] Missing {path}")
        continue

    add_vals = load_metric(path, "ADD")
    add_s_vals = load_metric(path, "ADD-S")
    rot_vals = load_metric(path, "rotation_error_deg")
    time_vals = load_metric(path, "registration_time")

    add_mean.append(np.mean(add_vals))
    add_s_mean.append(np.mean(add_s_vals))
    rot_mean.append(np.mean(rot_vals))
    time_mean.append(np.mean(time_vals))

# Create Correlation Matrix
import pandas as pd

data = {
    "ADD": add_mean,
    "ADD-S": add_s_mean,
    "Rotation Error": rot_mean,
    "Time": time_mean,
}

df = pd.DataFrame(data)

# Compute correlation matrix
corr_matrix = df.corr()

# Plot Heatmap
plt.figure(figsize=(6, 5))
sns.set(font_scale=1.2)
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    vmin=-1, vmax=1,
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title("Correlation Matrix of Metrics")
plt.tight_layout()

# Save to file
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_SVG, bbox_inches="tight")
plt.show()

print("Saved:", OUT_PDF)
print("Saved:", OUT_SVG)
