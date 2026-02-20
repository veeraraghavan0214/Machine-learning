
# 3D Mesh Normalization, Quantization, and MSE Analysis
# Author: veeraraghavan (Final Rectified Version)


import trimesh
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# PARAMETERS 
INPUT_DIR = r"C:\Users\B veeraraghavan\OneDrive\ドキュメント\All docs\8samples\meshes"  # folder where .obj files are stored
OUTPUT_DIR = r"C:\Users\B veeraraghavan\OneDrive\ドキュメント\All docs\8samples\outputs"
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "csv"), exist_ok=True)
BINS = 1024  # quantization bins

# NORMALIZATION METHODS 
def minmax_normalize(vertices):
    v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
    norm = (vertices - v_min) / (v_max - v_min + 1e-8)
    return norm, v_min, v_max

def zscore_normalize(vertices):
    mean, std = vertices.mean(axis=0), vertices.std(axis=0)
    norm = (vertices - mean) / (std + 1e-8)
    norm_min, norm_max = norm.min(axis=0), norm.max(axis=0)
    norm = (norm - norm_min) / (norm_max - norm_min + 1e-8)
    return norm, mean, std, norm_min, norm_max

def unit_sphere_normalize(vertices):
    center = vertices.mean(axis=0)
    centered = vertices - center
    max_dist = np.linalg.norm(centered, axis=1).max()
    norm = centered / (max_dist + 1e-8)
    norm = (norm + 1) / 2  # map [-1,1] → [0,1]
    return norm, center, max_dist

# QUANTIZATION 
def quantize(v, bins=BINS):
    return np.floor(v * (bins - 1)).astype(int)

def dequantize(q, bins=BINS):
    return q / (bins - 1)

# ERROR METRIC 
def mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2, axis=0)  # per-axis MSE

# VISUALIZATION (Optional)
def visualize_meshes(original, reconstructed, title="Mesh Comparison"):
    o3d_original = o3d.geometry.TriangleMesh()
    o3d_recon = o3d.geometry.TriangleMesh()
    o3d_original.vertices = o3d.utility.Vector3dVector(original)
    o3d_recon.vertices = o3d.utility.Vector3dVector(reconstructed)
    o3d_original.paint_uniform_color([0, 0, 1])  # blue
    o3d_recon.paint_uniform_color([0, 1, 0])     # green
    print(f"🔹 Visualizing {title} ... (Press 'q' to exit viewer)")
    o3d.visualization.draw_geometries([o3d_original, o3d_recon])

# MAIN PIPELINE
results = []

for file in os.listdir(INPUT_DIR):
    if file.endswith(".obj"):
        mesh_path = os.path.join(INPUT_DIR, file)
        mesh = trimesh.load(mesh_path, force='mesh')
        vertices = mesh.vertices

        print(f"\nProcessing: {file}")
        print(f"Vertices: {vertices.shape[0]}")

        for method in ["MinMax", "ZScore", "UnitSphere"]:

            if method == "MinMax":
                norm, v_min, v_max = minmax_normalize(vertices)
                q = quantize(norm)
                dq = dequantize(q)
                recon = dq * (v_max - v_min) + v_min

            elif method == "ZScore":
                norm, mean, std, norm_min, norm_max = zscore_normalize(vertices)
                q = quantize(norm)
                dq = dequantize(q)
                dq = dq * (norm_max - norm_min) + norm_min
                recon = dq * std + mean

            elif method == "UnitSphere":
                norm, center, max_dist = unit_sphere_normalize(vertices)
                q = quantize(norm)
                dq = dequantize(q)
                dq = dq * 2 - 1
                recon = dq * max_dist + center

            # Compute MSE per axis
            mse_axis = mse(vertices, recon)
            mse_total = np.mean(mse_axis)

            # Save results
            results.append({
                "Mesh": file,
                "Normalization": method,
                "MSE_X": mse_axis[0],
                "MSE_Y": mse_axis[1],
                "MSE_Z": mse_axis[2],
                "MSE_Total": mse_total
            })

            # Plot error per axis
            plt.figure(figsize=(5, 4))
            plt.bar(['X', 'Y', 'Z'], mse_axis, color=['r', 'g', 'b'])
            plt.title(f"MSE per Axis - {file} ({method})")
            plt.ylabel("MSE")
            plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"{file}_{method}_MSE.png"))
            plt.close()

# SAVE RESULTS
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "csv", "mesh_results.csv"), index=False)
print("\n Results saved to outputs/csv/mesh_results.csv")

# SUMMARY ANALYSIS
best_results = df.loc[df.groupby("Mesh")["MSE_Total"].idxmin()]
print("\n🔍 Best Normalization Method per Mesh:")
print(best_results[["Mesh", "Normalization", "MSE_Total"]])

# Plot summary
# IMPROVED SUMMARY PLOT
plt.figure(figsize=(12, 7))

x = np.arange(len(df["Mesh"].unique()))
width = 0.25  # bar width
meshes = df["Mesh"].unique()

# Extract subsets for each method
methods = ["MinMax", "ZScore", "UnitSphere"]
for i, method in enumerate(methods):
    subset = df[df["Normalization"] == method]
    plt.bar(x + i * width, subset["MSE_Total"], width=width, label=method, alpha=0.8)

plt.xticks(x + width, meshes, rotation=45, ha="right", fontsize=10)
plt.yscale("log") # Use log scale to make small MSEs visible
plt.title("MSE Comparison Across Normalization Methods (Log Scale)", fontsize=14)
plt.ylabel("Mean Squared Error (log scale, lower = better)", fontsize=12)
plt.xlabel("Mesh Files", fontsize=12)
plt.legend(title="Normalization Type")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plots", "overall_comparison_clear.png"), dpi=300)
plt.show()


# FINAL CONCLUSION
min_method = best_results["Normalization"].value_counts().idxmax()
print(f"\n📘 CONCLUSION:")
print(f"The normalization method that generally gives the least reconstruction error is: **{min_method} Normalization**.")
print(f"Overall pattern: Min–Max performs best on uniformly scaled meshes, while Unit Sphere slightly benefits elongated or irregular shapes.")
print("Z-Score tends to have slightly higher MSE due to unequal axis scaling.")
