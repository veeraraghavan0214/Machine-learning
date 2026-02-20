PROJECT TITLE:

3D Mesh Normalization, Quantization, and Error Analysis

OBJECTIVE:

This project demonstrates preprocessing steps used in 3D AI systems like SeamGPT to standardize mesh data.
We load 3D meshes, apply normalization, perform quantization, reconstruct the mesh, and evaluate information loss using Mean Squared Error (MSE).

DATASET:

Folder: meshes/
Contains 8 .obj 3D models:

branch.obj

cylinder.obj

explosive.obj

fence.obj

girl.obj

person.obj

table.obj

talwar.obj

Each file includes vertices (v x y z) and faces (f a b c) describing 3D object geometry.

REQUIREMENTS:
In VScode or any IDE Install dependencies (run once):

pip install trimesh open3d matplotlib pandas numpy

HOW TO RUN:

Place all .obj files in:
C:\Users\Desktop\8samples\meshes\

Run the script:

python mesh_analysis.py

Outputs will be generated in:

C:\Users\Desktop\8samples\outputs\
    ├── csv\mesh_results.csv
    └── plots\
        ├── overall_comparison_clear.png
        ├── <mesh>_<method>_MSE.png

What the Code Does:

Step	       Description
1. Load Mesh - Loads .obj files using trimesh and extracts vertex coordinates.
2. Normalize - Applies 3 methods:
• Min–Max ([0,1])
• Z-Score (standardize & rescale)
• Unit Sphere (fit in radius=1 sphere)
3. Quantize - Converts continuous coordinates to discrete bins (0–1023).
4. Reconstruct - Dequantize + Denormalize back to original scale.
5. Compute MSE - Measures how close reconstructed mesh is to original.
6. Visualize - Generates per-axis MSE bar charts, comparison plots, and final log-scale summary.

OUTPUTS:

CSV File: Summary of MSE (per axis and total) for all meshes & normalization types.

PLOTS:

MSE per axis (X, Y, Z) for each mesh

Overall MSE comparison (log scale)

Example (from CSV):

Mesh	Normalization	MSE_X	MSE_Y	MSE_Z	MSE_Total
branch.obj	MinMax	1.22e-07	1.19e-07	1.21e-07	1.20e-07
cylinder.obj	UnitSphere	3.05e-06	2.89e-06	2.91e-06	2.95e-06
girl.obj	ZScore	5.13e-06	4.98e-06	5.04e-06	5.05e-06

OBSERVATIONS:

Min–Max normalization consistently produced lowest MSE for most meshes.

Unit Sphere normalization was beneficial for elongated or irregular models (like talwar.obj, branch.obj).

Z-Score normalization had slightly higher errors due to variance-based scaling.

Using log-scale visualizations revealed even small MSE differences clearly.

The reconstruction process preserved geometry nearly perfectly (MSE values in range 1e-7–1e-5).

CONCLUSION:

The Min–Max normalization + 1024-bin quantization combination yielded the least reconstruction error, proving most effective for consistent and uniformly scaled meshes.
The preprocessing pipeline successfully demonstrated how normalization and quantization affect geometric fidelity and prepares data for 3D AI models like SeamGPT.