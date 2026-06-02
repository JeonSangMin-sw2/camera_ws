#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend to prevent PySide6/Qt platform conflicts
import matplotlib.pyplot as plt

def fit_circle_3d(points):
    """
    Fits a 3D circle to a set of 3D points using SVD.
    Returns:
        center (3,): 3D center of the circle
        normal (3,): 3D normal vector of the fitting plane
        radius (float): Radius of the fitted circle
        rmse (float): Root mean square error of points to the 3D circle
    """
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # 1. Fit plane using SVD (normal is the singular vector with the smallest singular value)
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[2, :]
    normal = normal / np.linalg.norm(normal)
    
    # Ensure normal points generally upwards for consistency
    if normal[2] < 0:
        normal = -normal
        
    # 2. Project points onto the fitting plane to get 2D coordinates
    # Construct local coordinate frame (u, v) on the fitting plane
    if abs(normal[0]) > 0.9:
        u = np.cross(normal, [0, 1, 0])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    pts_2d = np.zeros((len(points), 2))
    for i, p in enumerate(centered):
        pts_2d[i, 0] = np.dot(p, u)
        pts_2d[i, 1] = np.dot(p, v)
        
    # 3. Fit 2D circle: (u - uc)^2 + (v - vc)^2 = R^2
    # Linear system: u^2 + v^2 = 2*u*uc + 2*v*vc + (R^2 - uc^2 - vc^2)
    # A * x = B, where x = [2*uc, 2*vc, C]
    A = np.column_stack((pts_2d[:, 0], pts_2d[:, 1], np.ones(len(points))))
    B = pts_2d[:, 0]**2 + pts_2d[:, 1]**2
    
    x, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    uc = x[0] / 2.0
    vc = x[1] / 2.0
    radius = np.sqrt(x[2] + uc**2 + vc**2)
    
    # 4. Project 2D center back to 3D
    center = centroid + uc * u + vc * v
    
    # 5. Compute RMSE
    distances_to_center = np.linalg.norm(points - center, axis=1)
    rmse = np.sqrt(np.mean((distances_to_center - radius)**2))
    
    return center, normal, radius, rmse, u, v, uc, vc, pts_2d

def parse_sweep_file(filepath):
    """
    Parses a joint sweep text file.
    Format is expected to contain a header and comma-separated numeric rows:
    # Joint_Angle, Cam_X, Cam_Y, Cam_Z, Torso_X, Torso_Y, Torso_Z, EE_X, EE_Y, EE_Z
    """
    angles = []
    cam_pts = []
    torso_pts = []
    ee_pts = []
    
    if not os.path.exists(filepath):
        print(f"[WARN] File not found: {filepath}")
        return None
        
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                tokens = [float(x) for x in line.split(',')]
                if len(tokens) >= 10:
                    angles.append(tokens[0])
                    cam_pts.append(tokens[1:4])
                    torso_pts.append(tokens[4:7])
                    ee_pts.append(tokens[7:10])
            except ValueError:
                continue
                
    return {
        'angles': np.array(angles),
        'camera': np.array(cam_pts),
        'torso': np.array(torso_pts),
        'ee': np.array(ee_pts)
    }

def generate_circle_points_3d(center, normal, radius, u, v, num_points=100):
    """Generates 3D coordinates for a circle contour given normal and radius."""
    theta = np.linspace(0, 2*np.pi, num_points)
    circle_pts = []
    for t in theta:
        p = center + radius * (np.cos(t) * u + np.sin(t) * v)
        circle_pts.append(p)
    return np.array(circle_pts)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="3D Joint Sweep Marker Data Circle Fitting Debugger")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"], help="Arm side (left/right)")
    parser.add_argument("--axis-a", type=int, default=4, help="Axis number for sweep A (e.g. 4 for wrist yaw, 2 for shoulder)")
    parser.add_argument("--axis-b", type=int, default=6, help="Axis number for sweep B (e.g. 6 for wrist roll, 4 for elbow)")
    parser.add_argument("--frame", type=str, default="torso", choices=["camera", "torso", "ee"], help="Coordinate frame to plot (camera/torso/ee)")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    file_a = os.path.join(script_dir, f"sweep_points_{args.side}_joint_A_axis_{args.axis_a}.txt")
    file_b = os.path.join(script_dir, f"sweep_points_{args.side}_joint_B_axis_{args.axis_b}.txt")
    
    print("=" * 60)
    print("      3D JOINT CALIBRATION SWEEP DATA DEBUGGER")
    print("=" * 60)
    print(f"Loading files:\n - Axis A: {file_a}\n - Axis B: {file_b}")
    print(f"Plotting Frame: {args.frame.upper()}")
    
    data_a = parse_sweep_file(file_a)
    data_b = parse_sweep_file(file_b)
    
    if data_a is None or data_b is None:
        print("[ERROR] Failed to load one or both sweep logs. Please run a sweep first or check file paths.")
        return
        
    pts_a = data_a[args.frame]
    pts_b = data_b[args.frame]
    
    if len(pts_a) < 3 or len(pts_b) < 3:
        print("[ERROR] Not enough points in files to fit 3D circles (need at least 3 points).")
        return
        
    # Fit circles
    center_a, normal_a, r_a, rmse_a, u_a, v_a, uc_a, vc_a, _ = fit_circle_3d(pts_a)
    center_b, normal_b, r_b, rmse_b, u_b, v_b, uc_b, vc_b, _ = fit_circle_3d(pts_b)
    
    # Calculate geometric metrics
    center_dist = np.linalg.norm(center_a - center_b)
    angle_between_normals = np.degrees(np.arccos(np.clip(abs(np.dot(normal_a, normal_b)), -1.0, 1.0)))
    
    print("-" * 60)
    print("  [AXIS A FIT RESULTS]")
    print(f"    - Center: [{center_a[0]:.2f}, {center_a[1]:.2f}, {center_a[2]:.2f}] mm")
    print(f"    - Normal: [{normal_a[0]:.4f}, {normal_a[1]:.4f}, {normal_a[2]:.4f}]")
    print(f"    - Radius: {r_a:.2f} mm")
    print(f"    - Fitting RMSE: {rmse_a:.4f} mm")
    
    print("\n  [AXIS B FIT RESULTS]")
    print(f"    - Center: [{center_b[0]:.2f}, {center_b[1]:.2f}, {center_b[2]:.2f}] mm")
    print(f"    - Normal: [{normal_b[0]:.4f}, {normal_b[1]:.4f}, {normal_b[2]:.4f}]")
    print(f"    - Radius: {r_b:.2f} mm")
    print(f"    - Fitting RMSE: {rmse_b:.4f} mm")
    
    print("-" * 60)
    print("  [GEOMETRIC CONVERGENCE METRICS]")
    print(f"    - Center-to-Center Distance (Goal: minimize): {center_dist:.4f} mm")
    print(f"    - Angle between axis normals   (Goal: nominal): {angle_between_normals:.2f} deg")
    print("=" * 60)
    
    # Plotting: Robust 3-View Orthogonal Projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Generate 3D circle lines
    circle_pts_a = generate_circle_points_3d(center_a, normal_a, r_a, u_a, v_a)
    circle_pts_b = generate_circle_points_3d(center_b, normal_b, r_b, u_b, v_b)
    
    scale = min(r_a, r_b) * 0.4
    
    # Subplot 1: XY Projection (Top View)
    axes[0].scatter(pts_a[:, 0], pts_a[:, 1], c='red', s=15, alpha=0.7, label=f'Axis A Raw ({len(pts_a)} pts)')
    axes[0].scatter(pts_b[:, 0], pts_b[:, 1], c='blue', s=15, alpha=0.7, label=f'Axis B Raw ({len(pts_b)} pts)')
    axes[0].plot(circle_pts_a[:, 0], circle_pts_a[:, 1], 'r--', linewidth=2, label='Axis A Fit')
    axes[0].plot(circle_pts_b[:, 0], circle_pts_b[:, 1], 'b--', linewidth=2, label='Axis B Fit')
    axes[0].scatter([center_a[0]], [center_a[1]], c='darkred', marker='X', s=100, label='Center A')
    axes[0].scatter([center_b[0]], [center_b[1]], c='darkblue', marker='X', s=100, label='Center B')
    axes[0].plot([center_a[0], center_b[0]], [center_a[1], center_b[1]], color='purple', linestyle=':', linewidth=2, label=f'Center Dist: {center_dist:.2f}mm')
    axes[0].arrow(center_a[0], center_a[1], normal_a[0]*scale, normal_a[1]*scale, color='darkred', head_width=2, width=0.5)
    axes[0].arrow(center_b[0], center_b[1], normal_b[0]*scale, normal_b[1]*scale, color='darkblue', head_width=2, width=0.5)
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    axes[0].set_title('X-Y Projection (Top View)')
    axes[0].set_aspect('equal')
    axes[0].grid(True)
    axes[0].legend(fontsize=9)
    
    # Subplot 2: YZ Projection (Side View)
    axes[1].scatter(pts_a[:, 1], pts_a[:, 2], c='red', s=15, alpha=0.7, label='Axis A Raw')
    axes[1].scatter(pts_b[:, 1], pts_b[:, 2], c='blue', s=15, alpha=0.7, label='Axis B Raw')
    axes[1].plot(circle_pts_a[:, 1], circle_pts_a[:, 2], 'r--', linewidth=2, label='Axis A Fit')
    axes[1].plot(circle_pts_b[:, 1], circle_pts_b[:, 2], 'b--', linewidth=2, label='Axis B Fit')
    axes[1].scatter([center_a[1]], [center_a[2]], c='darkred', marker='X', s=100, label='Center A')
    axes[1].scatter([center_b[1]], [center_b[2]], c='darkblue', marker='X', s=100, label='Center B')
    axes[1].plot([center_a[1], center_b[1]], [center_a[2], center_b[2]], color='purple', linestyle=':', linewidth=2)
    axes[1].arrow(center_a[1], center_a[2], normal_a[1]*scale, normal_a[2]*scale, color='darkred', head_width=2, width=0.5)
    axes[1].arrow(center_b[1], center_b[2], normal_b[1]*scale, normal_b[2]*scale, color='darkblue', head_width=2, width=0.5)
    axes[1].set_xlabel('Y (mm)')
    axes[1].set_ylabel('Z (mm)')
    axes[1].set_title('Y-Z Projection (Side View)')
    axes[1].set_aspect('equal')
    axes[1].grid(True)
    axes[1].legend(fontsize=9)
    
    # Subplot 3: XZ Projection (Front View)
    axes[2].scatter(pts_a[:, 0], pts_a[:, 2], c='red', s=15, alpha=0.7, label='Axis A Raw')
    axes[2].scatter(pts_b[:, 0], pts_b[:, 2], c='blue', s=15, alpha=0.7, label='Axis B Raw')
    axes[2].plot(circle_pts_a[:, 0], circle_pts_a[:, 2], 'r--', linewidth=2, label='Axis A Fit')
    axes[2].plot(circle_pts_b[:, 0], circle_pts_b[:, 2], 'b--', linewidth=2, label='Axis B Fit')
    axes[2].scatter([center_a[0]], [center_a[2]], c='darkred', marker='X', s=100, label='Center A')
    axes[2].scatter([center_b[0]], [center_b[2]], c='darkblue', marker='X', s=100, label='Center B')
    axes[2].plot([center_a[0], center_b[0]], [center_a[2], center_b[2]], color='purple', linestyle=':', linewidth=2)
    axes[2].arrow(center_a[0], center_a[2], normal_a[0]*scale, normal_a[2]*scale, color='darkred', head_width=2, width=0.5)
    axes[2].arrow(center_b[0], center_b[2], normal_b[0]*scale, normal_b[2]*scale, color='darkblue', head_width=2, width=0.5)
    axes[2].set_xlabel('X (mm)')
    axes[2].set_ylabel('Z (mm)')
    axes[2].set_title('X-Z Projection (Front View)')
    axes[2].set_aspect('equal')
    axes[2].grid(True)
    axes[2].legend(fontsize=9)
    
    fig.suptitle(f"Orthogonal Multi-View Projections ({args.side.upper()} Arm, {args.frame.upper()} Frame)\nCenter Distance: {center_dist:.3f} mm | Axis Normals Angle: {angle_between_normals:.2f}°", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    plot_save_path = os.path.abspath(os.path.join(script_dir, f"debug_orthogonal_circles_{args.side}_{args.frame}.png"))
    plt.savefig(plot_save_path, dpi=150)
    print(f"\n[SAVED] Multi-view orthogonal projection diagram saved successfully to:\n       {plot_save_path}")
    print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
