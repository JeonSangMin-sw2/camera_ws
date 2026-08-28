import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_draw_v13_plot(res_6, res_5, res_4, unified_res, arm_side, save_path):
    fig = plt.figure(figsize=(16, 10))
    
    # 2x2 grid
    # Top-Left: Axis 6 (Roll) Fit
    # Top-Right: Axis 5 (Pitch) Fit
    # Bottom-Left: Axis 4 (Yaw) Fit
    # Bottom-Right: 3D Orthogonal Alignment Overview
    
    def plot_single(ax, res, title, color):
        if res is None:
            ax.set_title(f"{title}: No Data")
            ax.axis('off')
            return
        pts_2d = res.get('pts_2d')
        uc = res.get('uc_opt', 0.0)
        vc = res.get('vc_opt', 0.0)
        r = res.get('radius', 1.0)
        rmse = res.get('rmse', 0.0)
        if pts_2d is not None and len(pts_2d) > 0:
            ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=color, s=15, alpha=0.6, label='Raw Points')
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(uc + r*np.cos(theta), vc + r*np.sin(theta), 'r--', linewidth=2, label=f'Fit (r={r:.1f}mm)')
            ax.scatter([uc], [vc], c='darkred', marker='X', s=80, label='Center')
            ax.set_aspect('equal')
            ax.grid(True)
            ax.set_title(f"{title} (Radius: {r:.1f}mm, RMSE: {rmse:.3f}mm)", fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    plot_single(ax1, res_6, "Axis 6 (Wrist Roll) Sweep", "blue")
    plot_single(ax2, res_5, "Axis 5 (Wrist Pitch) Sweep", "green")
    plot_single(ax3, res_4, "Axis 4 (Wrist Yaw) Sweep", "purple")
    
    # Summary panel in ax4
    ax4.axis('off')
    summary_text = (
        f"=== UNIFIED 3-AXIS CALIBRATION SUMMARY ({arm_side.upper()} ARM) ===\n\n"
        f"1. Recommended Joint Offsets:\n"
        f"   * Joint 5 (Wrist Pitch): {unified_res.get('d5_opt_deg', 0.0):+.4f}°\n"
        f"   * Joint 6 (Wrist Roll) : {unified_res.get('d6_opt_deg', 0.0):+.4f}°\n\n"
        f"2. Marker Bracket Calibration:\n"
        f"   * Position (X, Y, Z)   : [{unified_res['x_e']:.2f}, {unified_res['y_e']:.2f}, {unified_res['z_e']:.2f}] mm\n"
        f"   * Orientation (R, P, Y): [{unified_res['roll_e']:.2f}°, {unified_res['pitch_e']:.2f}°, {unified_res['yaw_e']:.2f}°]\n\n"
        f"3. Quantitative Verification Metrics:\n"
        f"   * Orthogonality J4-J5  : {unified_res.get('ang_45', 0.0):.3f}° (Dev: {abs(unified_res.get('ang_45', 90.0)-90.0):.3f}°)\n"
        f"   * Orthogonality J5-J6  : {unified_res.get('ang_56', 0.0):.3f}° (Dev: {abs(unified_res.get('ang_56', 90.0)-90.0):.3f}°)\n"
        f"   * Orthogonality J4-J6  : {unified_res.get('ang_46', 0.0):.3f}° (Dev: {abs(unified_res.get('ang_46', 90.0)-90.0):.3f}°)\n"
        f"   * Max Radius Residual  : {unified_res.get('max_radius_err', 0.0):.3f} mm\n"
        f"   * Status               : {'CONVERGED (PASS)' if unified_res.get('converged', False) else 'WARNING'}\n"
    )
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))
    
    fig.suptitle(f"RB-Y1 v1.3 Spherical Wrist Simultaneous Calibration ({arm_side.upper()} Arm)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")

