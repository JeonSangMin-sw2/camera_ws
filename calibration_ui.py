import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np

from core.calibration_core import (
    create_robot,
    create_live_marker_transform,
    capture_one_sample as capture_robot_sample,
    get_arm_config,
    get_both_arm_config,
    get_head_config,
    load_npz_dataset,
    save_npz_dataset,
    split_arm_offsets,
    validate_dataset,
    generate_sim_measurements,
)
from core.calibration_optimizer import (
    DEFAULT_LAMBDA_CAM_POS,
    DEFAULT_LAMBDA_CAM_ROT,
    CalibrationOptimizer,
    QPCalibrationOptimizer,
)
from core.robot_motion import (
    AutoCollectionConfig,
    build_incremental_motion_plan,
    move_to_auto_ready_pose,
    execute_auto_motion_step,
    compute_fk,
)
from homeoffset_core import (
    apply_home_offset_from_json,
    move_robot_to_zero_pose,
)

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
WARNING_POSE_PATH = BASE_DIR / "warning_pose.png"
WARNING_POSE_CHECK_PATH = BASE_DIR / "warning_pose_check.png"

class CalibrationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Calibration UI")
        self.root.geometry("1200x680")

        self.robot = None
        self.dyn_model = None
        self.model = None
        self.marker_transform = None

        self.shared_arm_q_list = []
        self.shared_head_q_list = []
        self.shared_T_list = []
        self.head_move_count = 0
        self.auto_config = AutoCollectionConfig()
        self.auto_motion_plan = None
        self.auto_base_head_q = None
        self.auto_ready_done = False
        self.auto_motion_running = False
        self.auto_stop_requested = False
        self.auto_motion_after_id = None
        self.include_head_motion = True
        self.connected_servo_mode = "all"

        self.warning_img = None
        self.last_result_path = None
        self.last_dataset_path = None

        self.build_ui()
        self.update_head_pose_status()
        self.update_dev_mode_label()

    # ============================================================
    # UI
    # ============================================================
    def build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.user_tab = ttk.Frame(notebook)
        self.dev_tab = ttk.Frame(notebook)

        notebook.add(self.user_tab, text="User")
        notebook.add(self.dev_tab, text="Developer")

        self.build_user_tab()
        self.build_dev_tab()

    def build_user_tab(self):
        frm = self.user_tab

        # connection
        conn = ttk.LabelFrame(frm, text="Connection")
        conn.pack(fill="x", padx=10, pady=10)

        ttk.Label(conn, text="RPC IP").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.user_ip = tk.StringVar(value="192.168.30.1:50051")
        ttk.Entry(conn, textvariable=self.user_ip, width=30).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(conn, text="Model").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.user_model = tk.StringVar(value="a")
        ttk.Combobox(
            conn,
            textvariable=self.user_model,
            values=["a", "m"],
            state="readonly",
            width=8
        ).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        ttk.Button(conn, text="Connect", command=self.user_connect).grid(row=0, column=4, padx=5, pady=5)

        ttk.Label(conn, text="Servo On").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.user_servo_mode = tk.StringVar(value="all")
        ttk.Radiobutton(conn, text="All", variable=self.user_servo_mode, value="all").grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(conn, text="No Head", variable=self.user_servo_mode, value="no_head").grid(row=1, column=2, padx=5, pady=5, sticky="w")

        self.user_status = tk.StringVar(value="Disconnected")
        ttk.Label(conn, textvariable=self.user_status).grid(row=0, column=5, padx=10, pady=5, sticky="w")

        # setup
        setup = ttk.LabelFrame(frm, text="Setup")
        setup.pack(fill="x", padx=10, pady=10)

        ttk.Label(setup, text="Arm").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.user_calib_arm = tk.StringVar(value="both_arm")
        ttk.Combobox(setup, textvariable=self.user_calib_arm, values=["both_arm", "right_arm", "left_arm"], state="readonly", width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        cb_frm = ttk.Frame(setup)
        cb_frm.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.user_cal_with_head = tk.BooleanVar(value=True)
        self.user_cal_with_head_cb = ttk.Checkbutton(cb_frm, text="cal_with_head", variable=self.user_cal_with_head)
        self.user_cal_with_head_cb.pack(side="left", padx=2)
        self.user_use_camera_ext = tk.BooleanVar(value=True)
        ttk.Checkbutton(cb_frm, text="use_camera_ext", variable=self.user_use_camera_ext).pack(side="left", padx=2)

        ttk.Label(setup, text="Mode: live").grid(row=0, column=3, padx=20, pady=5, sticky="w")
        self.user_head_status = tk.StringVar(value="Auto Motion: 0/0")
        ttk.Label(setup, textvariable=self.user_head_status).grid(row=1, column=0, columnspan=5, padx=5, pady=5, sticky="w")

        # actions
        act = ttk.LabelFrame(frm, text="Actions")
        act.pack(fill="x", padx=10, pady=10)

        ttk.Button(act, text="1.Zero Pose Check", command=self.user_zero_pose_check).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(act, text="2.Init Pose", command=self.user_init_pose).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(act, text="3.Auto Motion", command=self.user_auto_motion).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(act, text="4.Record(Current)", command=self.user_record, padding=(80, 48)).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(act, text="5.Calculate", command=self.user_calculate).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(act, text="6.Apply Home Offset", command=self.user_apply_home_offset).grid(row=0, column=5, padx=5, pady=5)
        ttk.Button(act, text="7.Clear Samples", command=self.clear_samples).grid(row=0, column=6, padx=5, pady=5)
        ttk.Button(act, text="8.All Auto Motion", command=self.user_all_auto_motion).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(act, text="9.Stop", command=self.user_stop_auto_motion).grid(row=2, column=2, padx=5, pady=5)

        self.user_count = tk.StringVar(value="Samples: 0")
        ttk.Label(act, textvariable=self.user_count).grid(row=3, column=0, columnspan=7, padx=5, pady=5, sticky="w")
        
        # result/log
        logfrm = ttk.LabelFrame(frm, text="Log / Result")
        logfrm.pack(fill="both", expand=True, padx=10, pady=10)

        self.user_text = tk.Text(logfrm, height=20)
        self.user_text.pack(fill="both", expand=True, padx=5, pady=5)

    def build_dev_tab(self):
        frm = self.dev_tab

        # connection
        conn = ttk.LabelFrame(frm, text="Connection")
        conn.pack(fill="x", padx=10, pady=10)

        ttk.Label(conn, text="RPC IP").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dev_ip = tk.StringVar(value="192.168.30.1:50051")
        self.dev_ip_entry = ttk.Entry(conn, textvariable=self.dev_ip, width=30)
        self.dev_ip_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(conn, text="Model").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.dev_model = tk.StringVar(value="a")
        ttk.Combobox(conn, textvariable=self.dev_model, values=["a", "m"], state="readonly", width=8)\
            .grid(row=0, column=3, padx=5, pady=5, sticky="w")
        ttk.Button(conn, text="Connect", command=self.dev_connect).grid(row=0, column=4, padx=5, pady=5)

        ttk.Label(conn, text="Servo On").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dev_servo_mode = tk.StringVar(value="all")
        ttk.Radiobutton(conn, text="All", variable=self.dev_servo_mode, value="all").grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(conn, text="No Head", variable=self.dev_servo_mode, value="no_head").grid(row=1, column=2, padx=5, pady=5, sticky="w")

        self.dev_status = tk.StringVar(value="Disconnected")
        ttk.Label(conn, textvariable=self.dev_status).grid(row=0, column=5, padx=10, pady=5, sticky="w")

        # config
        cfg = ttk.LabelFrame(frm, text="Config")
        cfg.pack(fill="x", padx=10, pady=10)

        ttk.Label(cfg, text="Arm").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dev_calib_arm = tk.StringVar(value="both_arm")
        ttk.Combobox(cfg, textvariable=self.dev_calib_arm, values=["both_arm", "right_arm", "left_arm"], state="readonly", width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(cfg, text="Mode").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.dev_mode = tk.StringVar(value="live")
        mode_box = ttk.Combobox(cfg, textvariable=self.dev_mode, values=["live", "npz", "sim"], state="readonly", width=10)
        mode_box.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        mode_box.bind("<<ComboboxSelected>>", self.update_dev_mode_label)

        cb_row = ttk.Frame(cfg)
        cb_row.grid(row=0, column=4, columnspan=4, sticky="w")
        self.dev_cal_with_head = tk.BooleanVar(value=True)
        self.dev_cal_with_head_cb = ttk.Checkbutton(cb_row, text="cal_with_head", variable=self.dev_cal_with_head)
        self.dev_cal_with_head_cb.pack(side="left", padx=2)
        self.dev_use_camera_ext = tk.BooleanVar(value=True)
        ttk.Checkbutton(cb_row, text="use_camera_ext", variable=self.dev_use_camera_ext).pack(side="left", padx=2)
        self.dev_use_sag = tk.BooleanVar(value=False)
        ttk.Checkbutton(cb_row, text="use_sag", variable=self.dev_use_sag).pack(side="left", padx=2)

        ttk.Label(cfg, text="Path").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dev_path = tk.StringVar(value="result/dataset_YYYYMMDD_HHMMSS.npz")
        ttk.Entry(cfg, textvariable=self.dev_path, width=40).grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="w")

        self.dev_mode_info = tk.StringVar(value="In live mode, Auto Motion records once and All Auto Motion runs the full sweep; Stop interrupts between steps.")
        ttk.Label(cfg, textvariable=self.dev_mode_info).grid(row=1, column=4, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Label(cfg, text="lambda_cam").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.dev_lambda_cam_pos = tk.StringVar(value=str(DEFAULT_LAMBDA_CAM_POS))
        self.dev_lambda_cam_rot = tk.StringVar(value=str(DEFAULT_LAMBDA_CAM_ROT))
        ttk.Entry(cfg, textvariable=self.dev_lambda_cam_pos, width=6).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Entry(cfg, textvariable=self.dev_lambda_cam_rot, width=6).grid(row=2, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(cfg, text="Solver").grid(row=2, column=3, padx=5, pady=5, sticky="w")
        self.user_solver = tk.StringVar(value="Least Squares")
        self.solver_cb = ttk.Combobox(cfg, textvariable=self.user_solver, values=["Least Squares", "QP Solver"], state="readonly", width=12)
        self.solver_cb.grid(row=2, column=4, padx=5, pady=5, sticky="w")

        try:
            import qpsolvers
            has_qp = True
        except ImportError:
            has_qp = False
            
        if not has_qp:
            self.solver_cb["state"] = "disabled"
            
        ttk.Label(cfg, text="Auto Motion").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        
        auto_frm = ttk.Frame(cfg)
        auto_frm.grid(row=3, column=1, columnspan=6, sticky="w", pady=5)
        
        ttk.Label(auto_frm, text="Angle(deg):").pack(side="left", padx=(0, 2))
        self.dev_angle_step = tk.DoubleVar(value=5.0)
        ttk.Entry(auto_frm, textvariable=self.dev_angle_step, width=5).pack(side="left", padx=(0, 10))
        
        ttk.Label(auto_frm, text="Pos(m):").pack(side="left", padx=(0, 2))
        self.dev_pos_step = tk.DoubleVar(value=0.03)
        ttk.Entry(auto_frm, textvariable=self.dev_pos_step, width=5).pack(side="left", padx=(0, 10))
        
        ttk.Label(auto_frm, text="Max X(m):").pack(side="left", padx=(0, 2))
        self.dev_max_x = tk.DoubleVar(value=0.5)
        ttk.Entry(auto_frm, textvariable=self.dev_max_x, width=5).pack(side="left", padx=(0, 10))

        self.dev_est_samples = tk.StringVar(value="Est. Samples: 0")
        ttk.Label(auto_frm, textvariable=self.dev_est_samples, foreground="blue").pack(side="left", padx=(5, 0))

        self.dev_angle_step.trace_add("write", self.update_est_samples)
        self.dev_pos_step.trace_add("write", self.update_est_samples)
        self.dev_max_x.trace_add("write", self.update_est_samples)
        self.update_est_samples()

        self.dev_head_status = tk.StringVar(value="Auto Motion: 0/0")
        ttk.Label(cfg, textvariable=self.dev_head_status).grid(row=4, column=0, columnspan=7, padx=5, pady=5, sticky="w")

        # actions
        act = ttk.LabelFrame(frm, text="Actions")
        act.pack(fill="x", padx=10, pady=10)

        ttk.Button(act, text="1.Zero Pose Check", command=self.dev_zero_pose_check).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(act, text="2.Init Pose", command=self.dev_init_pose).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(act, text="3.Auto Motion", command=self.dev_auto_motion).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(act, text="4.Record(Current)", command=self.dev_record).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(act, text="5.Calculate", command=self.dev_calculate).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(act, text="6.Apply Home Offset", command=self.dev_apply_home_offset).grid(row=0, column=5, padx=5, pady=5)
        ttk.Button(act, text="7.Clear Samples", command=self.clear_samples).grid(row=0, column=6, padx=5, pady=5)
        ttk.Button(act, text="8.All Auto Motion", command=self.dev_all_auto_motion).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(act, text="9.Stop", command=self.dev_stop_auto_motion).grid(row=2, column=2, padx=5, pady=5)

        self.dev_count = tk.StringVar(value="Shared Samples: 0")
        ttk.Label(act, textvariable=self.dev_count).grid(row=3, column=0, columnspan=7, padx=5, pady=5, sticky="w")

        # result/log
        logfrm = ttk.LabelFrame(frm, text="Log / Result")
        logfrm.pack(fill="both", expand=True, padx=10, pady=10)

        self.dev_text = tk.Text(logfrm, height=20)
        self.dev_text.pack(fill="both", expand=True, padx=5, pady=5)

    # ============================================================
    # Common helpers
    # ============================================================

    # log
    def log(self, widget, msg):
        widget.insert("end", msg + "\n")
        widget.see("end")
        self.root.update_idletasks()

    # path
    def ensure_result_dir(self):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        return RESULT_DIR

    def build_output_paths(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.ensure_result_dir()
        dataset_path = result_dir / f"dataset_{timestamp}.npz"
        result_path = result_dir / f"result_{timestamp}.json"
        return dataset_path, result_path

    def get_latest_result_path(self):
        if self.last_result_path is not None and self.last_result_path.exists():
            return self.last_result_path

        result_dir = self.ensure_result_dir()
        result_files = sorted(
            result_dir.glob("result_*.json"),
            key=lambda file_path: file_path.stat().st_mtime,
            reverse=True,
        )
        if not result_files:
            raise RuntimeError(f"No calibration result JSON found in {result_dir}")

        self.last_result_path = result_files[0]
        return self.last_result_path
    
    def resolve_input_path(self, raw_path):
        input_path = Path(raw_path).expanduser()
        if input_path.is_absolute():
            return input_path
        return BASE_DIR / input_path
    
    # dataset sample
    def update_sample_counts(self):
        sample_count = len(self.shared_arm_q_list)
        self.user_count.set(f"Samples: {sample_count}")
        self.dev_count.set(f"Shared Samples: {sample_count}")

    def get_auto_pose_target_count(self):
        if self.auto_motion_plan is not None:
            return len(self.auto_motion_plan)
        return 0

    def update_head_pose_status(self):
        pose_target_count = self.get_auto_pose_target_count()
        pose_idx = min(self.head_move_count, pose_target_count)
        label = f"Auto Motion: {pose_idx}/{pose_target_count}"
        if not self.include_head_motion:
            label += " (headless)"
        self.user_head_status.set(label)
        self.dev_head_status.set(label)

    def clear_samples(self):
        self.stop_all_auto_motion_internal(cancel_robot=True)
        self.shared_arm_q_list.clear()
        self.shared_head_q_list.clear()
        self.shared_T_list.clear()
        self.head_move_count = 0
        self.auto_base_head_q = None
        self.auto_ready_done = False
        self.update_sample_counts()
        self.update_head_pose_status()
        self.log(self.user_text, "Shared samples cleared.")
        self.log(self.dev_text, "Shared samples cleared.")

    

    def get_active_arms(self, tab="user"):
        val = self.user_calib_arm.get() if tab == "user" else self.dev_calib_arm.get()
        if val == "both_arm":
            return ["right", "left"]
        elif val == "right_arm":
            return ["right"]
        elif val == "left_arm":
            return ["left"]
        return ["right", "left"]

    # robot 
    def connect_robot(self, ip, model_name, status_var, text_widget):
        try:
            self.include_head_motion = self.servo_mode_includes_head(self.connected_servo_mode)
            self.robot = create_robot(
                ip,
                model_name,
                power_regex=".*",
                servo_regex=self.servo_mode_to_regex(self.connected_servo_mode),
            )
            self.dyn_model = self.robot.get_dynamics()
            self.model = self.robot.model()
            
            if len(self.model.head_idx) == 0:
                self.user_cal_with_head.set(False)
                self.dev_cal_with_head.set(False)
                self.user_cal_with_head_cb.config(state="disabled")
                self.dev_cal_with_head_cb.config(state="disabled")
                self.log(text_widget, "No head joints detected. cal_with_head disabled and set to False (Torso base).")
            else:
                self.user_cal_with_head_cb.config(state="normal")
                self.dev_cal_with_head_cb.config(state="normal")

            self.auto_motion_running = False
            self.auto_stop_requested = False
            self.auto_motion_after_id = None
            self.auto_base_head_q = None
            self.auto_ready_done = False
            status_var.set("Connected")
            self.log(text_widget, f"Connected: {ip} (model={model_name}, servo={self.connected_servo_mode})")
            self.update_head_pose_status()
        except Exception as e:
            status_var.set("Disconnected")
            messagebox.showerror("Connection Error", str(e))
            self.log(text_widget, f"Connection failed: {e}")

    def servo_mode_to_regex(self, servo_mode):
        if servo_mode == "no_head":
            return r"torso_.*|right_arm_.*|left_arm_.*"
        return r".*"

    def servo_mode_includes_head(self, servo_mode):
        return servo_mode != "no_head"

    def get_capture_head_idx(self):
        if self.model is None:
            return None
        return get_head_config(self.model)["head_idx"]
    
    # robot motion
    def move_to_auto_init_pose(self, text_widget, tab="user"):
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.model is None:
            raise RuntimeError("Robot is not connected.")

        active_arms = self.get_active_arms(tab)

        self.log(text_widget, f"Moving to auto init pose (active_arms: {active_arms})...")
        move_to_auto_ready_pose(
            robot=self.robot,
            active_arms=active_arms,
            minimum_time=10.0,
            priority=self.auto_config.priority,
        )
        self.auto_ready_done = True
        self.auto_base_head_q = None

        if tab == "dev":
            try:
                self.auto_config.angle_step_deg = float(self.dev_angle_step.get())
                self.auto_config.position_step_m = float(self.dev_pos_step.get())
                self.auto_config.max_x = float(self.dev_max_x.get())
            except Exception as e:
                self.log(text_widget, f"Failed to read dev auto config: {e}. Using current values.")

        self.auto_motion_plan = None

        if self.include_head_motion:
            head_cfg = get_head_config(self.model)
            if head_cfg["head_idx"] is not None:
                self.auto_base_head_q = self.robot.get_state().position[head_cfg["head_idx"]].copy()
                self.log(text_widget, f"Auto base head pose (deg): {np.round(np.rad2deg(self.auto_base_head_q), 3)}")
            else:
                self.auto_base_head_q = None
                self.include_head_motion = False

        self.head_move_count = 0
        self.update_head_pose_status()
        self.update_est_samples()
        
        messagebox.showinfo(
            "Teaching Required",
            "Robot has moved to the initial pose.\n\n"
            "Please use the Teaching button to adjust the robot's pose so that the marker is clearly visible to the camera.\n"
            "Once adjusted, press 'Auto Motion' or 'All Auto Motion' to start the sequence."
        )

    def run_auto_motion_step(self, text_widget, tab="user"):
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.model is None:
            raise RuntimeError("Robot is not connected.")

        if self.dev_mode.get() != "sim" and self.marker_transform is None:
            self.marker_transform = create_live_marker_transform()

        pose_target = self.get_auto_pose_target_count()
        if self.head_move_count >= pose_target:
            self.log(text_widget, "All auto motions have already been executed.")
            return

        if not self.auto_ready_done:
            raise RuntimeError("Please move to Init Pose first.")

        # Re-build incremental motion plan based on the CURRENT (possibly teached) pose
        if self.auto_motion_plan is None or self.head_move_count == 0:
            if tab == "dev":
                try:
                    self.auto_config.angle_step_deg = float(self.dev_angle_step.get())
                    self.auto_config.position_step_m = float(self.dev_pos_step.get())
                    self.auto_config.max_x = float(self.dev_max_x.get())
                except Exception as e:
                    self.log(text_widget, f"Failed to read dev auto config: {e}. Using current values.")
            
            self.log(text_widget, f"Building motion plan based on current pose... (Angle={self.auto_config.angle_step_deg}deg, Pos={self.auto_config.position_step_m}m, MaxX={self.auto_config.max_x}m)")
            self.auto_motion_plan = build_incremental_motion_plan(
                self.robot, self.dyn_model, self.auto_config
            )
            self.update_head_pose_status()
            self.update_est_samples()

        if self.include_head_motion and self.auto_base_head_q is None:
            head_cfg = get_head_config(self.model)
            if head_cfg["head_idx"] is not None:
                self.auto_base_head_q = self.robot.get_state().position[head_cfg["head_idx"]].copy()
                self.log(text_widget, f"Auto base head pose (deg): {np.round(np.rad2deg(self.auto_base_head_q), 3)}")
            else:
                self.auto_base_head_q = None
                self.include_head_motion = False

        active_arms = self.get_active_arms(tab)
        motion_plan_step = self.auto_motion_plan[self.head_move_count]
        
        motion_info = execute_auto_motion_step(
            robot=self.robot,
            config=self.auto_config,
            motion_plan_step=motion_plan_step,
            active_arms=active_arms,
        )
        self.log(
            text_widget,
            f"Auto motion done: {motion_plan_step['desc']}",
        )

        q_arm, q_head, T_meas = self.capture_one_sample(text_widget)
        if q_arm is None:
            self.head_move_count += 1
            self.update_head_pose_status()
            if self.motion_test_mode.get():
                self.log(text_widget, "Capture failed after motion. (Motion Test: Continuing...)")
                return True
            else:
                self.log(text_widget, "Capture failed after motion. This pose is skipped.")
                return False

        self.shared_arm_q_list.append(q_arm)
        if q_head is not None:
            self.shared_head_q_list.append(q_head)
        self.shared_T_list.append(T_meas)
        self.head_move_count += 1
        self.update_sample_counts()
        self.update_head_pose_status()
        return True

    def move_to_next_auto_motion(self, text_widget):
        if self.auto_motion_running:
            raise RuntimeError("All Auto Motion is running. Press Stop first.")
        self.run_auto_motion_step(text_widget)

    def stop_all_auto_motion_internal(self, cancel_robot=False, reset_stop_requested=True):
        if self.auto_motion_after_id is not None:
            try:
                self.root.after_cancel(self.auto_motion_after_id)
            except Exception:
                pass
            self.auto_motion_after_id = None
        self.auto_motion_running = False
        if reset_stop_requested:
            self.auto_stop_requested = False

        if cancel_robot and self.robot is not None:
            try:
                self.robot.cancel_control()
            except Exception:
                pass

    def request_stop_all_auto_motion(self, text_widget):
        if not self.auto_motion_running and self.auto_motion_after_id is None:
            self.log(text_widget, "No All Auto Motion sequence is running.")
            self.stop_all_auto_motion_internal(cancel_robot=True)
            return

        self.auto_stop_requested = True
        self.stop_all_auto_motion_internal(cancel_robot=True, reset_stop_requested=False)
        self.log(text_widget, "Stop requested. Sent robot.cancel_control(); the all-auto sequence stops after the current step.")

    def run_all_auto_motion_sequence(self, text_widget, tab="user"):
        if self.auto_stop_requested:
            self.log(text_widget, "Auto Motion stopped by user.")
            self.auto_motion_running = False
            self.auto_stop_requested = False
            return

        pose_target = self.get_auto_pose_target_count()
        if self.head_move_count >= pose_target:
            self.log(text_widget, "All auto motions completed.")
            self.auto_motion_running = False
            return

        try:
            ok = self.run_auto_motion_step(text_widget, tab)
            if ok is False:
                self.log(text_widget, "Error capturing sample. Stopping.")
                self.auto_motion_running = False
                return
        except Exception as e:
            self.log(text_widget, f"Auto motion failed: {e}")
            self.auto_motion_running = False
            return

        self.auto_motion_after_id = self.root.after(
            10,
            lambda: self.run_all_auto_motion_sequence(text_widget, tab),
        )

    def move_to_all_auto_motions(self, text_widget, tab="user"):
        if not self.auto_ready_done:
            raise RuntimeError("Please move to Init Pose first.")

        if self.auto_motion_plan is None or len(self.auto_motion_plan) == 0:
            self.log(text_widget, "Motion plan is missing or empty. Re-building...")
            self.auto_motion_plan = build_incremental_motion_plan(
                self.robot, self.dyn_model, self.auto_config
            )

        pose_target = self.get_auto_pose_target_count()
        if self.head_move_count >= pose_target:
            self.log(text_widget, "All auto motions have already been executed.")
            return

        if self.auto_motion_running or self.auto_motion_after_id is not None:
            self.log(text_widget, "All Auto Motion is already running.")
            return

        self.auto_stop_requested = False
        self.auto_motion_running = True
        self.log(text_widget, "All Auto Motion started. Press Stop to interrupt between steps.")
        self.run_all_auto_motion_sequence(text_widget, tab)


    # record 
    def capture_one_sample(self, text_widget):
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")

        cfg = get_both_arm_config(self.model)
        head_idx = self.get_capture_head_idx()

        # In sim mode bypass camera and return dummy marker data
        if self.dev_mode.get() == "sim":
            state = self.robot.get_state()
            q_full = state.position.copy()
            q_arm = q_full[cfg["arm_idx"]].copy()
            q_head = q_full[head_idx].copy() if head_idx is not None else None
            # Return identity matrices as dummy marker measurements
            T_meas = np.stack([np.eye(4), np.eye(4)], axis=0)
            self.log(text_widget, "Sim/Test mode: Bypassing camera capture, using dummy marker data.")
        else:
            if self.marker_transform is None:
                self.marker_transform = create_live_marker_transform()

            q_arm, q_head, T_meas = capture_robot_sample(
                robot=self.robot,
                arm_idx=cfg["arm_idx"],
                marker_transform=self.marker_transform,
                head_idx=head_idx,
                side="all",
            )
        if T_meas is None:
            self.log(text_widget, "Marker not detected.")
            return None, None, None

        self.log(text_widget, f"Captured sample")
        self.log(text_widget, f"q_arm = {np.round(q_arm, 3)}")
        if q_head is not None:
            self.log(text_widget, f"q_head = {np.round(q_head, 3)}")
        else:
            self.log(text_widget, "q_head = None")
        self.log(text_widget, f"marker_right =\n{np.round(T_meas[0], 3)}")
        self.log(text_widget, f"marker_left =\n{np.round(T_meas[1], 3)}")
        return q_arm, q_head, T_meas

    # optimize
    def run_optimizer(
        self,
        active_arms,
        optimize_head,
        optimize_camera,
        q_arm_list,
        q_head_list,
        T_meas_list,
        result_path,
        text_widget,
        lambda_cam_pos=DEFAULT_LAMBDA_CAM_POS,
        lambda_cam_rot=DEFAULT_LAMBDA_CAM_ROT,
        solver_type="Least Squares",
        use_sag=False,
    ):
        if self.model is None:
            raise RuntimeError("Robot is not connected.")

        if len(active_arms) == 1:
            cfg = get_arm_config(self.model, active_arms[0])
            ee_links = {active_arms[0]: cfg["ee_link"]}
            ee_to_marker_nom = {active_arms[0]: cfg["ee_to_marker_nom"]}
        else:
            cfg = get_both_arm_config(self.model)
            ee_links = cfg["ee_links"]
            ee_to_marker_nom = cfg["ee_to_marker_nom"]

        head_cfg = get_head_config(self.model)
        
        ndof_val = 22 if (optimize_head and len(active_arms) == 2 and optimize_camera) else 22 # simple fallback
        
        if solver_type == "QP Solver":
            optimizer = QPCalibrationOptimizer(
                robot=self.robot,
                arm_idx=cfg["arm_idx"],
                ee_links=ee_links,
                mount_to_cam_nom=cfg["mount_to_cam_nom"],
                ee_to_marker_nom=ee_to_marker_nom,
                ndof=ndof_val,
                head_idx=head_cfg["head_idx"],
                lambda_cam_pos=lambda_cam_pos,
                lambda_cam_rot=lambda_cam_rot,
                use_sag=use_sag,
            )
        else:
            optimizer = CalibrationOptimizer(
                robot=self.robot,
                arm_idx=cfg["arm_idx"],
                ee_links=ee_links,
                mount_to_cam_nom=cfg["mount_to_cam_nom"],
                t5_to_cam_nom=cfg.get("t5_to_cam_nom"),
                ee_to_marker_nom=ee_to_marker_nom,
                active_arms=active_arms,
                optimize_arm=True,
                optimize_head=optimize_head,
                optimize_camera=optimize_camera,
                head_idx=head_cfg["head_idx"],
                use_head_kinematics=optimize_head,
                lambda_cam_pos=lambda_cam_pos,
                lambda_cam_rot=lambda_cam_rot,
                use_sag=use_sag,
            )

        q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, t5_to_cam_new = optimizer.optimize(
            q_arm_list,
            q_head_list,
            T_meas_list,
        )
        right_arm_offset, left_arm_offset = split_arm_offsets(q_arm_offset)

        t5_to_cam_new = [float(x) for x in t5_to_cam_new] if t5_to_cam_new else None
        mount_to_cam_new = [float(x) for x in mount_to_cam_new] if mount_to_cam_new else None

        self.log(text_widget, "\n===== RESULT =====")
        self.log(text_widget, f"lambda_cam_pos = {lambda_cam_pos}")
        self.log(text_widget, f"lambda_cam_rot = {lambda_cam_rot}")
        self.log(text_widget, "Right arm joint offset (deg):")
        self.log(text_widget, str(np.rad2deg(right_arm_offset)))
        if left_arm_offset is not None:
            self.log(text_widget, "Left arm joint offset (deg):")
            self.log(text_widget, str(np.rad2deg(left_arm_offset)))
        if q_head_offset is not None:
            self.log(text_widget, "Head joint offset (deg):")
            self.log(text_widget, str(np.rad2deg(q_head_offset)))
        
        if optimize_head:
            self.log(text_widget, "mount_to_cam xi:")
            self.log(text_widget, str(xi_cam))
            self.log(text_widget, "mount_to_cam_new:")
            self.log(text_widget, str(mount_to_cam_new))
        else:
            self.log(text_widget, "T5-to-camera xi:")
            self.log(text_widget, str(xi_cam))
            self.log(text_widget, "t5_to_cam_new:")
            self.log(text_widget, str(t5_to_cam_new))

        result_dict = {
            "joint_offset_deg": np.rad2deg(q_arm_offset).tolist(),
            "right_arm_joint_offset_deg": np.rad2deg(right_arm_offset).tolist(),
            "left_arm_joint_offset_deg": np.rad2deg(left_arm_offset).tolist() if left_arm_offset is not None else None,
            "head_joint_offset_deg": np.rad2deg(q_head_offset).tolist() if q_head_offset is not None else None,
            "xi_cam": np.array(xi_cam).tolist(),
        }

        if optimize_head:
            result_dict["xi_mount_cam"] = result_dict["xi_cam"]
        else:
            result_dict["xi_t5_cam"] = result_dict["xi_cam"]

        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=4)

        self.last_result_path = result_path
        self.log(text_widget, f"Result saved to {result_path}")

    # home offset
    def confirm_home_offset_action(self):
        popup = tk.Toplevel(self.root)
        popup.title("Warning")
        popup.geometry("650x620")
        popup.transient(self.root)
        popup.grab_set()

        result = {"ok": False}

        msg = (
            "Applying home offset will move the robot.\n"
            "The robot may move toward zero pose first.\n\n"
            "Before continuing, make sure:\n"
            "- the workspace is clear\n"
            "- there is no collision risk\n"
            "- the robot is in a safe condition to move\n"
        )

        ttk.Label(
            popup,
            text=msg,
            justify="left"
        ).pack(padx=15, pady=15, anchor="w")

        image_path = WARNING_POSE_PATH
        if image_path.exists():
            try:
                self.warning_img = tk.PhotoImage(file=str(image_path))
                self.warning_img = self.warning_img.subsample(3, 3)   
                ttk.Label(popup, image=self.warning_img).pack(padx=10, pady=10)
            except Exception:
                ttk.Label(popup, text=f"Failed to load image: {image_path}").pack(pady=10)
        else:
            ttk.Label(popup, text=f"Image not found: {image_path}").pack(pady=10)

        btn_frame = ttk.Frame(popup)
        btn_frame.pack(pady=15)

        def do_continue():
            result["ok"] = True
            popup.destroy()

        def do_cancel():
            popup.destroy()

        ttk.Button(btn_frame, text="Continue", command=do_continue).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Cancel", command=do_cancel).pack(side="left", padx=10)

        popup.wait_window()
        return result["ok"]

    def apply_home_offset_common(self, ip, model_name, arm, servo_regex, include_head, text_widget):
        result_path = self.get_latest_result_path()

        proceed = self.confirm_home_offset_action()
        if not proceed:
            self.log(text_widget, "Apply home offset cancelled.")
            return

        result = apply_home_offset_from_json(
            address=ip,
            model_name=model_name,
            arm=arm,
            json_path=str(result_path),
            power=".*",
            servo=servo_regex,
            include_head=include_head,
        )

        self.log(text_widget, "Home offset applied successfully.")
        self.log(text_widget, f"Arm: {result['arm']}")
        self.log(text_widget, f"Source: {result['source']}")
        self.log(text_widget, f"JSON: {result['json_path']}")
        if result.get("right_offset_deg") is not None:
            self.log(text_widget, f"Right Offset (deg): {result['right_offset_deg']}")
        if result.get("left_offset_deg") is not None:
            self.log(text_widget, f"Left Offset (deg): {result['left_offset_deg']}")
        self.log(text_widget, f"Offset (deg): {result['offset_deg']}")
        if result.get("head_offset_deg") is not None:
            self.log(text_widget, f"Head Offset (deg): {result['head_offset_deg']}")

    def show_zero_pose_check_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Zero Pose Check")
        popup.geometry("760x860")
        popup.transient(self.root)
        popup.grab_set()

        msg = (
            "The robot has moved to zero pose.\n\n"
            "Please compare the actual robot posture with the reference image.\n\n"
            "- If the posture matches the reference, you can proceed with data collection.\n"
            "- If the posture does not match and the two target joints appear outside the recommended range,\n"
            "  use direct teaching to move the robot to the recommended posture,\n"
            "  perform reset first, and then start data collection."
        )

        ttk.Label(popup, text=msg, justify="left").pack(padx=15, pady=15, anchor="w")

        image_paths = [WARNING_POSE_CHECK_PATH, WARNING_POSE_PATH]

        for image_path in image_paths:
            if image_path.exists():
                try:
                    img = tk.PhotoImage(file=str(image_path))
                    img = img.subsample(3, 3)
                    lbl = ttk.Label(popup, image=img)
                    lbl.image = img
                    lbl.pack(padx=10, pady=8)
                except Exception:
                    ttk.Label(popup, text=f"Failed to load image: {image_path}").pack(pady=5)

        ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=15)
        popup.wait_window()

    def zero_pose_check_common(self, ip, model_name, arm, servo_regex, include_head, text_widget):
        result = move_robot_to_zero_pose(
            address=ip,
            model_name=model_name,
            arm=arm,
            power=".*",
            servo=servo_regex,
            include_head=include_head,
        )

        self.log(text_widget, "\n===== ZERO POSE CHECK =====")
        self.log(text_widget, f"Arm: {result['arm']}")
        self.log(text_widget, result["message"])

        self.show_zero_pose_check_popup()

    # ============================================================
    # User tab
    # ============================================================
    def user_connect(self):
        self.connected_servo_mode = self.user_servo_mode.get()
        self.include_head_motion = self.servo_mode_includes_head(self.connected_servo_mode)
        self.connect_robot(self.user_ip.get(), self.user_model.get(), self.user_status, self.user_text)

    def user_zero_pose_check(self):
        try:
            servo_mode = self.user_servo_mode.get()
            self.zero_pose_check_common(
                ip=self.user_ip.get(),
                model_name=self.user_model.get(),
                arm=FIXED_CALIB_ARM,
                servo_regex=self.servo_mode_to_regex(servo_mode),
                include_head=self.servo_mode_includes_head(servo_mode),
                text_widget=self.user_text,
            )
        except Exception as e:
            messagebox.showerror("Zero Pose Check Error", str(e))
            self.log(self.user_text, f"Zero pose check failed: {e}")

    def user_init_pose(self):
        try:
            self.move_to_auto_init_pose(self.user_text, tab="user")
        except Exception as e:
            messagebox.showerror("Init Error", str(e))
            self.log(self.user_text, f"Init pose failed: {e}")

    def user_auto_motion(self):
        try:
            self.run_auto_motion_step(self.user_text, tab="user")
        except Exception as e:
            messagebox.showerror("Auto Motion Error", str(e))
            self.log(self.user_text, f"Auto motion failed: {e}")

    def user_all_auto_motion(self):
        try:
            self.move_to_all_auto_motions(self.user_text, tab="user")
        except Exception as e:
            messagebox.showerror("All Auto Motion Error", str(e))
            self.log(self.user_text, f"All auto motion failed: {e}")

    def user_stop_auto_motion(self):
        try:
            self.request_stop_all_auto_motion(self.user_text)
        except Exception as e:
            messagebox.showerror("Stop Error", str(e))
            self.log(self.user_text, f"Stop failed: {e}")

    def user_record(self):
        try:
            q_arm, q_head, T_meas = self.capture_one_sample(self.user_text)
            if q_arm is None:
                return
            self.shared_arm_q_list.append(q_arm)
            if q_head is not None:
                self.shared_head_q_list.append(q_head)
            self.shared_T_list.append(T_meas)
            self.update_sample_counts()
        except Exception as e:
            messagebox.showerror("Record Error", str(e))
            self.log(self.user_text, f"Record failed: {e}")

    def user_calculate(self):
        try:
            if len(self.shared_arm_q_list) == 0:
                messagebox.showwarning("Warning", "No recorded samples.")
                return

            q_arm_list = np.array(self.shared_arm_q_list)
            q_head_list = np.array(self.shared_head_q_list) if self.shared_head_q_list else None
            T_meas_list = np.array(self.shared_T_list)
            dataset_path, result_path = self.build_output_paths()

            arm_val = self.user_calib_arm.get()
            active_arms = ["right", "left"] if arm_val == "both_arm" else [arm_val.replace("_arm", "")]
            optimize_head = self.user_cal_with_head.get()
            optimize_camera = self.user_use_camera_ext.get()

            validate_dataset(q_arm_list, q_head_list, T_meas_list, optimize_head, active_arms)
            save_npz_dataset(dataset_path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)
            self.last_dataset_path = dataset_path
            self.log(self.user_text, f"Dataset saved to {dataset_path}")
            self.log(self.user_text, f"Optimization: arms={active_arms}, head={optimize_head}, cam={optimize_camera}")

            self.run_optimizer(
                active_arms=active_arms,
                optimize_head=optimize_head,
                optimize_camera=optimize_camera,
                q_arm_list=q_arm_list,
                q_head_list=q_head_list,
                T_meas_list=T_meas_list,
                result_path=result_path,
                text_widget=self.user_text,
                solver_type=self.user_solver.get(),
            )
        except Exception as e:
            messagebox.showerror("Calculate Error", str(e))
            self.log(self.user_text, f"Calculate failed: {e}")

    def user_apply_home_offset(self):
        try:
            servo_mode = self.user_servo_mode.get()
            arm_val = self.user_calib_arm.get()
            arm = "both" if arm_val == "both_arm" else arm_val.replace("_arm", "")
            self.apply_home_offset_common(
                ip=self.user_ip.get(),
                model_name=self.user_model.get(),
                arm=arm,
                servo_regex=self.servo_mode_to_regex(servo_mode),
                include_head=self.servo_mode_includes_head(servo_mode),
                text_widget=self.user_text,
            )
        except Exception as e:
            messagebox.showerror("Apply Home Offset Error", str(e))
            self.log(self.user_text, f"Apply home offset failed: {e}")

    # ============================================================
    # Developer tab
    # ============================================================
    def update_dev_mode_label(self, event=None):
        mode = self.dev_mode.get()
        if mode == "live":
            self.dev_mode_info.set("In live mode, Auto Motion records once and All Auto Motion runs the full sweep; Stop interrupts between steps.")
        elif mode == "sim":
            self.dev_ip.set("127.0.0.1:50051")
            self.dev_ip_entry.config(state="disabled")
            if self.dev_cal_with_head.get():
                self.dev_mode_info.set("sim mode uses simultaneous samples with head sweep. (IP locked to 127.0.0.1)")
            else:
                self.dev_mode_info.set("sim mode uses simultaneous samples. (IP locked to 127.0.0.1)")
        else:
            self.dev_ip_entry.config(state="normal")

    def dev_connect(self):
        self.connected_servo_mode = self.dev_servo_mode.get()
        self.include_head_motion = self.servo_mode_includes_head(self.connected_servo_mode)
        self.connect_robot(self.dev_ip.get(), self.dev_model.get(), self.dev_status, self.dev_text)

    def dev_zero_pose_check(self):
        try:
            servo_mode = self.dev_servo_mode.get()
            arm_val = self.dev_calib_arm.get()
            arm = "both" if arm_val == "both_arm" else arm_val.replace("_arm", "")
            self.zero_pose_check_common(
                ip=self.dev_ip.get(),
                model_name=self.dev_model.get(),
                arm=arm,
                servo_regex=self.servo_mode_to_regex(servo_mode),
                include_head=self.servo_mode_includes_head(servo_mode),
                text_widget=self.dev_text,
            )
        except Exception as e:
            messagebox.showerror("Zero Pose Check Error", str(e))
            self.log(self.dev_text, f"Zero pose check failed: {e}")

    def dev_init_pose(self):
        try:
            self.move_to_auto_init_pose(self.dev_text, tab="dev")
        except Exception as e:
            messagebox.showerror("Init Error", str(e))
            self.log(self.dev_text, f"Init pose failed: {e}")

    def dev_auto_motion(self):
        try:
            mode = self.dev_mode.get()
            if mode not in ["live", "sim"]:
                self.log(self.dev_text, "Auto motion is only available in live or sim mode.")
                return

            self.run_auto_motion_step(self.dev_text, tab="dev")
        except Exception as e:
            messagebox.showerror("Auto Motion Error", str(e))
            self.log(self.dev_text, f"Auto motion failed: {e}")

    def dev_all_auto_motion(self):
        try:
            mode = self.dev_mode.get()
            if mode not in ["live", "sim"]:
                self.log(self.dev_text, "All Auto motion is only available in live or sim mode.")
                return

            self.move_to_all_auto_motions(self.dev_text, tab="dev")
        except Exception as e:
            messagebox.showerror("All Auto Motion Error", str(e))
            self.log(self.dev_text, f"All auto motion failed: {e}")

    def dev_stop_auto_motion(self):
        try:
            self.request_stop_all_auto_motion(self.dev_text)
        except Exception as e:
            messagebox.showerror("Stop Error", str(e))
            self.log(self.dev_text, f"Stop failed: {e}")

    def dev_record(self):
        try:
            if self.dev_mode.get() != "live":
                messagebox.showwarning("Warning", "Record is available only in live mode.")
                return

            q_arm, q_head, T_meas = self.capture_one_sample(self.dev_text)
            if q_arm is None:
                return
            self.shared_arm_q_list.append(q_arm)
            if q_head is not None:
                self.shared_head_q_list.append(q_head)
            self.shared_T_list.append(T_meas)
            self.update_sample_counts()
        except Exception as e:
            messagebox.showerror("Record Error", str(e))
            self.log(self.dev_text, f"Record failed: {e}")

    def dev_get_lambda_cam(self):
        raw_value = self.dev_lambda_cam_pos.get().strip()
        if not raw_value:
            raise ValueError("lambda_cam is empty.")

        lambda_cam = float(raw_value)
        if lambda_cam < 0.0:
            raise ValueError("lambda_cam must be greater than or equal to 0.")

        return lambda_cam
    
    def dev_get_lambda_cam_rot(self):
        raw_value = self.dev_lambda_cam_rot.get().strip()
        if not raw_value:
            raise ValueError("lambda_cam is empty.")

        lambda_cam_rot = float(raw_value)
        if lambda_cam_rot < 0.0:
            raise ValueError("lambda_cam must be greater than or equal to 0.")

        return lambda_cam_rot


    def dev_calculate(self):
        try:
            mode = self.dev_mode.get()
            
            arm_val = self.dev_calib_arm.get()
            active_arms = ["right", "left"] if arm_val == "both_arm" else [arm_val.replace("_arm", "")]
            optimize_head = self.dev_cal_with_head.get()
            optimize_camera = self.dev_use_camera_ext.get()
            
            lambda_cam_pos = self.dev_get_lambda_cam()
            lambda_cam_rot = self.dev_get_lambda_cam_rot()
            if self.model is None:
                raise RuntimeError("Robot is not connected.")

            if not self.include_head_motion and optimize_head:
                optimize_head = False
                self.log(self.dev_text, "Headless mode selected; optimize_head changed to False.")

            if len(active_arms) == 1:
                cfg = get_arm_config(self.model, active_arms[0])
                ee_links = {active_arms[0]: cfg["ee_link"]}
                ee_to_marker_nom = {active_arms[0]: cfg["ee_to_marker_nom"]}
            else:
                cfg = get_both_arm_config(self.model)
                ee_links = cfg["ee_links"]
                ee_to_marker_nom = cfg["ee_to_marker_nom"]
                
            head_cfg = get_head_config(self.model)

            if mode == "live":
                if len(self.shared_arm_q_list) == 0:
                    messagebox.showwarning("Warning", "No recorded samples.")
                    return

                q_arm_list = np.array(self.shared_arm_q_list)
                q_head_list = np.array(self.shared_head_q_list) if self.shared_head_q_list else None
                T_meas_list = np.array(self.shared_T_list)

            elif mode == "npz":
                npz_path = self.resolve_input_path(self.dev_path.get())
                q_arm_list, q_head_list, T_meas_list = load_npz_dataset(npz_path)
                validate_dataset(q_arm_list, q_head_list, T_meas_list, optimize_head, active_arms)
                self.log(self.dev_text, f"Loaded npz: {npz_path}")
                self.log(self.dev_text, f"samples = {len(q_arm_list)}")

            else:  # sim
                sample_count = 100
                q_arm_list = np.random.uniform(-5, 5, (sample_count, 7 * len(active_arms)))
                if optimize_head:
                    q_head_list = np.column_stack([
                        np.random.uniform(np.deg2rad(-15.0), np.deg2rad(15.0), sample_count),
                        np.random.uniform(np.deg2rad(-15.0), np.deg2rad(15.0), sample_count),
                    ])
                else:
                    q_head_ref = self.robot.get_state().position[head_cfg["head_idx"]].copy()
                    q_head_list = np.tile(q_head_ref, (sample_count, 1))
                q_nominal = self.robot.get_state().position.copy()
                T_meas_list = generate_sim_measurements(
                    self.robot,
                    self.dyn_model,
                    q_arm_list,
                    q_head_list,
                    cfg["arm_idx"],
                    head_cfg["head_idx"],
                    q_nominal,
                    True, # optimize_arm
                    optimize_head,
                    optimize_camera,
                    active_arms,
                    ee_links,
                    cfg["mount_to_cam_nom"],
                    ee_to_marker_nom,
                    camera_link=head_cfg["camera_link"],
                )
                self.log(self.dev_text, f"Simulation dataset generated. samples = {sample_count}")

            dataset_path, result_path = self.build_output_paths()
            validate_dataset(q_arm_list, q_head_list, T_meas_list, optimize_head, active_arms)
            save_npz_dataset(dataset_path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)
            self.last_dataset_path = dataset_path
            self.log(self.dev_text, f"Dataset saved to {dataset_path}")

            self.run_optimizer(
                active_arms=active_arms,
                optimize_head=optimize_head,
                optimize_camera=optimize_camera,
                q_arm_list=q_arm_list,
                q_head_list=q_head_list,
                T_meas_list=T_meas_list,
                result_path=result_path,
                text_widget=self.dev_text,
                lambda_cam_pos=lambda_cam_pos,
                lambda_cam_rot=lambda_cam_rot,
                solver_type=self.user_solver.get(),
                use_sag=self.dev_use_sag.get(),
            )

        except Exception as e:
            messagebox.showerror("Calculate Error", str(e))
            self.log(self.dev_text, f"Calculate failed: {e}")

    def dev_apply_home_offset(self):
        try:
            servo_mode = self.dev_servo_mode.get()
            self.apply_home_offset_common(
                ip=self.dev_ip.get(),
                model_name=self.dev_model.get(),
                arm=FIXED_CALIB_ARM,
                servo_regex=self.servo_mode_to_regex(servo_mode),
                include_head=self.servo_mode_includes_head(servo_mode),
                text_widget=self.dev_text,
            )
        except Exception as e:
            messagebox.showerror("Apply Home Offset Error", str(e))
            self.log(self.dev_text, f"Apply home offset failed: {e}")

    # ============================================================
    # cleanup
    # ============================================================

    def on_close(self):
        try:
            self.stop_all_auto_motion_internal(cancel_robot=True)
            if self.marker_transform is not None:
                self.marker_transform.camera.monitoring(Flag=False)
        except Exception:
            pass
        self.root.destroy()


    def update_est_samples(self, *args):
        try:
            p = self.dev_pos_step.get()
            m = self.dev_max_x.get()
            if p <= 0:
                return

            current_x = 0.0
            if self.robot is not None and self.dyn_model is not None:
                try:
                    q_full = self.robot.get_state().position
                    _, T_fk = compute_fk(self.robot, self.dyn_model, q_full, "ee_right", "link_torso_5")
                    current_x = T_fk[0, 3]
                except:
                    pass
            
            # 절대 좌표 기준이므로 (max_x - current_x) 범위를 계산
            if m > current_x:
                cnt = 10 * (int((m - current_x) / p) + 1)
            else:
                cnt = 0
            self.dev_est_samples.set(f"Est. Samples: {cnt} (from X={current_x:.3f})")
        except:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
