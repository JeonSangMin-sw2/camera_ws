import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox
import threading
import time

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
    reset_home_offsets,
    check_calibration_state,
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
    reset_motion_state,
)
from homeoffset_core import (
    apply_home_offset_from_json,
    move_robot_to_zero_pose,
    movej,
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
        self.auto_motion_thread = None
        self.include_head_motion = True
        self.connected_servo_mode = "all"

        self.warning_img = None
        self.last_result_path = None
        self.last_dataset_path = None

        self.dataset_saved_in_session = False
        self.current_session_dataset_path = None

        self.build_ui()
        self.update_head_pose_status()
        self.update_dev_mode_label()

    # ============================================================
    # UI
    # ============================================================
    def build_ui(self):
        self.root.title("Marker Bracket Calibration System")
        self.root.geometry("1300x700")
        self.root.minsize(1300, 700)

        # Main horizontal container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(0, weight=1)

        # 1. Left/Middle Main Panel
        self.main_content = ttk.Frame(main_container)
        self.main_content.grid(row=0, column=0, sticky="nsew")

        # 2. Right Sidebar for "Options" (Starts hidden by default)
        self.sidebar = ttk.LabelFrame(main_container, text="Options", width=180)
        self.sidebar.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=10)
        self.sidebar.grid_remove() # Hide initially
        self.sidebar.grid_propagate(False) # Keep width fixed

        # Gather infrequently used checkboxes here
        self.dev_cal_with_head = tk.BooleanVar(value=True)
        self.dev_cal_with_head_cb = ttk.Checkbutton(self.sidebar, text="cal_with_head", variable=self.dev_cal_with_head)
        self.dev_cal_with_head_cb.pack(anchor="w", padx=15, pady=10)

        self.dev_use_sag = tk.BooleanVar(value=False)
        self.dev_use_sag_cb = ttk.Checkbutton(self.sidebar, text="use_sag", variable=self.dev_use_sag)
        self.dev_use_sag_cb.pack(anchor="w", padx=15, pady=10)

        self.build_main_ui(self.main_content)

    def toggle_sidebar(self):
        if self.sidebar.winfo_ismapped():
            self.sidebar.grid_remove()
        else:
            self.sidebar.grid()

    def on_servo_head_toggle(self):
        if not self.servo_head.get():
            self.dev_cal_with_head.set(False)
            self.dev_cal_with_head_cb.config(state="disabled")
        else:
            self.dev_cal_with_head_cb.config(state="normal")

    def build_main_ui(self, frm):
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(0, weight=0)
        frm.rowconfigure(1, weight=0)
        frm.rowconfigure(2, weight=0)
        frm.rowconfigure(3, weight=1)

        # 0. Options Button (Top-Left above Connection)
        ttk.Button(frm, text="⚙ Options", command=self.toggle_sidebar).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))

        # connection
        conn = ttk.LabelFrame(frm, text="Connection")
        conn.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        ttk.Label(conn, text="RPC IP").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dev_ip = tk.StringVar(value="192.168.30.1:50051")
        self.dev_ip_entry = ttk.Entry(conn, textvariable=self.dev_ip, width=30)
        self.dev_ip_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(conn, text="Model").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.dev_model = tk.StringVar(value="a")
        ttk.Combobox(conn, textvariable=self.dev_model, values=["a", "m"], state="readonly", width=8)\
            .grid(row=0, column=3, padx=5, pady=5, sticky="w")
        ttk.Button(conn, text="Connect", command=self.dev_connect).grid(row=0, column=4, padx=5, pady=5)
        self.dev_status = tk.StringVar(value="Disconnected")
        ttk.Label(conn, textvariable=self.dev_status).grid(row=0, column=5, padx=10, pady=5, sticky="w")

        ttk.Label(conn, text="Servo On").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.servo_body = tk.BooleanVar(value=True)
        self.servo_head = tk.BooleanVar(value=True)
        ttk.Checkbutton(conn, text="Body (torso/arms)", variable=self.servo_body).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(conn, text="Head", variable=self.servo_head, command=self.on_servo_head_toggle).grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # config
        cfg = ttk.LabelFrame(frm, text="Config")
        cfg.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        ttk.Label(cfg, text="Part").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dev_calib_part = tk.StringVar(value="both_arm")
        ttk.Combobox(cfg, textvariable=self.dev_calib_part, values=["both_arm", "right_arm", "left_arm", "torso"], state="readonly", width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(cfg, text="Mode").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.dev_mode = tk.StringVar(value="live")
        mode_box = ttk.Combobox(cfg, textvariable=self.dev_mode, values=["live", "npz", "sim"], state="readonly", width=10)
        mode_box.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        mode_box.bind("<<ComboboxSelected>>", self.update_dev_mode_label)

        ttk.Label(cfg, text="Path").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dev_path = tk.StringVar(value="result/dataset_YYYYMMDD_HHMMSS.npz")
        ttk.Entry(cfg, textvariable=self.dev_path, width=40).grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="w")

        self.dev_mode_info = tk.StringVar(value="In live mode, Auto Motion records once and All Auto Motion runs the full sweep; Stop interrupts between steps.")

        ttk.Label(cfg, text="Solver").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.dev_solver = tk.StringVar(value="Least Squares")
        self.solver_cb = ttk.Combobox(cfg, textvariable=self.dev_solver, values=["Least Squares", "QP Solver"], state="readonly", width=12)
        self.solver_cb.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        try:
            import qpsolvers
            has_qp = True
        except ImportError:
            has_qp = False
            
        if not has_qp:
            self.solver_cb["state"] = "disabled"
            
        self.dev_est_samples = tk.StringVar(value="Est. Samples: 0")
        ttk.Label(cfg, textvariable=self.dev_est_samples, foreground="blue").grid(row=3, column=1, columnspan=6, sticky="w", padx=5, pady=5)

        ttk.Label(cfg, text="Auto Motion Step").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        
        auto_frm = ttk.Frame(cfg)
        auto_frm.grid(row=4, column=1, columnspan=6, sticky="w", pady=5)
        
        ttk.Label(auto_frm, text="Angle(deg):").pack(side="left", padx=(0, 2))
        self.dev_angle_step = tk.DoubleVar(value=5.0)
        ttk.Entry(auto_frm, textvariable=self.dev_angle_step, width=5).pack(side="left", padx=(0, 10))
        
        ttk.Label(auto_frm, text="Pos(m):").pack(side="left", padx=(0, 2))
        self.dev_pos_step = tk.DoubleVar(value=0.03)
        ttk.Entry(auto_frm, textvariable=self.dev_pos_step, width=5).pack(side="left", padx=(0, 10))

        ttk.Label(auto_frm, text="Step(m):").pack(side="left", padx=(0, 2))
        self.dev_step_x = tk.DoubleVar(value=0.03)
        ttk.Entry(auto_frm, textvariable=self.dev_step_x, width=5).pack(side="left", padx=(0, 10))
        
        ttk.Label(auto_frm, text="Max X(m):").pack(side="left", padx=(0, 2))
        self.dev_max_x = tk.DoubleVar(value=0.4)
        ttk.Entry(auto_frm, textvariable=self.dev_max_x, width=5).pack(side="left", padx=(0, 10))

        self.dev_angle_step.trace_add("write", self.update_est_samples)
        self.dev_pos_step.trace_add("write", self.update_est_samples)
        self.dev_step_x.trace_add("write", self.update_est_samples)
        self.dev_max_x.trace_add("write", self.update_est_samples)
        self.update_est_samples()

        self.dev_head_status = tk.StringVar(value="Auto Motion: 0/0")
        ttk.Label(cfg, textvariable=self.dev_head_status).grid(row=5, column=0, columnspan=7, padx=5, pady=5, sticky="w")

        # actions
        act = ttk.LabelFrame(frm, text="Actions")
        act.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        act.columnconfigure(0, weight=1)
        act.columnconfigure(1, weight=1)

        ttk.Button(act, text="1) Zero Pose Check", command=self.dev_zero_pose_check).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="2) Init Pose", command=self.dev_init_pose).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="3-a-1) All Auto Motion", command=self.dev_all_auto_motion).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="3-a-2) Stop", command=self.dev_stop_auto_motion).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="3-b-1) Auto Motion", command=self.dev_auto_motion).grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="3-b-2) Record(Current)", command=self.dev_record).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="4-1) Calculate", command=self.dev_calculate).grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="4-2) Clear Samples", command=self.clear_samples).grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="5) Apply Home Offset", command=self.dev_apply_home_offset).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        ttk.Button(act, text="6) Home Offset Reset", command=self.dev_home_offset_reset).grid(row=5, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="7) Check Calibration State", command=self.dev_check_calibration_state).grid(row=5, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(act, text="8) Save Dataset", command=self.dev_save_dataset_manually).grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.dev_count = tk.StringVar(value="Shared Samples: 0")
        ttk.Label(act, textvariable=self.dev_count).grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # result/log
        logfrm = ttk.LabelFrame(frm, text="Log / Result")
        logfrm.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=10, pady=10)

        self.dev_text = tk.Text(logfrm, height=20)
        self.dev_text.pack(fill="both", expand=True, padx=5, pady=5)

    # ============================================================
    # Common helpers
    # ============================================================

    # log
    def log(self, widget, msg):
        def _log():
            widget.insert("end", msg + "\n")
            widget.see("end")
            self.root.update_idletasks()
        if threading.current_thread() is threading.main_thread():
            _log()
        else:
            self.root.after(0, _log)

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
        self.dev_head_status.set(label)

    def clear_samples(self):
        self.stop_all_auto_motion_internal(cancel_robot=True)
        reset_motion_state()
        self.shared_arm_q_list.clear()
        self.shared_head_q_list.clear()
        self.shared_T_list.clear()
        self.head_move_count = 0
        self.auto_base_head_q = None
        self.auto_ready_done = False
        self.dataset_saved_in_session = False
        self.current_session_dataset_path = None
        self.update_sample_counts()
        self.update_head_pose_status()
        self.log(self.dev_text, "Shared samples cleared.")

    def auto_save_current_dataset(self, text_widget):
        if len(self.shared_arm_q_list) == 0:
            return
        
        q_arm_list = np.array(self.shared_arm_q_list)
        q_head_list = np.array(self.shared_head_q_list) if self.shared_head_q_list else None
        T_meas_list = np.array(self.shared_T_list)
        
        # Auto-slice for single-arm mode if data has both
        active_arms = self.get_active_arms()
        optimize_head = self.dev_cal_with_head.get()
        if len(active_arms) == 1:
            if q_arm_list.shape[1] == 14:
                if active_arms[0] == "right":
                    q_arm_list = q_arm_list[:, :7]
                else:
                    q_arm_list = q_arm_list[:, 7:]
            if T_meas_list.ndim == 4 and T_meas_list.shape[1] == 2:
                if active_arms[0] == "right":
                    T_meas_list = T_meas_list[:, 0]
                else:
                    T_meas_list = T_meas_list[:, 1]
                    
        try:
            validate_dataset(q_arm_list, q_head_list, T_meas_list, optimize_head, active_arms)
            
            if not self.dataset_saved_in_session or self.current_session_dataset_path is None:
                dataset_path, _ = self.build_output_paths()
                self.current_session_dataset_path = dataset_path
                self.dataset_saved_in_session = True
                
            save_npz_dataset(self.current_session_dataset_path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)
            self.last_dataset_path = self.current_session_dataset_path
            self.log(text_widget, f"[Auto-Save] Dataset saved/updated in: {self.current_session_dataset_path}")
        except Exception as e:
            self.log(text_widget, f"[Auto-Save Error] {e}")

    

    def get_active_arms(self):
        val = self.dev_calib_part.get()
        if val == "both_arm":
            return ["right", "left"]
        elif val == "right_arm":
            return ["right"]
        elif val == "left_arm":
            return ["left"]
        elif val == "torso":
            return ["torso"]
        return ["right", "left"]

    def get_target_arm_str(self):
        val = self.dev_calib_part.get()
        if val == "both_arm":
            return "both"
        elif val == "torso":
            return "torso"
        return val.replace("_arm", "")

    # robot 
    def connect_robot(self, ip, model_name, status_var, text_widget, servo_regex):
        try:
            self.include_head_motion = self.servo_head.get()
            self.robot = create_robot(
                ip,
                model_name,
                power_regex=".*",
                servo_regex=servo_regex,
            )
            self.dyn_model = self.robot.get_dynamics()
            self.model = self.robot.model()
            
            if len(self.model.head_idx) == 0:
                self.dev_cal_with_head.set(False)
                self.dev_cal_with_head_cb.config(state="disabled")
                self.log(text_widget, "No head joints detected. cal_with_head disabled and set to False (Torso base).")
            else:
                self.dev_cal_with_head_cb.config(state="normal")

            self.auto_motion_running = False
            self.auto_stop_requested = False
            self.auto_motion_after_id = None
            self.auto_base_head_q = None
            self.auto_ready_done = False
            status_var.set("Connected")
            self.log(text_widget, f"Connected: {ip} (model={model_name}, servo_regex={servo_regex})")
            self.update_head_pose_status()
        except Exception as e:
            status_var.set("Disconnected")
            messagebox.showerror("Connection Error", str(e))
            self.log(text_widget, f"Connection failed: {e}")

    def get_capture_head_idx(self):
        if self.model is None:
            return None
        return get_head_config(self.model)["head_idx"]
    
    # robot motion
    def move_to_auto_init_pose_worker(self, text_widget, tab="dev"):
        try:
            if self.robot is None:
                raise RuntimeError("Robot is not connected.")
            if self.model is None:
                raise RuntimeError("Robot is not connected.")

            active_arms = self.get_active_arms()

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
                    self.auto_config.step_x_m = float(self.dev_step_x.get())
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
            self.root.after(0, self.update_head_pose_status)
            self.root.after(0, self.update_est_samples)
            
            self.root.after(0, lambda: messagebox.showinfo(
                "Teaching Required",
                "Robot has moved to the initial pose.\n\n"
                "Please use the Teaching button to adjust the robot's pose so that the marker is clearly visible to the camera.\n"
                "Once adjusted, press 'Auto Motion' or 'All Auto Motion' to start the sequence."
            ))
        except Exception as e:
            self.log(text_widget, f"Init pose failed: {e}")
            self.root.after(0, lambda err=e: messagebox.showerror("Init Error", str(err)))
        finally:
            self.auto_motion_running = False
            self.auto_motion_thread = None

    def run_auto_motion_step_worker(self, text_widget, tab="dev"):
        try:
            self.run_auto_motion_step_blocking(text_widget, tab)
        except Exception as e:
            self.log(text_widget, f"Auto motion failed: {e}")
            self.root.after(0, lambda err=e: messagebox.showerror("Auto Motion Error", str(err)))
        finally:
            self.auto_motion_running = False
            self.auto_motion_thread = None
            self.auto_save_current_dataset(text_widget)

    def run_auto_motion_step_blocking(self, text_widget, tab="dev"):
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.model is None:
            raise RuntimeError("Robot is not connected.")

        if self.dev_mode.get() != "sim" and self.marker_transform is None:
            self.marker_transform = create_live_marker_transform()

        pose_target = self.get_auto_pose_target_count()
        if self.head_move_count >= pose_target:
            self.log(text_widget, "All auto motions have already been executed.")
            return True

        if not self.auto_ready_done:
            raise RuntimeError("Please move to Init Pose first.")

        active_arms = self.get_active_arms()

        # Re-build incremental motion plan based on the CURRENT (possibly teached) pose
        if self.auto_motion_plan is None or self.head_move_count == 0:
            if tab == "dev":
                try:
                    self.auto_config.angle_step_deg = float(self.dev_angle_step.get())
                    self.auto_config.position_step_m = float(self.dev_pos_step.get())
                    self.auto_config.step_x_m = float(self.dev_step_x.get())
                    self.auto_config.max_x = float(self.dev_max_x.get())
                except Exception as e:
                    self.log(text_widget, f"Failed to read dev auto config: {e}. Using current values.")
            
            self.log(text_widget, f"Building motion plan based on current pose... (Angle={self.auto_config.angle_step_deg}deg, Pos={self.auto_config.position_step_m}m, StepX={self.auto_config.step_x_m}m, MaxX={self.auto_config.max_x}m)")
            self.auto_motion_plan = build_incremental_motion_plan(
                self.robot, self.dyn_model, self.auto_config, active_arms
            )
            self.root.after(0, self.update_head_pose_status)
            self.root.after(0, self.update_est_samples)

        if self.include_head_motion and self.auto_base_head_q is None:
            head_cfg = get_head_config(self.model)
            if head_cfg["head_idx"] is not None:
                self.auto_base_head_q = self.robot.get_state().position[head_cfg["head_idx"]].copy()
                self.log(text_widget, f"Auto base head pose (deg): {np.round(np.rad2deg(self.auto_base_head_q), 3)}")
            else:
                self.auto_base_head_q = None
                self.include_head_motion = False

        if self.auto_stop_requested:
            self.log(text_widget, "Auto Motion stopped by user.")
            return False

        motion_plan_step = self.auto_motion_plan[self.head_move_count]
        
        motion_info = execute_auto_motion_step(
            robot=self.robot,
            config=self.auto_config,
            motion_plan_step=motion_plan_step,
            active_arms=active_arms,
            include_head_motion=self.include_head_motion,
        )
        self.log(
            text_widget,
            f"Auto motion done: {motion_plan_step['desc']}",
        )

        if self.auto_stop_requested:
            self.log(text_widget, "Auto Motion stopped by user.")
            return False

        q_arm, q_head, T_meas = self.capture_one_sample(text_widget)
        if q_arm is None:
            self.head_move_count += 1
            self.root.after(0, self.update_head_pose_status)
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
        self.root.after(0, self.update_sample_counts)
        self.root.after(0, self.update_head_pose_status)
        return True

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
        reset_motion_state()

    def request_stop_all_auto_motion(self, text_widget):
        if not self.auto_motion_running and self.auto_motion_thread is None:
            self.log(text_widget, "No All Auto Motion sequence is running.")
            self.stop_all_auto_motion_internal(cancel_robot=True)
            return

        self.auto_stop_requested = True
        self.stop_all_auto_motion_internal(cancel_robot=True, reset_stop_requested=False)
        self.log(text_widget, "Stop requested. Sent robot.cancel_control(); the all-auto sequence stops after the current step.")

    def run_all_auto_motion_worker(self, text_widget, tab="dev"):
        try:
            pose_target = self.get_auto_pose_target_count()
            while self.head_move_count < pose_target:
                if self.auto_stop_requested:
                    self.log(text_widget, "Auto Motion stopped by user.")
                    break

                ok = self.run_auto_motion_step_blocking(text_widget, tab)
                if not ok:
                    if self.auto_stop_requested:
                        self.log(text_widget, "Auto Motion stopped by user.")
                        break
                    self.log(text_widget, f"Step capture failed and skipped. Continuing sequence...")

                # Sleep slightly between steps
                time.sleep(0.2)
            else:
                self.log(text_widget, "All auto motions completed.")
        except Exception as e:
            self.log(text_widget, f"All Auto Motion background worker error: {e}")
        finally:
            self.auto_motion_running = False
            self.auto_motion_thread = None
            self.auto_save_current_dataset(text_widget)

    def move_to_all_auto_motions(self, text_widget, tab="dev"):
        if not self.auto_ready_done:
            raise RuntimeError("Please move to Init Pose first.")

        if self.auto_motion_plan is None or len(self.auto_motion_plan) == 0:
            self.log(text_widget, "Motion plan is missing or empty. Re-building...")
            active_arms = self.get_active_arms()
            self.auto_motion_plan = build_incremental_motion_plan(
                self.robot, self.dyn_model, self.auto_config, active_arms
            )

        pose_target = self.get_auto_pose_target_count()
        if self.head_move_count >= pose_target:
            self.log(text_widget, "All auto motions have already been executed.")
            return

        if self.auto_motion_running or self.auto_motion_thread is not None:
            self.log(text_widget, "Another robot operation is already running.")
            return

        self.auto_stop_requested = False
        self.auto_motion_running = True
        self.log(text_widget, "All Auto Motion started in a background thread. Press Stop to cancel.")

        self.auto_motion_thread = threading.Thread(
            target=self.run_all_auto_motion_worker,
            args=(text_widget, tab)
        )
        self.auto_motion_thread.daemon = True
        self.auto_motion_thread.start()


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

        if solver_type == "QP Solver":
            optimizer = QPCalibrationOptimizer(
                robot=self.robot,
                arm_idx=cfg["arm_idx"],
                ee_links=ee_links,
                mount_to_cam_nom=cfg["mount_to_cam_nom"],
                head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
                ee_to_marker_nom=ee_to_marker_nom,
                head_idx=head_cfg["head_idx"],
                lambda_cam_pos=lambda_cam_pos,
                lambda_cam_rot=lambda_cam_rot,
                use_sag=use_sag,
                optimize_head=optimize_head,
                optimize_camera=optimize_camera,
                active_arms=active_arms,
                estimate_measurement_noise=True,
            )
        else:
            optimizer = CalibrationOptimizer(
                robot=self.robot,
                arm_idx=cfg["arm_idx"],
                ee_links=ee_links,
                mount_to_cam_nom=cfg["mount_to_cam_nom"],
                head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
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
                estimate_measurement_noise=True,
            )

        q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
            q_arm_list,
            q_head_list,
            T_meas_list,
        )
        
        # Correctly assign offsets based on active arms
        if len(active_arms) == 1:
            if active_arms[0] == "right":
                right_arm_offset = q_arm_offset
                left_arm_offset = None
            else:
                right_arm_offset = None
                left_arm_offset = q_arm_offset
        else:
            right_arm_offset = q_arm_offset[:7]
            left_arm_offset = q_arm_offset[7:]

        head_base_to_cam_new = [float(x) for x in head_base_to_cam_new] if head_base_to_cam_new else None
        mount_to_cam_new = [float(x) for x in mount_to_cam_new] if mount_to_cam_new else None

        self.log(text_widget, "\n===== RESULT =====")
        self.log(text_widget, f"lambda_cam_pos = {lambda_cam_pos}")
        self.log(text_widget, f"lambda_cam_rot = {lambda_cam_rot}")
        self.log(text_widget, f"measurement_noise = {optimizer.noise_estimator.format()}")
        
        if right_arm_offset is not None:
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
            self.log(text_widget, "head_base-to-camera xi:")
            self.log(text_widget, str(xi_cam))
            self.log(text_widget, "head_base_to_cam_new:")
            self.log(text_widget, str(head_base_to_cam_new))

        result_dict = {
            "joint_offset_deg": np.rad2deg(q_arm_offset).tolist(),
            "right_arm_joint_offset_deg": np.rad2deg(right_arm_offset).tolist() if right_arm_offset is not None else None,
            "left_arm_joint_offset_deg": np.rad2deg(left_arm_offset).tolist() if left_arm_offset is not None else None,
            "head_joint_offset_deg": np.rad2deg(q_head_offset).tolist() if q_head_offset is not None else None,
            "xi_cam": np.array(xi_cam).tolist(),
            "measurement_noise": optimizer.noise_estimator.as_dict(),
        }

        if optimize_head:
            result_dict["xi_mount_cam"] = result_dict["xi_cam"]
        else:
            result_dict["xi_head_base_cam"] = result_dict["xi_cam"]

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
    # Calibration Actions
    # ============================================================

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
        parts = []
        if self.servo_body.get():
            parts.append(r"mobile_.*|torso_.*|right_arm_.*|left_arm_.*")
        if self.servo_head.get():
            parts.append(r"head_.*")
        servo_regex = "|".join(parts) if parts else r"^$"

        self.include_head_motion = self.servo_head.get()
        self.connect_robot(
            ip=self.dev_ip.get(),
            model_name=self.dev_model.get(),
            status_var=self.dev_status,
            text_widget=self.dev_text,
            servo_regex=servo_regex,
        )

    def dev_zero_pose_check(self):
        try:
            parts = []
            if self.servo_body.get():
                parts.append(r"mobile_.*|torso_.*|right_arm_.*|left_arm_.*")
            if self.servo_head.get():
                parts.append(r"head_.*")
            servo_regex = "|".join(parts) if parts else r"^$"

            arm = self.get_target_arm_str()
            self.zero_pose_check_common(
                ip=self.dev_ip.get(),
                model_name=self.dev_model.get(),
                arm=arm,
                servo_regex=servo_regex,
                include_head=self.servo_head.get(),
                text_widget=self.dev_text,
            )
        except Exception as e:
            messagebox.showerror("Zero Pose Check Error", str(e))
            self.log(self.dev_text, f"Zero pose check failed: {e}")

    def dev_init_pose(self):
        if self.auto_motion_running or self.auto_motion_thread is not None:
            messagebox.showerror("Execution Error", "Another robot operation is currently running.")
            return
        self.auto_motion_running = True
        self.auto_motion_thread = threading.Thread(
            target=self.move_to_auto_init_pose_worker,
            args=(self.dev_text, "dev")
        )
        self.auto_motion_thread.daemon = True
        self.auto_motion_thread.start()

    def dev_auto_motion(self):
        try:
            mode = self.dev_mode.get()
            if mode not in ["live", "sim"]:
                self.log(self.dev_text, "Auto motion is only available in live or sim mode.")
                return

            if self.auto_motion_running or self.auto_motion_thread is not None:
                messagebox.showerror("Execution Error", "Another robot operation is currently running.")
                return

            self.auto_stop_requested = False
            self.auto_motion_running = True
            self.auto_motion_thread = threading.Thread(
                target=self.run_auto_motion_step_worker,
                args=(self.dev_text, "dev")
            )
            self.auto_motion_thread.daemon = True
            self.auto_motion_thread.start()
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
            self.auto_save_current_dataset(self.dev_text)
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
            self.auto_save_current_dataset(self.dev_text)
        except Exception as e:
            messagebox.showerror("Record Error", str(e))
            self.log(self.dev_text, f"Record failed: {e}")

    def dev_calculate(self):
        try:
            mode = self.dev_mode.get()
            
            arm_val = self.dev_calib_part.get()
            if arm_val == "torso":
                raise NotImplementedError("Torso calibration is not implemented yet.")
            active_arms = ["right", "left"] if arm_val == "both_arm" else [arm_val.replace("_arm", "")]
            optimize_head = self.dev_cal_with_head.get()
            optimize_camera = False
            
            lambda_cam_pos = 1.0
            lambda_cam_rot = 1.0
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
                
                # Auto-slice for single-arm mode if data has both
                if len(active_arms) == 1:
                    if q_arm_list.shape[1] == 14:
                        if active_arms[0] == "right":
                            q_arm_list = q_arm_list[:, :7]
                        else:
                            q_arm_list = q_arm_list[:, 7:]
                    if T_meas_list.ndim == 4 and T_meas_list.shape[1] == 2:
                        if active_arms[0] == "right":
                            T_meas_list = T_meas_list[:, 0]
                        else:
                            T_meas_list = T_meas_list[:, 1]

            elif mode == "npz":
                npz_path = self.resolve_input_path(self.dev_path.get())
                q_arm_list, q_head_list, T_meas_list = load_npz_dataset(npz_path)
                
                # Auto-slice for single-arm mode if data has both
                if len(active_arms) == 1:
                    if q_arm_list.shape[1] == 14:
                        if active_arms[0] == "right":
                            q_arm_list = q_arm_list[:, :7]
                        else:
                            q_arm_list = q_arm_list[:, 7:]
                    if T_meas_list.ndim == 4 and T_meas_list.shape[1] == 2:
                        if active_arms[0] == "right":
                            T_meas_list = T_meas_list[:, 0]
                        else:
                            T_meas_list = T_meas_list[:, 1]

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
                    cfg.get("head_base_to_cam_nom"),
                    ee_to_marker_nom,
                    camera_link=head_cfg["camera_link"],
                )
                self.log(self.dev_text, f"Simulation dataset generated. samples = {sample_count}")

            _, result_path = self.build_output_paths()
            validate_dataset(q_arm_list, q_head_list, T_meas_list, optimize_head, active_arms)
            if mode == "live":
                self.auto_save_current_dataset(self.dev_text)

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
                solver_type=self.dev_solver.get(),
                use_sag=self.dev_use_sag.get(),
            )

        except Exception as e:
            messagebox.showerror("Calculate Error", str(e))
            self.log(self.dev_text, f"Calculate failed: {e}")

    def dev_save_dataset_manually(self):
        if len(self.shared_arm_q_list) == 0:
            messagebox.showwarning("Warning", "No samples to save.")
            return
        
        from tkinter import filedialog
        selected_path = filedialog.asksaveasfilename(
            initialdir=str(RESULT_DIR),
            title="Save Dataset As",
            defaultextension=".npz",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        
        if not selected_path:
            return
            
        q_arm_list = np.array(self.shared_arm_q_list)
        q_head_list = np.array(self.shared_head_q_list) if self.shared_head_q_list else None
        T_meas_list = np.array(self.shared_T_list)
        
        active_arms = self.get_active_arms()
        optimize_head = self.dev_cal_with_head.get()
        if len(active_arms) == 1:
            if q_arm_list.shape[1] == 14:
                if active_arms[0] == "right":
                    q_arm_list = q_arm_list[:, :7]
                else:
                    q_arm_list = q_arm_list[:, 7:]
            if T_meas_list.ndim == 4 and T_meas_list.shape[1] == 2:
                if active_arms[0] == "right":
                    T_meas_list = T_meas_list[:, 0]
                else:
                    T_meas_list = T_meas_list[:, 1]
                    
        try:
            validate_dataset(q_arm_list, q_head_list, T_meas_list, optimize_head, active_arms)
            save_npz_dataset(selected_path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)
            self.last_dataset_path = selected_path
            self.log(self.dev_text, f"[Save] Dataset manually saved to: {selected_path}")
            messagebox.showinfo("Success", f"Dataset saved to:\n{selected_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
            self.log(self.dev_text, f"Manual save failed: {e}")

    def dev_apply_home_offset(self):
        try:
            parts = []
            if self.servo_body.get():
                parts.append(r"mobile_.*|torso_.*|right_arm_.*|left_arm_.*")
            if self.servo_head.get():
                parts.append(r"head_.*")
            servo_regex = "|".join(parts) if parts else r"^$"

            self.apply_home_offset_common(
                ip=self.dev_ip.get(),
                model_name=self.dev_model.get(),
                arm=self.get_target_arm_str(),
                servo_regex=servo_regex,
                include_head=self.servo_head.get(),
                text_widget=self.dev_text,
            )
        except Exception as e:
            messagebox.showerror("Apply Home Offset Error", str(e))
            self.log(self.dev_text, f"Apply home offset failed: {e}")

    def dev_home_offset_reset(self):
        if self.robot is None or self.model is None:
            messagebox.showerror("Error", "Robot is not connected.")
            return

        msg = (
            "Warning: Home Offset Reset will physically redefine the zero offset positions of your robot joints.\n\n"
            "Steps:\n"
            "1. Manually teach/move BOTH arms close to their home pose using direct teaching.\n"
            "2. Ensure the head is also centered/aligned if you want to reset head offsets.\n"
            "3. Click OK to start the process.\n\n"
            "During this, the control manager will disable, 48v power will cycle, and the robot connection will automatically restart."
        )
        if not messagebox.askokcancel("Confirm Home Offset Reset", msg):
            return

        # Start background worker thread
        t = threading.Thread(target=self.home_offset_reset_worker)
        t.daemon = True
        t.start()

    def home_offset_reset_worker(self):
        try:
            # Reconnect flow similar to apply_home_offset to avoid freezing
            old_robot = self.robot
            success = reset_home_offsets(
                old_robot,
                self.model,
                log_cb=lambda msg: self.log(self.dev_text, msg),
            )
            
            if old_robot is not None:
                try:
                    old_robot.disconnect()
                except Exception as e:
                    self.log(self.dev_text, f"Disconnect failed: {e}")
            self.robot = None
            self.model = None
            self.dyn_model = None
            self.dev_status.set("Disconnected")

            # 48v 끄고 2초 대기 (apply_home_offset 참고)
            self.log(self.dev_text, "Waiting for power cycle to complete (2 seconds)...")
            time.sleep(2.0)

            # Re-connect and initialize robot
            self.log(self.dev_text, "Re-connecting and initializing robot...")
            
            # Recreate connection with correct servo regex
            parts = []
            if self.servo_body.get():
                parts.append(r"mobile_.*|torso_.*|right_arm_.*|left_arm_.*")
            if self.servo_head.get():
                parts.append(r"head_.*")
            servo_regex = "|".join(parts) if parts else r"^$"

            self.include_head_motion = self.servo_head.get()
            self.robot = create_robot(
                self.dev_ip.get(),
                self.dev_model.get(),
                power_regex=".*",
                servo_regex=servo_regex,
            )
            self.dyn_model = self.robot.get_dynamics()
            self.model = self.robot.model()

            if len(self.model.head_idx) == 0:
                self.dev_cal_with_head.set(False)
                self.dev_cal_with_head_cb.config(state="disabled")
            else:
                self.dev_cal_with_head_cb.config(state="normal")

            self.auto_motion_running = False
            self.auto_stop_requested = False
            self.auto_motion_after_id = None
            self.auto_base_head_q = None
            self.auto_ready_done = False
            self.dev_status.set("Connected")
            self.log(self.dev_text, "Re-connection successful.")
            self.update_head_pose_status()

            # Move to zero pose after reset (apply_home_offset 참고)
            self.log(self.dev_text, "Moving to zero pose after home offset reset...")
            right_zero_pose = np.zeros(len(self.model.right_arm_idx))
            left_zero_pose = np.zeros(len(self.model.left_arm_idx))
            head_zero_pose = np.zeros(len(self.model.head_idx)) if self.servo_head.get() else None
            
            ok = movej(
                self.robot,
                right_arm=right_zero_pose,
                left_arm=left_zero_pose,
                head=head_zero_pose,
                minimum_time=5,
            )
            if not ok:
                self.log(self.dev_text, "Failed to move robot to zero pose after reset.")

            self.log(self.dev_text, "Home Offset Reset complete!")
            if success:
                messagebox.showinfo("Success", "Home Offset Reset completed successfully!")
            else:
                messagebox.showwarning("Warning", "Home Offset Reset finished, but some joints failed to reset. Please check the logs.")
        except Exception as e:
            self.log(self.dev_text, f"Home Offset Reset worker error: {e}")
            messagebox.showerror("Error", f"Home Offset Reset failed: {e}")

    def dev_check_calibration_state(self):
        if self.robot is None:
            messagebox.showerror("Error", "Robot is not connected.")
            return

        self.check_state_moved = False

        popup = tk.Toplevel(self.root)
        popup.title("Check Calibration State")
        popup.geometry("380x280")
        popup.transient(self.root)
        
        # Grid layout for inputs
        ttk.Label(popup, text="X Position (m):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        x_var = tk.StringVar(value="0.35")
        x_entry = ttk.Entry(popup, textvariable=x_var, width=15)
        x_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        ttk.Label(popup, text="Y Position (m):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        y_var = tk.StringVar(value="0.0")
        y_entry = ttk.Entry(popup, textvariable=y_var, width=15)
        y_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        ttk.Label(popup, text="Z Position (m):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        z_var = tk.StringVar(value="0.0")
        z_entry = ttk.Entry(popup, textvariable=z_var, width=15)
        z_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        ttk.Label(popup, text="Y Offset (m):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        offset_var = tk.StringVar(value="0.175")
        offset_entry = ttk.Entry(popup, textvariable=offset_var, width=15)
        offset_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # Action Buttons frame
        btn_frm = ttk.Frame(popup)
        btn_frm.grid(row=4, column=0, columnspan=2, pady=20)

        status_lbl = ttk.Label(popup, text="Status: Ready", foreground="blue")
        status_lbl.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        def on_move():
            try:
                x = float(x_var.get())
                y = float(y_var.get())
                z = float(z_var.get())
                offset = float(offset_var.get())
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid floating-point numbers.")
                return

            status_lbl.config(text="Status: Moving...", foreground="orange")
            
            def move_worker():
                try:
                    active_arms = self.get_active_arms()
                    
                    # Delegate logic directly to check_calibration_state in core
                    check_calibration_state(
                        self.robot,
                        self.dev_model.get(),
                        active_arms,
                        [x, y, z],
                        offset,
                        log_cb=lambda msg: self.log(self.dev_text, f"[Check State] {msg}"),
                        skip_ready=self.check_state_moved
                    )
                    
                    self.check_state_moved = True
                    self.log(self.dev_text, "[Check State] Symmetrical move completed successfully.")
                    popup.after(0, lambda: status_lbl.config(text="Status: Move OK", foreground="green"))
                except Exception as ex:
                    self.log(self.dev_text, f"[Check State Error] {ex}")
                    popup.after(0, lambda err=ex: status_lbl.config(text="Status: Error", foreground="red"))

            t = threading.Thread(target=move_worker)
            t.daemon = True
            t.start()

        def on_close():
            self.check_state_moved = False
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", on_close)

        ttk.Button(btn_frm, text="Move", command=on_move).pack(side="left", padx=10)
        ttk.Button(btn_frm, text="Close", command=on_close).pack(side="left", padx=10)

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
            step_x = self.dev_step_x.get()
            m = self.dev_max_x.get()
            if p <= 0 or step_x <= 0:
                return

            current_x = 0.0
            if self.robot is not None and self.dyn_model is not None:
                try:
                    # Sync config values for build_incremental_motion_plan
                    self.auto_config.angle_step_deg = float(self.dev_angle_step.get())
                    self.auto_config.position_step_m = float(self.dev_pos_step.get())
                    self.auto_config.step_x_m = float(self.dev_step_x.get())
                    self.auto_config.max_x = float(self.dev_max_x.get())

                    active_arms = self.get_active_arms()
                    temp_plan = build_incremental_motion_plan(
                        self.robot, self.dyn_model, self.auto_config, active_arms
                    )
                    cnt = len(temp_plan)

                    # Compute current_x for display
                    q_full = self.robot.get_state().position
                    _, T_fk = compute_fk(self.robot, self.dyn_model, q_full, "ee_right", "link_torso_5")
                    current_x = T_fk[0, 3]

                    self.dev_est_samples.set(f"Est. Samples: {cnt} (from X={current_x:.3f})")
                    return
                except:
                    pass

            # Fallback/Offline estimation:
            # We exactly generate 33 steps per X-step (12 joint steps + 1 restore step + 12 RPY steps + 8 YZ steps = 33)
            current_x = 0.3
            if m > current_x:
                cnt = 33 * (int((m - current_x) / step_x) + 1)
            else:
                cnt = 0
            self.dev_est_samples.set(f"Est. Samples: {cnt} (approx)")
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
