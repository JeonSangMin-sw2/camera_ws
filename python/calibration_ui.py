import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np
import rby1_sdk as rby

from calibration_core import (
    create_robot,
    create_live_marker_transform,
    capture_one_sample as capture_robot_sample,
    get_arm_config,
    get_head_config,
    load_npz_dataset,
    save_npz_dataset,
    validate_dataset_for_ndof,
    generate_sim_measurements,
    CalibrationOptimizer,
)
from homeoffset_core import (
    apply_home_offset_from_json,
    move_robot_to_zero_pose,
)

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
WARNING_POSE_PATH = BASE_DIR / "warning_pose.png"
WARNING_POSE_CHECK_PATH = BASE_DIR / "warning_pose_check.png"
HEAD_SWEEP_COUNT_TARGET = 20
HEAD_SWEEP_RANGE_DEG = {
    "head0": (-15.0, 15.0),
    "head1": (-15.0, 15.0),
}

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

        self.warning_img = None
        self.last_result_path = None
        self.last_dataset_path = None

        self._build_ui()
        self.update_head_pose_status()
        self._update_dev_mode_label()

    # ============================================================
    # UI
    # ============================================================


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

    def zero_pose_check_common(self, ip, model_name, arm, text_widget):
        result = move_robot_to_zero_pose(
            address=ip,
            model_name=model_name,
            arm=arm,
            power=".*",
            servo=".*",
        )

        self.log(text_widget, "\n===== ZERO POSE CHECK =====")
        self.log(text_widget, f"Arm: {result['arm']}")
        self.log(text_widget, result["message"])

        self.show_zero_pose_check_popup()
        
    def user_zero_pose_check(self):
        try:
            self.zero_pose_check_common(
                ip=self.user_ip.get(),
                model_name=self.user_model.get(),
                arm=self.user_arm.get(),
                text_widget=self.user_text,
            )
        except Exception as e:
            messagebox.showerror("Zero Pose Check Error", str(e))
            self.log(self.user_text, f"Zero pose check failed: {e}")


    def dev_zero_pose_check(self):
        try:
            self.zero_pose_check_common(
                ip=self.dev_ip.get(),
                model_name=self.dev_model.get(),
                arm=self.dev_arm.get(),
                text_widget=self.dev_text,
            )
        except Exception as e:
            messagebox.showerror("Zero Pose Check Error", str(e))
            self.log(self.dev_text, f"Zero pose check failed: {e}")

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.user_tab = ttk.Frame(notebook)
        self.dev_tab = ttk.Frame(notebook)

        notebook.add(self.user_tab, text="User")
        notebook.add(self.dev_tab, text="Developer")

        self._build_user_tab()
        self._build_dev_tab()

    def _build_user_tab(self):
        frm = self.user_tab

        # connection
        conn = ttk.LabelFrame(frm, text="Connection")
        conn.pack(fill="x", padx=10, pady=10)

        ttk.Label(conn, text="RPC IP").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.user_ip = tk.StringVar(value="localhost:50051")
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

        self.user_status = tk.StringVar(value="Disconnected")
        ttk.Label(conn, textvariable=self.user_status).grid(row=0, column=5, padx=10, pady=5, sticky="w")

        # setup
        setup = ttk.LabelFrame(frm, text="Setup")
        setup.pack(fill="x", padx=10, pady=10)

        ttk.Label(setup, text="Arm").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.user_arm = tk.StringVar(value="right")
        ttk.Combobox(
            setup,
            textvariable=self.user_arm,
            values=["right", "left"],
            state="readonly",
            width=10
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(setup, text="Mode: live").grid(row=0, column=2, padx=20, pady=5, sticky="w")
        ttk.Label(setup, text="ndof: auto (13 / 15)").grid(row=0, column=3, padx=20, pady=5, sticky="w")
        self.user_head_status = tk.StringVar(value="Head Move: 0/20")
        ttk.Label(setup, textvariable=self.user_head_status).grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="w")

        # actions
        act = ttk.LabelFrame(frm, text="Actions")
        act.pack(fill="x", padx=10, pady=10)

        ttk.Button(act, text="1.Zero Pose Check", command=self.user_zero_pose_check).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(act, text="2.Next Head Pose", command=self.user_next_head_pose).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(act, text="3.Record", command=self.user_record, padding=(80, 48)).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(act, text="4.Calculate", command=self.user_calculate).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(act, text="5.Apply Home Offset", command=self.user_apply_home_offset).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(act, text="6.Clear Samples", command=self.clear_samples).grid(row=0, column=5, padx=5, pady=5)

        self.user_count = tk.StringVar(value="Samples: 0")
        ttk.Label(act, textvariable=self.user_count).grid(row=0, column=6, padx=20, pady=5, sticky="w")
        
        

        # result/log
        logfrm = ttk.LabelFrame(frm, text="Log / Result")
        logfrm.pack(fill="both", expand=True, padx=10, pady=10)

        self.user_text = tk.Text(logfrm, height=20)
        self.user_text.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_dev_tab(self):
        frm = self.dev_tab

        # connection
        conn = ttk.LabelFrame(frm, text="Connection")
        conn.pack(fill="x", padx=10, pady=10)

        ttk.Label(conn, text="RPC IP").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dev_ip = tk.StringVar(value="localhost:50051")
        ttk.Entry(conn, textvariable=self.dev_ip, width=30).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(conn, text="Model").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.dev_model = tk.StringVar(value="a")
        ttk.Combobox(conn, textvariable=self.dev_model, values=["a", "m"], state="readonly", width=8)\
            .grid(row=0, column=3, padx=5, pady=5, sticky="w")
        ttk.Button(conn, text="Connect", command=self.dev_connect).grid(row=0, column=4, padx=5, pady=5)

        self.dev_status = tk.StringVar(value="Disconnected")
        ttk.Label(conn, textvariable=self.dev_status).grid(row=0, column=5, padx=10, pady=5, sticky="w")

        # config
        cfg = ttk.LabelFrame(frm, text="Config")
        cfg.pack(fill="x", padx=10, pady=10)

        ttk.Label(cfg, text="Arm").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dev_arm = tk.StringVar(value="right")
        ttk.Combobox(cfg, textvariable=self.dev_arm, values=["right", "left"], state="readonly", width=10)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(cfg, text="Mode").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.dev_mode = tk.StringVar(value="live")
        mode_box = ttk.Combobox(cfg, textvariable=self.dev_mode, values=["live", "npz", "sim"], state="readonly", width=10)
        mode_box.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        mode_box.bind("<<ComboboxSelected>>", self._update_dev_mode_label)

        ttk.Label(cfg, text="ndof").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.dev_ndof = tk.IntVar(value=13)
        ndof_box = ttk.Combobox(cfg, textvariable=self.dev_ndof, values=[2, 6, 7, 9, 13, 15], state="readonly", width=10)
        ndof_box.grid(row=0, column=5, padx=5, pady=5, sticky="w")
        ndof_box.bind("<<ComboboxSelected>>", self._update_dev_mode_label)

        ttk.Label(cfg, text="Path").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dev_path = tk.StringVar(value="result/dataset_YYYYMMDD_HHMMSS.npz")
        ttk.Entry(cfg, textvariable=self.dev_path, width=40).grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="w")

        self.dev_mode_info = tk.StringVar(value="Record button is used only in live mode.")
        ttk.Label(cfg, textvariable=self.dev_mode_info).grid(row=1, column=4, columnspan=2, padx=5, pady=5, sticky="w")

        self.dev_head_status = tk.StringVar(value="Head Move: 0/20")
        ttk.Label(cfg, textvariable=self.dev_head_status).grid(row=2, column=0, columnspan=6, padx=5, pady=5, sticky="w")

        # actions
        act = ttk.LabelFrame(frm, text="Actions")
        act.pack(fill="x", padx=10, pady=10)

        ttk.Button(act, text="1.Zero Pose Check", command=self.dev_zero_pose_check).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(act, text="2.Next Head Pose", command=self.dev_next_head_pose).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(act, text="3.Record", command=self.dev_record).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(act, text="4.Calculate", command=self.dev_calculate).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(act, text="5.Apply Home Offset", command=self.dev_apply_home_offset).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(act, text="6.Clear Samples", command=self.clear_samples).grid(row=0, column=5, padx=5, pady=5)

        self.dev_count = tk.StringVar(value="Shared Samples: 0")
        ttk.Label(act, textvariable=self.dev_count).grid(row=0, column=6, padx=20, pady=5, sticky="w")



        # result/log
        logfrm = ttk.LabelFrame(frm, text="Log / Result")
        logfrm.pack(fill="both", expand=True, padx=10, pady=10)

        self.dev_text = tk.Text(logfrm, height=20)
        self.dev_text.pack(fill="both", expand=True, padx=5, pady=5)

    # ============================================================
    # Common helpers
    # ============================================================

    def log(self, widget, msg):
        widget.insert("end", msg + "\n")
        widget.see("end")
        self.root.update_idletasks()

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

    def update_sample_counts(self):
        sample_count = len(self.shared_arm_q_list)
        self.user_count.set(f"Samples: {sample_count}")
        self.dev_count.set(f"Shared Samples: {sample_count}")

    def update_head_pose_status(self):
        pose_idx = min(self.head_move_count, HEAD_SWEEP_COUNT_TARGET)
        label = f"Head Move: {pose_idx}/{HEAD_SWEEP_COUNT_TARGET}"
        self.user_head_status.set(label)
        self.dev_head_status.set(label)

    def resolve_input_path(self, raw_path):
        input_path = Path(raw_path).expanduser()
        if input_path.is_absolute():
            return input_path
        return BASE_DIR / input_path

    def clear_samples(self):
        self.shared_arm_q_list.clear()
        self.shared_head_q_list.clear()
        self.shared_T_list.clear()
        self.head_move_count = 0
        self.update_sample_counts()
        self.update_head_pose_status()
        self.log(self.user_text, "Shared samples cleared.")
        self.log(self.dev_text, "Shared samples cleared.")

    def connect_robot(self, ip, model_name, status_var, text_widget):
        try:
            self.robot = create_robot(ip, model_name)
            self.dyn_model = self.robot.get_dynamics()
            self.model = self.robot.model()
            status_var.set("Connected")
            self.log(text_widget, f"Connected: {ip} (model={model_name})")
            self.update_head_pose_status()
        except Exception as e:
            status_var.set("Disconnected")
            messagebox.showerror("Connection Error", str(e))
            self.log(text_widget, f"Connection failed: {e}")

    def move_head_to_pose(self, pose_deg, text_widget):
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")

        pose_rad = np.deg2rad(np.array(pose_deg, dtype=np.float64))
        cmd = (
            rby.RobotCommandBuilder()
            .set_command(
                rby.ComponentBasedCommandBuilder().set_head_command(
                    rby.JointPositionCommandBuilder()
                    .set_position(pose_rad)
                    .set_minimum_time(2.0)
                )
            )
        )
        rv = self.robot.send_command(cmd, 5).get()
        if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError("Failed to move head to the requested pose.")

        self.log(text_widget, f"Head moved to {pose_deg} deg")

    def move_head_to_next_pose(self, text_widget):
        pose_deg = (
            float(np.random.uniform(*HEAD_SWEEP_RANGE_DEG["head0"])),
            float(np.random.uniform(*HEAD_SWEEP_RANGE_DEG["head1"])),
        )
        self.move_head_to_pose(pose_deg, text_widget)
        self.head_move_count += 1
        self.update_head_pose_status()

    def capture_one_sample(self, arm, text_widget):
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")

        if self.marker_transform is None:
            self.marker_transform = create_live_marker_transform()

        if self.model is None:
            raise RuntimeError("Robot is not connected.")

        cfg = get_arm_config(self.model, arm)
        head_cfg = get_head_config(self.model)
        q_arm, q_head, T_meas = capture_robot_sample(
            robot=self.robot,
            arm_idx=cfg["arm_idx"],
            marker_transform=self.marker_transform,
            head_idx=head_cfg["head_idx"],
            side=arm,
        )
        if T_meas is None:
            self.log(text_widget, "Marker not detected.")
            return None, None, None

        self.log(text_widget, f"Captured sample")
        self.log(text_widget, f"q_arm = {np.round(q_arm, 3)}")
        self.log(text_widget, f"q_head = {np.round(q_head, 3)}")
        self.log(text_widget, f"marker =\n{np.round(T_meas, 3)}")
        return q_arm, q_head, T_meas

    def run_optimizer(self, arm, ndof, q_arm_list, q_head_list, T_meas_list, result_path, text_widget):
        if self.model is None:
            raise RuntimeError("Robot is not connected.")

        cfg = get_arm_config(self.model, arm)
        head_cfg = get_head_config(self.model)

        optimizer = CalibrationOptimizer(
            robot=self.robot,
            arm_idx=cfg["arm_idx"],
            ee_link=cfg["ee_link"],
            mount_to_cam_nom=cfg["mount_to_cam_nom"],
            ee_to_marker_nom=cfg["ee_to_marker_nom"],
            ndof=ndof,
            head_idx=head_cfg["head_idx"],
        )

        q_arm_offset, q_head_offset, xi_t5_cam, t5_to_cam_new = optimizer.optimize(q_arm_list, q_head_list, T_meas_list)

        t5_to_cam_new = [float(x) for x in t5_to_cam_new]

        self.log(text_widget, "\n===== RESULT =====")
        self.log(text_widget, "Arm joint offset (deg):")
        self.log(text_widget, str(np.rad2deg(q_arm_offset)))
        if q_head_offset is not None:
            self.log(text_widget, "Head joint offset (deg):")
            self.log(text_widget, str(np.rad2deg(q_head_offset)))
        self.log(text_widget, "T5-to-camera xi:")
        self.log(text_widget, str(xi_t5_cam))
        self.log(text_widget, "t5_to_cam_new:")
        self.log(text_widget, str(t5_to_cam_new))

        result_dict = {
            "joint_offset_deg": np.rad2deg(q_arm_offset).tolist(),
            "head_joint_offset_deg": np.rad2deg(q_head_offset).tolist() if q_head_offset is not None else None,
            "xi_t5_cam": np.array(xi_t5_cam).tolist(),
            "xi_cam": np.array(xi_t5_cam).tolist()
        }

        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=4)

        self.last_result_path = result_path
        self.log(text_widget, f"Result saved to {result_path}")

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

    def apply_home_offset_common(self, ip, model_name, arm, text_widget):
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
            servo=".*",
        )

        self.log(text_widget, "Home offset applied successfully.")
        self.log(text_widget, f"Arm: {result['arm']}")
        self.log(text_widget, f"Source: {result['source']}")
        self.log(text_widget, f"JSON: {result['json_path']}")
        self.log(text_widget, f"Offset (deg): {result['offset_deg']}")
        if result.get("head_offset_deg") is not None:
            self.log(text_widget, f"Head Offset (deg): {result['head_offset_deg']}")

    # ============================================================
    # User tab
    # ============================================================

    def user_connect(self):
        self.connect_robot(self.user_ip.get(), self.user_model.get(), self.user_status, self.user_text)

    def user_next_head_pose(self):
        try:
            self.move_head_to_next_pose(self.user_text)
        except Exception as e:
            messagebox.showerror("Head Move Error", str(e))
            self.log(self.user_text, f"Head move failed: {e}")

    def user_record(self):
        try:
            q_arm, q_head, T_meas = self.capture_one_sample(self.user_arm.get(), self.user_text)
            if q_arm is None:
                return
            self.shared_arm_q_list.append(q_arm)
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
            ndof = 15 if q_head_list is not None and np.ptp(q_head_list, axis=0).max() > np.deg2rad(1.0) else 13

            save_npz_dataset(dataset_path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)
            self.last_dataset_path = dataset_path
            self.log(self.user_text, f"Dataset saved to {dataset_path}")
            self.log(self.user_text, f"Optimization ndof = {ndof}")

            self.run_optimizer(
                arm=self.user_arm.get(),
                ndof=ndof,
                q_arm_list=q_arm_list,
                q_head_list=q_head_list,
                T_meas_list=T_meas_list,
                result_path=result_path,
                text_widget=self.user_text,
            )
        except Exception as e:
            messagebox.showerror("Calculate Error", str(e))
            self.log(self.user_text, f"Calculate failed: {e}")

    def user_apply_home_offset(self):
        try:
            self.apply_home_offset_common(
                ip=self.user_ip.get(),
                model_name=self.user_model.get(),
                arm=self.user_arm.get(),
                text_widget=self.user_text,
            )
        except Exception as e:
            messagebox.showerror("Apply Home Offset Error", str(e))
            self.log(self.user_text, f"Apply home offset failed: {e}")

    # ============================================================
    # Developer tab
    # ============================================================

    def _update_dev_mode_label(self, event=None):
        mode = self.dev_mode.get()
        if mode == "live":
            self.dev_mode_info.set("Record button is used only in live mode.")
        elif mode == "npz":
            self.dev_mode_info.set("Path is used in npz mode.")
        else:
            if int(self.dev_ndof.get()) in (2, 9, 15):
                self.dev_mode_info.set("sim mode uses default random arm + head samples.")
            else:
                self.dev_mode_info.set("sim mode uses default random arm samples.")

    def dev_connect(self):
        self.connect_robot(self.dev_ip.get(), self.dev_model.get(), self.dev_status, self.dev_text)

    def dev_next_head_pose(self):
        try:
            self.move_head_to_next_pose(self.dev_text)
        except Exception as e:
            messagebox.showerror("Head Move Error", str(e))
            self.log(self.dev_text, f"Head move failed: {e}")

    def dev_record(self):
        try:
            if self.dev_mode.get() != "live":
                messagebox.showwarning("Warning", "Record is available only in live mode.")
                return

            q_arm, q_head, T_meas = self.capture_one_sample(self.dev_arm.get(), self.dev_text)
            if q_arm is None:
                return
            self.shared_arm_q_list.append(q_arm)
            self.shared_head_q_list.append(q_head)
            self.shared_T_list.append(T_meas)
            self.update_sample_counts()
        except Exception as e:
            messagebox.showerror("Record Error", str(e))
            self.log(self.dev_text, f"Record failed: {e}")

    def dev_calculate(self):
        try:
            mode = self.dev_mode.get()
            ndof = int(self.dev_ndof.get())
            arm = self.dev_arm.get()
            if self.model is None:
                raise RuntimeError("Robot is not connected.")

            cfg = get_arm_config(self.model, arm)
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
                validate_dataset_for_ndof(ndof, q_arm_list, q_head_list, T_meas_list)
                self.log(self.dev_text, f"Loaded npz: {npz_path}")
                self.log(self.dev_text, f"samples = {len(q_arm_list)}")

            else:  # sim
                sample_count = 100
                q_arm_list = np.random.uniform(-5, 5, (sample_count, 7))
                if ndof in (2, 9, 15):
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
                    ndof,
                    cfg["ee_link"],
                    cfg["mount_to_cam_nom"],
                    cfg["ee_to_marker_nom"],
                    camera_link=head_cfg["camera_link"],
                )
                self.log(self.dev_text, f"Simulation dataset generated. samples = {sample_count}, ndof = {ndof}")

            dataset_path, result_path = self.build_output_paths()
            save_npz_dataset(dataset_path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)
            self.last_dataset_path = dataset_path
            self.log(self.dev_text, f"Dataset saved to {dataset_path}")

            self.run_optimizer(
                arm=arm,
                ndof=ndof,
                q_arm_list=q_arm_list,
                q_head_list=q_head_list,
                T_meas_list=T_meas_list,
                result_path=result_path,
                text_widget=self.dev_text,
            )

        except Exception as e:
            messagebox.showerror("Calculate Error", str(e))
            self.log(self.dev_text, f"Calculate failed: {e}")

    def dev_apply_home_offset(self):
        try:
            self.apply_home_offset_common(
                ip=self.dev_ip.get(),
                model_name=self.dev_model.get(),
                arm=self.dev_arm.get(),
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
            if self.marker_transform is not None:
                self.marker_transform.camera.monitoring(Flag=False)
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
