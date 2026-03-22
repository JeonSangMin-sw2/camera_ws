import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from calibration_core import (
    create_robot,
    load_npz_dataset,
    generate_sim_measurements,
    CalibrationOptimizer,
    Marker_Transform,
)
from homeoffset_core import (
    apply_home_offset_from_json,
    move_robot_to_zero_pose,
)

class CalibrationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Calibration UI")
        self.root.geometry("900x680")

        self.robot = None
        self.dyn_model = None
        self.model = None
        self.marker_transform = None

        self.user_q_list = []
        self.user_T_list = []

        self.dev_q_list = []
        self.dev_T_list = []

        self.warning_img = None

        self._build_ui()

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

        image_paths = ["warning_pose_check.png", "warning_pose.png"]

        for image_path in image_paths:
            if os.path.exists(image_path):
                try:
                    img = tk.PhotoImage(file=image_path)
                    img = img.subsample(3, 3)
                    lbl = ttk.Label(popup, image=img)
                    lbl.image = img
                    lbl.pack(padx=10, pady=8)
                except Exception:
                    ttk.Label(popup, text=f"Failed to load image: {image_path}").pack(pady=5)

        ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=15)
        popup.wait_window()

    def zero_pose_check_common(self, ip, arm, text_widget):
        result = move_robot_to_zero_pose(
            address=ip,
            model_name="m",
            arm=arm,
            power=".*",
            servo="^(?!.*head).*",
        )

        self.log(text_widget, "\n===== ZERO POSE CHECK =====")
        self.log(text_widget, f"Arm: {result['arm']}")
        self.log(text_widget, result["message"])

        self.show_zero_pose_check_popup()
        
    def user_zero_pose_check(self):
        try:
            self.zero_pose_check_common(
                ip=self.user_ip.get(),
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
        ttk.Button(conn, text="Connect", command=self.user_connect).grid(row=0, column=2, padx=5, pady=5)

        self.user_status = tk.StringVar(value="Disconnected")
        ttk.Label(conn, textvariable=self.user_status).grid(row=0, column=3, padx=10, pady=5, sticky="w")

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
        ttk.Label(setup, text="ndof: 13").grid(row=0, column=3, padx=20, pady=5, sticky="w")

        # actions
        act = ttk.LabelFrame(frm, text="Actions")
        act.pack(fill="x", padx=10, pady=10)

        ttk.Button(act, text="1.Zero Pose Check", command=self.user_zero_pose_check).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(act, text="2.Record", command=self.user_record).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(act, text="3.Calculate", command=self.user_calculate).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(act, text="4.Apply Home Offset", command=self.user_apply_home_offset).grid(row=0, column=3, padx=5, pady=5)

        self.user_count = tk.StringVar(value="Samples: 0")
        ttk.Label(act, textvariable=self.user_count).grid(row=0, column=4, padx=20, pady=5, sticky="w")
        
        

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
        ttk.Button(conn, text="Connect", command=self.dev_connect).grid(row=0, column=2, padx=5, pady=5)

        self.dev_status = tk.StringVar(value="Disconnected")
        ttk.Label(conn, textvariable=self.dev_status).grid(row=0, column=3, padx=10, pady=5, sticky="w")

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
        ttk.Combobox(cfg, textvariable=self.dev_ndof, values=[6, 7, 13], state="readonly", width=10)\
            .grid(row=0, column=5, padx=5, pady=5, sticky="w")

        ttk.Label(cfg, text="Path").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dev_path = tk.StringVar(value="captured_dataset.npz")
        ttk.Entry(cfg, textvariable=self.dev_path, width=40).grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="w")

        self.dev_mode_info = tk.StringVar(value="Record button is used only in live mode.")
        ttk.Label(cfg, textvariable=self.dev_mode_info).grid(row=1, column=4, columnspan=2, padx=5, pady=5, sticky="w")

        # actions
        act = ttk.LabelFrame(frm, text="Actions")
        act.pack(fill="x", padx=10, pady=10)

        ttk.Button(act, text="1.Zero Pose Check", command=self.dev_zero_pose_check).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(act, text="2.Record", command=self.dev_record).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(act, text="3.Calculate", command=self.dev_calculate).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(act, text="4.Apply Home Offset", command=self.dev_apply_home_offset).grid(row=0, column=3, padx=5, pady=5)

        self.dev_count = tk.StringVar(value="Live Samples: 0")
        ttk.Label(act, textvariable=self.dev_count).grid(row=0, column=4, padx=20, pady=5, sticky="w")



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

    def get_arm_config(self, arm):
        if self.model is None:
            raise RuntimeError("Robot is not connected.")

        if arm == "right":
            return {
                "ARM_IDX": self.model.right_arm_idx,
                "ee_link": "ee_right",
                "tool_to_cam_nom": [
                    0.01079, -0.094527, -0.028914,
                    154.992754, -0.269972, -179.718444
                ],
            }
        else:
            return {
                "ARM_IDX": self.model.left_arm_idx,
                "ee_link": "ee_left",
                "tool_to_cam_nom": [
                    -0.009187, 0.094257, -0.028313,
                    154.667827, -0.320824, -0.268186
                ],
            }

    def connect_robot(self, ip, status_var, text_widget):
        try:
            self.robot = create_robot(ip)
            self.dyn_model = self.robot.get_dynamics()
            self.model = self.robot.model()
            status_var.set("Connected")
            self.log(text_widget, f"Connected: {ip}")
        except Exception as e:
            status_var.set("Disconnected")
            messagebox.showerror("Connection Error", str(e))
            self.log(text_widget, f"Connection failed: {e}")

    def capture_one_sample(self, arm, text_widget):
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")

        if self.marker_transform is None:
            self.marker_transform = Marker_Transform(
                Stereo=True,
                serial_number=None,
                monitoring=False
            )

        cfg = self.get_arm_config(arm)

        state = self.robot.get_state()
        q_full = state.position.copy()
        q_cmd = q_full[cfg["ARM_IDX"][:7]].copy()

        result = self.marker_transform.get_marker_transform(sampling_time=2)
        if result is None:
            self.log(text_widget, "Marker not detected.")
            return None, None

        T_meas = np.array(result).reshape(4, 4)
        self.log(text_widget, f"Captured sample")
        self.log(text_widget, f"q = {np.round(q_cmd, 3)}")
        self.log(text_widget, f"marker =\n{np.round(T_meas, 3)}")
        return q_cmd, T_meas

    def run_optimizer(self, arm, ndof, q_cmd_list, T_meas_list, text_widget):
        cfg = self.get_arm_config(arm)

        optimizer = CalibrationOptimizer(
            robot=self.robot,
            arm_idx=cfg["ARM_IDX"],
            ee_link=cfg["ee_link"],
            tool_to_cam_nom=cfg["tool_to_cam_nom"],
            ndof=ndof,
        )

        q_offset, xi_cam, tool_to_cam_new = optimizer.optimize(q_cmd_list, T_meas_list)

        tool_to_cam_new = [float(x) for x in tool_to_cam_new]

        self.log(text_widget, "\n===== RESULT =====")
        self.log(text_widget, "Joint offset (deg):")
        self.log(text_widget, str(np.rad2deg(q_offset)))
        self.log(text_widget, "Camera xi:")
        self.log(text_widget, str(xi_cam))
        self.log(text_widget, "tool_to_cam_new:")
        self.log(text_widget, str(tool_to_cam_new))

        result_dict = {
            "joint_offset_deg": np.rad2deg(q_offset).tolist(),
            "xi_cam": np.array(xi_cam).tolist()
        }

        with open("calibration_result.json", "w") as f:
            json.dump(result_dict, f, indent=4)

        self.log(text_widget, "Result saved to calibration_result.json")

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

        image_path = "warning_pose.png"
        if os.path.exists(image_path):
            try:
                self.warning_img = tk.PhotoImage(file=image_path)
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

    def apply_home_offset_common(self, ip, arm, text_widget):
        if not os.path.exists("calibration_result.json"):
            raise RuntimeError("calibration_result.json not found.")

        proceed = self.confirm_home_offset_action()
        if not proceed:
            self.log(text_widget, "Apply home offset cancelled.")
            return

        result = apply_home_offset_from_json(
            address=ip,
            model_name="m",
            arm=arm,
            json_path="calibration_result.json",
            power=".*",
            servo="^(?!.*head).*",
        )

        self.log(text_widget, "Home offset applied successfully.")
        self.log(text_widget, f"Arm: {result['arm']}")
        self.log(text_widget, f"Source: {result['source']}")
        self.log(text_widget, f"JSON: {result['json_path']}")
        self.log(text_widget, f"Offset (deg): {result['offset_deg']}")

    # ============================================================
    # User tab
    # ============================================================

    def user_connect(self):
        self.connect_robot(self.user_ip.get(), self.user_status, self.user_text)

    def user_record(self):
        try:
            q_cmd, T_meas = self.capture_one_sample(self.user_arm.get(), self.user_text)
            if q_cmd is None:
                return
            self.user_q_list.append(q_cmd)
            self.user_T_list.append(T_meas)
            self.user_count.set(f"Samples: {len(self.user_q_list)}")
        except Exception as e:
            messagebox.showerror("Record Error", str(e))
            self.log(self.user_text, f"Record failed: {e}")

    def user_calculate(self):
        try:
            if len(self.user_q_list) == 0:
                messagebox.showwarning("Warning", "No recorded samples.")
                return

            q_cmd_list = np.array(self.user_q_list)
            T_meas_list = np.array(self.user_T_list)

            np.savez_compressed(
                "captured_dataset.npz",
                q=q_cmd_list,
                marker=T_meas_list
            )
            self.log(self.user_text, "Dataset saved to captured_dataset.npz")

            self.run_optimizer(
                arm=self.user_arm.get(),
                ndof=13,
                q_cmd_list=q_cmd_list,
                T_meas_list=T_meas_list,
                text_widget=self.user_text,
            )
        except Exception as e:
            messagebox.showerror("Calculate Error", str(e))
            self.log(self.user_text, f"Calculate failed: {e}")

    def user_apply_home_offset(self):
        try:
            self.apply_home_offset_common(
                ip=self.user_ip.get(),
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
            self.dev_mode_info.set("sim mode uses random q_cmd_list with size 100.")

    def dev_connect(self):
        self.connect_robot(self.dev_ip.get(), self.dev_status, self.dev_text)

    def dev_record(self):
        try:
            if self.dev_mode.get() != "live":
                messagebox.showwarning("Warning", "Record is available only in live mode.")
                return

            q_cmd, T_meas = self.capture_one_sample(self.dev_arm.get(), self.dev_text)
            if q_cmd is None:
                return
            self.dev_q_list.append(q_cmd)
            self.dev_T_list.append(T_meas)
            self.dev_count.set(f"Live Samples: {len(self.dev_q_list)}")
        except Exception as e:
            messagebox.showerror("Record Error", str(e))
            self.log(self.dev_text, f"Record failed: {e}")

    def dev_calculate(self):
        try:
            mode = self.dev_mode.get()
            ndof = int(self.dev_ndof.get())
            arm = self.dev_arm.get()
            cfg = self.get_arm_config(arm)

            if mode == "live":
                if len(self.dev_q_list) == 0:
                    messagebox.showwarning("Warning", "No recorded samples.")
                    return

                q_cmd_list = np.array(self.dev_q_list)
                T_meas_list = np.array(self.dev_T_list)

                np.savez_compressed(
                    "captured_dataset.npz",
                    q=q_cmd_list,
                    marker=T_meas_list
                )
                self.log(self.dev_text, "Dataset saved to captured_dataset.npz")

            elif mode == "npz":
                q_cmd_list, T_meas_list = load_npz_dataset(self.dev_path.get())
                self.log(self.dev_text, f"Loaded npz: {self.dev_path.get()}")
                self.log(self.dev_text, f"size = {np.size(q_cmd_list)}")

            else:  # sim
                q_cmd_list = np.random.uniform(-5, 5, (100, 7))
                q_nominal = self.robot.get_state().position.copy()
                T_meas_list = generate_sim_measurements(
                    self.robot,
                    self.dyn_model,
                    q_cmd_list,
                    cfg["ARM_IDX"],
                    q_nominal,
                    ndof,
                    cfg["ee_link"],
                    cfg["tool_to_cam_nom"],
                )
                self.log(self.dev_text, "Simulation dataset generated.")

            self.run_optimizer(
                arm=arm,
                ndof=ndof,
                q_cmd_list=q_cmd_list,
                T_meas_list=T_meas_list,
                text_widget=self.dev_text,
            )

        except Exception as e:
            messagebox.showerror("Calculate Error", str(e))
            self.log(self.dev_text, f"Calculate failed: {e}")

    def dev_apply_home_offset(self):
        try:
            self.apply_home_offset_common(
                ip=self.dev_ip.get(),
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