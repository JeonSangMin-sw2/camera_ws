"""UI-independent execution owner for robot calibration and camera acquisition."""
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock
import logging

import numpy as np

from core.robot.robot_core import RobotOperations
from core.robot.motion import AutoCollectionConfig
from .MarkerCalibrator import MarkerCalibrator
from .JointCalibrator import JointCalibrator
from .HeadCameraCalibrator import HeadCameraCalibrator
from .IntrinsicsCalibrator import IntrinsicsCalibrator
from .observation import ObservationSource
from .sequences.result import SequenceContext, SequenceCancelled, require_converged_joint


class CalibrationCore:
    def __init__(self, observer=None, robot=None, on_event=None, camera_factory=None):
        self.stop_event = Event()
        self._run_lock = Lock()
        self._state_lock = Lock()
        self._contexts = []
        self._last_result = None
        self._prepared = False
        self._executing = False
        self._closed = False
        self.robot_address = None
        self.robot_model_name = None
        self.on_event = on_event
        self.camera_factory = camera_factory
        self.robot = robot
        self.model = robot.model() if robot is not None else None
        self.robot_version = "1.2"
        self.include_head_motion = True
        self.apply_joint_offset_flag = True
        self.joint_offsets_store = {side: {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0} for side in ("right", "left")}
        self.last_home_reset_path = None
        self.last_result_path = None
        self.auto_config = AutoCollectionConfig()
        self.prompt_teaching = None
        self.observer = ObservationSource(observer, self.stop_event) if observer is not None else None
        self.marker_calibrator = MarkerCalibrator(self.observer, robot)
        self.joint_calibrator = JointCalibrator(self.observer, robot)
        self.head_camera_calibrator = HeadCameraCalibrator(self.observer, robot)
        self.intrinsics_calibrator = IntrinsicsCalibrator()

    @property
    def calibrators(self):
        return (self.marker_calibrator, self.joint_calibrator, self.head_camera_calibrator)

    @property
    def is_busy(self):
        return self._run_lock.locked()

    @property
    def is_simulated(self):
        return bool(self.observer is not None and self.observer.sim)

    def emit(self, kind, value):
        if self.on_event is not None:
            try:
                self.on_event(kind, value)
            except Exception:
                # A display/log consumer must not break motion cleanup or strand
                # the execution lock. The result is still available to the caller.
                logging.getLogger(__name__).exception("Core event consumer failed: %s", kind)

    def log_msg(self, message):
        self.emit("log", message)

    def emit_detection(self, detected):
        # Internal observations may still carry validity; UI need not show an indicator.
        self.emit("detection", bool(detected))

    def get_robot_version(self):
        return self.robot_version

    def bind_robot(self, robot, version="1.2"):
        if self._closed:
            raise RuntimeError("Calibration core is closed")
        if self._run_lock.locked():
            raise RuntimeError("Cannot replace robot during a calibration sequence")
        self.robot, self.robot_version = robot, str(version).removeprefix("v")
        self.model = robot.model() if robot is not None else None
        for calibrator in self.calibrators:
            calibrator.robot, calibrator.robot_version = robot, self.robot_version
        if self.observer is not None:
            self.observer.bind_robot(robot, self.robot_version)

    def connect_robot(self, address, model, **options):
        if self._closed:
            raise RuntimeError("Calibration core is closed")
        if self._run_lock.locked():
            raise RuntimeError("Cannot reconnect during a sequence")
        self.robot_address, self.robot_model_name = address, model
        from core.robot.robot_core import connect_robot_session
        robot = connect_robot_session(address, model, log_callback=self.log_msg, **options)
        self.bind_robot(robot)
        return robot

    def disconnect_robot(self):
        if self.is_busy:
            raise RuntimeError("Cannot disconnect during a sequence")
        RobotOperations.terminate_robot(self.robot)
        self.bind_robot(None)

    def connect_camera(self, **options):
        if self._closed:
            raise RuntimeError("Calibration core is closed")
        if self._run_lock.locked():
            raise RuntimeError("Cannot reconnect camera during a calibration sequence")
        if self.observer is not None:
            self.observer.close()
            self.observer = None
            for calibrator in self.calibrators:
                calibrator.marker_st = None
        from core.marker_detection import Marker_Transform
        if self.camera_factory is not None:
            options["camera_factory"] = self.camera_factory
        engine = Marker_Transform(**options)
        self.observer = ObservationSource(engine, self.stop_event)
        self.observer.bind_robot(self.robot, self.robot_version)
        for calibrator in self.calibrators:
            calibrator.marker_st = self.observer

    def get_monitor_snapshot(self):
        if self.observer is None or self.observer.camera is None:
            return {"image": None, "frame_id": 0, "frame_timestamp": None,
                    "temperature": None, "temperature_timestamp": None, "exposure": None,
                    "connected": False, "error": ""}
        return self.observer.camera.get_monitor_snapshot()

    def set_camera_exposure(self, value, auto_exposure=False):
        if self.observer is None:
            return False
        return self.observer.set_camera_exposure(value, auto_exposure=auto_exposure)

    def accept_marker_result(self, arm, result):
        values = [result["x_e"] / 1000, result["y_e"] / 1000, result["z_e"] / 1000,
                  result["roll_e"], result["pitch_e"], result["yaw_e"]]
        for calibrator in self.calibrators:
            calibrator.camera_config[f"Tf_to_marker_{arm}"] = values.copy()
        if "opt_delta_5" in result:
            self.joint_offsets_store[arm]["joint5"] = float(result["opt_delta_5"])
            self.joint_offsets_store[arm]["joint6"] = float(result["opt_delta_6"])

    def update_marker_transforms(self, left, right):
        if self.is_busy:
            raise RuntimeError("Cannot change marker transforms during a sequence")
        if self.observer is not None:
            with self.observer.lock:
                engine = self.observer.engine
                for side, value in (("left", left), ("right", right)):
                    engine.markers_config[f"Tf_to_marker_{side}"] = list(value)
                    setattr(engine, f"Tf_to_marker_tf_{side}", engine.make_transform(value))

    def prepare_run(self):
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("Another calibration sequence is already running")
        if self._closed:
            self._run_lock.release()
            raise RuntimeError("Calibration core is closed")
        self._prepared = True
        self.stop_event.clear()
        for calibrator in self.calibrators:
            calibrator.stop_requested = False
            calibrator.include_head_motion = self.include_head_motion
            calibrator.partial_data = {}

    def stop_check(self):
        if self.stop_event.is_set():
            raise SequenceCancelled()

    def checkpoint(self, key, value):
        if self._contexts:
            self._contexts[-1].checkpoint(key, value)

    def cancel(self):
        self.stop_event.set()
        for calibrator in self.calibrators:
            calibrator.stop_requested = True
        if self._executing and self.robot is not None:
            try:
                from core.robot.robot_core import cancel_control
                cancel_control(self.robot, self.robot_address, self.robot_model_name)
            except Exception as error:
                self.log_msg(f"Cancel command failed: {error}")

    def get_run_status(self):
        with self._state_lock:
            return deepcopy(self._last_result)

    def _progress(self, result):
        with self._state_lock:
            self._last_result = result.snapshot()
        self.emit("progress", result.snapshot())

    def run(self, name, *, prepared=False, **options):
        if not prepared:
            self.prepare_run()
        with self._state_lock:
            if not self._prepared:
                raise RuntimeError("Sequence must be prepared exactly once")
            self._prepared = False
            self._executing = True
        try:
            from core.robot.robot_core import motion_cancellation
            with motion_cancellation(self.stop_event):
                result = self._execute(name, options)
            self._progress(result)
        finally:
            self._executing = False
            self._run_lock.release()
        self.emit("result", result.snapshot())
        return result

    def _execute(self, name, options):
        ctx = SequenceContext(name, self.stop_event, self._progress)
        self._contexts.append(ctx)
        if self.observer is not None:
            self.observer.on_observation = self._record_observation
        self._progress(ctx.result)
        try:
            ctx.check_cancelled()
            needs_observation = name in ("marker", "joint", "step1", "collect", "full") or (
                name == "step2" and options.get("samples") is None)
            if needs_observation and (self.robot is None or self.observer is None):
                raise RuntimeError("Robot and camera/marker source must be connected before acquisition")
            if name == "home":
                from core.robot.home_offset import HomeOffsetController, save_home_reset_baseline_json, reset_current_pose_home_offsets
                from core.storage import StoragePaths
                controller = HomeOffsetController(self.robot, self.log_msg)
                task = options.pop("task_type")
                if task == "reset":
                    baseline, data = save_home_reset_baseline_json(
                        self.robot, self.model, StoragePaths.root / "config",
                        model_name=options.pop("model_name", self.robot_model_name),
                        include_head=options.get("include_head", True))
                    self.last_home_reset_path = str(baseline)
                    ctx.checkpoint("baseline", data)
                    value = reset_current_pose_home_offsets(self.robot, self.model, log_cb=self.log_msg, **options)
                elif task == "move_zero":
                    value = controller.move_home_offset_candidate_path(**options)
                elif task == "move_check":
                    value = controller.move_to_check_position_candidate_path(**options)
                elif task == "apply":
                    value = controller.apply_current_pose_home_offset(**options)
                else:
                    raise ValueError(f"Unknown Home Offset action: {task}")
                ctx.checkpoint("home", value)
                ctx.check_cancelled()
                if not value.get("success", value.get("status") != "failed"):
                    raise RuntimeError(value.get("error", "Home Offset action failed"))
                ctx.complete("home", value)
            elif name in ("zero_pose", "check_state", "draw_square"):
                from core.robot.motion import move_to_zero_pose, check_calibration_state, execute_draw_square_trajectory
                if name == "zero_pose":
                    value = move_to_zero_pose(self.robot, self.model, log_cb=self.log_msg, **options)
                elif name == "check_state":
                    value = check_calibration_state(self.robot, log_cb=self.log_msg, **options)
                else:
                    value = execute_draw_square_trajectory(self.robot, log_cb=self.log_msg, **options)
                ctx.check_cancelled()
                if value is False:
                    raise RuntimeError("Robot action failed")
                ctx.complete("motion", value)
            elif name == "step2_ready":
                from core.robot.motion import move_to_auto_ready_pose, verify_and_align_head_at_ready_pose
                arms = options.get("active_arms", ["right", "left"])
                priority = options.get("priority", 10)
                move_to_auto_ready_pose(self.robot, arms, priority=priority,
                                       include_head_motion=self.include_head_motion,
                                       robot_version=self.robot_version)
                ctx.check_cancelled()
                verify_and_align_head_at_ready_pose(
                    self.robot, self.observer, self.model, arms, priority,
                    include_head_motion=self.include_head_motion,
                    prompt_teaching_cb=self.prompt_teaching, log_cb=self.log_msg,
                    on_head_aligned_cb=lambda q: ctx.checkpoint("head_pose", q))
                ctx.check_cancelled()
                ctx.complete("ready", True)
            elif name == "ready":
                kind = options.pop("kind", "marker")
                calibrator = {"marker": self.marker_calibrator, "joint": self.joint_calibrator,
                              "head": self.head_camera_calibrator}[kind]
                if kind == "head":
                    options["stop_event"] = self.stop_event
                value = calibrator.perform_move_to_ready_pose(log_callback=self.log_msg, **options)
                ctx.check_cancelled()
                if not value:
                    raise RuntimeError("Ready pose failed")
                ctx.complete("ready", True)
            elif name == "full_ready":
                for side in ("right", "left"):
                    ctx.check_cancelled()
                    c = self.marker_calibrator if self.marker_calibrator.is_v13() else self.joint_calibrator
                    mode = "marker" if c is self.marker_calibrator else "wrist_pitch"
                    if not c.perform_move_to_ready_pose(side, mode, log_callback=self.log_msg):
                        ctx.check_cancelled()
                        raise RuntimeError(f"Ready pose failed: {side}")
                    ctx.complete(side, True)
            elif name == "manual_head":
                value = self.joint_calibrator.movej(self.robot, head=options["position"], minimum_time=1.5)
                ctx.check_cancelled()
                if not value:
                    raise RuntimeError("Head movement failed")
                ctx.complete("motion", True)
            elif name == "intrinsics":
                self.intrinsics_calibrator.stop_event = self.stop_event
                value = self.intrinsics_calibrator.run_calibration_with_images(options["images"], None)
                ctx.checkpoint("intrinsics", getattr(self.intrinsics_calibrator, "partial_data", {}))
                ctx.check_cancelled()
                if not value:
                    raise RuntimeError("Intrinsics calibration failed")
                ctx.complete("intrinsics", {"camera_matrix": self.intrinsics_calibrator.cameraMatrix,
                                            "dist_coeffs": self.intrinsics_calibrator.distCoeffs})
            elif name == "step1":
                from .sequences.step1 import execute_step1_sequence
                result = execute_step1_sequence(
                    self.joint_calibrator, self.marker_calibrator, self.joint_offsets_store,
                    stop_event=self.stop_event, log_callback=self.log_msg,
                    status_callback=self.emit_detection,
                    bracket_finished_callback=lambda value: self.emit("bracket", value),
                    joint_finished_callback=lambda value: self.emit("joint", value),
                    context=ctx, **options,
                )
                ctx.result = result
            elif name == "marker":
                from .sequences.marker import run_marker
                run_marker(self, ctx, **options)
            elif name == "joint":
                value = self.joint_calibrator.perform_joint_calibration(
                    log_callback=self.log_msg, status_callback=self.emit_detection, **options)
                require_converged_joint(value, ctx, "joint_result")
                ctx.complete("joint", value)
            elif name == "step1_5":
                from .sequences.step1_5 import run_step1_5
                run_step1_5(self, ctx, **options)
            elif name == "collect":
                from .sequences.collection import run_collection
                run_collection(self, ctx, **options)
            elif name == "optimize":
                from .sequences.step2 import optimize_step2
                result = optimize_step2(self, *options.get("args", ()), **options.get("kwargs", {}))
                ctx.checkpoint("optimization", result)
                ctx.check_cancelled()
                ctx.complete("optimization", result)
            elif name == "step2":
                from .sequences.collection import run_collection
                from .sequences.step2 import optimize_step2
                samples = options.get("samples")
                if samples is None:
                    collection_options = dict(options.get("collection", {}))
                    collection_options.setdefault("prepare", True)
                    samples = run_collection(self, ctx, **collection_options)
                ctx.check_cancelled()
                solve = dict(options.get("optimization", {}))
                if not samples:
                    raise RuntimeError("No samples supplied for Step 2")
                head_samples = [s.get("q_head") for s in samples]
                has_head_samples = self.include_head_motion and any(q is not None for q in head_samples)
                if has_head_samples and any(q is None for q in head_samples):
                    raise RuntimeError("Step 2 samples contain incomplete head joint data")
                q_head_list = np.asarray(head_samples, dtype=float) if has_head_samples else None
                solve.setdefault("active_arms", ["right", "left"])
                solve.setdefault("optimize_head", has_head_samples)
                solve.setdefault("optimize_camera", True)
                solve.setdefault("result_path", self.default_result_path())
                q_arm_array = np.asarray([s["q_arm"] for s in samples])
                T_meas_array = np.asarray([s["marker"] for s in samples])
                # Keep the raw samples next to the result (dataset_<timestamp>.npz) so any run can
                # be re-optimized / compared offline; the GUI auto-save only covers manual collection.
                if options.get("samples") is None:
                    from core.calibration.data import save_npz_dataset
                    result_file = Path(solve["result_path"])
                    dataset_path = result_file.with_name(result_file.name.replace("result_", "dataset_", 1)).with_suffix(".npz")
                    try:
                        save_npz_dataset(dataset_path, q_arm=q_arm_array, T_meas=T_meas_array, q_head=q_head_list)
                        self.log_msg(f"[Step2] Dataset saved: {dataset_path}")
                        ctx.checkpoint("dataset_path", str(dataset_path))
                    except Exception as error:
                        self.log_msg(f"[Step2][WARN] Dataset save failed: {error}")
                value = optimize_step2(self, q_arm_list=q_arm_array,
                                      q_head_list=q_head_list,
                                      T_meas_list=T_meas_array, **solve)
                ctx.checkpoint("optimization", value)
                ctx.check_cancelled()
                ctx.complete("optimization", value)
            elif name == "full":
                from .sequences.full import run_full
                run_full(self, ctx, **options)
            else:
                raise ValueError(f"Unknown calibration sequence: {name}")
            if ctx.result.status == "running":
                ctx.check_cancelled()
                ctx.result.finish("completed")
        except SequenceCancelled as error:
            ctx.result.partial.update(error.partial)
            ctx.result.finish("cancelled", str(error))
        except Exception as error:
            ctx.result.partial.update(getattr(error, "partial", {}))
            ctx.result.finish("cancelled" if self.stop_event.is_set() else "failed", str(error))
        finally:
            for key, c in zip(("marker", "joint", "head"), self.calibrators):
                if c.partial_data:
                    ctx.result.partial[key] = deepcopy(c.partial_data)
            self._contexts.pop()
        return ctx.result.snapshot()

    def _record_observation(self, value, frame):
        if self._contexts and value is not None:
            observations = self._contexts[-1].result.partial.setdefault("observations", [])
            observations.append({"marker": deepcopy(value), "frame": frame})

    @staticmethod
    def default_result_path():
        from datetime import datetime
        from core.storage import CONFIG_PATHS
        return str(Path(CONFIG_PATHS["result_dir"]) / ("result_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"))

    def accept_head_result(self, result):
        self.head_camera_calibrator.calibrated_results = deepcopy(result)
        if not result.get("skipped"):
            self.joint_offsets_store["head"] = deepcopy(result.get("head_offsets_deg", {}))
            for calibrator in self.calibrators:
                if "calibrated_mount_to_cam" in result:
                    calibrator.camera_config["mount_to_cam"] = deepcopy(result["calibrated_mount_to_cam"])

    def close(self):
        if self._closed:
            return
        self.cancel()
        with self._state_lock:
            if self._prepared and not self._executing:
                self._prepared = False
                self._run_lock.release()
        if not self._run_lock.acquire(timeout=5.0):
            raise RuntimeError("Calibration is stopping; wait before closing the application")
        try:
            if self.observer is not None:
                self.observer.close()
            if self.robot is not None and not RobotOperations.terminate_robot(self.robot):
                raise RuntimeError("Robot session could not be disconnected")
            self.observer, self.robot, self.model = None, None, None
            for calibrator in self.calibrators:
                calibrator.marker_st, calibrator.robot = None, None
            self._closed = True
        finally:
            self._run_lock.release()
