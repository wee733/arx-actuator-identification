from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arx_actuator_identification.actuator import JointActuatorParams
from arx_actuator_identification.b2arx.joint_map import (
    CONTROLLED_JOINT_NAMES,
    DEFAULT_ARM_POS,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    Q_DOWN,
)
from arx_actuator_identification.dataset import SweepDataset


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAAC_USD = PACKAGE_ROOT / "arx_actuator_identification" / "my_robot" / "my_b2arx" / "my_robot.usd"


@dataclass(frozen=True)
class IsaacReplayConfig:
    num_envs: int = 6144
    sim_dt: float = 1.0 / 200.0
    env_spacing: float = 4.0
    usd_path: str | None = None
    device: str = "cuda:0"
    headless: bool = True
    fixed_base_height: float = 0.55
    effort_limit_sim: float = 64.0
    velocity_limit_sim: float = 100.0
    leg_effort_limit_sim: float = 320.0
    base_joint_stiffness: float = 0.0
    base_joint_damping: float = 0.0

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.sim_dt <= 0.0:
            raise ValueError("sim_dt must be positive")
        if self.env_spacing <= 0.0:
            raise ValueError("env_spacing must be positive")
        if self.effort_limit_sim <= 0.0:
            raise ValueError("effort_limit_sim must be positive")


@dataclass
class BatchReplayResult:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray


class IsaacBatchReplayBackend:
    def __init__(self, cfg: IsaacReplayConfig | None = None):
        self.cfg = IsaacReplayConfig() if cfg is None else cfg
        self._worker: _IsaacReplayWorker | None = None

    def replay_batch(
        self,
        dataset: SweepDataset,
        candidates: list[JointActuatorParams],
    ) -> BatchReplayResult:
        if self._worker is None:
            runtime = _import_isaac_runtime(headless=self.cfg.headless, device=self.cfg.device)
            self._worker = _IsaacReplayWorker(self.cfg, runtime)
        return self._worker.replay_batch(dataset, candidates)

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def reset_scene(self) -> None:
        # Keep the Isaac/Kit app alive, but force the next replay to rebuild the
        # SimulationContext and Articulation. Reusing one PhysX scene across all
        # six joints can leave the next joint's first sim.step spinning after a
        # long joint-1 search.
        if self._worker is not None:
            self._worker.reset_scene()


def resolve_isaac_usd_path(path: str | Path | None = None) -> Path:
    resolved = DEFAULT_ISAAC_USD if path is None else Path(path).expanduser()
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Isaac USD not found: {resolved}. Restore the local asset at {DEFAULT_ISAAC_USD} "
            "or pass --usd-path explicitly; see docs/assets.md."
        )
    return resolved


def _import_isaac_runtime(headless: bool = True, device: str | None = None):
    try:
        import torch
        from isaaclab.app import AppLauncher

        if device and str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(str(device))

        launcher_kwargs = {"headless": bool(headless)}
        if device is not None and "device" in inspect.signature(AppLauncher).parameters:
            launcher_kwargs["device"] = str(device)
        app_launcher = AppLauncher(**launcher_kwargs)
        simulation_app = app_launcher.app

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim.schemas import ArticulationRootPropertiesCfg
    except ImportError as exc:
        raise RuntimeError(
            "IsaacLab runtime is required for Isaac replay. Run this command through "
            "your selected `isaaclab.sh -p ...` so IsaacLab and Isaac Sim are on PYTHONPATH; "
            "see docs/versions.md for the legacy backend's compatibility limits."
        ) from exc

    return SimpleNamespace(
        torch=torch,
        AppLauncher=AppLauncher,
        simulation_app=simulation_app,
        sim_utils=sim_utils,
        IdealPDActuatorCfg=IdealPDActuatorCfg,
        Articulation=Articulation,
        ArticulationCfg=ArticulationCfg,
        ArticulationRootPropertiesCfg=ArticulationRootPropertiesCfg,
    )


class _IsaacReplayWorker:
    def __init__(self, cfg: IsaacReplayConfig, runtime):
        self.cfg = cfg
        self.runtime = runtime
        self.torch = runtime.torch
        self.device = cfg.device
        self.usd_path = resolve_isaac_usd_path(cfg.usd_path)
        self.sim = None
        self.robot = None
        self._controlled_joint_idx = None
        self._arm_joint_idx = None
        self._gripper_joint_idx = None
        self._env_origins = None
        self._scene_counter = 0
        self._scene_root: str | None = None

    def replay_batch(
        self,
        dataset: SweepDataset,
        candidates: list[JointActuatorParams],
    ) -> BatchReplayResult:
        if dataset is None:
            raise ValueError("dataset is required")
        dataset.validate()
        if not candidates:
            raise ValueError("candidates must not be empty")
        if len(candidates) > self.cfg.num_envs:
            raise ValueError("candidates length must be <= cfg.num_envs")

        if self.sim is None or self.robot is None:
            self._setup_sim(num_envs=self.cfg.num_envs)
        return self._run_replay(dataset, candidates)

    def _setup_sim(self, num_envs: int) -> None:
        torch = self.torch
        sim_utils = self.runtime.sim_utils
        self._scene_counter += 1
        self._scene_root = f"/World/Replay_{self._scene_counter}"
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.cfg.sim_dt, device=self.device))
        sim_utils.create_prim(self._scene_root, "Xform")
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func(f"{self._scene_root}/Light", light_cfg)
        self._env_origins = _define_origins(torch, num_envs, self.cfg.env_spacing, self.device)
        for env_idx in range(num_envs):
            sim_utils.create_prim(
                f"{self._scene_root}/Env_{env_idx}",
                "Xform",
                translation=self._env_origins[env_idx].tolist(),
            )

        robot_cfg = self._make_robot_cfg().replace(prim_path=f"{self._scene_root}/Env_.*/Robot")
        self.robot = self.runtime.Articulation(robot_cfg)
        self.sim.reset()
        self.robot.update(float(self.cfg.sim_dt))
        controlled_idx = []
        for joint_name in CONTROLLED_JOINT_NAMES:
            idx, _ = self.robot.find_joints(joint_name)
            if len(idx) != 1:
                raise RuntimeError(f"Joint '{joint_name}' maps to {len(idx)} DoFs, expected 1")
            controlled_idx.append(idx[0])
        self._controlled_joint_idx = torch.tensor(controlled_idx, dtype=torch.long, device=self.device)
        self._arm_joint_idx = self._controlled_joint_idx[12:]

        gripper_idx = []
        for joint_name in GRIPPER_JOINT_NAMES:
            idx, _ = self.robot.find_joints(joint_name)
            if len(idx) != 1:
                raise RuntimeError(f"Joint '{joint_name}' maps to {len(idx)} DoFs, expected 1")
            gripper_idx.append(idx[0])
        self._gripper_joint_idx = torch.tensor(gripper_idx, dtype=torch.long, device=self.device)

    def _make_robot_cfg(self):
        sim_utils = self.runtime.sim_utils
        ArticulationCfg = self.runtime.ArticulationCfg
        IdealPDActuatorCfg = self.runtime.IdealPDActuatorCfg
        ArticulationRootPropertiesCfg = self.runtime.ArticulationRootPropertiesCfg
        joint_pos = {
            name: float(value)
            for name, value in zip(CONTROLLED_JOINT_NAMES, np.concatenate([Q_DOWN, DEFAULT_ARM_POS]))
        }
        joint_pos.update(
            {name: float(value) for name, value in zip(GRIPPER_JOINT_NAMES, GRIPPER_CLOSED_POS)}
        )
        return ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(self.usd_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.cfg.fixed_base_height),
                joint_pos=joint_pos,
            ),
            actuators={
                "all_joints": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit=float(self.cfg.effort_limit_sim),
                    effort_limit_sim=float(self.cfg.effort_limit_sim),
                    velocity_limit=float(self.cfg.velocity_limit_sim),
                    velocity_limit_sim=float(self.cfg.velocity_limit_sim),
                    stiffness=float(self.cfg.base_joint_stiffness),
                    damping=float(self.cfg.base_joint_damping),
                )
            },
        )

    def _run_replay(self, dataset: SweepDataset, candidates: list[JointActuatorParams]) -> BatchReplayResult:
        torch = self.torch
        assert self.sim is not None and self.robot is not None
        assert self._controlled_joint_idx is not None and self._arm_joint_idx is not None
        assert self._gripper_joint_idx is not None

        num_envs = len(candidates)
        active_env_ids = torch.arange(num_envs, dtype=torch.long, device=self.device)
        sample_count = len(dataset.t)
        active_joint = int(dataset.active_joint)
        active_joint_id = self._arm_joint_idx[active_joint : active_joint + 1]

        # 每次 replay 前做完整物理世界重置，清除上一轮积累的数值漂移/约束违反状态。
        # 不加此重置时 PhysX 求解器在第二轮第一个 sim.step() 里陷入 CPU 自旋。
        self.sim.reset()
        self.robot.update(float(self.cfg.sim_dt))

        controlled_initial = np.concatenate([Q_DOWN, self._initial_arm_position(dataset)])
        joint_pos = self.robot.data.default_joint_pos[:num_envs].clone()
        joint_vel = self.robot.data.default_joint_vel[:num_envs].clone()
        controlled_initial_t = torch.tensor(controlled_initial, dtype=joint_pos.dtype, device=self.device)
        joint_pos[:, self._controlled_joint_idx] = controlled_initial_t.unsqueeze(0)
        joint_vel[:, self._controlled_joint_idx] = 0.0
        gripper_closed_t = torch.tensor(GRIPPER_CLOSED_POS, dtype=joint_pos.dtype, device=self.device)
        joint_pos[:, self._gripper_joint_idx] = gripper_closed_t.unsqueeze(0)
        joint_vel[:, self._gripper_joint_idx] = 0.0

        root_state = self.robot.data.default_root_state[:num_envs].clone()
        active_env_origins = self._env_origins[:num_envs] if self._env_origins is not None else None
        if active_env_origins is not None:
            root_state[:, :3] += active_env_origins
        root_state[:, 2] = self.cfg.fixed_base_height + (
            active_env_origins[:, 2] if active_env_origins is not None else 0.0
        )
        root_state[:, 3] = 1.0
        root_state[:, 4:7] = 0.0
        root_state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=active_env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=active_env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=active_env_ids)
        self.robot.reset(active_env_ids)
        self.robot.update(float(self.cfg.sim_dt))

        kp = torch.tensor([param.kp for param in candidates], dtype=joint_pos.dtype, device=self.device)
        kd = torch.tensor([param.kd for param in candidates], dtype=joint_pos.dtype, device=self.device)
        tau_limit = torch.tensor([param.tau_limit for param in candidates], dtype=joint_pos.dtype, device=self.device)
        filter_alpha = torch.tensor([param.filter_alpha for param in candidates], dtype=joint_pos.dtype, device=self.device)
        tau_lag_alpha = torch.tensor([param.tau_lag_alpha for param in candidates], dtype=joint_pos.dtype, device=self.device)
        delay_steps = [int(param.delay_steps) for param in candidates]
        max_delay = max(delay_steps) if delay_steps else 0

        # 纯 IdealPD：本套 PhysX articulation 臂已重力中性(实测)，不加重力前馈。
        q_cmd_np = np.asarray(dataset.q_cmd[:, active_joint], dtype=float)
        initial_cmd = float(q_cmd_np[0])
        command_queue = torch.full((max_delay + 1, num_envs), initial_cmd, dtype=joint_pos.dtype, device=self.device)
        filtered_cmd = torch.full((num_envs,), initial_cmd, dtype=joint_pos.dtype, device=self.device)
        filtered_tau = torch.zeros((num_envs,), dtype=joint_pos.dtype, device=self.device)
        delay_index = torch.tensor(delay_steps, dtype=torch.long, device=self.device)

        q_out = torch.zeros((num_envs, sample_count), dtype=joint_pos.dtype, device=self.device)
        qd_out = torch.zeros_like(q_out)
        tau_out = torch.zeros_like(q_out)
        dt_values = np.diff(dataset.t)
        nominal_dt = float(np.median(dt_values)) if len(dt_values) else float(self.cfg.sim_dt)
        effort = torch.zeros((num_envs, joint_pos.shape[1]), dtype=joint_pos.dtype, device=self.device)

        # Single-joint replay must freeze every non-active DoF, not only the
        # policy-controlled 18 DoFs. The R5a USD contains two extra wrist/gripper
        # DoFs (R5a_joint7/8); leaving them free corrupts the fitted response of
        # wrist joints.
        all_joint_ids = torch.arange(joint_pos.shape[1], dtype=torch.long, device=self.device)
        inactive_joint_ids = all_joint_ids[all_joint_ids != int(active_joint_id[0])]
        inactive_pos = joint_pos[:, inactive_joint_ids].clone()
        inactive_vel = torch.zeros_like(inactive_pos)

        for sample_idx, raw_command in enumerate(q_cmd_np):
            active_q = self.robot.data.joint_pos[:num_envs, active_joint_id].squeeze(1)
            active_qd = self.robot.data.joint_vel[:num_envs, active_joint_id].squeeze(1)
            q_out[:, sample_idx] = active_q
            qd_out[:, sample_idx] = active_qd

            command_queue = torch.roll(command_queue, shifts=1, dims=0)
            command_queue[0, :] = float(raw_command)
            delayed_cmd = command_queue[delay_index, torch.arange(num_envs, device=self.device)]
            filtered_cmd = filter_alpha * delayed_cmd + (1.0 - filter_alpha) * filtered_cmd

            raw_tau = kp * (filtered_cmd - active_q) - kd * active_qd
            raw_tau = torch.clamp(raw_tau, min=-tau_limit, max=tau_limit)
            filtered_tau = tau_lag_alpha * raw_tau + (1.0 - tau_lag_alpha) * filtered_tau
            effort.zero_()
            effort[:, int(active_joint_id[0])] = filtered_tau

            self.robot.set_joint_effort_target(effort, env_ids=active_env_ids)
            step_dt = float(dt_values[sample_idx]) if sample_idx < len(dt_values) else nominal_dt
            substeps = max(1, int(round(step_dt / float(self.cfg.sim_dt))))
            for _ in range(substeps):
                self.robot.write_joint_state_to_sim(
                    inactive_pos,
                    inactive_vel,
                    joint_ids=inactive_joint_ids,
                    env_ids=active_env_ids,
                )
                self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=active_env_ids)
                self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=active_env_ids)
                self.robot.write_data_to_sim()
                self.sim.step()
                self.robot.update(float(self.cfg.sim_dt))

            self.robot.write_joint_state_to_sim(
                inactive_pos,
                inactive_vel,
                joint_ids=inactive_joint_ids,
                env_ids=active_env_ids,
            )
            self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=active_env_ids)
            self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=active_env_ids)
            self.robot.update(float(self.cfg.sim_dt))
            tau_out[:, sample_idx] = filtered_tau

        return BatchReplayResult(
            t=np.asarray(dataset.t, dtype=float).copy(),
            q=q_out.detach().cpu().numpy(),
            qd=qd_out.detach().cpu().numpy(),
            tau=tau_out.detach().cpu().numpy(),
        )

    def _initial_arm_position(self, dataset: SweepDataset) -> np.ndarray:
        measured = np.asarray(dataset.q[0, :6], dtype=float)
        if measured.shape == (6,) and np.all(np.isfinite(measured)):
            return measured.copy()
        return DEFAULT_ARM_POS.copy()

    def close(self) -> None:
        # Isaac Sim/Kit can block for a long time while tearing down plugins in
        # headless or driver-limited environments. The CLI writes outputs before
        # calling close(), so keep close() non-blocking.
        self.reset_scene()

    def reset_scene(self) -> None:
        # Drop scene handles so the next replay builds a fresh world while
        # preserving the already-launched Isaac app/runtime.
        scene_root = self._scene_root
        self.sim = None
        self.robot = None
        self._controlled_joint_idx = None
        self._arm_joint_idx = None
        self._gripper_joint_idx = None
        self._env_origins = None
        self._scene_root = None
        if scene_root is None:
            return
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is not None and stage.GetPrimAtPath(scene_root).IsValid():
                stage.RemovePrim(scene_root)
        except Exception:
            # Best effort only. _setup_sim uses a fresh namespace every time, so
            # stale prims cannot collide even if the stage refuses removal.
            pass


def _define_origins(torch, num_origins: int, spacing: float, device: str):
    env_origins = torch.zeros((num_origins, 3), device=device)
    num_cols = int(np.floor(np.sqrt(num_origins)))
    num_cols = max(1, num_cols)
    num_rows = int(np.ceil(num_origins / num_cols))
    xx, yy = torch.meshgrid(torch.arange(num_rows, device=device), torch.arange(num_cols, device=device), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2.0
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2.0
    return env_origins
