from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arx_actuator_identification.actuator import JointActuatorParams, SimplePdActuator
from arx_actuator_identification.b2arx.config import B2arxMujocoConfig, load_b2arx_mujoco_config
from arx_actuator_identification.b2arx.joint_map import (
    CONTROLLED_JOINT_NAMES,
    DEFAULT_ARM_POS,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    Q_DOWN,
    arm_joint_name,
)
from arx_actuator_identification.backends import ReplayResult
from arx_actuator_identification.dataset import SweepDataset


@dataclass
class MujocoReplayOptions:
    use_q_down: bool = True
    fixed_base_height: float | None = None
    substep: bool = True


class B2arxMujocoReplayBackend:
    def __init__(self, cfg: B2arxMujocoConfig, options: MujocoReplayOptions | None = None):
        self.cfg = cfg
        self.options = MujocoReplayOptions() if options is None else options

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        options: MujocoReplayOptions | None = None,
    ) -> "B2arxMujocoReplayBackend":
        return cls(load_b2arx_mujoco_config(config_path), options=options)

    def replay(self, dataset: SweepDataset, params: JointActuatorParams) -> ReplayResult:
        dataset.validate()
        mujoco = _import_mujoco()
        model, data, qposadr, dofadr, gripper_qposadr, gripper_dofadr, act_map = self._setup_model(mujoco)
        active_joint = dataset.active_joint
        active_name = arm_joint_name(active_joint)
        active_controlled_idx = CONTROLLED_JOINT_NAMES.index(active_name)
        active_actuator_id = act_map[active_name]
        active_dof_id = int(dofadr[active_controlled_idx])

        controlled_initial = self._initial_controlled_positions(dataset)
        self._reset_state(
            mujoco,
            model,
            data,
            qposadr,
            dofadr,
            gripper_qposadr,
            gripper_dofadr,
            act_map,
            controlled_initial,
        )

        actuator = SimplePdActuator([params])
        actuator.reset(np.array([dataset.q_cmd[0, active_joint]], dtype=float))

        # 重力前馈：每帧用独立 gravity_data 在 qvel=0 下 mj_forward 取纯重力力矩，
        # 避免 data.qfrc_bias 混入速度相关(科氏/离心)项。gravity_scale=0 时跳过(省算)。
        gravity_scale = float(params.gravity_scale)
        gravity_data = mujoco.MjData(model) if gravity_scale != 0.0 else None

        q_values: list[float] = []
        qd_values: list[float] = []
        tau_values: list[float] = []
        dt_values = np.diff(dataset.t)
        nominal_dt = float(np.median(dt_values)) if len(dt_values) else float(model.opt.timestep)

        for sample_idx, command in enumerate(dataset.q_cmd[:, active_joint]):
            self._lock_base(data)
            self._lock_inactive_joints(data, qposadr, dofadr, controlled_initial, active_controlled_idx)
            self._lock_gripper_joints(data, gripper_qposadr, gripper_dofadr)

            q = float(data.qpos[int(qposadr[active_controlled_idx])])
            qd = float(data.qvel[active_dof_id])
            q_values.append(q)
            qd_values.append(qd)
            tau_g = 0.0
            if gravity_data is not None:
                tau_g = gravity_scale * self._gravity_torque(
                    mujoco, model, gravity_data, data, active_dof_id
                )
            tau = actuator.step_with_feedforward(
                np.array([command], dtype=float),
                np.array([q]),
                np.array([qd]),
                np.array([tau_g]),
            )[0]
            data.ctrl[active_actuator_id] = float(tau)
            self._set_gripper_actuator_targets(data, act_map)

            step_dt = float(dt_values[sample_idx]) if sample_idx < len(dt_values) else nominal_dt
            self._step_for_duration(
                mujoco,
                model,
                data,
                step_dt,
                lock_callback=lambda: self._lock_gripper_joints(data, gripper_qposadr, gripper_dofadr),
            )

            self._lock_base(data)
            self._lock_inactive_joints(data, qposadr, dofadr, controlled_initial, active_controlled_idx)
            self._lock_gripper_joints(data, gripper_qposadr, gripper_dofadr)
            tau_values.append(float(data.actuator_force[active_actuator_id]))

        return ReplayResult(
            t=dataset.t.copy(),
            q=np.asarray(q_values),
            qd=np.asarray(qd_values),
            tau=np.asarray(tau_values),
        )

    def _setup_model(self, mujoco):
        model = mujoco.MjModel.from_xml_path(str(self.cfg.xml_path))
        data = mujoco.MjData(model)
        model.opt.timestep = float(self.cfg.simulation_dt)

        joint_ids = []
        qposadr = []
        dofadr = []
        for joint_name in CONTROLLED_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"joint {joint_name} is missing from {self.cfg.xml_path}")
            joint_ids.append(joint_id)
            qposadr.append(int(model.jnt_qposadr[joint_id]))
            dofadr.append(int(model.jnt_dofadr[joint_id]))

        gripper_qposadr = []
        gripper_dofadr = []
        for joint_name in GRIPPER_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"joint {joint_name} is missing from {self.cfg.xml_path}")
            gripper_qposadr.append(int(model.jnt_qposadr[joint_id]))
            gripper_dofadr.append(int(model.jnt_dofadr[joint_id]))

        act_map: dict[str, int] = {}
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id][0])
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                act_map[joint_name] = actuator_id

        for idx, joint_name in enumerate(CONTROLLED_JOINT_NAMES):
            if joint_name not in act_map:
                raise ValueError(f"actuator for {joint_name} is missing from {self.cfg.xml_path}")
            actuator_id = act_map[joint_name]
            model.actuator_gainprm[actuator_id][0] = 1.0
            model.actuator_biasprm[actuator_id][:] = 0.0
            model.actuator_forcerange[actuator_id][0] = -float(self.cfg.effort_limits[idx])
            model.actuator_forcerange[actuator_id][1] = float(self.cfg.effort_limits[idx])
            model.actuator_forcelimited[actuator_id] = 1
            model.dof_armature[int(dofadr[idx])] = float(self.cfg.joint_armature)

        return (
            model,
            data,
            np.asarray(qposadr, dtype=np.int32),
            np.asarray(dofadr, dtype=np.int32),
            np.asarray(gripper_qposadr, dtype=np.int32),
            np.asarray(gripper_dofadr, dtype=np.int32),
            act_map,
        )

    def _initial_controlled_positions(self, dataset: SweepDataset) -> np.ndarray:
        arm_initial = DEFAULT_ARM_POS.copy()
        measured = np.asarray(dataset.q[0, :6], dtype=float)
        if measured.shape == (6,) and np.all(np.isfinite(measured)):
            arm_initial = measured.copy()
        arm_initial[dataset.active_joint] = float(dataset.q[0, dataset.active_joint])
        if self.options.use_q_down:
            return np.concatenate([Q_DOWN, arm_initial])
        controlled = self.cfg.default_angles.copy()
        controlled[12:] = arm_initial
        return controlled

    def _reset_state(
        self,
        mujoco,
        model,
        data,
        qposadr,
        dofadr,
        gripper_qposadr,
        gripper_dofadr,
        act_map: dict[str, int],
        controlled_initial: np.ndarray,
    ) -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = self._base_height()
        data.qpos[3] = 1.0
        data.qpos[4:7] = 0.0
        data.qvel[0:6] = 0.0
        for idx, value in enumerate(controlled_initial):
            data.qpos[int(qposadr[idx])] = float(value)
            data.qvel[int(dofadr[idx])] = 0.0
        data.ctrl[:] = 0.0
        self._lock_gripper_joints(data, gripper_qposadr, gripper_dofadr)
        self._set_gripper_actuator_targets(data, act_map)
        mujoco.mj_forward(model, data)

    def _base_height(self) -> float:
        if self.options.fixed_base_height is not None:
            return float(self.options.fixed_base_height)
        return float(self.cfg.init_base_height)

    def _lock_base(self, data) -> None:
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = self._base_height()
        data.qpos[3] = 1.0
        data.qpos[4:7] = 0.0
        data.qvel[0:6] = 0.0

    def _lock_inactive_joints(
        self,
        data,
        qposadr: np.ndarray,
        dofadr: np.ndarray,
        controlled_initial: np.ndarray,
        active_controlled_idx: int,
    ) -> None:
        for idx, value in enumerate(controlled_initial):
            if idx == active_controlled_idx:
                continue
            data.qpos[int(qposadr[idx])] = float(value)
            data.qvel[int(dofadr[idx])] = 0.0

    def _lock_gripper_joints(
        self,
        data,
        gripper_qposadr: np.ndarray,
        gripper_dofadr: np.ndarray,
    ) -> None:
        for idx, value in enumerate(GRIPPER_CLOSED_POS):
            data.qpos[int(gripper_qposadr[idx])] = float(value)
            data.qvel[int(gripper_dofadr[idx])] = 0.0

    def _set_gripper_actuator_targets(self, data, act_map: dict[str, int]) -> None:
        for joint_name, value in zip(GRIPPER_JOINT_NAMES, GRIPPER_CLOSED_POS):
            actuator_id = act_map.get(joint_name)
            if actuator_id is not None:
                data.ctrl[actuator_id] = float(value)

    def _step_for_duration(self, mujoco, model, data, duration: float, lock_callback=None) -> None:
        if duration <= 0.0:
            if lock_callback is not None:
                lock_callback()
            mujoco.mj_step(model, data)
            if lock_callback is not None:
                lock_callback()
            return
        steps = max(1, int(round(duration / float(model.opt.timestep))))
        for _ in range(steps):
            if lock_callback is not None:
                lock_callback()
            mujoco.mj_step(model, data)
        if lock_callback is not None:
            lock_callback()

    def _gravity_torque(self, mujoco, model, gravity_data, data, dof_id: int) -> float:
        # 纯重力力矩 G(q)：复制当前位形，速度清零再 mj_forward，
        # 这样 qfrc_bias 只含重力(无科氏/离心)。返回该 dof 上的广义重力力。
        # 约定符号：qfrc_bias = G(q)+C(q,qd)，qvel=0 后即 G(q)；运动方程
        # M*qddot + qfrc_bias = qfrc_applied，故抵消重力所需的前馈 tau_ff = +G(q)。
        gravity_data.qpos[:] = data.qpos
        gravity_data.qvel[:] = 0.0
        mujoco.mj_forward(model, gravity_data)
        return float(gravity_data.qfrc_bias[dof_id])


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo Python package is required for the mujoco backend") from exc
    return mujoco
