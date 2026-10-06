# arx-sysid

Real-to-sim actuator identification for the ARX R5 arm, aimed at policies
trained in Isaac Lab 3.0 on Newton / MuJoCo-Warp. The repository covers the
whole loop: collect excitation data on the robot, identify actuator
parameters for the simulator, and check them against held-out trajectories in
the same physics the policy will train in.

The arm is position-controlled through the ARX SDK; the robot reports joint
position, a motor-side velocity and motor current, but no joint torque. The
identified parameters are therefore *simulation-equivalent*: they make the
simulated arm follow the same commands the same way, given the inertia in the
simulation model. See [What the parameters mean](#what-the-parameters-mean).

## Layout

| Path | Contents |
| --- | --- |
| `tools/collect/` | C++ collector built against the ARX SDK; runs on the robot computer |
| `src/arx_sysid/` | Python package: data loading, plant construction, identification, replay |
| `src/arx_sysid/netmlp/` | Experimental per-joint neural actuator (Isaac Lab `ActuatorNetMLP` contract) |
| `config/` | Identification configs and the PD baseline being compared against |
| `models/b2arx.xml` | MJCF of the B2 + R5a robot; only the arm's kinematics and inertia are used |
| `scripts/run.sh` | Runs the CLI from a checkout with a chosen interpreter |
| `legacy/` | Original single-joint PD identification, kept for reference only |

Captures go in `data/<RUN_ID>/csv/`, results in `runs/`; neither is tracked.

## 1. Collect on the robot

The collector reproduces the deployment command path: a target written at the
policy rate (50 Hz), an exponential moving average stepped at 1 kHz, and a
`setJointPositions` call on every millisecond. It logs, at 100 Hz, the raw and
smoothed commands, joint positions, SDK velocities and motor currents.

```bash
cmake -S tools/collect -B build/collect -DARX_SDK_ROOT=/path/to/arx_r5_controller
cmake --build build/collect -j

ARX=./build/collect/arx_collect
URDF=/path/to/arx_r5_controller/X5liteaa0.urdf
OUT=data/$(date +%Y%m%d_%H%M%S)/csv && mkdir -p "$OUT"

# One chirp and one step capture per joint (zero-based joint index in file names).
for j in 0 1 2 3 4 5; do
  $ARX --urdf $URDF --can can1 --out $OUT/joint${j}_chirp.csv --profile chirp --joint $j --amp 0.03
  $ARX --urdf $URDF --can can1 --out $OUT/joint${j}_step.csv  --profile step  --joint $j --amp 0.045
done

# Static poses for current-to-torque calibration (keep the base level).
# Example poses only: check them for self-collision on your mounting first.
$ARX --urdf $URDF --can can1 --out $OUT/static.csv --profile static --static-move 6 \
  --static-poses "0,0.6,0.6,0,0,0;0,0.6,1.4,0.6,0,0;0,0.6,2,0.9,0,0;0,1.2,1.6,0,0,0;0,1.8,1.2,-0.9,-1,0;0,1.2,1.2,0.9,-1,0;0,1,0.8,0,0,0"
```

`--ema-alpha` must equal the deployed EMA coefficient (default 0.05 at 1 kHz,
about 20 ms). Every run starts by ramping to the start pose
(`--start-pose`, default `0,1,0.8,0,0,0`), aborts on a joint limit or a
tracking error above `--max-track-err`, and returns to the start pose on exit
or Ctrl-C. Each CSV gets a `.meta.yaml` with the full configuration.

## 2. Identify

Requires Python 3.12 with Newton 1.6, Warp 1.17, MuJoCo 3.12 and PyTorch 2.11
(an Isaac Lab 3.0 environment already has them). Edit `config/greybox.yaml` to
list your capture directories, then:

```bash
ARX_SYSID_PYTHON=/path/to/python bash scripts/run.sh identify --run-dir runs/my_fit
```

This takes about 25 minutes on CPU (GPUs are hidden from the workers) and
writes `params.yaml`, `report.yaml`, response plots for the step test and the
chirp validation blocks, and per-sample traces.

### Method

All six joints are simulated together in a fixed-base, gravity-neutral arm
with the source inertia and zero passive damping, the same plant the policy
sees. Every 5 ms tick applies the Isaac Lab IdealPD law

    tau = clip(kp * (q_cmd(t - delay) + bias - q) - kd * qd, +-effort)

with the command delay in physics ticks, plus MuJoCo dry friction
(`frictionloss`). All six encoders are scored, so the inertial coupling
between joints constrains the fit.

Two details decide most of the accuracy:

- **Friction stiffness.** MuJoCo enforces dry friction as a soft constraint.
  With its default `solimp` d0 = 0.9 a loaded joint keeps creeping towards
  its target, where the real gearbox stops dead. d0 is identified per joint
  (`friction_impedance`); the data pushes it to 0.987–0.9997, i.e. close to
  hard Coulomb friction. It is a per-dof MuJoCo parameter, so it carries over
  to MJWarp on the GPU.
- **Target bias.** Each joint holds a constant few-mrad offset from its
  command regardless of the step direction (encoder zero or residual gravity
  compensation). It is fitted as an offset added to the target.

Armature is not fitted: position data cannot separate it from kp, kd and
friction (see below).

Gradients come from a differentiable reduced model (`greybox.py`): first-order
inertia about the capture pose, the matching Coriolis terms, and MuJoCo's
frictionloss solve with the identified impedance. It matches the official
Newton/MuJoCo replay to within 0.1 mrad, with default and stiff friction.
Every reported score comes from the official replay (`replay.py`).

Data use is fixed in advance: the first 70 % of each chirp fits parameters,
the last 30 % selects among model structures (`pd`, `pd_friction`,
`pd_stiction`, `pd_stiction_bias`; the simplest within 5 % of the best
all-joint RMSE wins), and step captures are replayed once at the end as the
test. A fit is reported as accepted only when every step trajectory's
active-joint RMSE is within 5 % of the baseline's.

### Using the result in Isaac Lab / Newton

`params.yaml` lists, per joint, the values for `DelayedPDActuatorCfg`
(`stiffness`, `damping`, `friction`, `armature`, `effort_limit_sim`,
`min_delay` = `max_delay` in physics ticks) and two values that config class
does not carry:

- `mjc_solimpfriction_d0`: set the joint's `mjc:solimpfriction` first element
  in the USD, or `model.mujoco.solimpfriction[dof, 0]` followed by
  `solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)`;
- `target_bias_rad`: add it to the joint position target.

Keep the simulation as identified: Newton with MuJoCo-Warp, `implicitfast`,
a 5 ms step with one substep, explicit PD (not `ImplicitActuator`), the
post-EMA command as the target, zero joint damping and armature, and gravity
either disabled for the arm or compensated with the same model.

## 3. Calibrate current to torque (optional)

Position data cannot separate a gain from the inertia it drives (see below).
Current can: at rest the motor holds gravity, which the model's masses and
centres of mass predict well. Fit a torque constant and offset per joint from
the static capture:

```bash
bash scripts/run.sh calibrate-current --capture data/<RUN_ID>/csv/static.csv --output runs/torque_constants.yaml
```

With the example poses the model predicts gravity-load spans of about 14, 7.6,
2.5 and 0.65 N·m on joints 2–5. Joint 1 is vertical and joint 6 rolls about
the tool axis, so gravity never loads them and their constants stay
unresolved; joint 5's span is small, so expect a loose fit there.

## What the parameters mean

For one joint, position follows

    acceleration = (kp/m) * (target - q) - (kd/m) * velocity - friction/m

so encoder data only determines kp, kd and friction *relative to* the joint's
inertia m. If the simulation uses the same inertia as the identification, the
ratios are right and the simulated arm moves like the real one, which is what
real-to-sim needs. The absolute values matter when forces that do not go
through the PD act on the arm: payloads, contacts, base accelerations. In
particular the model has no rotor inertia, so small wrist joints come out with
small gains. Supplying a known `armature` (rotor inertia times gear ratio
squared) and re-identifying, or calibrating current as above, fixes the scale.

Isaac Lab on Newton treats joint friction as a soft constraint: below the
friction limit a joint still creeps slowly, where a real gearbox sticks. A
policy simulator that needs true stiction must apply it inside the actuator.

## NetMLP (experimental)

`netmlp-train` / `netmlp-validate` train one MLP per joint with the Isaac Lab
`ActuatorNetMLP` input contract and export `.pt` / `.pt2`. Without torque
labels it learns from inverse-dynamics estimates and position rollouts, which
inherit the model inertia error and amplify encoder noise. Prefer the grey-box
fit unless current-based torque labels are available.

## Data format

`joint{j}_{profile}.csv` (`j` zero-based) with columns `t`, `active_joint`,
`q_cmd_raw0..5`, `q_cmd_smooth0..5`, `q_real0..5`, `qd_real0..5` and, from
schema v2, `cur0..5`; static captures add `pose_index`, `recording`. The SDK
velocity is motor-side: about 4.5x joint velocity on joints 1–3 and 1.5x on
4–6, so the identification differentiates position instead. Joints in configs
and parameter files are one-based (`R5a_joint1..6`).
