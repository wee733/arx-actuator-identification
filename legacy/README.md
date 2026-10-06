# Legacy single-joint PD identification (unmaintained)

This directory keeps the first identification pipeline for reference. It is
not maintained, not tested, and not installed with `arx-sysid`.

It fits one joint at a time with every other joint held fixed, searching kp,
kd, a command delay and optional filters by replaying candidates in a PhysX
(Isaac Lab 2.x) or full-robot MuJoCo scene. Its results became the baseline in
`config/baselines/pd_20260605.yaml`.

It was replaced by the coupled grey-box identification in `src/arx_sysid`
because:

- holding the other joints fixed hides the strong inertial coupling of the R5
  arm (a single-joint chirp swings neighbouring joints by 20-30 mrad);
- without a friction term the fit explains stiction as a long command delay
  and an under-damped loop, which rings in simulation but not on the robot;
- the PhysX backend does not run on Isaac Lab 3.0 / Newton, which the target
  policy training uses.

Paths inside these files refer to the original repository layout
(`arx_actuator_identification/`, `models/b2arx_mj/`) and to local robot assets
that are not distributed.
