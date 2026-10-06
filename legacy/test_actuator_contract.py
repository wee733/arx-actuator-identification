import numpy as np

from arx_actuator_identification.actuator import JointActuatorParams, SimplePdActuator


def test_delay_preserves_initial_target_and_total_effort_is_clamped():
    actuator = SimplePdActuator([JointActuatorParams(
        kp=4, kd=1, delay_steps=2, filter_alpha=1, tau_limit=3,
    )])
    actuator.reset(np.array([0.0]))
    outputs = [actuator.step_with_feedforward(np.array([command]), np.array([0.0]),
                np.array([0.0]), np.array([1.0]))[0] for command in [1, 2, 3]]
    np.testing.assert_allclose(outputs, [1, 1, 3])


def test_velocity_feedback_opposes_motion():
    actuator = SimplePdActuator([JointActuatorParams(
        kp=10, kd=2, delay_steps=0, filter_alpha=1, tau_limit=20,
    )])
    np.testing.assert_allclose(actuator.step(np.array([0.0]), np.array([0.0]), np.array([3.0])), [-6])
