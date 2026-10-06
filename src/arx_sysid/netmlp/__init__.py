"""Experimental per-joint neural actuator (Isaac Lab ActuatorNetMLP contract).

Without torque or current labels it is trained on inverse-dynamics estimates
and position rollouts; prefer the grey-box identification unless current data
is available.
"""
