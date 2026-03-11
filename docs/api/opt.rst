Optimization
============

.. automodule:: exp_mpc.stewart_min.opt
   :no-members:

.. currentmodule:: exp_mpc.stewart_min.opt

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   Weights
   ExpWeights
   CostTerms
   TrainState

Functions
---------

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   acc_cost_arr
   omega_cost_arr
   leg_boundary_cost_arr
   joint_angles
   joint_angle_boundary_cost_arr
   roll_boundary_cost_arr
   pitch_boundary_cost_arr
   yaw_boundary_cost_arr
   yaw_dot_boundary_cost_arr
   control_cost_arr
   cost_flat_jax
   cost_and_grad_flat_jax
   lbfgs_cost
   train_step_with_cost_jax
   train_step_with_cost
