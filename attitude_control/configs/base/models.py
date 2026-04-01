mlp_pid = dict(
    type='MLPConfiguredPIDController',
    hidden_dim=256,
    task_invariant=True,
)
torque_agent = dict(
    type='AttitudeControlModel',
    hidden_dim=256,
    time_invariant=True,
    deterministic=True,
)
learnable_mrp = dict(
    type='LMRP',
    num_sat=64,
)
torque_agent_frozen = dict(
    type='AttitudeControlModel',
    hidden_dim=256,
    time_invariant=True,
    deterministic=True,
    update_normalizer=False,
)

pid = dict(
    type='MRPFeedbackController',
    k=2.3,
    ki=0.000212,
    p=18.,
    dt=1.,
)
