environment = dict(
    num_envs=512,
)

model = dict(
    hidden_dim=256,
    with_integral_limit=False,
)

runner = dict(
    model=model,
    callbacks=[
        dict(type='LoggerRegistry.GitLogger'),
        dict(
            type='ResetCallback',
            sample_task_interval=1,
            sample_constellation_interval=5,
        ),
        dict(type='MonitorRegistry.BatteryMonitor'),
        dict(type='MonitorRegistry.AttitudeErrorsMonitor'),
        dict(type='MonitorRegistry.TorqueMonitor'),
    ],
    environment=environment,
    episode_length=180,
    total_episode=1,
    full_grad=False,
    progress_bar=False,
)
