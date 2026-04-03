environment = dict(
    num_envs=512,
)

model = dict(
    hidden_dim=256,
    with_integral_limit=True,
)
episode_num = 1000
warmup_episode = 100
optimizer = dict(
    type='torch.optim.Adam',
    lr=1e-4,
    betas=(0.7, 0.95),
)
cos_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=episode_num,
    eta_min=5e-5,
)
step_scheduler = dict(
    type='SquentialLR',
    schedulers=[
        dict(
            type='LinearLR',
            start_factor=0.0001,
            end_factor=1.0,
            total_iters=warmup_episode - 1
        ),
        dict(
            type='MultiStepLR',
            milestones=[300, 600],
            gamma=0.1,
        )
    ],
    milestones=[warmup_episode],
)

runner = dict(
    model=model,
    callbacks=[
        dict(type='LoggerRegistry.GitLogger'),
        dict(type='LoggerRegistry.TensorboardLogger'),
        dict(
            type='ResetCallback',
            sample_task_interval=1,
            sample_constellation_interval=5,
        ),
        dict(
            type='LossRegistry.LossCollector',
            losses=dict(
                attitude_loss=dict(type='AttitudeLoss', weight=1.0),
                battery_loss=dict(type='BatteryLoss', weight=0.01),
                motion_loss=dict(type='MotionLoss', weight=1.0, threshold=0.1)
            )
        ),
        dict(type='OptimizeCallback'),
        dict(
            type='LRSchedulerCallback',
            lr_scheduler_config=step_scheduler,
        ),
        dict(
            type='LoggerRegistry.CheckpointLogger',
            interval=20,
        ),
        dict(type='LoggerRegistry.LogCallback', file_logging=True),
        dict(type='MonitorRegistry.BatteryMonitor'),
        dict(type='MonitorRegistry.AttitudeErrorsMonitor'),
        dict(type='MonitorRegistry.TorqueMonitor'),
    ],
    environment=environment,
    optimizer=optimizer,
    episode_length=180,
    total_episode=episode_num,
    full_grad=False,
    progress_bar=False,
)
