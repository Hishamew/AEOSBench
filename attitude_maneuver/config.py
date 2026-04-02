environment = dict(
    num_envs=128,
)

model = dict(
    hidden_dim=256,
)

optimizer = dict(
    type='torch.optim.Adam',
    lr=1e-3,
    betas=(0.7, 0.95),
)
lr_scheduler = dict(
    type='SequentialLR',
    schedulers=[
        dict(
            type='LinearLR',
            start_factor=0.1,
            end_factor=1.0,
            total_iters=10,
        ),
        dict(
            type='MultiStepLR',
            milestones=[40, 80],
            gamma=0.1,
        ),
    ],
    milestones=[10],
)

runner = dict(
    model=model,
    callbacks=[
        dict(type='LoggerRegistry.GitLogger'),
        dict(type='LoggerRegistry.TensorboardCallback'),
        dict(
            type='LossRegistry.LossCollector',
            losses=dict(
                attitude_loss=dict(type='AttitudeLoss', weight=1.0),
                battery_loss=dict(type='BatteryLoss', weight=1.0),
            )
        ),
        dict(type='OptimizeCallback'),
        dict(
            type='LRSchedulerCallback',
            lr_scheduler_config=lr_scheduler,
        ),
        dict(
            type='LoggerRegistry.CheckpointCallback',
            interval=20,
        ),
        dict(type='MonitorRegistry.BatteryMonitor'),
        dict(type='MonitorRegistry.AttitudeErrorsMonitor'),
        dict(type='MonitorRegistry.TorqueMonitor'),
    ],
    environment=environment,
    optimizer=optimizer,
    episode_length=180,
    total_episode=100,
    full_grad=False,
)
