agent_torque_adam = dict(
    type='torch.optim.Adam',
    betas=(0.7, 0.95),
    lr=2e-3,
    scheduler=dict(
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
    ),
)
mlp_pid_adam = dict(
    type='torch.optim.Adam',
    betas=(0.7, 0.95),
    lr=5e-4,
    scheduler=dict(
        type='CosineAnnealingLR',
        T_max=400,
        eta_min=5e-5,
    ),
)
