from algo.configs.base.benchmarks import att_maintain as benchmark
from algo.configs.base.environments import att_maintain as environment
from algo.configs.base.models import pid as model
from algo.configs.base.optims import agent_torque_adam as optim

# model.update(wheels_aware=True)

# NOTE: In with window task, rollout length should be divisible by episode_lenth
# NOTE: if episode_length is changed, add same change to environment config
train = dict(
    loss=dict(
        attitude_loss=1.,
        battery_loss=0.1,
        nominal_loss=10.,
        motion_loss=1.,
    ),
    task_sampler=dict(
        sample_task=True,  # Using environment sampler to indicate whether
        sample_constellation=True,  # to train a generalized model
    ),
    monitor=[],
    rollout_length=180,
    episode_length=360,
    total_step=400 * 360,
    log_dir="./.test",
    gamma=1.,
    gamma_step=300,
    save_period=20 * 360,
    evaluate_initial_model=True,
)
