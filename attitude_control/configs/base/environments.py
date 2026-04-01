att_maintain = dict(
    type='AttitudeMaintainEnvironment',
    num_envs=512,
    # train_dir='algo/benchmark/train',
    with_window=True,
)

location_pointing = dict(
    type='LocationPointingEnvironment',
    num_envs=512,
    train_dir='algo/benchmark/train',
    initial_ephemeris_path='algo/data/initial_ephemeris.pth',
)

spacecraft_pointing = dict(
    type='SpacecraftPointingEnvironment',
    num_envs=512,
    train_dir='algo/benchmark/train',
)
