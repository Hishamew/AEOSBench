dir_path = 'algo/benchmark'

att_maintain = dict(
    type='AttitudeMaintainBenchmark',
    dir_path=dir_path,
)
location_pointing = dict(
    type='LocationPointingBenchmark',
    dir_path=dir_path,
    initial_ephemeris_path='algo/data/initial_ephemeris.pth',
)

spacecraft_pointing = dict(
    type='SpacecraftPointingBenchmark',
    dir_path=dir_path,
)
