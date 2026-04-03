import torch
from attitude_maneuver.constants import INITIAL_EPHEMERIS_PATH
from attitude_maneuver.environment import AttitudeControlConstellation
from satsim.architecture import Timer

from constellation.constants import TASKSETS_ROOT, TIMESTAMP
from constellation.data import Constellation, TaskSet

TASKSET_PATH = TASKSETS_ROOT / 'mrp.json'
TASKSET = TaskSet.load(str(TASKSET_PATH))

if __name__ == '__main__':
    constellation = Constellation.sample_mrp()
    timer = Timer(1.)
    simulator = AttitudeControlConstellation(
        timer,
        constellation,
        TIMESTAMP,
        TASKSET,
    )
    timer.reset()
    ephemeris = simulator.get_earth_ephemeris(None)
    torch.save(ephemeris, str(INITIAL_EPHEMERIS_PATH))
