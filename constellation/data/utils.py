__all__ = [
    'concat_constellations',
    'concat_tasksets',
]
import dataclasses
from dataclasses import fields

from .constellations import Constellation, Satellite
from .orbits import Orbit
from .tasksets import Task, TaskSet


def concat_constellations(
    constellations: list[Constellation]
) -> Constellation:
    """Concatenate multiple constellations into one constellation."""
    satellites = []
    for constellation in constellations:
        satellites.extend(constellation.sort())

    # NOTE: use dataclasses.asdict will recurrently convert dataclass instances to dict
    # which is not what we want here
    reid_constellation = Constellation()
    satellites_fields = fields(Satellite)
    for idx, sat in enumerate(satellites):
        sat_dict = dict()
        for field in satellites_fields:
            if 'id' in field.name:
                sat_dict[field.name] = idx
                continue
            if field.name == 'orbit':
                new_orbit = dataclasses.asdict(getattr(sat, field.name))
                new_orbit['id_'] = idx
                sat_dict[field.name] = Orbit(**new_orbit)
                continue

            sat_dict[field.name] = getattr(sat, field.name)
        satellite = Satellite(**sat_dict)
        reid_constellation[satellite.id_] = satellite

    return reid_constellation


def concat_tasksets(tasksets: list[TaskSet]) -> TaskSet:
    """Concatenate multiple tasksets into one taskset."""
    tasks = []
    for taskset in tasksets:
        tasks.extend(taskset)

    reid_tasks = TaskSet()
    for idx, task in enumerate(tasks):
        task_dict = dataclasses.asdict(task)
        task_dict['id_'] = idx
        reid_tasks.append(Task(**task_dict))

    return reid_tasks
