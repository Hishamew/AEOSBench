__all__ = [
    'InertiaTuple',
    'ReactionWheelConfig',
    'ReactionWheelConfigGroup',
    'ReactionWheel',
    'ReactionWheelGroup',
    'ReactionWheelGroups',
    'SatelliteConfig',
    'Constellation',
]
import dataclasses
import math
import random
from collections import UserList
from typing import Iterable, Literal, Self, TypedDict

import torch

from satsim.architecture import constants
from satsim.data.orbits import (OrbitalElement, OrbitalElements, OrbitDict,
                                elem2rv)
from satsim.simulation.gravity import Ephemeris
from satsim.simulation.reaction_wheels import ReactionWheelRegistry
from satsim.utils import LLA2PCPF, PCPF2LLA

InertiaTuple = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


class ReactionWheelConfig(TypedDict):
    power: float
    rw_speed_init: float


ReactionWheelConfigGroup = tuple[
    ReactionWheelConfig,
    ReactionWheelConfig,
    ReactionWheelConfig,
]


@dataclasses.dataclass
class ReactionWheel:
    type: str
    power: float  # watt
    rw_speed_init: float  # rad/s

    @classmethod
    def sample(cls) -> 'ReactionWheelGroup':
        type_ = random.choice(list(ReactionWheelRegistry.keys()))
        return tuple(
            cls(
                type_,
                round(random.uniform(5., 7.)),
                round(random.uniform(-100, 100)),
            ) for _ in range(3))

    @property
    def data(self) -> float:
        return self.power


ReactionWheelGroup = tuple[ReactionWheel, ReactionWheel, ReactionWheel]


class ReactionWheelGroups(UserList[ReactionWheelGroup]):

    @classmethod
    def from_dicts(cls, configs: Iterable[ReactionWheelConfigGroup]) -> Self:
        return cls(
            [tuple(ReactionWheel(**c) for c in config) for config in configs])

    @classmethod
    def sample(cls, n: int) -> Self:
        return cls([ReactionWheel.sample() for _ in range(n)])

    def to_dicts(self) -> list[ReactionWheelConfigGroup]:
        return [[dataclasses.asdict(rw) for rw in rw_group]
                for rw_group in self]

    def to_tensor(self) -> torch.Tensor:
        data = [[rw.data for rw in rw_group] for rw_group in self]
        return torch.tensor(data)


class SatelliteConfig(TypedDict):
    id: int
    mass: float
    inertia: InertiaTuple
    reaction_wheels: ReactionWheelConfigGroup


@dataclasses.dataclass
class Satellite:
    id: int
    mass: float
    inertia: InertiaTuple
    reaction_wheels: ReactionWheelGroup

    def to_dict(self) -> SatelliteConfig:
        return SatelliteConfig(
            id=self.id,
            mass=self.mass,
            inertia=self.inertia,
            reaction_wheels=[
                dataclasses.asdict(rw) for rw in self.reaction_wheels
            ],
        )

    @classmethod
    def from_dict(
        cls,
        config: SatelliteConfig,
    ) -> Self:
        reaction_wheels = tuple(
            ReactionWheel(**{
                'type': 'HoneywellHR12Small',
                **rw_config
            }) for rw_config in config['reaction_wheels'])
        return cls(
            id=config['id'],
            mass=config['mass'],
            inertia=config['inertia'],
            reaction_wheels=reaction_wheels,
        )

    @classmethod
    def sample(cls, id: int) -> Self:
        reaction_wheels = ReactionWheel.sample()

        mass = round(random.uniform(50, 200), 3)

        # According to the definition of inertia matrix
        # it must satisfy Triangle inequality constraint
        # i.e. Ixx + Iyy ≥ Izz, Iyy + Izz ≥ Ixx, Izz + Ixx ≥ Iyy
        Ixx = round(random.uniform(50, 200), 6)
        Iyy = round(random.uniform(50, 200), 6)

        # min_Izz = max(50, abs(Ixx - Iyy))
        # max_Izz = min(200, Ixx + Iyy)

        Izz = round(random.uniform(50, 200), 6)

        inertias = [Ixx, Iyy, Izz]
        random.shuffle(inertias)

        inertia = (inertias[0], 0, 0, 0, inertias[1], 0, 0, 0, inertias[2])

        return cls(
            id,
            mass,
            inertia,
            reaction_wheels,
        )

    def to_tensor(self) -> torch.Tensor:
        mass = torch.tensor([self.mass])
        inertia = torch.tensor(
            [self.inertia[0], self.inertia[4], self.inertia[8]])
        return torch.cat([mass, inertia], dim=-1)


class Constellation(UserList[Satellite]):

    @property
    def num_satellite(self) -> int:
        return len(self)

    def to_dicts(self) -> list[SatelliteConfig]:
        return [sat.to_dict() for sat in self]

    @classmethod
    def from_dicts(
        cls,
        configs: list[SatelliteConfig],
    ) -> Self:
        return cls([Satellite.from_dict(config) for config in configs])

    @classmethod
    def sample(cls, n: int) -> Self:
        satellites = [Satellite.sample(idx) for idx in range(n)]
        return cls(satellites)

    def get_mass(self) -> torch.Tensor:
        return torch.tensor([sat.mass for sat in self])

    def get_inertia(self) -> torch.Tensor:
        return torch.tensor([sat.inertia for sat in self]).view(-1, 3, 3)

    def to_tensor(self) -> torch.Tensor:
        return torch.stack([sat.to_tensor() for sat in self], dim=0)


Attitude = tuple[float, float, float]


class AttitudeMaintainTaskConfig(TypedDict):
    id: int
    start_time: int
    end_time: int
    attitude_RN: Attitude


@dataclasses.dataclass
class AttitudeMaintainTask:
    id: int
    attitude_RN: Attitude
    start_time: int
    end_time: int

    def to_dict(self) -> list[AttitudeMaintainTaskConfig]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(
        cls,
        task_dict: AttitudeMaintainTaskConfig,
    ) -> 'AttitudeMaintainTask':
        return cls(**task_dict)

    @classmethod
    def sample(
        cls,
        id: int,
    ) -> 'AttitudeMaintainTask':
        short_or_long = random.choice([0, 1])
        task_duration = random.randint(
            60, 80) if short_or_long else random.randint(10, 30)

        task_start_time = random.randint(100, 260 - task_duration)
        task_end_time = task_start_time + task_duration

        attitude_RN = torch.randn(3)
        attitude_RN_axis = attitude_RN / (attitude_RN.norm() + 1e-8)
        angle = random.uniform(torch.pi / 6, torch.pi)
        attitude_RN = attitude_RN_axis * math.tan(angle / 4)
        attitude_RN = attitude_RN.tolist()
        attitude_RN = [round(n, 6) for n in attitude_RN]

        return cls(
            id,
            attitude_RN,
            task_start_time,
            task_end_time,
        )


class AttitudeMaintainTaskset(UserList[AttitudeMaintainTask]):

    @classmethod
    def sample(cls, n: int) -> 'AttitudeMaintainTaskset':
        tasks = [AttitudeMaintainTask.sample(idx) for idx in range(n)]
        return cls(tasks)

    def to_dicts(self) -> list[AttitudeMaintainTaskConfig]:
        return [task.to_dict() for task in self]

    @classmethod
    def from_dicts(
        cls,
        task_dicts: list[AttitudeMaintainTaskConfig],
    ) -> 'AttitudeMaintainTaskset':
        tasks = [AttitudeMaintainTask.from_dict(d) for d in task_dicts]
        return cls(tasks)

    def get_target_attitudes(self) -> torch.Tensor:
        return torch.tensor([task.attitude_RN for task in self])

    def get_time_windows(self) -> torch.Tensor:
        return torch.tensor([[task.start_time, task.end_time]
                             for task in self])


class LocationPointingTaskConfig(TypedDict):
    id: int
    latitude: float
    longitude: float
    broken_wheels: Literal[0, 1, 2]


@dataclasses.dataclass
class LocationPointingTask:
    id: int
    latitude: float
    longitude: float
    broken_wheels: Literal[0, 1, 2]

    def to_dict(self) -> LocationPointingTaskConfig:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(
        cls,
        task_dict: LocationPointingTaskConfig,
    ) -> Self:
        return cls(**task_dict)


class LocationPointingTaskset(UserList[LocationPointingTask]):

    @classmethod
    def sample(
        cls,
        spacecraft_oes: OrbitalElements,
        initial_ephemeris: Ephemeris,
    ) -> Self:
        position_BP_N, _ = elem2rv(constants.MU_EARTH * 1e9, spacecraft_oes)
        direction_cosine_matrix_PN = initial_ephemeris[
            'direction_cosine_matrix_CN']
        position_BP_P = torch.einsum(
            '...ij,...j->...i ',
            direction_cosine_matrix_PN.to(position_BP_N),
            position_BP_N,
        )
        lla_spacecraft = PCPF2LLA(
            position_BP_P,
            constants.REQ_EARTH * 1e3,
            constants.REQ_EARTH * 1e3,
        )
        latitude_sat, longitude_sat, _ = lla_spacecraft.unbind(-1)

        # L stands for task point, O stands for Sub-Satellite point
        # For task to be seen, the difference made to longitude and latitude \delta
        # must following such restraint
        # arccos(cos^2(\delta)) < arccos(R/H) ->
        # cos^2(\delta) > R/H
        # where H is the height of satellite and R is radius of earth
        # with the most great R/H with 6.378 / 6.8
        # we have \delta < arccos(sqrt(R/H)) = 0.252, so we take 0.1 as example
        delta_limit = 0.2
        random_task_point_latitude = latitude_sat + torch.rand_like(
            latitude_sat) * delta_limit - delta_limit / 2
        random_task_point_longitude = longitude_sat + torch.rand_like(
            longitude_sat) * delta_limit - delta_limit / 2

        latitude = [round(la, 5) for la in random_task_point_latitude.tolist()]
        longitude = [
            round(lo, 5) for lo in random_task_point_longitude.tolist()
        ]
        broken_wheels = [
            random.choice([0, 1, 2]) for _ in range(len(latitude))
        ]

        return cls([
            LocationPointingTask(
                id=idx,
                latitude=latitude[idx],
                longitude=longitude[idx],
                broken_wheels=broken_wheels[idx],
            ) for idx in range(len(latitude))
        ])

    def to_dicts(self) -> list[LocationPointingTaskConfig]:
        return [task.to_dict() for task in self]

    @classmethod
    def from_dicts(
        cls,
        task_dicts: list[LocationPointingTaskConfig],
    ) -> Self:
        tasks = [LocationPointingTask.from_dict(d) for d in task_dicts]
        return cls(tasks)

    def get_task_coordinate(self) -> torch.Tensor:
        return LLA2PCPF(
            torch.tensor([t.latitude for t in self]),
            torch.tensor([t.longitude for t in self]),
            0.,
            constants.REQ_EARTH * 1e3,
            constants.REQ_EARTH * 1e3,
        )

    def get_reaction_wheel_availability(self) -> torch.Tensor:
        broken_mask = torch.nn.functional.one_hot(
            torch.tensor([t.broken_wheels for t in self]),
            num_classes=3,
        ).to(torch.bool)

        return ~broken_mask


class SpacecraftPointingTaskConfig(TypedDict):
    id: int
    chief_spacecraft_oe: OrbitDict


@dataclasses.dataclass
class SpacecraftPointingTask:
    id: int
    chief_spacecraft_oe: OrbitalElement

    def to_dict(self) -> SpacecraftPointingTaskConfig:
        chief_spacecraft_oe_dict = self.chief_spacecraft_oe.to_dict()
        return SpacecraftPointingTaskConfig(
            id=self.id,
            chief_spacecraft_oe=chief_spacecraft_oe_dict,
        )

    @classmethod
    def from_dicts(
        cls,
        task_dicts: SpacecraftPointingTaskConfig,
    ) -> 'SpacecraftPointingTask':
        return cls(
            id=task_dicts['id'],
            chief_spacecraft_oe=OrbitalElement.from_dict(
                task_dicts['chief_spacecraft_oe']),
        )

    @classmethod
    def sample(
        cls,
        id: int,
        deputy_spacecraft_oe: OrbitalElement,
    ) -> 'SpacecraftPointingTask':

        oe = deputy_spacecraft_oe
        ahead_or_behind = random.choice([-1, 1])
        delta_true_anomaly = ahead_or_behind * random.uniform(0.1, 0.2)

        true_anomaly = oe.true_anomaly + delta_true_anomaly

        chief_oe = OrbitalElement(
            oe.id,
            oe.semi_major_axis,
            oe.eccentricity,
            oe.inclination,
            oe.right_ascension_of_the_ascending_node,
            oe.argument_of_perigee,
            true_anomaly,
        )

        return cls(id, chief_oe)


class SpacecraftPointingTaskset(UserList[SpacecraftPointingTask]):

    @classmethod
    def sample(
        cls,
        spacecraft_oes: OrbitalElements,
    ) -> 'SpacecraftPointingTaskset':
        tasks = [
            SpacecraftPointingTask.sample(idx, oe)
            for idx, oe in enumerate(spacecraft_oes)
        ]
        return cls(tasks)

    def to_dicts(self) -> list[SpacecraftPointingTaskConfig]:
        return [task.to_dict() for task in self]

    @classmethod
    def from_dicts(
        cls,
        task_dicts: list[SpacecraftPointingTaskConfig],
    ) -> 'SpacecraftPointingTaskset':
        tasks = [SpacecraftPointingTask.from_dicts(d) for d in task_dicts]
        return cls(tasks)

    def get_orbital_elements(self) -> OrbitalElements:
        return OrbitalElements([task.chief_spacecraft_oe for task in self])
