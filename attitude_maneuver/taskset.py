import dataclasses
import math
import random
from collections import UserList
from typing import Literal, Self, TypedDict

import torch
from satsim.architecture import constants
from satsim.simulation.gravity import Ephemeris
from satsim.utils import LLA2PCPF, PCPF2LLA

from constellation.data import (
    Constellation,
    Coordinate,
    SensorType,
    Task,
    TaskSet,
)

DELTA_LIMIT = 12  # degree


@dataclasses.dataclass(frozen=True)
class LocationPointingTask(Task):

    @classmethod
    def sample(cls, id_: int, sc_lat: float, sc_lon: float) -> Self:
        latitude = sc_lat + random.random() * DELTA_LIMIT - DELTA_LIMIT / 2
        longitude = sc_lon + random.random() * DELTA_LIMIT - DELTA_LIMIT / 2
        return cls(
            id_,
            0,
            360,
            360,
            Coordinate(
                latitude,
                longitude,
            ),
            SensorType.VISIBLE,
        )


class LocationPointingTaskset(TaskSet):

    @classmethod
    def sample(
        cls,
        constellation: Constellation,
        initial_ephemeris: Ephemeris,
    ) -> Self:
        rs = []
        for sat in constellation.sort():
            r, _ = sat.rv
            r = torch.from_numpy(r)
            rs.append(r)
        position_BP_N = torch.stack(rs)

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

        return cls([
            LocationPointingTask.sample(i, lat, lon) for i, (lat, lon) in
            enumerate(zip(latitude_sat.tolist(), longitude_sat.tolist()))
        ])
