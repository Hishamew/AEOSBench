import dataclasses
from typing import NamedTuple, TypedDict

Coordinate = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
# position in meter, lat and lon in rad
# all in reference to ECEF


class SatelliteData(TypedDict):
    id: int
    position: list[Coordinate] | Coordinate
    orientation: Quaternion
    taskId: int


class TaskData(TypedDict):
    id: int
    position: Coordinate
    lat: float
    lon: float
    completion: float  # [0,100]
    satId: list[int]


class SocketDict(TypedDict):
    time_step: int
    task_datas: list[TaskData]
    satellites_data: list[SatelliteData]


class Socket(TypedDict):
    time_step: int
    task_datas: list[TaskData]
    satellites_data: list[SatelliteData]


class SimulationInitializationConfig(TypedDict):
    ID: str
    satellites: list[SatelliteData]
    tasks: list[TaskData]
