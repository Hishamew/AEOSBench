__all__ = [
    'build_reaction_wheel',
]

from typing import Callable, Self, TypeAlias

from satsim.architecture import constants
from satsim.simulation.reaction_wheels import (
    HoneywellHR12Large,
    HoneywellHR12Medium,
    HoneywellHR12Small,
    HoneywellHR16Large,
    HoneywellHR16Medium,
    HoneywellHR16Small,
    ReactionWheel,
)

from ...data import ReactionWheel as RWConfig


class HoneywellHR14Small(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            mass=7.5,
            max_momentum=25.,
            max_torque=0.2,
            max_angular_velocity=6000. * constants.RPM,
            **kwargs,
        )


class HoneywellHR14Medium(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            mass=8.5,
            max_momentum=50.,
            max_torque=0.2,
            max_angular_velocity=6000. * constants.RPM,
            **kwargs,
        )


class HoneywellHR14Large(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            mass=10.6,
            max_momentum=75.,
            max_torque=0.2,
            max_angular_velocity=6000. * constants.RPM,
            **kwargs,
        )


HoneywellHR12: TypeAlias = HoneywellHR12Large | HoneywellHR12Medium | HoneywellHR12Small
HoneywellHR14: TypeAlias = HoneywellHR14Large | HoneywellHR14Medium | HoneywellHR14Small
HoneywellHR16: TypeAlias = HoneywellHR16Large | HoneywellHR16Medium | HoneywellHR16Small


def build(
    build_cls: type[ReactionWheel],
    config: RWConfig,
) -> ReactionWheel:
    return build_cls.build(
        mech_to_elec_efficiency=config.efficiency,
        base_power=config.power,
        angular_velocity_init=config.rw_speed_init * constants.RPM,
    )


def honeywell_hr12_builder(config: RWConfig) -> type[HoneywellHR12]:
    mapping: dict[int, type[HoneywellHR12]] = {
        12: HoneywellHR12Small,
        25: HoneywellHR12Medium,
        50: HoneywellHR12Large
    }
    build_cls = mapping[int(config.max_momentum)]
    return build_cls


def honeywell_hr14_builder(config: RWConfig) -> type[HoneywellHR14]:
    mapping: dict[int, type[HoneywellHR14]] = {
        25: HoneywellHR14Small,
        50: HoneywellHR14Medium,
        75: HoneywellHR14Large
    }
    build_cls = mapping[int(config.max_momentum)]
    return build_cls


def honeywell_hr16_builder(config: RWConfig) -> type[HoneywellHR16]:
    mapping: dict[int, type[HoneywellHR16]] = {
        50: HoneywellHR16Small,
        75: HoneywellHR16Medium,
        100: HoneywellHR16Large
    }
    build_cls = mapping[int(config.max_momentum)]
    return build_cls


builder_mapping: dict[str, Callable[[RWConfig], ReactionWheel]] = {
    'Honeywell_HR12': honeywell_hr12_builder,
    'Honeywell_HR14': honeywell_hr14_builder,
    'Honeywell_HR16': honeywell_hr16_builder,
}


def build_reaction_wheel(config: RWConfig) -> ReactionWheel:
    builder = builder_mapping[config.rw_type]
    build_cls = builder(config)
    return build(build_cls, config)
