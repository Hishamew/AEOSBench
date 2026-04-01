__all__ = [
    'BaseConstellationStateDict',
    'BaseConstellation',
    'DynamicObservation',
]
import random
from typing import NotRequired, TypedDict, TypeVar, cast

import torch

from satsim.architecture import Module, Timer, constants
from satsim.attitude_guidance import GuidanceOutput
from satsim.simulation.base import BatteryStateDict
from satsim.simulation.gravity import (Ephemeris, GravityField,
                                       PointMassGravityBody, SpiceInterface)
from satsim.simulation.power import SimpleBattery
from satsim.simulation.reaction_wheels import (HoneywellHR12Small,
                                               ReactionWheelRegistry,
                                               ReactionWheels, concat)
from satsim.simulation.spacecraft import (IntegrateMethod, Spacecraft,
                                          SpacecraftStateDict,
                                          SpacecraftStateOutput)
from satsim.utils import move_to

from ..data import Constellation, InertiaTuple, ReactionWheelGroups
from .pertubation import (GravityGradientEffector, NoisePool,
                          PerturbationForceTorque, SensorErrorModel)


class DynamicObservation(TypedDict):
    reaction_wheels_speed: torch.Tensor
    spacecraft_angular_velocity: torch.Tensor
    spacecraft_attitude: torch.Tensor
    battery_percentage: torch.Tensor
    current_time: float
    attitude_guide_info: GuidanceOutput


def attitude_init(position_BP_N: torch.Tensor):
    position_PB_N = -position_BP_N
    position_PB_N_unit = torch.nn.functional.normalize(position_PB_N, dim=-1)
    implied_sensor_direction = torch.tensor([0, 0, 1.])

    cos_angle = torch.einsum(
        '...i, i -> ...',
        position_PB_N_unit,
        implied_sensor_direction,
    ).unsqueeze(-1)
    angle = torch.acos(cos_angle)

    axis = implied_sensor_direction.unsqueeze(0).cross(position_PB_N_unit,
                                                       dim=-1)
    axis = torch.nn.functional.normalize(axis, dim=-1)

    attitude_BN = axis * torch.tan(angle / 4.)
    return attitude_BN


class BaseConstellationStateDict(TypedDict):
    _battery: BatteryStateDict
    _spacecraft: SpacecraftStateDict
    _attitude_error_model: NotRequired[SensorErrorModel]
    _transition_error_model: NotRequired[SensorErrorModel]


T = TypeVar(
    'T',
    bound=BaseConstellationStateDict,
)


class BaseConstellation(Module[T]):

    def __init__(
        self,
        timer: Timer,
        constellation: Constellation,
        position_BP_N_init: torch.Tensor,
        velocity_BP_N_init: torch.Tensor,
        attitude_BN_init: torch.Tensor | None = None,
        angular_velocity_BN_B_init: torch.Tensor | None = None,
        use_noise: bool = True,
        use_perturbation: bool = True,
        use_gravity_gradient: bool = True,
        noise_pool: NoisePool | None = None,
        integrate_method: IntegrateMethod = 'RK',
    ) -> None:
        super().__init__(timer=timer)
        self._n = constellation.num_satellite
        if attitude_BN_init is None:
            attitude_BN_init = attitude_init(position_BP_N_init)
        self._setup_spacecraft(
            constellation,
            integrate_method,
            position_BP_N_init,
            velocity_BP_N_init,
            attitude_BN_init,
            angular_velocity_BN_B_init,
        )

        self._battery = SimpleBattery(
            timer=self._timer,
            storage_capacity=torch.full([self._n], 20000),
            stored_charge_percentage_init=torch.full([self._n], 0.8),
        )

        if use_gravity_gradient:
            gravity_gradient = GravityGradientEffector(timer=self._timer, )
            self._spacecraft.state_effectors.update(
                dict(_gravity_gradient=gravity_gradient))

        self._use_noise = use_noise
        if use_noise:
            attitude_error_factor = 1 / 360 * torch.pi / 180  # 1/360 degree
            angular_velocity_error_factor = 0.05 * torch.pi / 180.0  # 1/20 degree/s
            attitude_error_bound = 1e-4
            angular_velocity_error_bound = 1e-3
            process_noise_matrix = torch.diag_embed(
                torch.tensor([
                    attitude_error_factor,
                    attitude_error_factor,
                    attitude_error_factor,
                    angular_velocity_error_factor,
                    angular_velocity_error_factor,
                    angular_velocity_error_factor,
                ]))
            walkbound = torch.tensor([
                attitude_error_bound,
                attitude_error_bound,
                attitude_error_bound,
                angular_velocity_error_bound,
                angular_velocity_error_bound,
                angular_velocity_error_bound,
            ])

            attitude_noise_pool = None if noise_pool is None else noise_pool[
                'attitude_error']
            self._attitude_error_model = SensorErrorModel(
                torch.ones(self.num_satellite, dtype=torch.bool),
                process_noise_matrix,
                walkbound,
                seed=random.randint(0, 100000),
                recorded_noise=attitude_noise_pool,
                timer=self._timer,
            )

            position_error_factor = 5.0  # 5 meters
            velocity_error_factor = 0.035  # 0.035 m/s
            process_noise_matrix = torch.diag_embed(
                torch.tensor([
                    position_error_factor,
                    position_error_factor,
                    position_error_factor,
                    velocity_error_factor,
                    velocity_error_factor,
                    velocity_error_factor,
                ]))
            walkbound = torch.tensor([10., 10., 10., 0.01, 0.01, 0.01])

            transition_noise_pool = None if noise_pool is None else noise_pool[
                'transition_error']
            self._transition_error_model = SensorErrorModel(
                torch.ones(self.num_satellite, dtype=torch.bool),
                process_noise_matrix,
                walkbound,
                seed=random.randint(0, 100000),
                recorded_noise=transition_noise_pool,
                timer=self._timer)

        if use_perturbation:
            recorded_perturbation_torque = None if noise_pool is None else noise_pool[
                'perturbation_torque']
            recorded_perturbation_force = None if noise_pool is None else noise_pool[
                'perturbation_force']
            perturbation = PerturbationForceTorque(
                self.num_satellite,
                random.randint(0, 100000),
                timer=self._timer,
                recorded_noise_torque=recorded_perturbation_torque,
                recorded_noise_force=recorded_perturbation_force,
            )
            self._spacecraft.state_effectors.update(
                dict(_perturbation=perturbation))

    @property
    def reaction_wheels(self) -> ReactionWheels:
        return self._spacecraft.state_effectors['_reaction_wheels']

    @property
    def num_satellite(self) -> int:
        return self._n

    @property
    def spice_interface(self) -> SpiceInterface:
        return cast(SpiceInterface,
                    self._spacecraft.gravity_field.spice_interface)

    def _get_new_gravity_field(self) -> GravityField:
        sun = PointMassGravityBody.create_sun(timer=self._timer)
        earth = PointMassGravityBody.create_earth(
            timer=self._timer,
            is_central=True,
        )

        spice_interface = SpiceInterface(
            timer=self._timer,
            utc_time_init=constants.UTC_TIME_START,
        )

        return GravityField(
            timer=self._timer,
            spice_interface=spice_interface,
            gravity_bodies=[sun, earth],
        )

    def _setup_spacecraft(
        self,
        constellation: Constellation,
        integrate_method: IntegrateMethod,
        position_BP_N_init: torch.Tensor,
        velocity_BP_N_init: torch.Tensor,
        attitude_BN_init: torch.Tensor | None = None,
        angular_velocity_BN_B_init: torch.Tensor | None = None,
    ) -> None:
        gravity_field = self._get_new_gravity_field()
        reaction_wheels = self._build_reaction_wheels_from_config(
            constellation)

        inertias = constellation.get_inertia()
        mass = constellation.get_mass()
        self._spacecraft = Spacecraft(
            timer=self._timer,
            mass=mass,
            moment_of_inertia_matrix_wrt_body_point=inertias,
            position_BP_N=position_BP_N_init,
            velocity_BP_N=velocity_BP_N_init,
            attitude_BN=attitude_BN_init,
            angular_velocity_BN_B=angular_velocity_BN_B_init,
            gravity_field=gravity_field,
            state_effectors=dict(_reaction_wheels=reaction_wheels),
            integrate_method=integrate_method,
        )

    def _build_reaction_wheels_from_config(
        self,
        constellation: Constellation,
    ) -> ReactionWheels:
        reaction_wheels_groups = [sat.reaction_wheels for sat in constellation]

        reaction_wheels_0 = []
        reaction_wheels_1 = []
        reaction_wheels_2 = []

        for rw0, rw1, rw2 in reaction_wheels_groups:

            reaction_wheel_type = ReactionWheelRegistry[rw0.type]
            reaction_wheels_0.append(
                reaction_wheel_type.build(
                    mech_to_elec_efficiency=getattr(rw0, 'efficiency', -1),
                    base_power=0.,
                    angular_velocity_init=rw0.rw_speed_init,
                ))
            reaction_wheels_1.append(
                reaction_wheel_type.build(
                    mech_to_elec_efficiency=getattr(rw1, 'efficiency', -1),
                    base_power=0.,
                    angular_velocity_init=rw1.rw_speed_init,
                ))
            reaction_wheels_2.append(
                reaction_wheel_type.build(
                    mech_to_elec_efficiency=getattr(rw2, 'efficiency', -1),
                    base_power=0.,
                    angular_velocity_init=rw2.rw_speed_init,
                ))

        reaction_wheel_0 = concat(reaction_wheels_0)
        reaction_wheel_1 = concat(reaction_wheels_1)
        reaction_wheel_2 = concat(reaction_wheels_2)
        return ReactionWheels(
            timer=self._timer,
            reaction_wheels=[
                reaction_wheel_0, reaction_wheel_1, reaction_wheel_2
            ],
        )

    def _simple_motor_torque_assign(
        self,
        torque: torch.Tensor,
    ) -> torch.Tensor:
        return -torque

    def _apply_torque(
        self,
        state_dict: T,
        motor_torque: torch.Tensor,
    ) -> tuple[
            T,
            SpacecraftStateOutput,
            Ephemeris,
            torch.Tensor,
            torch.Tensor,
    ]:
        battery_state_dict = state_dict['_battery']

        reaction_wheels_state_dict = state_dict['_spacecraft'][
            '_state_effectors']['_reaction_wheels']
        reaction_wheels_state_dict, (
            battery_state_dict, ) = self.reaction_wheels(
                state_dict=reaction_wheels_state_dict,
                battery_state_dict=battery_state_dict,
                motor_torque=motor_torque,
            )
        state_dict['_spacecraft']['_state_effectors'][
            '_reaction_wheels'] = reaction_wheels_state_dict

        if '_perturbation' in self._spacecraft.state_effectors:
            perturbation_state_dict = state_dict['_spacecraft'][
                '_state_effectors']['_perturbation']
            perturbation_state_dict, _ = self._spacecraft.state_effectors[
                '_perturbation'](perturbation_state_dict)

        spacecraft_state_output: SpacecraftStateOutput
        spacecraft_state_dict = state_dict['_spacecraft']
        spacecraft_state_dict, spacecraft_state_output = self._spacecraft(
            spacecraft_state_dict)
        state_dict['_spacecraft'] = spacecraft_state_dict
        attitude_BN = spacecraft_state_dict['_hub']['dynamic_params'][
            'attitude_BN']
        angular_velocity_BN_B = spacecraft_state_dict['_hub'][
            'dynamic_params']['angular_velocity_BN_B']

        sun_ephemeris: Ephemeris
        earth_ephemeris: Ephemeris
        _, (sun_ephemeris, ) = self.spice_interface(names=['SUN'])
        _, (earth_ephemeris, ) = self.spice_interface(names=['EARTH'])

        # Need to align device location and data type of the ephemeris
        sun_ephemeris = move_to(
            sun_ephemeris,
            attitude_BN,
        )
        earth_ephemeris = move_to(
            earth_ephemeris,
            attitude_BN,
        )

        battery_state_dict = state_dict['_battery']
        battery_state_dict, _ = self._battery(state_dict=battery_state_dict)
        state_dict['_battery'] = battery_state_dict

        if self._use_noise:
            attitude_error_model_state = state_dict['_attitude_error_model']
            attitude_error_model_state, (
                attitude_BN,
                angular_velocity_BN_B,
            ) = self._attitude_error_model(
                attitude_error_model_state,
                attitude_BN,
                angular_velocity_BN_B,
            )
            state_dict['_attitude_error_model'] = attitude_error_model_state

            transition_error_model_state = state_dict[
                '_transition_error_model']
            transition_error_model_state, (
                position_BN_N,
                velocity_BN_N,
            ) = self._transition_error_model(
                transition_error_model_state,
                spacecraft_state_output.position_BN_N,
                spacecraft_state_output.velocity_BN_N,
            )
            state_dict[
                '_transition_error_model'] = transition_error_model_state

            spacecraft_state_output = SpacecraftStateOutput(
                position_BN_N,
                velocity_BN_N,
                spacecraft_state_output.angular_acceleration_BN_B,
            )

        return state_dict, spacecraft_state_output, earth_ephemeris, attitude_BN, angular_velocity_BN_B

    def _assembly_observation(
        self,
        state_dict: T,
        guidance_output: GuidanceOutput,
        attitude_BN: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
    ) -> DynamicObservation:

        reaction_wheels_speed = state_dict['_spacecraft']['_state_effectors'][
            '_reaction_wheels']['dynamic_params']['angular_velocity'].squeeze(
                -2)
        hub_dynam = state_dict['_spacecraft']['_hub']['dynamic_params']
        angular_velocity = hub_dynam['angular_velocity_BN_B']
        attitude = hub_dynam['attitude_BN']
        if attitude.dim() == 1:
            attitude = attitude.expand_as(angular_velocity)

        battery_state_dict = state_dict['_battery']
        percentage = battery_state_dict['stored_charge_percentage'].unsqueeze(
            -1)
        guidance_output = GuidanceOutput(*(t for t in guidance_output))

        observation = DynamicObservation(
            reaction_wheels_speed=reaction_wheels_speed,
            spacecraft_angular_velocity=angular_velocity_BN_B,
            spacecraft_attitude=attitude_BN,
            battery_percentage=percentage,
            current_time=self._timer.time,
            attitude_guide_info=guidance_output,
        )
        return observation
