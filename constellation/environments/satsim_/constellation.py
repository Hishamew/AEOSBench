__all__ = [
    'SatsimConstellationStateDict',
    'SatsimConstellation',
]
from typing import TypedDict, TypeVar, cast

import torch
from satsim.architecture import Module, Timer, constants
from satsim.attitude_control import MRPFeedback, MRPFeedbackStateDict
from satsim.attitude_guidance import (
    GuidanceOutput,
    LocationPointing,
    LocationPointingStateDict,
    ReferenceAttitudeOutput,
)
from satsim.enviroment.ground_location import AccessState
from satsim.enviroment.ground_mapping import (
    GroundMapping,
    GroundMappingStateDict,
)
from satsim.simulation.base import BatteryStateDict
from satsim.simulation.eclipse import compute_shadow_factor
from satsim.simulation.gravity import (
    Ephemeris,
    GravityField,
    PointMassGravityBody,
    SpiceInterface,
)
from satsim.simulation.power import (
    SimpleBattery,
    SimplePowerSink,
    SimplePowerSinkStateDict,
    SimpleSolarPanel,
    SimpleSolarPanelStateDict,
)
from satsim.simulation.reaction_wheels import (
    HoneywellHR12Small,
    ReactionWheels,
    ReactionWheelsStateDict,
    concat,
)
from satsim.simulation.spacecraft import (
    IntegrateMethod,
    Spacecraft,
    SpacecraftStateDict,
    SpacecraftStateOutput,
)
from satsim.utils import LLA2PCPF, move_to

from ...data import Constellation, TaskSet
from ...data.actions import Actions
from ...data.constellations import Satellites
from ..basilisk.constants import UNIT_VECTOR_Z
from .utils import convert_to_utc, is_utc_datetime_str


class SatsimConstellationStateDict(TypedDict):
    _battery: BatteryStateDict
    _spacecraft: SpacecraftStateDict
    _solar_panel: SimpleSolarPanelStateDict
    _power_sink: SimplePowerSinkStateDict
    _location_pointing: LocationPointingStateDict
    _ground_location: GroundMappingStateDict
    _mrp_control: MRPFeedbackStateDict


T = TypeVar(
    'T',
    bound=SatsimConstellationStateDict,
)


# Task point and coordinate is marked as T
# Tracking target and coordinate is marked as L
class SatsimConstellation(Module[SatsimConstellationStateDict]):

    def __init__(
        self,
        timer: Timer,
        constellation: Constellation,
        standard_time_init: str,
        taskset: TaskSet,
    ) -> None:
        super().__init__(timer=timer)

        self._n = len(constellation)
        sorted_satellites = sorted(
            [sat for sat in constellation.values()],
            key=lambda x: x.id_,
        )
        self._satellites = sorted_satellites
        self._standard_time_init = standard_time_init
        self._taskset = taskset

        self.setup_basic_spacecraft()
        self.setup_additional_power_system()
        self.setup_target_tracking()

    def setup_basic_spacecraft(self) -> None:
        gravity_field = self._get_new_gravity_field(self._standard_time_init)
        reaction_wheels = self._build_reaction_wheels_from_config(
            self._satellites
        )

        inertias = [sat.inertia for sat in self._satellites]
        mass = [sat.mass for sat in self._satellites]
        inertias = torch.tensor(inertias).view(-1, 3, 3)
        mass = torch.tensor(mass)

        rs = []
        vs = []
        for sat in self._satellites:
            r, v = sat.rv
            r, v = torch.from_numpy(r), torch.from_numpy(v)
            rs.append(r)
            vs.append(v)
        position_BP_N_init = torch.stack(rs)
        velocity_BP_N_init = torch.stack(vs)

        attitude_BN_init = torch.tensor([
            sat.mrp_attitude_bn for sat in self._satellites
        ])
        # TODO: Check
        angular_velocity_BN_B_init = torch.zeros_like(attitude_BN_init)

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
        )

        capacity = torch.tensor([
            sat.battery.capacity for sat in self._satellites
        ])
        percentage_init = torch.tensor([
            sat.battery.percentage for sat in self._satellites
        ])

        self._battery = SimpleBattery(
            timer=self._timer,
            storage_capacity=capacity,
            stored_charge_percentage_init=percentage_init,
        )

    def _get_new_gravity_field(self, standard_time_init: str) -> GravityField:
        standard_time_init = convert_to_utc(standard_time_init)
        sun = PointMassGravityBody.create_sun(timer=self._timer)
        earth = PointMassGravityBody.create_earth(
            timer=self._timer,
            is_central=True,
        )

        spice_interface = SpiceInterface(
            timer=self._timer,
            utc_time_init=standard_time_init,
        )

        return GravityField(
            timer=self._timer,
            spice_interface=spice_interface,
            gravity_bodies=[sun, earth],
        )

    def _build_reaction_wheels_from_config(
        self,
        satellites: Satellites,
    ) -> ReactionWheels:
        reaction_wheels_groups = [sat.reaction_wheels for sat in satellites]

        reaction_wheels_0 = []
        reaction_wheels_1 = []
        reaction_wheels_2 = []

        for rw0, rw1, rw2 in reaction_wheels_groups:
            reaction_wheels_0.append(
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=rw0.efficiency,
                    base_power=rw0.power,
                    angular_velocity_init=rw0.rw_speed_init,
                )
            )
            reaction_wheels_1.append(
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=rw1.efficiency,
                    base_power=rw1.power,
                    angular_velocity_init=rw1.rw_speed_init,
                )
            )
            reaction_wheels_2.append(
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=rw2.efficiency,
                    base_power=rw2.power,
                    angular_velocity_init=rw2.rw_speed_init,
                )
            )

        reaction_wheel_0 = concat(reaction_wheels_0)
        reaction_wheel_1 = concat(reaction_wheels_1)
        reaction_wheel_2 = concat(reaction_wheels_2)
        return ReactionWheels(
            timer=self._timer,
            reaction_wheels=[
                reaction_wheel_0, reaction_wheel_1, reaction_wheel_2
            ],
        )

    def setup_additional_power_system(self) -> None:
        directions = torch.tensor([
            sat.solar_panel.direction for sat in self._satellites
        ])
        areas = torch.tensor([
            sat.solar_panel.area for sat in self._satellites
        ])
        effs = torch.tensor([
            sat.solar_panel.efficiency for sat in self._satellites
        ])

        self._solar_panel = SimpleSolarPanel(
            timer=self._timer,
            panel_normal_B_B=directions,
            panel_area=areas,
            panel_efficiency=effs,
        )

        camera_effs = torch.tensor([
            sat.sensor.power for sat in self._satellites
        ])
        self._power_sink = SimplePowerSink(
            timer=self._timer,
            power_efficiency=camera_effs,
        )
        self.register_buffer(
            '_camera_switch',
            torch.tensor([sat.sensor.enabled for sat in self._satellites]),
        )
        self.register_buffer(
            '_camera_on',
            torch.tensor([sat.sensor.enabled for sat in self._satellites]),
        )

    def setup_target_tracking(self) -> None:
        sensor_pointing_directions = torch.tensor(
            [UNIT_VECTOR_Z for _ in range(self.num_satellites)],
            dtype=torch.get_default_dtype(),
        )
        half_view = torch.tensor([
            sat.sensor.half_field_of_view for sat in self._satellites
        ])

        self._location_pointing = LocationPointing(
            timer=self._timer,
            pointing_direction_B_B=sensor_pointing_directions,
        )

        position_TP_P = torch.tensor(self._taskset.coordinates_ecef)  #
        self.register_buffer(
            '_position_TP_P',
            position_TP_P,
            persistent=False,
        )
        self.register_buffer(
            '_position_LP_P',
            torch.zeros(self.num_satellites, 3),
        )
        self.register_buffer(
            '_with_target',
            torch.zeros(self.num_satellites, dtype=torch.bool),
        )
        self._ground_mapping = GroundMapping(
            timer=self._timer,
            minimum_elevation=torch.tensor([
                0. for _ in range(self.num_satellites)
            ]),
            half_field_of_view=half_view,
            camera_direction_B_B=sensor_pointing_directions,
        )

        k = torch.tensor([sat.mrp_control.k for sat in self._satellites])
        ki = torch.tensor([sat.mrp_control.ki for sat in self._satellites])
        p = torch.tensor([sat.mrp_control.p for sat in self._satellites])
        integral_limit = torch.tensor([
            sat.mrp_control.integral_limit for sat in self._satellites
        ])
        self._mrp_control = MRPFeedback(
            timer=self._timer,
            k=k,
            ki=ki,
            p=p,
            integral_limit=integral_limit,
        )

    @property
    def satellites(self) -> Satellites:
        return self._satellites

    @property
    def num_satellites(self) -> int:
        return self._n

    @property
    def sensor_type(self) -> torch.Tensor:
        return torch.tensor([sat.sensor.type_ for sat in self._satellites])

    @property
    def reaction_wheels(self) -> ReactionWheels:
        return self.spacecraft.state_effectors['_reaction_wheels']

    @property
    def spice_interface(self) -> SpiceInterface:
        return cast(
            SpiceInterface, self.spacecraft.gravity_field.spice_interface
        )

    @property
    def camera_switch(self) -> torch.Tensor:
        return self.get_buffer('_camera_switch')

    @property
    def camera_on(self) -> torch.Tensor:
        return self.get_buffer('_camera_on')

    @property
    def task_points(self) -> torch.Tensor:
        """All Task Points Buffer."""
        return self.get_buffer('_position_TP_P')

    @property
    def tracking_target(self) -> torch.Tensor:
        """Target Point Buffer.
        """
        return self.get_buffer('_position_LP_P')

    @property
    def with_target(self) -> torch.Tensor:
        return self.get_buffer('_with_target')

    @property
    def spacecraft(self) -> Spacecraft:
        return self._spacecraft

    @property
    def battery(self) -> SimpleBattery:
        return self._battery

    @property
    def solar_panel(self) -> SimpleSolarPanel:
        return self._solar_panel

    @property
    def power_sink(self) -> SimplePowerSink:
        return self._power_sink

    @property
    def location_pointing(self) -> LocationPointing:
        return self._location_pointing

    @property
    def ground_mapping(self) -> GroundMapping:
        """GroundMapping module is used for all target location
        access states monitoring.
        LocationPointing target is
        calculated through calculate_location_inertial_position
        """
        return self._ground_mapping

    @property
    def mrp_control(self) -> MRPFeedback:
        return self._mrp_control

    def take_actions(
        self,
        actions: Actions,
    ) -> None:
        toggle, with_target, target = actions.to_tensors()
        latitude, longitude = target.unbind(-1)
        self._turn_switch(toggle)

        position_LP_P = LLA2PCPF(
            latitude,
            longitude,
            torch.zeros_like(latitude),
            constants.REQ_EARTH * 1e3,
            constants.REQ_EARTH * 1e3,
        )
        self._update_tracking_target(
            position_LP_P.to(self.tracking_target),
            with_target.to(self.with_target),
        )

    def _update_tracking_target(
        self,
        new_targets: torch.Tensor,
        with_target: torch.BoolTensor,
    ) -> None:
        self._position_LP_P = new_targets
        self._with_target = with_target

    def _turn_switch(self, toggle: torch.Tensor) -> None:
        self._camera_switch = self.camera_switch.bitwise_xor(toggle)

    def _simple_motor_torque_assign(
        self,
        torque: torch.Tensor,
    ) -> torch.Tensor:
        return -torque

    def calculate_location_inertial_position(
        self,
        earth_ephemeris: Ephemeris,
    ) -> torch.Tensor:
        """Calculate Tracking Target coordinate as in
        position_LN_N
        """
        direction_cosine_matrix_PN = earth_ephemeris[
            'direction_cosine_matrix_CN']
        position_PN_N = earth_ephemeris['position_CN_N']
        position_LP_N = torch.einsum(
            '...ij, ...i -> ...j',
            direction_cosine_matrix_PN,
            self.tracking_target,
        )
        return position_PN_N + position_LP_N

    def get_earth_ephemeris(
        self,
        target: torch.Tensor | torch.device | torch.dtype,
    ) -> Ephemeris:
        earth_ephemeris: Ephemeris
        _, (earth_ephemeris, ) = self.spice_interface(names=['EARTH'])

        earth_ephemeris = move_to(
            earth_ephemeris,
            target,
        )
        return earth_ephemeris

    def get_task_access(
        self,
        state_dict: SatsimConstellationStateDict,
    ) -> AccessState:
        """Task access socket.
        This method can be called whenever needed and do not require any simulation prerequist.
        """
        attitude_BN = state_dict['_spacecraft']['_hub']['dynamic_params'][
            'attitude_BN']
        position_BP_N = state_dict['_spacecraft']['_hub']['dynamic_params'][
            'position_BP_N']
        velocity_BP_N = state_dict['_spacecraft']['_hub']['dynamic_params'][
            'velocity_BP_N']
        earth_ephemeris = self.get_earth_ephemeris(attitude_BN)

        ## Ugly temporal fix, initial ephemeris loading should be implemented in reset
        if not hasattr(
            self.spacecraft.gravity_field, '_gravity_bodies_ephemeris'
        ):
            self.spacecraft.gravity_field._load_ephemeris(attitude_BN)
            self.spacecraft.gravity_field._calculate_next_position()

        position_BN_N, velocity_BN_N = self.spacecraft.gravity_field.update_inertial_position_and_velocity(
            position_BP_N,
            velocity_BP_N,
        )

        _, (access_state, _, _) = self.ground_mapping(
            ephemeris=earth_ephemeris,
            position_BN_N=position_BN_N,
            velocity_BN_N=velocity_BN_N,
            attitude_BN=attitude_BN,
            position_LP_P=self.task_points,
            equatorial_radius=constants.REQ_EARTH * 1e3,
            polar_radius=constants.REQ_EARTH * 1e3,
        )
        return access_state

    def forward(
        self, state_dict: SatsimConstellationStateDict
    ) -> tuple[SatsimConstellationStateDict, tuple[Ephemeris]]:
        # Start by integrating time to next step
        spacecraft_state_dict = state_dict['_spacecraft']
        spacecraft_state_dict, spacecraft_output = self.spacecraft(
            spacecraft_state_dict
        )
        spacecraft_state_dict: SpacecraftStateDict
        spacecraft_output: SpacecraftStateOutput
        state_dict['_spacecraft'] = spacecraft_state_dict

        # get spacecraft state
        attitude_BN = spacecraft_state_dict['_hub']['dynamic_params'][
            'attitude_BN']
        angular_velocity_BN_B = spacecraft_state_dict['_hub'][
            'dynamic_params']['angular_velocity_BN_B']
        wheel_speeds = spacecraft_state_dict['_state_effectors'][
            '_reaction_wheels']['dynamic_params']['angular_velocity']

        # target tracking system update
        sun_ephemeris: Ephemeris
        earth_ephemeris: Ephemeris
        _, (sun_ephemeris, ) = self.spice_interface(names=['SUN'])
        _, (earth_ephemeris, ) = self.spice_interface(names=['EARTH'])

        sun_ephemeris = move_to(
            sun_ephemeris,
            attitude_BN,
        )
        earth_ephemeris = move_to(
            earth_ephemeris,
            attitude_BN,
        )

        position_LN_N = self.calculate_location_inertial_position(
            earth_ephemeris
        )

        location_pointing_state_dict = state_dict['_location_pointing']
        location_pointing_state_dict, (
            guidance, reference
        ) = self.location_pointing(
            location_pointing_state_dict,
            position_LN_N=position_LN_N,
            position_BN_N=spacecraft_output.position_BN_N,
            attitude_BN=attitude_BN,
            angular_velocity_BN_B=angular_velocity_BN_B,
        )
        location_pointing_state_dict: LocationPointingStateDict
        guidance: GuidanceOutput
        reference: ReferenceAttitudeOutput
        state_dict['_location_pointing'] = location_pointing_state_dict

        mrp_control_state_dict = state_dict['_mrp_control']
        mrp_control_state_dict, (control_torque, _) = self.mrp_control(
            mrp_control_state_dict,
            sigma_BR=guidance.attitude_BR,
            omega_BR_B=guidance.angular_velocity_BR_B,
            omega_RN_B=guidance.angular_acceleration_RN_B,
            domega_RN_B=guidance.angular_acceleration_RN_B,
            wheel_speeds=wheel_speeds,
            inertia_spacecraft_point_b_in_body=self.spacecraft.hub.
            moment_of_inertia_matrix_wrt_body_point,
            reaction_wheels_inertia_wrt_spin=self.reaction_wheels.
            moment_of_inertia_wrt_spin,
            reaction_wheels_spin_axis=self.reaction_wheels.spin_axis_in_body,
        )
        mrp_control_state_dict: MRPFeedbackStateDict
        control_torque: torch.Tensor
        state_dict['_mrp_control'] = mrp_control_state_dict

        # power supply system update
        battery_state_dict = state_dict['_battery']

        factors = compute_shadow_factor(
            sun_ephemeris['position_CN_N'],
            earth_ephemeris['position_CN_N'],
            spacecraft_output.position_BN_N,
            torch.tensor([constants.REQ_EARTH * 1e3]),
        )

        ## solar panel is a non-stated module
        solar_panel_state_dict = state_dict['_solar_panel']
        solar_panel_state_dict, (_, battery_state_dict) = self.solar_panel(
            solar_panel_state_dict,
            position_BN_N=spacecraft_output.position_BN_N,
            position_SN_N=sun_ephemeris['position_CN_N'],
            attitude_BN=attitude_BN,
            shadow_factor=factors,
            battery_state_dict=battery_state_dict,
        )

        ## power sink is a non-stated module
        power_sink_state_dict = state_dict['_power_sink']
        power_sink_state_dict, (
            camera_on,
            battery_state_dict,
        ) = self.power_sink(
            power_sink_state_dict,
            turn_on=self.camera_switch,
            battery_state_dict=battery_state_dict,
        )
        camera_on: torch.BoolTensor
        self._camera_on = camera_on

        motor_torque = self._simple_motor_torque_assign(control_torque)
        reaction_wheels_state_dict = state_dict['_spacecraft'][
            '_state_effectors']['_reaction_wheels']
        reaction_wheels_state_dict, (battery_state_dict,
                                     ) = self.reaction_wheels(
                                         reaction_wheels_state_dict,
                                         battery_state_dict=battery_state_dict,
                                         motor_torque=motor_torque,
                                     )
        reaction_wheels_state_dict: ReactionWheelsStateDict
        state_dict['_spacecraft']['_state_effectors'][
            '_reaction_wheels'] = reaction_wheels_state_dict

        battery_state_dict, _ = self.battery(battery_state_dict)
        battery_state_dict: BatteryStateDict
        state_dict['_battery'] = battery_state_dict

        return state_dict, (earth_ephemeris, camera_on)
