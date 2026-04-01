from typing import TypedDict

import torch

from satsim.architecture import constants
from satsim.attitude_guidance import (GuidanceOutput, LocationPointing,
                                      LocationPointingStateDict,
                                      ReferenceAttitudeOutput)
from satsim.data import elem2rv
from satsim.simulation.gravity import Ephemeris
from satsim.simulation.spacecraft import (Spacecraft, SpacecraftStateDict,
                                          SpacecraftStateOutput)

from ..data import (AttitudeMaintainTaskset, LocationPointingTaskset,
                    SpacecraftPointingTaskset)
from .base_constellation import (BaseConstellation, BaseConstellationStateDict,
                                 DynamicObservation)
from .guidance import (SimpleInertial3DGuide, SimpleInertial3DGuideStateDict,
                       SpacecraftPointing, SpacecraftPointingStateDict)


class LocationPointingConstellationStateDict(BaseConstellationStateDict):
    _location_pointing: LocationPointingStateDict


class LocationPointingConstellation(
        BaseConstellation[LocationPointingConstellationStateDict]):

    def __init__(
        self,
        task_config: LocationPointingTaskset,
        use_wheels_failure: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialization of AttitudeControlConstellation (attitude-controlled satellite constellation)

        Initializes an attitude-controlled satellite constellation instance, which inherits from
        `BaseConstellation`. This method completes key initializations including:
        1. Calculation of satellite initial orbit position and velocity from orbital elements
        2. Calculation of initial attitude (pointing to target location)
        3. Initialization of on-board subsystems (battery, attitude pointing controller)
        4. Inheritance and configuration of parent class `BaseConstellation`

        Args:
            timer (Timer): Time management module for simulation. Provides core time functions such as
                simulation timing, step advancement, and reset, which are used by subsystems like
                `SimpleBattery` and `LocationPointing` for time-dependent logic.
            constellation (BaseConstellationConfig): Configuration object for the satellite constellation.
                Contains core constellation parameters required by the parent class `BaseConstellation`,
                such as the number of satellites, satellite mass properties (e.g., moment of inertia),
                and reaction wheel hardware parameters (e.g., moment of inertia).
            orbits (OrbitalElements): Orbital elements set for all satellites in the constellation.
                Used to calculate the initial inertial position (`position_BP_N`) and initial inertial
                velocity (`velocity_BP_N`) of each satellite via the `elem2rv` function.
            position_LP_P (torch.Tensor): Position vector of the target in the Earth-Centered
                Earth-Fixed (ECEF) coordinate system (P).
                - Tensor shape: `[num_satellite, 3]` (num_satellite = number of satellites in the constellation;
                3 corresponds to x, y, z coordinates in ECEF)
                - Role: Serves as the target reference for the attitude pointing controller (`LocationPointing`),
                guiding satellites to point to this location.

        Note:
            - `timer` and `constellation` are essential parameters for the parent class `BaseConstellation`
            and must be passed via `*args` or `**kwargs` (consistent with the parent class's parameter order).
            - The initial attitude (`attitude_BN_init`) is calculated by the `attitude_init` function, which
            ensures the satellite initially points to the target location (`position_LP_P`).
        """
        super().__init__(
            *args,
            **kwargs,
        )
        position_LP_P = task_config.get_task_coordinate()
        self.register_buffer('_position_LP_P', position_LP_P, persistent=False)

        self._location_pointing = LocationPointing(
            timer=self._timer,
            pointing_direction_B_B=torch.tensor([0, 0, 1.]).expand(
                self.num_satellite, 3),
        )

        self._use_wheels_failure = use_wheels_failure
        reaction_wheels_available = (
            task_config.get_reaction_wheel_availability() if use_wheels_failure
            else torch.ones(self.num_satellite, 3, dtype=torch.bool))

        self.register_buffer(
            '_reaction_wheels_available',
            reaction_wheels_available,
        )

    @property
    def reaction_wheels_available(self) -> None:
        return self.get_buffer('_reaction_wheels_available')

    @property
    def position_LP_P(self) -> torch.Tensor:
        return self.get_buffer('_position_LP_P')

    @property
    def reference_attitude(self) -> ReferenceAttitudeOutput:
        return self._reference_output_buffer

    def calculate_location_inertial_position(
        self,
        earth_ephemeris: Ephemeris,
    ) -> torch.Tensor:
        direction_cosine_matrix_PN = earth_ephemeris[
            'direction_cosine_matrix_CN']
        position_PN_N = earth_ephemeris['position_CN_N']
        position_LP_N = torch.einsum(
            '...ij, ...i -> ...j',
            direction_cosine_matrix_PN,
            self.position_LP_P,
        )
        return position_PN_N + position_LP_N

    def forward(
        self,
        state_dict: LocationPointingConstellationStateDict,
        torque: torch.Tensor,
    ) -> tuple[LocationPointingConstellationStateDict, tuple[
            DynamicObservation,
            Ephemeris,
    ]]:
        motor_torque = self._simple_motor_torque_assign(torque)

        # TODO: Test run to disable z axis
        if self._use_wheels_failure:
            simple_mask = torch.ones(self.num_satellite, 3, dtype=torch.bool)
            simple_mask[..., 2] = False
            motor_torque = torch.where(
                simple_mask,
                motor_torque,
                0.,
            )

        # TODO: fix failure wheels to be x

        (
            state_dict,
            spacecraft_state_output,
            earth_ephemeris,
            attitude_BN,
            angular_velocity_BN_B,
        ) = self._apply_torque(
            state_dict,
            motor_torque,
        )

        position_LN_N = self.calculate_location_inertial_position(
            earth_ephemeris)

        guidance_state_dict = state_dict['_location_pointing']
        guidance_output: GuidanceOutput
        guidance_state_dict, (
            guidance_output,
            reference_output,
        ) = self._location_pointing(
            state_dict=guidance_state_dict,
            position_LN_N=position_LN_N,
            position_BN_N=spacecraft_state_output.position_BN_N,
            attitude_BN=attitude_BN,
            angular_velocity_BN_B=angular_velocity_BN_B,
        )
        state_dict['_location_pointing'] = guidance_state_dict

        self._reference_output_buffer: ReferenceAttitudeOutput = reference_output

        observation = self._assembly_observation(
            state_dict,
            guidance_output,
            attitude_BN,
            angular_velocity_BN_B,
        )

        return state_dict, (
            observation,
            earth_ephemeris,
        )


class AttitudeMaintainConstellationStateDict(BaseConstellationStateDict):
    _simple_inertial_3d: SimpleInertial3DGuideStateDict


class AttitudeMaintainConstellation(
        BaseConstellation[AttitudeMaintainConstellationStateDict]):

    def __init__(
        self,
        task_config: AttitudeMaintainTaskset,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        attitude_RN = task_config.get_target_attitudes()
        self._simple_inertial_3d = SimpleInertial3DGuide(
            timer=self._timer,
            attitude_RN=attitude_RN,
        )

    @property
    def attitude_RN(self) -> torch.Tensor:
        return self._simple_inertial_3d.attitude_RN

    def forward(
        self,
        state_dict: AttitudeMaintainConstellationStateDict,
        torque: torch.Tensor,
    ) -> tuple[LocationPointingConstellationStateDict, tuple[
            DynamicObservation,
            Ephemeris,
    ]]:
        motor_torque = self._simple_motor_torque_assign(torque)

        (
            state_dict,
            spacecraft_state_output,
            earth_ephemeris,
            attitude_BN,
            angular_velocity_BN_B,
        ) = self._apply_torque(
            state_dict,
            motor_torque,
        )

        guidance_state_dict = state_dict['_simple_inertial_3d']
        guidance_output: GuidanceOutput
        guidance_state_dict, guidance_output = self._simple_inertial_3d(
            state_dict=guidance_state_dict,
            attitude_BN=attitude_BN,
            angular_velocity_BN_B=angular_velocity_BN_B,
        )
        state_dict['_simple_inertial_3d'] = guidance_state_dict

        observation = self._assembly_observation(
            state_dict,
            guidance_output,
            attitude_BN,
            angular_velocity_BN_B,
        )

        return state_dict, (
            observation,
            earth_ephemeris,
        )


class SpacecraftPointingConstellationStateDict(BaseConstellationStateDict):
    _spacecraft_pointing: SpacecraftPointingStateDict
    _chief_spacecraft: SpacecraftStateDict


class SpacecraftPointingConstellation(
        BaseConstellation[SpacecraftPointingConstellationStateDict]):

    def __init__(
        self,
        task_config: SpacecraftPointingTaskset,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        chief_spacecraft_orbit = task_config.get_orbital_elements()
        position_CP_N, velocity_CP_N = elem2rv(
            constants.MU_EARTH * 1e9,
            chief_spacecraft_orbit,
        )
        self._chief_spacecraft = Spacecraft(
            timer=self._timer,
            mass=self._spacecraft.hub.mass,
            moment_of_inertia_matrix_wrt_body_point=self._spacecraft.hub.
            moment_of_inertia_matrix_wrt_body_point,
            position_BP_N=position_CP_N,
            velocity_BP_N=velocity_CP_N,
            attitude_BN=torch.zeros_like(position_CP_N),
            angular_velocity_BN_B=torch.zeros_like(position_CP_N),
            gravity_field=self._get_new_gravity_field(),
        )
        self._spacecraft_pointing = SpacecraftPointing(timer=self._timer)

    def forward(
        self,
        state_dict: SpacecraftPointingStateDict,
        torque: torch.Tensor,
    ) -> tuple[LocationPointingConstellationStateDict, tuple[
            DynamicObservation,
            Ephemeris,
    ]]:
        motor_torque = self._simple_motor_torque_assign(torque)

        (
            state_dict,
            deputy_spacecraft_state_output,
            earth_ephemeris,
            attitude_BN,
            angular_velocity_BN_B,
        ) = self._apply_torque(
            state_dict,
            motor_torque,
        )

        chief_spacecraft_state_output: SpacecraftStateOutput
        chief_spacecraft_state_dict = state_dict['_chief_spacecraft']
        chief_spacecraft_state_dict, chief_spacecraft_state_output = self._chief_spacecraft(
            chief_spacecraft_state_dict)
        state_dict['_chief_spacecraft'] = chief_spacecraft_state_dict

        guidance_state_dict = state_dict['_spacecraft_pointing']
        guidance_output: GuidanceOutput
        guidance_state_dict, guidance_output = self._spacecraft_pointing(
            state_dict=guidance_state_dict,
            attitude_BN=attitude_BN,
            angular_velocity_BN_B=angular_velocity_BN_B,
            position_BN_N=deputy_spacecraft_state_output.position_BN_N,
            position_CN_N=chief_spacecraft_state_output.position_BN_N,
        )
        state_dict['_spacecraft_pointing'] = guidance_state_dict

        observation = self._assembly_observation(
            state_dict,
            guidance_output,
            attitude_BN,
            angular_velocity_BN_B,
        )

        return state_dict, (
            observation,
            earth_ephemeris,
        )
