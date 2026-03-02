__all__ = [
    'SatsimEnvironment',
]

import todd
import torch
from Basilisk.utilities import macros, orbitalMotion
from satsim.architecture import Timer, constants
from satsim.utils import dict_recursive_apply

from ...constants import INTERVAL, TIMESTAMP
from ...data import Actions, Constellation, TaskSet
from ...data.constellations import (
    Battery,
    ReactionWheel,
    Satellite,
    Satellites,
    Sensor,
)
from ...data.orbits import Orbit
from ..base import BaseEnvironment
from .constellation import SatsimConstellation


class SatsimEnvironment(BaseEnvironment):

    def __init__(
        self,
        *args,
        standard_time_init: str = TIMESTAMP,
        constellation: Constellation,
        all_tasks: TaskSet,
        backend: torch.device | None = None,
        fp_precision: torch.dtype = torch.float64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # env set
        # RANK = int(os.environ.get('RANK', '0'))
        # device_count = torch.cuda.device_count()
        # self._device = torch.device(RANK % device_count)
        if backend is not None:
            self._backend = backend
        else:
            self._backend = torch.device('cuda' if todd.Store.cuda else 'cpu')
        self._fp_precision = fp_precision

        self._authentic_timer = Timer(INTERVAL, self._start_time)

        self._simulator = SatsimConstellation(
            self._authentic_timer,
            constellation,
            standard_time_init,
            all_tasks,
        )

        self._constellation_data = constellation
        self._authentic_timer.reset()
        self._simulator_state_dict = self._simulator.reset()
        self._simulator_state_dict = dict_recursive_apply(
            self._simulator_state_dict,
            lambda x: x.to(self._backend, dtype=torch.float64),
        )
        self._simulator.to(self._backend, dtype=torch.float64)

    @property
    def num_satellites(self) -> int:
        return self._simulator.num_satellites

    @property
    def authentic_timer(self) -> Timer:
        return self._authentic_timer

    @property
    def start_time(self) -> int:
        return self._start_time

    def get_constellation(self) -> Constellation:
        satellites: Satellites = []
        inertias = self._simulator_state_dict['_spacecraft']['mass_props'][
            'moment_of_inertia_matrix_wrt_body_point']
        masses = self._simulator_state_dict['_spacecraft']['mass_props']['mass'
                                                                         ]
        position_BP_N = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']['position_BP_N']
        velocity_BP_N = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']['velocity_BP_N']
        attitude_BN = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']['attitude_BN']
        battery_percentages = self._simulator_state_dict['_battery'][
            'stored_charge_percentage']
        wheel_speeds: torch.Tensor = self._simulator_state_dict['_spacecraft'][
            '_state_effectors']['_reaction_wheels']['dynamic_params'][
                'angular_velocity']
        wheel_speeds = wheel_speeds.squeeze()

        if battery_percentages.dim() != 1:
            battery_percentages = battery_percentages.expand_as(masses)

        for idx, satellite in enumerate(self._simulator.satellites):
            r_BP_N = position_BP_N[idx].cpu().numpy()
            v_BP_N = velocity_BP_N[idx].cpu().numpy()
            orbital_elements = orbitalMotion.rv2elem(
                constants.MU_EARTH * 1e9,
                r_BP_N,
                v_BP_N,
            )
            # eccentricity and semi major axis is np.array for some reason
            # we convert them to float to avoid fp precision conflict
            orbit = Orbit(
                satellite.orbit_id,
                float(orbital_elements.e),
                float(orbital_elements.a),
                orbital_elements.i / macros.D2R,
                orbital_elements.Omega / macros.D2R,
                orbital_elements.omega / macros.D2R,
            )

            sensor = Sensor(
                satellite.sensor.type_,
                self._simulator.camera_switch[idx].item(),
                satellite.sensor.half_field_of_view,
                satellite.sensor.power,
            )
            battery = Battery(
                satellite.battery.capacity,
                battery_percentages[idx].item(),
            )
            reaction_wheels = satellite.reaction_wheels
            wheel_speeds_idx = wheel_speeds[idx]
            reaction_wheels = tuple([
                ReactionWheel(
                    rw.rw_type,
                    rw.rw_direction,
                    rw.max_momentum,
                    wheel_speeds_idx[i].item(),
                    rw.power,
                    rw.efficiency,
                ) for i, rw in enumerate(reaction_wheels)
            ])

            socket_satellite = Satellite(
                satellite.id_,
                tuple(inertias[idx].view(-1).tolist()),
                masses[idx].item(),
                satellite.center_of_mass,
                satellite.orbit_id,
                orbit,
                satellite.solar_panel,
                sensor,
                battery,
                reaction_wheels,
                satellite.mrp_control,
                orbital_elements.f / macros.D2R,
                tuple(attitude_BN[idx].tolist()),
            )
            satellites.append(socket_satellite)

        return Constellation({
            satellite.id_: satellite
            for satellite in satellites
        })

    def take_actions(self, actions: Actions) -> None:
        self._simulator.take_actions(actions)

    def step(self) -> None:
        self._simulator_state_dict, _ = self._simulator(
            self._simulator_state_dict
        )
        self.authentic_timer.step()

    def is_visible(self, tasks: TaskSet) -> torch.Tensor:  # TODO: No need
        access_state = self._simulator.get_task_access(
            self._simulator_state_dict
        )
        has_access = access_state.has_access  # [n_p, n_sc]
        camera_on = self._simulator.camera_on.expand_as(
            has_access
        )  # [n_p, n_sc]

        task_sensor_type = torch.tensor(
            [task.sensor_type for task in tasks],
            device=self._backend,
        ).unsqueeze(1)  # [n_p,1]
        satellite_sensor_type = self._simulator.sensor_type.unsqueeze(0).to(
            self._backend
        )  # [1, n_sc]
        sensor_match = task_sensor_type == satellite_sensor_type

        return (has_access & camera_on
                & sensor_match).transpose(0, 1).to(torch.get_default_device())

    def get_earth_rotation(self) -> torch.Tensor:
        earth_ephmeris = self._simulator.get_earth_ephemeris(None)
        return earth_ephmeris['direction_cosine_matrix_CN'].squeeze()
