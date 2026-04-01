__all__ = [
    'SatsimEnvironment',
]

import einops
import todd
import torch
from satsim.architecture import Timer, constants
from satsim.utils import dict_recursive_apply

from ...constants import INTERVAL, R2D, TIMESTAMP
from ...data import Actions, Constellation, Coordinate, Orbit, TaskSet
from ...data.constellations import (
    Battery,
    ReactionWheel,
    Satellite,
    Satellites,
    Sensor,
)
from ..base import BaseEnvironment
from .constellation import SatsimConstellation
from .utils import rv2elem, rv2elem_torch


class SatsimEnvironment(BaseEnvironment):

    def __init__(
        self,
        *args,
        standard_time_init: str = TIMESTAMP,
        constellation: Constellation,
        all_tasks: TaskSet,
        backend: torch.device | None = None,
        fp_precision: torch.dtype = torch.float64,
        reset_trigger_threshold: float = 1e-4,
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

        self._reset_trigger_threshold = reset_trigger_threshold

        self._constellation_data = constellation
        self._authentic_timer.reset()
        self._simulator_state_dict = self._simulator.reset()
        self.previous_targets = [None for _ in range(self.num_satellites)]
        self._task_ids = torch.tensor(
            [-1 for _ in range(self.num_satellites)],
            device=self._backend,
        )

        self._simulator_state_dict = dict_recursive_apply(
            self._simulator_state_dict,
            lambda x: x.to(self._backend, dtype=torch.float64),
        )
        self._simulator.to(self._backend, dtype=torch.float64)

    @property
    def current_task_id(self) -> torch.Tensor:
        return self._task_ids

    @property
    def previous_targets(self) -> list[Coordinate | None]:
        return self._previous_targets

    @previous_targets.setter
    def previous_targets(self, value: list[Coordinate | None]) -> None:
        self._previous_targets = value

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
        wheel_speeds = wheel_speeds.squeeze(1)

        if battery_percentages.dim() != 1:
            battery_percentages = battery_percentages.expand_as(masses)

        for idx, satellite in enumerate(self._simulator.satellites):
            r_BP_N = position_BP_N[idx].cpu().numpy(
            )  # TODO: change to torch version
            v_BP_N = velocity_BP_N[idx].cpu().numpy()
            orbital_elements = rv2elem(
                constants.MU_EARTH * 1e9,
                r_BP_N,
                v_BP_N,
            )
            # eccentricity and semi major axis is np.array for some reason
            # we convert them to float to avoid fp precision conflict
            orbit = Orbit(
                satellite.orbit_id,
                orbital_elements['e'],
                orbital_elements['a'],
                orbital_elements['i'] * R2D,
                orbital_elements['Omega'] * R2D,
                orbital_elements['omega'] * R2D,
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
                    wheel_speeds_idx[i].item() / constants.RPM,
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
                orbital_elements['f'] * R2D,
                tuple(attitude_BN[idx].tolist()),
            )
            satellites.append(socket_satellite)

        return Constellation({
            satellite.id_: satellite
            for satellite in satellites
        })

    def _reset_integrators(self, need_reset_mask: torch.Tensor) -> None:
        integral_sigma = self._simulator_state_dict['_mrp_control'][
            'integral_sigma']
        integral_sigma = torch.where(
            need_reset_mask.unsqueeze(1),
            torch.zeros_like(integral_sigma),
            integral_sigma,
        )
        self._simulator_state_dict['_mrp_control']['integral_sigma'
                                                   ] = integral_sigma

    def take_actions(self, actions: Actions) -> None:
        reset_flags = []
        targets = [action.target_location for action in actions]
        for p_t, target in zip(self.previous_targets, targets):
            if target is None and p_t is None:
                reset_flags.append(False)
                continue
            elif target is None or p_t is None:
                reset_flags.append(True)
                continue
            x, y = target
            px, py = p_t

            flag = (
                abs(x - px) > self._reset_trigger_threshold
                or abs(y - py) > self._reset_trigger_threshold
            )
            reset_flags.append(flag)

        need_reset_mask = torch.tensor(reset_flags, device=self._backend)
        if need_reset_mask.any():
            self._reset_integrators(need_reset_mask)
        self.previous_targets = targets

        toggles = torch.tensor(
            [a.toggle for a in actions],
            device=self._backend,
            dtype=torch.bool,
        )
        target_LLA = torch.tensor(
            [
                a.target_location if a.target_location is not None else
                (0., 0.) for a in actions
            ],
            device=self._backend,
            dtype=self._fp_precision,
        )
        with_target = torch.tensor(
            [a.target_location is not None for a in actions],
            device=self._backend,
            dtype=torch.bool,
        )

        self._simulator.take_actions(
            toggles,
            target_LLA,
            with_target,
        )

    def take_actions_with_tensor(
        self,
        task_indices: torch.Tensor,
        tasks: TaskSet,
    ) -> None:
        assert task_indices.shape == (self.num_satellites, )
        valid_tasks = tasks
        with_target = (task_indices != -1) & (task_indices < len(valid_tasks))

        target_location_LLA = self._simulator.tracking_target.new_tensor([
            (0., 0.) if i == -1 or i >= len(tasks) else tasks[i].coordinate
            for i in task_indices
        ])

        new_task_ids = task_indices.new_tensor([
            -1 if i == -1 or i >= len(tasks) else tasks[i].id_
            for i in task_indices
        ])
        old_task_ids = self._task_ids

        reset = old_task_ids != new_task_ids
        self._task_ids = new_task_ids

        sensor_enabled = self._simulator.camera_switch
        toggles = with_target.bitwise_xor(sensor_enabled)
        if reset.any():
            self._reset_integrators(reset)
        self._simulator.take_actions(
            toggles,
            target_location_LLA,
            with_target,
        )

    def step(self) -> None:
        self._simulator_state_dict, _ = self._simulator(
            self._simulator_state_dict
        )
        self.authentic_timer.step()

    def is_visible(self, tasks: TaskSet) -> torch.Tensor:
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
        visibility = (has_access & camera_on
                      & sensor_match).transpose(0, 1).to(
                          torch.get_default_device()
                      )
        return visibility

    def get_earth_rotation(self) -> torch.Tensor:
        earth_ephmeris = self._simulator.get_earth_ephemeris(None)
        return einops.rearrange(
            earth_ephmeris['direction_cosine_matrix_CN'], '1 n m -> n m'
        )

    def get_observation(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        battery_percentages = self._simulator_state_dict['_battery'][
            'stored_charge_percentage']
        reaction_wheel_speeds: torch.Tensor = self._simulator_state_dict[
            '_spacecraft']['_state_effectors']['_reaction_wheels'][
                'dynamic_params']['angular_velocity']
        reaction_wheel_speeds = einops.rearrange(
            reaction_wheel_speeds, 'ns 1 nw -> ns nw'
        )

        position_BP_N = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']['position_BP_N']
        velocity_BP_N = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']['velocity_BP_N']
        orbital_elements = rv2elem_torch(
            constants.MU_EARTH * 1e9,
            position_BP_N,
            velocity_BP_N,
        )
        # orbit_dynamic = torch.stack(
        #     [
        #         orbital_elements['e'],
        #         orbital_elements['a'],
        #         orbital_elements['i'] * R2D,
        #         orbital_elements['Omega'] * R2D,
        #         orbital_elements['omega'] * R2D,
        #         orbital_elements['f'] * R2D,
        #     ],
        #     dim=1,
        # )
        attitude_BN = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']['attitude_BN']

        if battery_percentages.dim() != 1:
            battery_percentages = einops.repeat(
                battery_percentages,
                '... -> ... ns',
                ns=self.num_satellites,
            )
        dynamic_data = torch.cat(
            [
                einops.rearrange(battery_percentages, 'ns -> ns 1'),
                reaction_wheel_speeds / constants.RPM,
                einops.rearrange(orbital_elements['f'] * R2D, 'ns -> ns 1'),
                attitude_BN,
            ],
            dim=1,
        )

        sensor_type, static_data = self._constellation_data.static_to_tensor()
        # new_orbit = torch.stack(
        #     [
        #         orbital_elements['e'],
        #         orbital_elements['a'],
        #         orbital_elements['i'] * R2D,
        #         orbital_elements['Omega'] * R2D,
        #         orbital_elements['omega'] * R2D,
        #     ],
        #     dim=-1,
        # )
        # static_data[..., 13:18] = new_orbit

        data = torch.cat(
            [
                static_data,
                dynamic_data,
            ],
            dim=1,
        )

        return (
            sensor_type,
            self._simulator.camera_switch,
            data,
        )
