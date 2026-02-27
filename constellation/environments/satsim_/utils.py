from datetime import datetime

import torch


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

    axis = implied_sensor_direction.unsqueeze(0).cross(
        position_PB_N_unit, dim=-1
    )
    axis = torch.nn.functional.normalize(axis, dim=-1)

    attitude_BN = axis * torch.tan(angle / 4.)
    return attitude_BN


def is_utc_datetime_str(input_str, formats=None):
    default_formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S+00:00",
        "%Y-%m-%d %H:%M:%S+00:00",
    ]
    target_formats = formats if formats else default_formats

    for fmt in target_formats:
        try:
            dt = datetime.strptime(input_str, fmt)
            return True
        except ValueError:

            continue
    return False


def convert_to_utc(datetime_str: str):
    naive_dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
    utc_dt = naive_dt

    utc_str_with_z = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    utc_str_with_offset = utc_dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    return utc_str_with_z
