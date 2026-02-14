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
