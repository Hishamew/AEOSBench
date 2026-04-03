__all__ = [
    'compatible_load',
]
import torch
from torch import nn


def compatible_load(
    model: nn.Module,
    state_dict: dict,
    ki_manual_normalize: bool = False,
) -> None:
    """
    Load the state dict into the model, make compatible for both
    with_integral_limit and without_integral_limit versions of the model.

    Args:
        model (nn.Module): The model to load the state dict into.
        state_dict (dict): The state dict to load.
    """
    model_state_dict = model.state_dict()
    output_dim = model_state_dict['mlp.2.weight'].shape[0]

    state_dict_output_dim = state_dict['mlp.2.weight'].shape[0]
    if output_dim == state_dict_output_dim:
        # The output dim matches, we can load directly
        model.load_state_dict(state_dict)
        return

    if ki_manual_normalize:
        state_dict['mlp.2.weight'][1, :] *= 1e-4
        state_dict['mlp.2.bias'][1] *= 1e-4

    if output_dim == 4 and state_dict_output_dim == 3:
        state_dict['mlp.2.weight'] = torch.cat(
            [
                state_dict['mlp.2.weight'],
                torch.zeros(1, state_dict['mlp.2.weight'].shape[1])
            ],
            dim=0,
        )
        state_dict['mlp.2.bias'] = torch.cat(
            [
                state_dict['mlp.2.bias'],
                torch.tensor([1e-3], dtype=state_dict['mlp.2.bias'].dtype)
            ],
            dim=0,
        )
    elif output_dim == 3 and state_dict_output_dim == 4:
        weight = state_dict['mlp.2.weight']
        bias = state_dict['mlp.2.bias']
        # Remove the last row of the weight and bias
        state_dict['mlp.2.weight'] = weight[:-1, :]
        state_dict['mlp.2.bias'] = bias[:-1]

    model.load_state_dict(state_dict, strict=False)
