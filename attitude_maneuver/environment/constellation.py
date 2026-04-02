__all__ = [
    'AttitudeControlConstellation',
    'AttitudeControlConstellationStateDict',
]
import torch

from constellation.environments import SatsimConstellation
from constellation.environments.satsim_.constellation import (
    SatsimConstellationStateDict,
)

from .learnable_mrp_control import (
    LearnableMRPControl,
    LearnableMRPControlStateDict,
)


class AttitudeControlConstellationStateDict(SatsimConstellationStateDict):
    _mrp_control: LearnableMRPControlStateDict


class AttitudeControlConstellation(SatsimConstellation):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mrp_control = LearnableMRPControl(timer=self._timer)

    def configure_pid(
        self,
        state_dict: SatsimConstellationStateDict,
        k: torch.Tensor,
        ki: torch.Tensor,
        p: torch.Tensor,
        integral_limit: torch.Tensor,
        full_grad: bool = False,
    ) -> SatsimConstellationStateDict:
        reference = state_dict['_mrp_control']['integral_sigma']
        mrp_control_state_dict = LearnableMRPControlStateDict(
            integral_sigma=reference.new_zeros((k.size(0), 3)),
            k=k.to(reference),
            ki=ki.to(reference),
            p=p.to(reference),
            integral_limit=integral_limit.to(reference),
        )
        state_dict['_mrp_control'] = mrp_control_state_dict

        if full_grad:
            return state_dict

        def detach_forward_hook(module, args, kwargs: dict[str, torch.Tensor]):
            state_dict: LearnableMRPControlStateDict = args[0]
            integral_sigma = state_dict['integral_sigma']
            state_dict['integral_sigma'] = integral_sigma.detach()

            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.detach()

            return tuple([state_dict]), kwargs

        self.register_forward_pre_hook(detach_forward_hook, with_kwargs=True)

        return state_dict
