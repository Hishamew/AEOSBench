import torch

from .base import BaseEvaluator
from .constellation import BaseConstellationStateDict
from .environment import AttitudeControlEnvironment, Loss
from .registry import EvaluatorRegistry


@EvaluatorRegistry.register_()
class PowerEvaluator(BaseEvaluator):

    def __call__(
        self,
        environment: AttitudeControlEnvironment,
        state_dict: BaseConstellationStateDict,
        loss_info: Loss,
    ) -> None:
        self._most_recent_power_cost = state_dict['_battery'][
            'stored_charge_percentage']

    def _evaluate(self, ) -> tuple[dict[str, float], str]:
        power_reward_mean = self._most_recent_power_cost.mean().item()
        power_reward_std = self._most_recent_power_cost.std().item()

        evaluate_result = dict(
            power_reward_mean=power_reward_mean,
            power_reward_std=power_reward_std,
        )
        log = (f"Power Reward - Mean: {power_reward_mean:.8f}, \n"
               f"Std: {power_reward_std:.8f}")

        return evaluate_result, log


@EvaluatorRegistry.register_('AIE', 'AttitudeErrorEvaluator')
class AverageIntegralAttitudeErrorEvaluator(BaseEvaluator):

    def _init_state(self) -> None:
        self._attitude_reward = 0.
        self._reward_count = 0.

    def __call__(
        self,
        environment: AttitudeControlEnvironment,
        state_dict: BaseConstellationStateDict,
        loss_info: Loss,
    ) -> None:

        attitude_penalty = environment.calculate_real_attitude_error()
        attitude_penalty = 4 * torch.atan(attitude_penalty.norm(dim=-1))
        mask = loss_info['loss_mask']

        self._attitude_reward += attitude_penalty * mask
        self._reward_count += mask

    def _evaluate(self) -> tuple[dict[str, float], str]:
        att_benchmark = -self._attitude_reward / self._reward_count

        attitude_reward_mean = (att_benchmark).mean().item()
        attitude_reward_std = (att_benchmark).std().item()

        evaluate_result = dict(
            attitude_reward_mean=attitude_reward_mean,
            attitude_reward_std=attitude_reward_std,
        )
        log = (f"Attitude Reward - Mean: {attitude_reward_mean:.8f}, \n"
               f"Std: {attitude_reward_std:.8f}")

        return evaluate_result, log


@EvaluatorRegistry.register_('TDPA')
class TimeDomainPerformanceEvaluator(BaseEvaluator):

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self._threshold = threshold

    def _init_state(self) -> None:
        self._cached_attitude_error = []

    def __call__(
        self,
        environment: AttitudeControlEnvironment,
        state_dict: BaseConstellationStateDict,
        loss_info: Loss,
    ) -> None:

        attitude_penalty = environment.calculate_real_attitude_error()
        attitude_error_in_rad = 4 * torch.atan(attitude_penalty.norm(dim=-1))

        self._cached_attitude_error.append(attitude_error_in_rad)

    def _compute_time_maneuver(
        self,
        curves: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        sequence_length = curves.shape[1]
        mask = curves > self._threshold
        indices = torch.arange(sequence_length,
                               device=curves.device).unsqueeze(0) + 1
        masked_indices = mask * indices
        time_maneuver = masked_indices.max(dim=1)[0]

        failed_mask = mask.all(dim=-1)

        time_maneuver[failed_mask] = sequence_length

        indices = indices - 1
        aie_mask = indices >= time_maneuver.unsqueeze(dim=1)
        average_integral_error = -(aie_mask * curves).sum(1) / (
            sequence_length + 1 - time_maneuver)

        average_integral_error[failed_mask] = -curves[failed_mask, -1]

        return time_maneuver, average_integral_error

    def _compute_time_response(
        self,
        curves: torch.Tensor,
    ) -> torch.Tensor | None:
        mask = curves <= self._threshold
        time_response = torch.argmax(mask.int(), dim=1)
        if not mask.any(dim=1).all():
            return None

        return time_response

    def _evaluate(self) -> tuple[dict[str, float], str]:
        attitude_errors = torch.stack(self._cached_attitude_error, dim=1)

        time_response = self._compute_time_response(attitude_errors)
        time_maneuver, average_integral_error = self._compute_time_maneuver(
            attitude_errors)

        log = ""
        evaluation_result = dict()
        if time_response is not None:
            time_response_mean = time_response.float().mean().item()
            time_response_std = time_response.float().std().item()
            evaluation_result.update(
                time_maneuver_mean=time_response_mean,
                time_maneuver_std=time_response_std,
            )
            log += (
                f"Attitude response time at {self._threshold} rad threshold: \n"
                f"Mean: {time_response_mean:.8f}, \n"
                f"Std: {time_response_std:.8f}\n")
        else:
            log += "Model failed to response to attitude error.\n"

        if time_maneuver is not None:
            time_maneuver_mean = time_maneuver.float().mean().item()
            time_maneuver_std = time_maneuver.float().std().item()

            steady_state_error_mean = average_integral_error.mean().item()
            steady_state_error_std = average_integral_error.std().item()
            evaluation_result.update(
                time_maneuver_mean=time_maneuver_mean,
                time_maneuver_std=time_maneuver_std,
                steady_state_error_mean=steady_state_error_mean,
                steady_state_error_std=steady_state_error_std,
            )

            log += (
                f"Attitude maneuver time at {self._threshold} rad threshold: \n"
                f"Mean: {time_maneuver_mean:.8f}, \n"
                f"Std: {time_maneuver_std:.8f}. \n"
                f"Average integral error at {self._threshold} rad threshold: \n"
                f"Mean: {steady_state_error_mean:.8f} \n"
                f"Std: {steady_state_error_std:.8f}\n")
        else:
            log += "Model failed to maneuver attitude.\n"

        if time_response is not None and time_maneuver is not None:
            time_overmodulation = time_maneuver - time_response

            time_overmodulation_mean = time_overmodulation.float().mean().item(
            )
            time_overmodulation_std = time_overmodulation.float().std().item()
            evaluation_result.update(
                time_overmodulation_mean=time_overmodulation_mean,
                time_overmodulation_std=time_overmodulation_std,
            )

            log += (f"Overmodulation time:\n"
                    f"Mean: {time_overmodulation_mean:.8f}, \n"
                    f"Std: {time_overmodulation_std:.8f}.\n")

            tr_expand = time_response.unsqueeze(1)
            tm_expand = time_maneuver.unsqueeze(1)
            seq_len = attitude_errors.size(1)
            seq_indices = torch.arange(
                seq_len, device=attitude_errors.device).unsqueeze(0)
            mask = (seq_indices >= tr_expand) & (seq_indices < tm_expand)

            masked_errors = attitude_errors.masked_fill(~mask, -torch.inf)
            max_error = masked_errors.max(dim=1)[0]

            sigma = torch.clamp(max_error - self._threshold,
                                min=0) / self._threshold * 100
            sigma_mean = sigma.mean().item()
            sigma_std = sigma.std().item()

            evaluation_result.update(sigma_percent_mean=sigma_mean,
                                     sigma_percent_std=sigma_std)
            log += (f"sigma% (Overthreshold Attitude Error Ratio): \n"
                    f"Mean: {sigma_mean:.8f}%, \n"
                    f"Std: {sigma_std:.8f}%")

        return evaluation_result, log
