from typing import List
import numpy as np

from docent.util.const import kB


class BaseLambdaSampler:
    def __init__(self):
        self.current_lambda = 0


    def update_lambda(self):
        self.current_lambda = self._next_lambda


    def is_stop_iter(self):
        return self.current_lambda == 1.


    def reset_sampler(self):
        self.current_lambda = 0.


class FixedLambdaSampler(BaseLambdaSampler):
    def __init__(
        self,
        lambda_list: List[float],
    ):
        super().__init__()
        lambda_list.sort()
        if lambda_list[0] != 0:
            lambda_list = [0] + lambda_list

        if lambda_list[-1] != 1:
            lambda_list += [1]

        self.lambda_list = lambda_list


    def step_next_lambda(self, **kwargs):
        current_idx = self.lambda_list.index(self.current_lambda)
        next_idx = current_idx + 1
        self._next_lambda = self.lambda_list[next_idx]


class LinearLambdaSampler(FixedLambdaSampler):
    def __init__(
        self,
        n_lambda: int,
    ):
        lambda_list = np.linspace(0, 1, n_lambda).tolist()
        super().__init__(lambda_list=lambda_list)


class QuadratureLambdaSampler(FixedLambdaSampler):
    def __init__(
        self,
        n_lambda: int,
    ):
        x_quad, w_quad = np.polynomial.legendre.leggauss(n_lambda)  # [-1, 1]
        x_quad = ((x_quad + 1) / 2).tolist()  # [0, 1]
        self.w_quad = (w_quad / 2).tolist()  # [0, 1], save for quadrature
        super().__init__(lambda_list=x_quad)


class AdaptiveLambdaSampler(BaseLambdaSampler):
    def __init__(
        self,
        target_overlap: float,
    ):
        super().__init__()
        self.target_overlap = target_overlap
        self.overlap = None


    def step_next_lambda(self, beta, free_energy_list=None):
        lambda_switch, overlap = solve_lambda(
            beta=beta,
            free_energy_list=free_energy_list,
            lambda_ref=self.current_lambda,
            target_overlap=self.target_overlap,
        )
        self._next_lambda = np.clip(lambda_switch, 0, 1)
        self.overlap = overlap


def calculate_boltzmann_weight_at_lambda(
    free_energy_list,
    beta,
    lambda_switch,
    lambda_ref=0,
):
    beta_free = np.array(
        [
            beta * f['internal_energy'] - f['vib_entropy'] / kB
            for f in free_energy_list
        ]
    )
    diff = (lambda_switch - lambda_ref) * beta_free
    diff -= min(diff)
    exp = np.exp(-diff)
    return exp / np.sum(exp)


def calculate_overlap_at_lambda(
    beta,
    lambda_switch,
    lambda_ref=0,
    free_energy_list=None,
):
    if free_energy_list is None:
        return 1.
    prob = calculate_boltzmann_weight_at_lambda(
        free_energy_list, beta, lambda_switch, lambda_ref
    )
    overlap = prob * len(prob)
    overlap[overlap > 1.] = 1.
    return np.mean(overlap)


def solve_lambda(
    beta,
    free_energy_list=None,
    lambda_ref=0.,
    target_overlap=0.7,
):
    CONV = 0.01
    MAX_ATTEMPT = 1000

    if (
        overlap := calculate_overlap_at_lambda(
            beta, 1., lambda_ref, free_energy_list
        )
    ) > target_overlap + CONV:
        return 1., overlap
    lambda_1, lambda_2 = lambda_ref, 1.

    for _ in range(MAX_ATTEMPT):
        lambda_switch = (lambda_1 + lambda_2) / 2
        overlap = calculate_overlap_at_lambda(
            beta, lambda_switch, lambda_ref, free_energy_list
        )
        if abs(overlap - target_overlap) < CONV:
            break
        if (target_overlap - overlap) > 0:
            # if target > overlap, search near lambda_ref
            # change lambda_2
            lambda_2 = lambda_switch
        else:
            # if target < overlap, search far from lambda_ref
            lambda_1 = lambda_switch

    return lambda_switch, overlap

