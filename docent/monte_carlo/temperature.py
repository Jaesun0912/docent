from typing import List, Union
import numpy as np

from docent.util.const import kB, T_MIN


class BaseScheduler:
    def __init__(
        self,
        n_cycles: int = float('inf'),
        t_low: float = -1.,
    ):
        self.max_cycle = n_cycles
        self.t_low = t_low
        if 1/self.max_cycle == 0. and t_low < 0.:
            raise ValueError(
                'Infinite temperature schedule!'
                + ' You should specify either t_low > 0 or n_cycles < inf.'
            )

        self.cycle = 0


    def update_cycle(self):
        self.cycle += 1


    def step_next_temperature(self):
        raise NotImplementedError('One should specify this method')


    def is_stop_iter(self):
        raise NotImplementedError('One should specify this method')


class ConstantTemperatureScheduler(BaseScheduler):
    """ Constant temperature used in PT"""
    def __init__(
        self,
        temperatures: List[Union[float, int]],
        n_cycles: int = float('inf'),
    ):
        super().__init__(n_cycles=n_cycles)
        self.temperatures = temperatures


    def step_next_temperature(self, energy_list=None):
        pass

    def is_stop_iter(self):
        return self.cycle >= self.max_cycle


class LinearTemperatureScheduler(BaseScheduler):
    """ Deterministic, t_low always given """
    def __init__(
        self,
        delta: float,
        n_replica: int,
        temperature: float = float('inf'),
        n_cycles: int = float('inf'),
        t_low: float = -1.,
        inv_mode: bool = False,
    ):
        super().__init__(
            n_cycles=n_cycles,
            t_low=t_low,
        )
        assert delta > 0.
        self.temperatures = [temperature]*n_replica
        self.delta = delta
        self.inv_mode = inv_mode


    def step_next_temperature(self, energy_list=None):
        if self.inv_mode:
            beta = temp2beta(self.temperatures[0])
            beta += self.delta
            self.temperatures = [beta2temp(beta)]*len(self.temperatures)
        else:
            self.temperatures = [t-self.delta for t in self.temperatures]
        if self.cycle == self.max_cycle - 1:
            self.temperatures = [self.t_low]*len(self.temperatures)


    def is_stop_iter(self):
        return self.cycle >= self.max_cycle


class AdaptiveTemperatureScheduler(BaseScheduler):
    def __init__(
        self,
        overlap: float,
        n_replica: int,
        temperature: float = float('inf'),
        n_cycles: int = float('inf'),
        t_low: float = -1.,
    ):
        super().__init__(
            n_cycles=n_cycles,
            t_low=t_low,
        )
        self.temperatures = [temperature]*n_replica
        self.overlap = overlap


    def step_next_temperature(self, energy_list):
        beta_ref = temp2beta(self.temperatures[0])
        beta = solve_beta(energy_list, beta_ref, self.overlap)
        self.temperatures = [max(beta2temp(beta), self.t_low)]*len(self.temperatures)


    def is_stop_iter(self):
        return self.cycle >= self.max_cycle or self.temperatures[0] <= self.t_low
        

def temp2beta(temp):
    beta = float('inf') if temp == 0. else 1/kB/temp
    return beta


def beta2temp(beta):
    return temp2beta(beta)


def calculate_boltzmann_weight(
    energy_list,
    beta,
    beta_ref=0.
):
    energy_list = np.array(energy_list) - min(energy_list)
    # beta > beta_ref, energy_list > 0
    exp = np.exp(-(beta-beta_ref)*energy_list)
    return exp / sum(exp)


def calculate_overlap(
    energy_list,
    beta,
    beta_ref=0.
):
    boltz_w = calculate_boltzmann_weight(energy_list, beta, beta_ref)
    boltz = boltz_w * len(boltz_w)
    boltz[boltz > 1.] = 1.
    return np.mean(boltz)


def solve_beta(
    energy_list,
    beta_ref=0.,
    overlap=0.7,
):
    CONV = 0.01
    MAX_ATTEMPT = 1000
    # binary search
    beta_1 = beta_ref  # overlap = 1
    beta_2 = temp2beta(T_MIN)  # overlap ~ 0

    if calculate_overlap(energy_list, beta_2, beta_ref) > overlap + CONV:
        # Solution in T < T_MIN (= 1 K), usually unphysical
        return beta_2

    for _ in range(MAX_ATTEMPT):
        beta = (beta_1 + beta_2)/2
        alpha = calculate_overlap(energy_list, beta, beta_ref)
        if abs(overlap-alpha) < CONV:
            break
        if overlap - alpha > 0:
            beta_2 = beta  # search lower T
        else:
            beta_1 = beta  # search higher T

    return beta


def temp_range_from_mc_params(
    n_temperatures: int,
    t_low: float = 0.,
    t_high: float = float('inf'),
    inv_mode: bool = True,
    **kwargs
):
    if inv_mode:
        assert t_low > 0. # inv temp mode, t_low must be finite
        return 1/np.linspace(1/float(t_high), 1/float(t_low), n_temperatures)
    else:
        assert 1/t_high > 0. # temp mode, t_high must be finite
        return np.linspace(float(t_high), float(t_low), n_temperatures)

