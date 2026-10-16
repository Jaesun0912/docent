from typing import List
import numpy as np
import math
import pickle
from ase.io import write

from docent.structure.orbit import Crystal
from docent.util.utils import get_statistics_of_list, add_if_is_not_adaptive
from docent.util.calc import get_energy_of_atoms_list, get_free_energy_of_atoms_list
from docent.util.const import kB
from docent.monte_carlo.temperature import (
    ConstantTemperatureScheduler, 
    LinearTemperatureScheduler,
    AdaptiveTemperatureScheduler,
    MultiStepTemperatureScheduler,
    temp2beta,
    beta2temp,
    calculate_boltzmann_weight,
    temp_range_from_mc_params,
    solve_beta
)
from docent.monte_carlo.lambda_switch import (
    FixedLambdaSampler,
    LinearLambdaSampler,
    QuadratureLambdaSampler,
    AdaptiveLambdaSampler,
    calculate_boltzmann_weight_at_lambda,
    calculate_overlap_at_lambda,
)


class Replica:
    def __init__(
        self,
        crystals: List[Crystal],
        save_unique_minima: int = 10,
    ):
        self.crystals = crystals
        self.n_minima = save_unique_minima
        self.unique_minima = {}
        self.rx_attempt = np.zeros(len(crystals))
        self.rx_accept = np.zeros(len(crystals))
        self.mc_attempt = np.zeros(len(crystals))
        self.mc_accept = np.zeros(len(crystals))
        self.mc_step = 0
        self.energy_recorder = [[] for _ in range(len(crystals))]
        self.free_energy_recorder = [[] for _ in range(len(crystals))]
        self.mc_recorder = {
            'energy': {(float('inf'), 0): []},
            'free_energy': {(float('inf'), 0): []},
            'num_resampled': {(float('inf'), 0): []},
        }


    @property
    def cycle(self):
        return self.t_scheduler.cycle


    @cycle.setter
    def _set_cycle(self):
        self.cycle = self.sc.cycle


    def get_current_lambda(self):
        if self.lambda_sampler is None:
            return 0.
        return float(self.lambda_sampler.current_lambda)


    def init_replica(self, calc, free_calc, disorder_only=False):
        # Only called in the initialization of MC step
        atoms_list = []
        for crystal in self.crystals:
            atoms_list.append(crystal.to_ase_atoms(disorder_only=disorder_only))
        energy_list = get_energy_of_atoms_list(atoms_list, calc)
        free_energy_list = (
            get_free_energy_of_atoms_list(
                atoms_list, float('inf'), True, free_calc
            ) if free_calc is not None
            else [dict()] * len(atoms_list)
        )
        self.mc_recorder['n_atoms'] = len(self.crystals[0].to_ase_atoms())
        self.mc_recorder['entropy_conf_inf'] = self.crystals[0].entropy
        for idx, (energy, free_energy, crystal) in enumerate(
            zip(energy_list, free_energy_list, self.crystals)
        ):
            self.mc_recorder['energy'][(float('inf'), 0.0)].append(energy)
            self.mc_recorder['free_energy'][(float('inf'), 0.0)].append(free_energy)
            crystal.energy = energy
            crystal.info.update(free_energy)
            self.energy_recorder[idx].append(energy)
            self.free_energy_recorder[idx].append(free_energy)
            if energy in sorted(energy_list)[:self.n_minima]:
                self.unique_minima[f'R{idx+1}_init'] = crystal.copy()


    def init_t_scheduler_from_config(self, config):
        # Should be called after init_replica
        tconfig = config['mc_params'].copy()
        tconfig['inv_mode'] = tconfig['t_schedule_mode'].lower() == 'beta'
        if config['mc_method'] == 'pt':
            if isinstance(th:=tconfig['t_high'], str) and th.lower() == 'adaptive':
                beta = solve_beta(
                    energy_list=[c.energy for c in self.crystals],
                    overlap=tconfig['pa_overlap'],
                )
                tconfig['t_high'] = beta2temp(beta)
            temperatures = temp_range_from_mc_params(**tconfig)
            self.t_scheduler = ConstantTemperatureScheduler(
                temperatures, tconfig['n_cycles'],
            )

        else:  # pa
            t_high_low_key_pairs, scheduler_list = [], []
            if tconfig.get('lambda_sampler', None) is not None:
                t_high_low_key_pairs += [('t_inf', 't_high')]
            t_high_low_key_pairs += [('t_high', 't_low')]

            for (t_high_key, t_low_key) in t_high_low_key_pairs:
                t_kwargs = {'n_replica': tconfig['n_replicas']}
                if t_high_key == 't_inf' or isinstance(th:=tconfig[t_high_key], str):
                    beta = 0.
                    tconfig[t_high_key] = beta2temp(beta)
                t_kwargs['temperature'] = float(tconfig[t_high_key])

                add_if_is_not_adaptive(tconfig, t_low_key, t_kwargs, 't_low')
                add_if_is_not_adaptive(tconfig, 'n_cycles', t_kwargs)

                if (tmode := tconfig['t_schedule_mode'].lower()) == 'adaptive':
                    t_kwargs['overlap'] = tconfig['pa_overlap']
                    scheduler_list.append(AdaptiveTemperatureScheduler(**t_kwargs))
                else:
                    # t_low always given
                    add_if_is_not_adaptive(tconfig, 'inv_mode', t_kwargs)
                    t_space = temp_range_from_mc_params(**tconfig)
                    t_kwargs['delta'] = (
                        temp2beta(t_space[1]) - temp2beta(t_space[0])
                        if t_kwargs['inv_mode']
                        else t_space[0] - t_space[1]
                    )
                    scheduler_list.append(LinearTemperatureScheduler(**t_kwargs))

            if len(scheduler_list) == 1:
                self.t_scheduler = scheduler_list[0]
            else:
                self.t_scheduler = MultiStepTemperatureScheduler(scheduler_list)


    def init_lambda_sampler_from_config(self, config):
        lconfig = config['mc_params'].copy()
        if (sampler_type:=lconfig.get('lambda_sampler', None)) is None:
            self.lambda_sampler = None
            return

        if sampler_type.lower() == 'fixed':
            self.lambda_sampler = FixedLambdaSampler(lconfig['lambda_list'])
        elif sampler_type.lower() == 'linear':
            self.lambda_sampler = LinearLambdaSampler(lconfig['n_lambda'])
        elif sampler_type.lower() == 'quadrature':
            self.lambda_sampler = QuadratureLambdaSampler(lconfig['n_lambda'])
        elif sampler_type.lower() == 'adaptive':
            self.lambda_sampler = AdaptiveLambdaSampler(lconfig['lambda_overlap'])
        else:
            raise NotImplementedError


    def _get_maximum_minima_energy(self):
        return max([c.energy for c in self.unique_minima.values()])


    def _update_minima_dct(self, crystal, replica_idx):
        max_en = self._get_maximum_minima_energy()
        max_key = [
            key for key, m_crystal in self.unique_minima.items()
            if m_crystal.energy == max_en
        ][0]
        del self.unique_minima[max_key]
        self.unique_minima[f'R{replica_idx+1}C{self.cycle}M{self.mc_step}'] = crystal


    def process_single_mc_step(self, calc, free_calc, disorder_only=False, mode='exchange'):
        exchange_info, new_atoms_list = [], []
        for crystal in self.crystals:
            if mode == 'exchange':
                orbit_idx, i, j, pos_idx = crystal.random_exchange_random_idx()
                exchange_info.append((orbit_idx, j, i, pos_idx))
            else:
                # this will not work!
                crystal.random_permute_random_idx()
            new_atoms_list.append(crystal.to_ase_atoms(disorder_only=disorder_only))

        new_energy_list = get_energy_of_atoms_list(new_atoms_list, calc)
        lambda_switch = self.get_current_lambda()
        require_free_energy_calculation = free_calc is not None and lambda_switch != 0
        new_free_energy_list = (
            get_free_energy_of_atoms_list(
                new_atoms_list, self.t_scheduler.temperatures, True, free_calc
            ) if require_free_energy_calculation
            else [dict()] * len(new_atoms_list)
        )
        minimum_energy_indices = sorted(
            range(len(new_energy_list)), key=lambda i: new_energy_list[i]
        )[:self.n_minima]
        for idx, (
                temperature,
                new_energy,
                new_free_energy_info,
                crystal,
                exchange_idx
            ) in enumerate(
                zip(
                    self.t_scheduler.temperatures,
                    new_energy_list,
                    new_free_energy_list,
                    self.crystals,
                    exchange_info
                )
        ):
            #new_crystal.energy = new_energy
            if (
                idx in minimum_energy_indices
                and new_energy < self._get_maximum_minima_energy()
            ):
                #self._update_minima_dct.copy()deepcopy(new_crystal), idx)
                minima = crystal.copy()
                minima.energy = new_energy
                self._update_minima_dct(minima, idx)

            old_energy = crystal.energy
            old_free_energy_info = crystal.info
            if 'vib_entropy' in new_free_energy_info:
                diff = (
                    (new_energy - old_energy) * temp2beta(temperature)
                    + (
                        new_free_energy_info['internal_energy']
                        - old_free_energy_info['internal_energy']
                    ) * temp2beta(temperature) * lambda_switch
                    + (
                        old_free_energy_info['vib_entropy']
                        - new_free_energy_info['vib_entropy']
                    ) / kB * lambda_switch
                )
            else:
                diff = (new_energy - old_energy) * temp2beta(temperature)
            prob = 1 if diff < 0 \
                else math.exp(-diff)
            rand_num = np.random.rand()  # [0, 1)
            self.mc_attempt[idx] += 1

            if prob > rand_num:
                #accepted_crystals.append(deepcopy(new_crystal))
                #accepted_crystals.append(new_crystal)
                crystal.energy = new_energy
                crystal.info.update(new_free_energy_info)
                self.mc_accept[idx] += 1
                self.energy_recorder[idx].append(new_energy)
                self.free_energy_recorder[idx].append(new_free_energy_info)
                stamp = 'acc'
            else:
                #accepted_crystals.append(deepcopy(old_crystal))
                #accepted_crystals.append(old_crystal)
                crystal.exchange_idx_deterministic_idx(*exchange_idx)
                self.energy_recorder[idx].append(old_energy)
                self.free_energy_recorder[idx].append(old_free_energy_info)
                stamp = 'dec'

        self.mc_step += 1


    def infer_free_energy(self, free_calc, disorder_only=False):
        temp = self.t_scheduler.temperatures[0]
        atoms_list = []
        for crystal in self.crystals:
            atoms_list.append(crystal.to_ase_atoms(disorder_only=disorder_only))
        free_energy_list = get_free_energy_of_atoms_list(
                atoms_list, temp, True, free_calc,
        )
        for idx, (free_energy, crystal) in enumerate(
            zip(free_energy_list, self.crystals)
        ):
            #self.mc_recorder['free_energy'][(float('inf'), 0)].append(free_energy)
            crystal.info.update(free_energy)
            self.free_energy_recorder[idx].append(free_energy)


    def process_parallel_tempering(self):
        for idx in range((self.cycle-1)%2, len(self.crystals)-1, 2):
            self.rx_attempt[idx] += 1
            self.rx_attempt[idx+1] += 1
            t_i = self.t_scheduler.temperatures[idx]
            t_j = self.t_scheduler.temperatures[idx+1]
            beta_i = temp2beta(t_i)
            beta_j = temp2beta(t_j)
            e_i = self.crystals[idx].energy
            e_j = self.crystals[idx+1].energy
            prob = 1 if (val := (e_i-e_j)*(beta_i-beta_j)) > 0 \
                else math.exp(val)
            rand_num = np.random.rand()
            if prob > rand_num:
                self.crystals[idx], self.crystals[idx+1] =\
                    self.crystals[idx+1], self.crystals[idx]
                self.rx_accept[idx] += 1
                self.rx_accept[idx+1] += 1


    def process_population_annealing(self, free_calc=None):
        probs = []
        num_rep = len(self.crystals)
        t_orig = self.t_scheduler.temperatures[0]
        lambda_switch = self.get_current_lambda()
        require_free_energy_calculation = free_calc is not None and lambda_switch != 0

        self.t_scheduler.step_next_temperature(
            [c.energy for c in self.crystals],
            [c.info for c in self.crystals] if require_free_energy_calculation else None,
        )
        t_new = self.t_scheduler._next_temperatures[0]


        new_free_energy_list = None

        if require_free_energy_calculation:
            atoms_list = [crystal.to_ase_atoms() for crystal in self.crystals]
            new_free_energy_list = get_free_energy_of_atoms_list(
                atoms_list, t_new, True, free_calc
            )
        #self.resampled_free_energy_recorder = new_free_energy_list
        probs = calculate_boltzmann_weight(
            energy_list=[c.energy for c in self.crystals],
            beta=temp2beta(t_new),
            beta_ref=temp2beta(t_orig),
            free_energy_list=new_free_energy_list,
            ref_free_energy_list=[c.info for c in self.crystals],
        )
        n_samples = np.array([math.floor(num_rep*p) for p in probs])
        assert num_rep >= sum(n_samples)
        n_samples += np.random.multinomial(num_rep-sum(n_samples), probs)
        assert num_rep == sum(n_samples)
        crystals = []
        for idx, n in enumerate(n_samples):
            for _ in range(n):
                resampled = self.crystals[idx].copy()
                new_free_energy = (
                    dict() if new_free_energy_list is None
                    else new_free_energy_list[idx]
                )
                resampled.info.update(new_free_energy)
                crystals.append(resampled)
            self.rx_accept[idx] = n
            self.rx_attempt[idx] = num_rep
        self.crystals = crystals


    def require_mc_for_lambda_switching(self, skip_mc_overlap):
        if self.lambda_sampler is None:
            return None

        if len(self.t_scheduler) == 2:  # T_inf -> T_h -> T_l
            if sum(self.t_scheduler._get_stop_iter_list()) == 0:
                # T_inf -> T_h stage
                return None

        beta = temp2beta(self.t_scheduler.temperatures[0])
        free_energy_list = [c.info for c in self.crystals]
        current_lambda = self.lambda_sampler.current_lambda
        overlap = calculate_overlap_at_lambda(  # overlap of lambda = 0 and 1
            beta=beta,
            lambda_switch=1-current_lambda,
            lambda_ref=current_lambda,
            free_energy_list=free_energy_list,
        )

        print(overlap)

        return overlap < skip_mc_overlap


    def process_lambda_resampling(self):
        probs = []
        num_rep = len(self.crystals)
        lambda_orig = self.lambda_sampler.current_lambda
        beta = temp2beta(self.t_scheduler.temperatures[0])

        free_energy_list = [c.info for c in self.crystals]

        self.lambda_sampler.step_next_lambda(
            beta, free_energy_list,
        )
        probs = calculate_boltzmann_weight_at_lambda(
            free_energy_list=free_energy_list,
            beta=beta,
            lambda_switch=self.lambda_sampler._next_lambda,
            lambda_ref=lambda_orig,
        )
        n_samples = np.array([math.floor(num_rep*p) for p in probs])
        assert num_rep >= sum(n_samples)
        n_samples += np.random.multinomial(num_rep-sum(n_samples), probs)
        assert num_rep == sum(n_samples)
        crystals = []
        for idx, n in enumerate(n_samples):
            for _ in range(n):
                resampled = self.crystals[idx].copy()
                # resampled.info.update(new_free_energy_list[idx])
                crystals.append(resampled)
            self.rx_accept[idx] = n
            self.rx_attempt[idx] = num_rep
        self.crystals = crystals


    def get_energy_statistics(self):
        stat_dct = {}
        final = []
        for idx, energy_list in enumerate(self.energy_recorder):
            final.append(energy_list[-1])
            stat_dct[f'R{idx+1}'] = get_statistics_of_list(energy_list)
        stat_dct['Final'] = get_statistics_of_list(final)
        return stat_dct


    def get_accept_statistics(self):
        stat_dct = {'Final': {'MC accept': '----', 'RX accept': '----'}}
        for idx, (mc_acc, mc_att, rx_acc, rx_att) in enumerate(
            zip(
                self.mc_accept,
                self.mc_attempt,
                self.rx_accept,
                self.rx_attempt
            )
        ):
            mc_string = f'{int(mc_acc)}/{int(mc_att)}' if mc_att != 0 else '----'
            rx_string = f'{int(rx_acc)}/{int(rx_att)}' if rx_att != 0 else '----'
            stat_dct[f'R{idx+1}'] = {'MC accept': mc_string, 'RX accept': rx_string}
        return stat_dct


    def get_temperature_lambda_statistics(self, mc_method='pa'):
        stat_dct = {}
        current_lambda = '----' if mc_method != 'pa' else self.get_current_lambda()
        final_t = '----' if mc_method != 'pa' else self.t_scheduler.temperatures[0]
        stat_dct['Final'] = {'T (K)': final_t, 'lambda': current_lambda}

        for idx, t in enumerate(self.t_scheduler.temperatures):
            stat_dct[f'R{idx+1}'] = {'T (K)': t, 'lambda': current_lambda}

        return stat_dct


    def get_statistics(self, mc_method='pa'):
        en_stat_dct = self.get_energy_statistics()
        acc_stat_dct = self.get_accept_statistics()
        t_lambda_dct = self.get_temperature_lambda_statistics(mc_method)
        for key, val in en_stat_dct.items():
            val.update(acc_stat_dct[key])
            val.update(t_lambda_dct[key])

        return en_stat_dct


    def update_mc_result(self):
        current_lambda = round(self.get_current_lambda(), 3)
        for (
            temperature,
            energy_list,
            free_energy_list,
            n_resampled,
        ) in zip(
            self.t_scheduler.temperatures,
            self.energy_recorder,
            self.free_energy_recorder,
            self.rx_accept,
        ):
            temperature = round(temperature, 3)
            key = (temperature, current_lambda)

            if key not in self.mc_recorder['energy']:
                self.mc_recorder['energy'][key] = []
                self.mc_recorder['free_energy'][key] = []
                self.mc_recorder['num_resampled'][key] = []
            self.mc_recorder['energy'][key].append(energy_list[-1])
            self.mc_recorder['free_energy'][key].append(free_energy_list[-1])
            self.mc_recorder['num_resampled'][key].append(n_resampled)


    def prepare_next_cycle(self):
        self.mc_step = 0
        self.mc_attempt = np.zeros(len(self.crystals))
        self.mc_accept = np.zeros(len(self.crystals))
        self.energy_recorder = [[] for _ in range(len(self.crystals))]
        self.free_energy_recorder = [[] for _ in range(len(self.crystals))]


    def save_replica_ase_atoms(self, fpath):
        atoms_list = []
        for idx, crystal in enumerate(self.crystals):
            info = {
                'replica_num': idx+1,
                'mc_step': self.mc_step,
                'cycle': self.cycle,
            }
            crystal.info.update(info)
            atoms = crystal.to_ase_atoms()
            atoms_list.append(atoms)

        write(fpath, atoms_list)


    def save_minima_ase_atoms(self, fpath):
        atoms_list = []
        for key, crystal in self.unique_minima.items():
            crystal.info.update({'src': key})
            atoms = crystal.to_ase_atoms()
            atoms_list.append(atoms)

        write(fpath, atoms_list)

    def save_mc_result(self, fpath):
        with open(fpath, 'wb') as f:
            pickle.dump(self.mc_recorder, f)

