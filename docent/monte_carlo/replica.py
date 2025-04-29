from typing import List
import numpy as np
import math
from copy import deepcopy
from ase.io import write

from docent.structure.orbit import Crystal
from docent.util.utils import get_statistics_of_list, add_if_is_not_adaptive
from docent.util.calc import get_energy_of_atoms_list
from docent.util.const import kB
from docent.monte_carlo.temperature import (
    ConstantTemperatureScheduler, 
    LinearTemperatureScheduler,
    AdaptiveTemperatureScheduler,
    temp2beta,
    beta2temp,
    calculate_boltzmann_weight,
    temp_range_from_mc_params,
    solve_beta
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


    @property
    def cycle(self):
        return self.t_scheduler.cycle


    @cycle.setter
    def _set_cycle(self):
        self.cycle = self.sc.cycle


    def calc_energy_of_replica(self, calc):
        # Only called in the initialization of MC step
        atoms_list = []
        for crystal in self.crystals:
            atoms_list.append(crystal.to_ase_atoms())
        energy_list = get_energy_of_atoms_list(atoms_list, calc)
        for idx, (energy, crystal) in enumerate(zip(energy_list, self.crystals)):
            crystal.energy = energy
            self.energy_recorder[idx].append(energy)
            if energy in sorted(energy_list)[:self.n_minima]:
                self.unique_minima[f'R{idx+1}_init'] = deepcopy(crystal)


    def init_t_scheduler_from_config(self, config):
        # Should be called after calc_energy_of_replica
        tconfig = config['mc_params'].copy()
        tconfig['inv_mode'] = tconfig['t_schedule_mode'].lower() == 'beta'
        if config['mc_method'] == 'pt':
            temperatures = temp_range_from_mc_params(**tconfig)
            self.t_scheduler = ConstantTemperatureScheduler(
                temperatures, tconfig['n_cycles'],
            )

        else:  # pa
            t_kwargs = {'n_replica': tconfig['n_replicas']}
            if isinstance(th:=tconfig['t_high'], str) and th.lower() == 'adaptive':
                beta = solve_beta(
                    energy_list=[c.energy for c in self.crystals],
                    overlap=tconfig['pa_overlap'],
                )
                tconfig['t_high'] = beta2temp(beta)
                print(beta, tconfig['t_high'], [c.energy for c in self.crystals], flush=True)
            t_kwargs['temperature'] = float(tconfig['t_high'])

            add_if_is_not_adaptive(tconfig, 't_low', t_kwargs)
            add_if_is_not_adaptive(tconfig, 'n_cycles', t_kwargs)

            if (tmode := tconfig['t_schedule_mode'].lower()) == 'adaptive':
                t_kwargs['overlap'] = tconfig['pa_overlap']
                self.t_scheduler = AdaptiveTemperatureScheduler(**t_kwargs)
            else:
                # t_low always given
                add_if_is_not_adaptive(tconfig, 'inv_mode', t_kwargs)
                t_space = temp_range_from_mc_params(**tconfig)
                t_kwargs['delta'] = (
                    temp2beta(t_space[1]) - temp2beta(t_space[0])
                    if t_kwargs['inv_mode']
                    else t_space[0] - t_space[1]
                )
                self.t_scheduler = LinearTemperatureScheduler(**t_kwargs)


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


    def process_single_mc_step(self, calc, mode='exchange'):
        old_crystals, new_crystals, new_atoms_list = [], [], []
        for crystal, temperature in zip(self.crystals, self.t_scheduler.temperatures):
            old_crystal = deepcopy(crystal)
            old_crystals.append(old_crystal)
            if mode == 'exchange':
                crystal.random_exchange_random_idx()
            else:
                crystal.random_permute_random_idx()
            new_crystal = deepcopy(crystal)
            new_crystals.append(new_crystal)
            new_atoms_list.append(new_crystal.to_ase_atoms())

        new_energy_list = get_energy_of_atoms_list(new_atoms_list, calc)
        accepted_crystals = []
        for idx, (temperature, new_energy, new_crystal, old_crystal) in enumerate(
            zip(self.t_scheduler.temperatures, new_energy_list, new_crystals, old_crystals)
        ):
            new_crystal.energy = new_energy
            if new_energy < self._get_maximum_minima_energy():
                self._update_minima_dct(deepcopy(new_crystal), idx)

            old_energy = old_crystal.energy
            prob = 1 if new_energy < old_energy \
                else math.exp(-(new_energy-old_energy)*temp2beta(temperature))
            rand_num = np.random.rand()  # [0, 1)
            self.mc_attempt[idx] += 1
            if prob > rand_num:
                accepted_crystals.append(deepcopy(new_crystal))
                self.mc_accept[idx] += 1
                self.energy_recorder[idx].append(new_energy)
            else:
                accepted_crystals.append(deepcopy(old_crystal))
                self.energy_recorder[idx].append(old_energy)

        self.mc_step += 1
        self.crystals = accepted_crystals


    def process_parallel_tempering(self):
        for idx in range(self.cycle%2, len(self.crystals)-1, 2):
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


    def process_population_annealing(self):
        probs = []
        num_rep = len(self.crystals)
        t_orig = self.t_scheduler.temperatures[0]
        self.t_scheduler.step_next_temperature([c.energy for c in self.crystals])
        t_new = self.t_scheduler.temperatures[0]
        probs = calculate_boltzmann_weight(
            energy_list=[c.energy for c in self.crystals],
            beta=temp2beta(t_new),
            beta_ref=temp2beta(t_orig),
        )
        n_samples = np.array([math.floor(num_rep*p) for p in probs])
        assert num_rep >= sum(n_samples)
        n_samples += np.random.multinomial(num_rep-sum(n_samples), probs)
        assert num_rep == sum(n_samples)
        crystals = []
        for idx, n in enumerate(n_samples):
            crystals += [deepcopy(self.crystals[idx]) for _ in range(n)]
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


    def get_statistics(self):
        en_stat_dct = self.get_energy_statistics()
        acc_stat_dct = self.get_accept_statistics()
        for key, val in en_stat_dct.items():
            val.update(acc_stat_dct[key])
        return en_stat_dct


    def prepare_next_cycle(self):
        self.mc_step = 0
        self.t_scheduler.update_cycle()
        self.mc_attempt = np.zeros(len(self.crystals))
        self.mc_accept = np.zeros(len(self.crystals))
        self.energy_recorder = [[] for _ in range(len(self.crystals))]


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

