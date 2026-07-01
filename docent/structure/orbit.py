import numpy as np
import math
import itertools
from typing import Union, List, Dict
from tqdm import tqdm
from numba import jit

import pickle

from docent.structure.site import (
    VirtualSite,
    CombinedSite,
    group_eq_sites,
    sites_from_pymatgen
)
from docent.structure.lattice import(
    get_possible_lattice_transforms,
    calc_abc,
    calc_cubicity
)
from docent.util.utils import (
    combination,
    get_occupation_dict,
)
from docent.util.const import FULLY_OCCUPY, MAX_SUPERCELL_MUL

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
#from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element


def _get_total_by_elem(label_dict):
    total = {}
    for info in label_dict.values():
        for elem, prop in info.items():
            if elem not in total:
                total[elem] = 0.
            total[elem] += prop
    return total


class Orbit:
    def __init__(
        self,
        eq_sites: Union[List[VirtualSite], List[CombinedSite]],
        occupation_dict: Dict[str, Dict[str, int]],  # {'Mg1': {'Mg': 4, 'Ca': 2}}
    ):
        self.eq_sites = eq_sites
        self.occupation_dict = occupation_dict
        self.occupation_by_elem = _get_total_by_elem(occupation_dict)
        total_occ = sum(self.occupation_by_elem.values())
        #total_sites = sum([len(site) for site in eq_sites])
        self.multiplicity = len(eq_sites)
        self.has_substitutional = len(self.occupation_by_elem) > 1
        self.has_vacancy = total_occ < self.multiplicity
        self.liquid_like = total_occ > self.multiplicity
        self.is_combined = isinstance(eq_sites[0], CombinedSite)
        self.is_disordered = \
            self.has_substitutional or self.has_vacancy or self.is_combined
        self._allowed_exchanges = None


    def __str__(self):
        name = self.get_disorder_symbol()
        site = str(self.eq_sites[0])
        return f'{len(self.eq_sites)}x{site} {name} {self.occupation_dict}'


    def copy(self):
        orbit = self.__class__(
            eq_sites = [site.copy() for site in self.eq_sites],
            occupation_dict = self.occupation_dict.copy()
        )
        orbit._allowed_exchanges = self._allowed_exchanges.copy() if self._allowed_exchanges is not None else None
        return orbit


    def calculate_entropy(self):
        if self.liquid_like:  # can not calculate entropy for liquid-like orbit
            return np.nan

        fractions = [n/self.multiplicity for n in self.occupation_by_elem.values()]
        if self.has_vacancy:
            fractions.append(1-sum(self.occupation_by_elem.values())/self.multiplicity)
        entropy = 0
        for f in fractions:
            entropy -= f*math.log(f)
        return entropy * self.multiplicity


    def get_disorder_symbol(self):
        name = []
        if self.has_substitutional:
            name.append('S')
        if self.has_vacancy:
            name.append('V')
        if self.is_combined:
            name.append('P')
        if not self.is_disordered:
            name.append('O')
        name = ''.join(sorted(name))
        return name


    def get_num_every_combination(self):
        site_occu = []
        for label_occupation in self.occupation_dict.values():
            site_occu += list(label_occupation.values())
        site_combination = combination(
            sum([len(site.allowed_labels) for site in self.eq_sites]),
            site_occu
        )
        pos_combination = 1
        for label, label_pos in self.eq_sites[0].allowed_positions.items():
            for elem, pos in label_pos.items():
                pos_combination *= (len(pos)**self.occupation_dict[label][elem])

        return site_combination * pos_combination


    def random_permute(self):
        remain_label_idx = np.arange(len(self.eq_sites))
        for eq_site in self.eq_sites:
            eq_site.vacant_site()
        for label, label_occupation in self.occupation_dict.items():
            label_number = sum(label_occupation.values())
            selected_label_idx = np.random.choice(
                remain_label_idx, label_number, replace=False
            )
            label_mask = ~np.isin(remain_label_idx, selected_label_idx)
            remain_label_idx = remain_label_idx[label_mask]
            remain_idx = selected_label_idx.copy()
            for element, number in label_occupation.items():
                selected_idx = np.random.choice(
                    remain_idx, number, replace=False
                )
                mask = ~np.isin(remain_idx, selected_idx)
                remain_idx = remain_idx[mask]

                for idx in selected_idx:
                    self.eq_sites[idx].occupy_site(element=element, label=label)


    def _get_possible_exchanges(self):
        if self._allowed_exchanges is not None:  # already calculated
            return
        total_idx = np.arange(len(self.eq_sites))
        is_occ = np.array([site.is_occ for site in self.eq_sites])
        labels = np.array([site.occ_label for site in self.eq_sites])
        elems = np.array([site.occ_element for site in self.eq_sites])
        npos = np.array([len(site.allowed_positions[l][e]) if o else 0 for site, l, e, o in zip(self.eq_sites, labels, elems, is_occ)])

        pair_cond = total_idx[:, None] <= total_idx[None, :]  # avoid i > j
        vac_cond = (is_occ[:, None] | is_occ[None, :])  # no exchange between vacancy
        pos_cond = np.diag(npos != 1) | (total_idx[:, None] != total_idx[None, :])
        same_cond = (total_idx[:, None] == total_idx[None, :]) | (labels[:, None] != labels[None, :]) | (elems[:, None] != elems[None, :])
        cond = pair_cond & vac_cond & pos_cond & same_cond
        self._allowed_exchanges = np.column_stack(np.where(cond))


    def _get_possible_exchanges_legacy(self):
        if self._allowed_exchanges is not None:  # already calculated
            return
        total_idx = np.arange(len(self.eq_sites))
        allowed_exchanges = []
        for i in total_idx:
            if not self.eq_sites[i].is_occ:
                continue  # avoid exchange btw vacancies.
            for j in total_idx:
                label_i = self.eq_sites[i].occ_label
                elem_i = self.eq_sites[i].occ_element
                if j < i and self.eq_sites[j].is_occ:
                    # (j, i) pair is already considered, skip (i, j)
                    continue
                if i == j and len(self.eq_sites[i].allowed_positions[label_i][elem_i]) == 1:
                    # same site & no position to exchange
                    continue

                label_j = self.eq_sites[j].occ_label
                elem_j = self.eq_sites[j].occ_element
                if i != j and label_i == label_j and elem_i == elem_j:
                    # case for exchange btw the same label & elements
                    continue
                allowed_exchanges.append([i, j])
        self._allowed_exchanges = np.array(allowed_exchanges)


    def random_exchange(self):
        # this should be called after occupation (e.g. with random_permute)
        self._get_possible_exchanges()
        i, j = self._allowed_exchanges[
            np.random.choice(np.arange(len(self._allowed_exchanges)), 1)[0]
        ]
        pos_idx = self.exchange_by_index(i, j)
        return i, j, pos_idx


    def exchange_by_index(self, i, j, pos_idx=None):
        if i == j:  # positional disorder exchange
            site = self.eq_sites[i]
            elem = site.occ_element
            label = site.occ_label
            orig_pos_idx = site.occ_position_idx
            if pos_idx is None:
                pos_indices = np.arange(len(site.allowed_positions[label][elem]))
                mask = ~np.isin(pos_indices, orig_pos_idx)
                pos_idx = np.random.choice(pos_indices[mask], 1)[0]
            site.occupy_site(site.occ_element, label, pos_idx)
            return orig_pos_idx

        else:
            site1 = self.eq_sites[i]
            site2 = self.eq_sites[j]

            if not site1.is_occ:  # exchange with vacancy
                site1.occupy_site(
                    site2.occ_element, site2.occ_label, site2.occ_position_idx
                )
                site2.vacant_site()

            elif not site2.is_occ:  # exchange with vacancy
                site2.occupy_site(
                    site1.occ_element, site1.occ_label, site1.occ_position_idx
                )
                site1.vacant_site()

            else:  # exchange btw elements
                elem1 = site1.occ_element
                label1 = site1.occ_label
                pos_idx_1 = site1.occ_position_idx
                elem2 = site2.occ_element
                label2 = site2.occ_label
                pos_idx_2 = site2.occ_position_idx

                site1.occupy_site(elem2, label2, pos_idx_2)
                site2.occupy_site(elem1, label1, pos_idx_1)

            mask_i = self._allowed_exchanges == i
            mask_j = self._allowed_exchanges == j
            self._allowed_exchanges[mask_i] = j
            self._allowed_exchanges[mask_j] = i


class Crystal:
    def __init__(
        self,
        orbits: List[Orbit],
        cell: List[float],
        for_copy: bool = False,
    ):
        self.orbits = orbits
        self.disordered_orbit_idx = []
        self.ordered_orbit_idx = []
        self.is_occupied = False
        self.energy = 0
        self.info = {}
        self.cell = cell
        self.entropy = 0

        for idx, orbit in enumerate(orbits):
            if orbit.is_disordered:
                self.disordered_orbit_idx.append(idx)
            else:
                self.ordered_orbit_idx.append(idx)

            if for_copy:
                continue
            entropy = orbit.calculate_entropy()
            self.entropy += entropy


    def __str__(self):
        string = ''
        for orbit in self.orbits:
            string += '-'*50 + '\n'
            string += str(orbit) + '\n'
            string += '-'*50 + '\n'
            for site in orbit.eq_sites:
                string += f'{site} {site.occ_element}\n'
        return string
            

    def __len__(self):
        natoms = 0
        for orbit in self.orbits:
            natoms += sum(orbit.occupation_by_elem.values())
        return int(natoms)


    def copy(self):
        crystal = self.__class__(
            orbits=[orbit.copy() for orbit in self.orbits],
            cell=np.array(self.cell).copy().tolist(),
            for_copy=True,
        )
        crystal.is_occupied = self.is_occupied
        crystal.energy = self.energy
        crystal.info = self.info.copy()
        crystal.entropy = self.entropy
        return crystal


    def calculate_entropy(self):
        entropy = 0
        for orbit in self.orbits:
            if orbit.liquid_like:  # Can not calculate liquid-like orbit
                return None
            entropy += orbit.calculate_entropy()
        return entropy


    def random_generate_structure(self):
        self.is_occupied = True
        for idx in self.disordered_orbit_idx:
            self.random_permute_deterministic_idx(idx)
        for idx in self.ordered_orbit_idx:
            orbit = self.orbits[idx]
            for site in orbit.eq_sites:
                label = list(site.allowed_labels)[0]
                elem = list(site.allowed_elements[label])[0]
                site.occupy_site(elem)


    def random_permute_deterministic_idx(self, idx):
        assert idx in self.disordered_orbit_idx
        self.orbits[idx].random_permute()


    def exchange_idx_deterministic_idx(self, idx, i, j, pos_idx=None):
        assert idx in self.disordered_orbit_idx
        self.orbits[idx].exchange_by_index(i, j, pos_idx)


    def random_permute_random_idx(self):
        idx = np.random.choice(self.disordered_orbit_idx, 1)[0]
        self.random_permute_deterministic_idx(idx)


    def random_exchange_random_idx(self):
        idx = np.random.choice(self.disordered_orbit_idx, 1)[0]
        i, j, pos_idx = self.orbits[idx].random_exchange()
        return idx, i, j, pos_idx


    def get_disorder_symbol(self):
        name = []
        for idx in self.disordered_orbit_idx:
            orbit = self.orbits[idx]
            if orbit.has_substitutional:
                name.append('S')
            if orbit.has_vacancy:
                name.append('V')
            if orbit.is_combined:
                name.append('P')

        if len(name) == 0:
            return 'O'
        else:
            return ''.join(sorted(list(set(name))))


    def get_disorder_orbit_symbols(self):
        dsyms = []
        for orbit in self.orbits:
            dsyms.append(orbit.get_disorder_symbol())
        return dsyms


    def to_ase_atoms(self, disorder_only=False, assign_site_info=False):
        assert self.is_occupied
        elems = []
        positions = []
        site_info_keys = ['label', 'occ_label', 'disorder', 'oxidation']
        site_info = {key: [] for key in site_info_keys}
        #for idx in self.disordered_orbit_idx:
        for idx, orbit in enumerate(self.orbits):
            #orbit = self.orbits[idx]
            if disorder_only and idx in self.ordered_orbit_idx:
                continue
            for site in orbit.eq_sites:
                if not site.is_occ:
                    continue
                elems.append(site.occ_element)
                positions.append(site.occ_position)
                if assign_site_info:
                    site_info['label'].append(site.label)
                    site_info['occ_label'].append(site.occ_label)
                    site_info['disorder'].append(orbit.get_disorder_symbol())
                    site_info['oxidation'].append(site.oxidation[site.occ_label][site.occ_element])

        """
        for idx in self.ordered_orbit_idx:
            orbit = self.orbits[idx]
            for site in orbit.eq_sites:
                elem = list(site.allowed_elements)[0]
                pos = site.allowed_positions[elem][0]
                elems.append(elem)
                positions.append(pos)
        """

        sorted_idx = []
        elem_list = sorted(list(set(elems)))
        for e in elem_list:
            for idx, elem in enumerate(elems):
                if elem == e:
                    sorted_idx.append(idx)

        formula = ''.join([elems[idx] for idx in sorted_idx])
        positions = [positions[idx] for idx in sorted_idx]
        atoms = Atoms(formula, positions=positions, pbc=True, cell=self.cell)
        calc_result = {'energy': self.energy}
        calc = SinglePointCalculator(atoms, **calc_result)
        atoms = calc.get_atoms()
        atoms.info = self.info.copy()

        if assign_site_info:
            for key, array in site_info.items():
                array = [array[idx] for idx in sorted_idx]
                atoms.set_array(key, np.array(array))

        return atoms


    def get_num_every_combination(self):
        num = 1
        for orbit in self.orbits:
            num *= orbit.get_num_every_combination()
        return num


def supercell_from_pymatgen(stct, matrix, config):
    supercell = stct.make_supercell(matrix, in_place=False)
    super_sites = sites_from_pymatgen(supercell, config)
    super_eq_site_dct = group_eq_sites(super_sites)
    orbits = []
    for eq_site in super_eq_site_dct.values():
        occu_dct = get_occupation_dict(eq_site, 1, 1, 1)
        orbits.append(Orbit(eq_site, occu_dct))
    return Crystal(orbits, supercell.lattice.matrix)


def supercell_from_occupation_dicts(stct, config, matrix, occupation_dicts):
    supercell = stct.make_supercell(matrix, in_place=False)
    super_sites = sites_from_pymatgen(supercell, config)
    super_eq_site_dct = group_eq_sites(super_sites)
    orbits = []
    for slabel, eq_site in super_eq_site_dct.items():
        occu_dct = occupation_dicts[slabel]
        orbits.append(Orbit(eq_site, occu_dct))
    return Crystal(orbits, supercell.lattice.matrix)


def get_valid_occupation_number_v1(
    charges,
    denums,
    target_occs,
    possible_combination,
):
    #TODO: make it numba jit
    success = None
    min_loss = float('inf')
    for nocc_comb in possible_combination:
        loss = 0
        totchg = 0
        for tar, chg, noccu, denum in zip(target_occs, charges, nocc_comb, denums):
            loss += abs(tar - noccu/denum)
            totchg += chg*noccu
        # loss = np.sum(np.abs((np.array(target_occs) - np.array(nocc_comb)/np.array(denums))))
        # totchg = np.sum(np.array(charges)*np.array(nocc_comb))
        if totchg == 0 and loss < min_loss:
            min_loss = loss
            success = nocc_comb

    return success, min_loss


@jit(nopython=True, cache=True)
def _get_valid_occupation_number(
    charges,
    denums,
    target_occs,
    possible_combination,
):
    loss = np.sum(np.abs(target_occs - possible_combination/denums), axis=1)
    totchg = np.sum(charges*possible_combination, axis=1)

    success = totchg==0
    if len(totchg[success]) == 0:
        return None, None

    min_loss = np.min(loss[success])
    success_idx = np.argwhere(loss==min_loss)[0][0]
    return possible_combination[success_idx], min_loss


#@jit(nopython=True, cache=True)
def get_valid_occupation_number(
    occ_nums_comb,
    muls,  # supercell * site
    orbit_indices,
    max_ox,
    min_ox,
):
    """
    occ_sum_by_orbit = np.zeros(
        (occ_nums_comb.shape[0], orbit_indices.max() + 1),
        dtype=occ_nums_comb.dtype
    )
    np.add.at(occ_sum_by_orbit, (slice(None), orbit_indices), occ_nums_comb)
    """
    #occ_sum_by_orbit = np.vstack([np.bincount(orbit_indices, row) for row in occ_nums_comb])
    #survive = (occ_sum_by_orbit <= muls).all(axis=1)
    #if len(survive) == 0:  # no possible occupation
    #    return None

    #occ_nums_comb = occ_nums_comb[survive]
    max_charge = np.sum(occ_nums_comb*max_ox, axis=1)
    if np.all(max_ox == min_ox):  # no float ox
        min_charge = max_charge
    else:
        min_charge = np.sum(occ_nums_comb*min_ox, axis=1)
    
    mask = (max_charge < 0) | (min_charge > 0)
    if np.all(mask):  # no possible charge
        return None

    return occ_nums_comb[~mask]


#@jit(nopython=True, cache=True)
def get_best_occupation_and_l2_loss(
    occ_numbs_comb,
    muls,
    target_occs,
    weights,
):
    result = occ_numbs_comb / muls
    squared_error = np.power(result-target_occs, 2)
    weighted_error = squared_error * weights
    mse = np.sum(weighted_error, axis=1)
    min_loss = np.min(mse)
    min_idx = np.argwhere(mse==min_loss)[0][0]
    return occ_numbs_comb[min_idx], min_loss


#@jit(nopython=True, cache=True)
def get_best_occupation_and_l1_loss(
    occ_numbs_comb,
    muls,
    target_occs,
    weights,
):
    result = occ_numbs_comb / muls
    abs_error = np.abs(result-target_occs)
    weighted_error = abs_error * weights
    mae = np.sum(weighted_error, axis=1)
    min_loss = np.min(mae)
    min_idx = np.argwhere(mae==min_loss)[0][0]
    return occ_numbs_comb[min_idx], min_loss


def _filter_impossible_comb(
    possible_ranges,
    muls,
    orbit_indices,
):
    orbit_range = {}
    possible_combs = []
    for idx, orbit_idx in enumerate(orbit_indices):
        if orbit_idx not in orbit_range:
            orbit_range[orbit_idx] = []
        orbit_range[orbit_idx].append(possible_ranges[idx])

    for idx, mul in enumerate(muls):
        all_comb = np.array(list(itertools.product(*orbit_range[idx])))
        if len(all_comb) == 0:
            return []
        mask = np.sum(all_comb, axis=1) > mul
        possible_comb = all_comb[~mask].tolist()
        possible_combs.append(possible_comb)

    return [list(itertools.chain.from_iterable(comb)) for comb in itertools.product(*possible_combs)]


def get_occupation_with_charge_balance_and_loss(
    eq_site_dct,
    sup_mul,
    ox_tol = 0.05,
    max_occ_tol = 1.05,
    attempt_occ_tol = 0.1,
    unit='orbit',
    loss_type = 'l2',  # one of l2, l1
    weight_type = None,  # one of None, 'multiplicity', 'scattering'
    gloss = float('inf'),  # for fast filtering
):
    occupation_dicts = {slabel: {} for slabel in eq_site_dct}
    slabels, labels, elems = [], [], []
    possible_ranges, target_occs, weights = [], [], []
    muls, orbit_indices = [], []
    max_ox, min_ox = [], []
    for orbit_idx, (site_label, eq_sites) in enumerate(eq_site_dct.items()):
        eq_site = eq_sites[0]
        site_mul = len(eq_sites)
        total_occu = 0
        muls.append(site_mul*sup_mul)
        for label, species in eq_site.species.items():
            oxidations = eq_site.oxidation[label]
            for elem in species:
                occu = species[elem]
                total_occu += occu
                ox = oxidations[elem]
                ox = 0 if ox is None else ox
                tol = ox_tol if int(ox) != ox else 0
                max_ox.append(ox+tol)
                min_ox.append(ox-tol)
                if weight_type is None:
                    weight = 1
                elif weight_type == 'multiplicity':
                    weight = site_mul
                else:
                    weight = site_mul*Element(elem).Z

                if total_occu > max_occ_tol:  # can not fill liquid-like structure
                    print(site_mul, eq_site, total_occu, max_occ_tol, label)
                    return None, None, -1  
                if occu > FULLY_OCCUPY:
                    possible_range = [sup_mul*site_mul]
                else:
                    interval = site_mul if unit=='orbit' else 1
                    possible_range = []
                    for n in list(range(0, sup_mul*site_mul+1, interval)):
                        diff = abs(n/(sup_mul*site_mul) - occu)
                        if diff > attempt_occ_tol:
                            continue
                        if loss_type == 'l2':
                            single_loss = (weight*diff)**2
                        else:
                            single_loss = weight * abs(diff)
                        if single_loss > gloss:
                            continue
                        possible_range.append(n)

                possible_ranges.append(possible_range)
                target_occs.append(occu)
                orbit_indices.append(orbit_idx)
                weights.append(weight)

                slabels.append(site_label)
                labels.append(label)
                elems.append(elem)

    #occ_nums_comb = list(itertools.product(*possible_ranges))
    occ_nums_comb = _filter_impossible_comb(
        possible_ranges,
        muls,
        orbit_indices,
    )
    if len(occ_nums_comb) == 0:  # can not find range
        #print(sup_mul, -2)
        return None, None, -2
    occ_nums_comb = get_valid_occupation_number(
        np.array(occ_nums_comb),
        np.array(muls),
        np.array(orbit_indices),
        np.array(max_ox),
        np.array(min_ox),
    )
    if occ_nums_comb is None:  # no possible combination matches charge balance
        #print(sup_mul, -3)
        return None, None, -3
    if loss_type == 'l2':
        occ_nums, loss = get_best_occupation_and_l2_loss(
            np.array(occ_nums_comb),
            np.array(muls)[orbit_indices],
            np.array(target_occs),
            np.power(np.array(weights), 2),
        )
    else:
        occ_nums, loss = get_best_occupation_and_l1_loss(
            np.array(occ_nums_comb),
            np.array(muls)[orbit_indices],
            np.array(target_occs),
            np.array(weights),
        )
    if min(occ_nums) < 1:  # something is not occupied
        return None, None, -4

    for slabel, label, elem, occu in zip(slabels, labels, elems, occ_nums):
        if label not in occupation_dicts[slabel]:
            occupation_dicts[slabel][label] = {}
        occupation_dicts[slabel][label][elem] = occu

    return occupation_dicts, loss, None


def get_occupation_with_charge_balance(
    eq_site_dct,
    multiplier,
    tolerance=0.,
    unit='orbit'
):
    occupation_dicts = {slabel: {} for slabel in eq_site_dct}
    slabels, labels, elems, denums, target_occs, charges, possible_num_occs = \
        [], [], [], [], [], [], []
    min_charge, max_charge = 0, 0   
    for site_label, eq_sites in eq_site_dct.items():
        eq_site = eq_sites[0]
        base_mul = multiplier if unit == 'orbit' else multiplier*len(eq_sites)
        remain_mul = len(eq_sites) if unit == 'orbit' else 1
        for label, species in eq_site.species.items():
            oxidations = eq_site.oxidation[label]
            for elem in species:
                occu = species[elem]
                ox = oxidations[elem]
                ox = 0 if ox is None else ox
                if occu > FULLY_OCCUPY:
                    possible_range = [base_mul*remain_mul]
                else:
                    possible_min = max(math.floor((occu-tolerance)*base_mul), 1)
                    possible_max = min(math.ceil((occu+tolerance)*base_mul), base_mul)
                    possible_range = [
                        v*remain_mul for v in range(possible_min, possible_max+1)
                        if v/base_mul > occu-tolerance and v/base_mul < occu+tolerance
                    ]
                if len(possible_range) == 0:
                    print(multiplier, 'no range')
                    return None, None
                if ox > 0:
                    max_charge += ox*max(possible_range)
                    min_charge += ox*min(possible_range)
                else:
                    max_charge += ox*min(possible_range)
                    min_charge += ox*max(possible_range)

                slabels.append(site_label)
                denums.append(base_mul*remain_mul)
                labels.append(label)
                elems.append(elem)
                target_occs.append(occu)
                charges.append(ox)
                possible_num_occs.append(possible_range)

    if max_charge < 0 or min_charge > 0:
        print(multiplier, 'no chg', max_charge, min_charge)
        return None, None
    possible_combination = list(itertools.product(*possible_num_occs))
    valid_occu, loss = get_valid_occupation_number(
        np.array(charges),
        np.array(denums),
        np.array(target_occs),
        np.array(possible_combination)
    )

    if valid_occu is None:
        return None, None

    for slabel, label, elem, occu in zip(slabels, labels, elems, valid_occu):
        if label not in occupation_dicts[slabel]:
            occupation_dicts[slabel][label] = {}
        occupation_dicts[slabel][label][elem] = occu

    #print(multiplier, loss)
    return occupation_dicts, loss


def sanitize_to_neutral(
    eq_site_dct,
    relative_error=False,
    multiplicity_power=0,
    weight_atomic_number=False,
    clip=None
):
    coeff_matrix = np.zeros((len(eq_site_dct)+1, len(eq_site_dct)+1))
    tot_chg = 0.
    slabels, labels, elems, lagrange_indices = [], [], [], []
    for idx, (slabel, eq_sites) in enumerate(eq_site_dct.items()):
        eq_site = eq_sites[0]
        mul = len(eq_sites)
        tot_occu = 0.
        for label, species in eq_site.species.items():
            for elem, occu in species.items():
                chg = eq_site.oxidation[label][elem]
                chg = 0 if chg is None else chg
                tot_occu += occu
                tot_chg += chg*mul*occu
                rel = occu**2 if relative_error else 1
                w_atom = Element(elem).Z if weight_atomic_number else 1
                coeff_matrix[0][0] += (chg*mul)**2 / (mul**multiplicity_power) / (w_atom**2) * rel
                coeff_matrix[idx+1][0] += chg*mul / (mul**multiplicity_power) / (w_atom**2) * rel
                coeff_matrix[0][idx+1] += chg*mul / (mul**multiplicity_power) / (w_atom**2) * rel
                coeff_matrix[idx+1][idx+1] += 1 / (mul**multiplicity_power) / (w_atom**2) * rel

                slabels.append(slabel)
                labels.append(label)
                elems.append(elem)
                lagrange_indices.append(idx+1)
        if tot_occu < FULLY_OCCUPY:
            rel = (1. - tot_occu)**2 if relative_error else 1
            coeff_matrix[idx+1][idx+1] += 1 / (mul**multiplicity_power) * rel  # vacancy contribution

    fit = np.zeros(len(eq_site_dct)+1)
    fit[0] = -tot_chg
    lagrange_mul = np.matmul(np.linalg.inv(coeff_matrix),fit)

    for slabel, label, elem, idx in zip(slabels, labels, elems, lagrange_indices):
        chg = eq_site_dct[slabel][0].oxidation[label][elem]
        mul = len(eq_site_dct[slabel])
        delta = (lagrange_mul[0]*chg*mul + lagrange_mul[idx]) / (mul**multiplicity_power)
        if relative_error:
            occu = eq_site_dct[slabel][0].species[label][elem]
            delta *= occu**2
        if weight_atomic_number:
            delta /= (Element(elem).Z ** 2)
        if clip is not None:
            delta = float(np.clip(delta, -abs(clip), abs(clip))) 
        for eq_site in eq_site_dct[slabel]:
            eq_site.species[label][elem] += delta


def get_valid_occu_dct_and_lattice_transform(parser, config):
    stct = parser.parse_structures(primitive=False)[0]
    sites = sites_from_pymatgen(stct, config)
    eq_site_dct = group_eq_sites(sites)

    exp_natom_per_cell = max(stct.composition.num_atoms, 1e-6)

    # sanitize oxidation state?
    
    crit = config.get(
        'supercell_criterion',
        {
            'mul':
                {'min': 1, 'max': MAX_SUPERCELL_MUL},
            'natom':
                {'min': 0, 'max': float('inf')},
        }
    )
    transforms, valid_occu_dct, final_err = None, None, None

    if isinstance(config['supercell_mode'], str):
        global_loss = float('inf')
        selection_loss = float('inf')

        min_mul = math.ceil(
            max(
                crit['mul']['min'],
                crit['natom'].get('min', 0)/exp_natom_per_cell
            )
        )
        max_mul = math.floor(
            min(
                crit['mul']['max'],
                crit['natom'].get('max', float('inf'))/exp_natom_per_cell
            )
        )
        if min_mul > max_mul:
            final_err = 0
        for sup_mul in range(min_mul, max_mul+1):
            occupation_dicts, loss, err_msg = get_occupation_with_charge_balance_and_loss(
                eq_site_dct,
                sup_mul,
                ox_tol=config.get('oxidation_state_tol', 0.05),
                attempt_occ_tol = config.get('attempt_occ_tol', 0.1),
                max_occ_tol = config.get('max_occ_tol', 1.05),
                unit=config.get('supercell_unit', 'orbit'),
                loss_type=config.get('supercell_loss_type', 'l1'),
                weight_type=config.get('supercell_weight_type', 1),
                gloss=global_loss
            )

            if occupation_dicts is None:  # fail to generate
                final_err = err_msg
                continue

            transforms = get_possible_lattice_transforms(stct, sup_mul)
            if 'latt' in crit:
                sup_lattices = transforms @ stct.lattice.matrix
                min_lat = crit['latt'].get('min', 0)
                max_lat = crit['latt'].get('max', float('inf'))
                abcs = calc_abc(sup_lattices)
                select = np.all(abcs > min_lat) & np.all(abcs < max_lat)
                transforms = transforms[select]

            if 'cubicity' in crit:
                sup_lattices = transforms @ stct.lattice.matrix
                min_cub = crit['cubicity'].get('min', -float('inf'))
                max_cub = crit['cubicity'].get('max', 1)
                cubs = calc_cubicity(sup_lattices)
                select = (cubs > min_cub) & (cubs < max_cub)
                transforms = transforms[select]

            if len(transforms) == 0:
                final_err = -5  # no possible lattices
                continue

            if 'entropy' in crit:
                sup = supercell_from_occupation_dicts(
                    stct, config, transforms[0], occupation_dicts
                )
                min_ent = crit['entropy'].get('min', 0)
                max_ent = crit['entropy'].get('max', float('inf'))
                entropy = sup.calculate_entropy()
                if entropy < min_ent:
                    continue
                if entropy > max_ent:
                    break

            if 'permutation' in crit:
                sup = supercell_from_occupation_dicts(
                    stct, config, transforms[0], occupation_dicts
                )
                min_perm = crit['permutation'].get('min', 1)
                max_perm = crit['permutation'].get('max', float('inf'))
                perm = sup.get_num_every_combination()
                if perm < min_perm:
                    continue
                if perm > max_perm:
                    break

            loss_crit = config.get('supercell_selection', 'size')
            if loss_crit == 'size' or loss < 1e-4:
                valid_occu_dct = occupation_dicts
                break

            elif loss_crit == 'error' and  loss < selection_loss:
                global_loss = loss
                selection_loss = loss
                valid_occu_dct = occupation_dicts

            elif loss_crit == 'product' and loss*total_mul < selection_loss:
                gloabl_loss = loss
                selection_loss = loss*total_mul
                valid_occu_dct = occupation_dicts

    else:
        matrix = np.array(config['supercell_mode']).astype(int)
        if matrix.ndim == 1:
            matrix = np.diag(matrix)
        base_mul = np.linalg.det(matrix)
        valid_occu_dct = {slabel: {} for slabel in eq_site_dct.keys()}
        for site_label, eq_sites in eq_site_dct.items():
            eq_site = eq_sites[0]
            remain_mul = len(eq_sites)
            for label, species in eq_site.species.items():
                if label not in valid_occu_dct[site_label]:
                    valid_occu_dct[site_label][label] = {}
                for elem in species:
                    occu = species[elem]
                    valid_occu_dct[site_label][label][elem] = \
                        round(occu*base_mul*remain_mul)
        transforms = np.array([matrix])

    return valid_occu_dct, transforms, final_err


def crystal_from_pymatgen_parser(parser, config):
    stct = parser.parse_structures(primitive=False)[0]
    sites = sites_from_pymatgen(stct, config)
    eq_site_dct = group_eq_sites(sites)

    # sanitize oxidation state?
    
    crit = config.get(
        'supercell_criterion', {'latt': {'min': 1, 'max': MAX_SUPERCELL_MUL}}
    )
    matrix, valid_occu_dict = None, None
    if config['supercell_mode'].lower() == 'auto':
        min_n = crit.get('latt', {}).get('min', 1)
        max_n = crit.get('latt', {}).get('max', MAX_SUPERCELL_MUL)
        matrices = list(
            itertools.combinations_with_replacement(
                range(min_n, max_n+1), 3
            )
        )
        sup_mul_comb = {na*nb*nc: [] for na, nb, nc in matrices}
        for na, nb, nc in matrices:
            sup_mul_comb[na*nb*nc].extend(list(itertools.permutations([na, nb, nc])))
        """
        with open('/data2_1/sunny990912/disorder/py_monte_carlo/example/comb.pkl', 'rb') as f:
            sup_mul_comb = pickle.load(f)
        """

        gloss = float('inf')
        selection_loss = float('inf')
        final_err = None
        keys = sorted(sup_mul_comb.keys())
        for sup_mul in keys:
            possible_abcs = sup_mul_comb[sup_mul]
            if 'mul' in crit:
                min_m = crit['mul'].get('min', 1)
                max_m = crit['mul'].get('max', float('inf'))
                if sup_mul < min_m or sup_mul > max_m:
                    continue
            occupation_dicts, loss, err_msg = get_occupation_with_charge_balance_and_loss(
                eq_site_dct,
                sup_mul,
                ox_tol=config.get('oxidation_state_tol', 0.05),
                attempt_occ_tol = config.get('attempt_occ_tol', 0.1),
                max_occ_tol = config.get('max_occ_tol', 1.05),
                unit=config.get('supercell_unit', 'orbit'),
                loss_type=config.get('supercell_loss_type', 'l1'),
                weight_type=config.get('supercell_weight_type', 1),
                gloss=gloss
            )
            if occupation_dicts is None:  # fail to generate
                final_err = err_msg
                continue

            sup = supercell_from_occupation_dicts(stct, config, possible_abcs[0], occupation_dicts)
            if 'natom' in crit:
                min_nat = crit['natom'].get('min', 1)
                max_nat = crit['natom'].get('max', float('inf'))
                if len(sup) < min_nat:
                    continue
                if len(sup) > max_nat:
                    break

            if 'entropy' in crit:
                min_ent = crit['entropy'].get('min', 0)
                max_ent = crit['entropy'].get('max', float('inf'))
                entropy = sup.calculate_entropy()
                if entropy < min_ent:
                    continue
                if entropy > max_ent:
                    break

            if 'permutation' in crit:
                min_perm = crit['permutation'].get('min', 1)
                max_perm = crit['permutation'].get('max', float('inf'))
                perm = sup.get_num_every_combination()
                if perm < min_perm:
                    continue
                if perm > max_perm:
                    break

            loss_crit = config.get('supercell_selection', 'size')
            if loss_crit == 'size' or loss < 1e-4:
                matrix = possible_abcs
                valid_occu_dct = occupation_dicts
                break

            elif loss_crit == 'error' and  loss < selection_loss:
                gloss = loss
                selection_loss = loss
                matrix = possible_abcs
                valid_occu_dct = occupation_dicts

            elif loss_crit == 'product' and loss*total_mul < selection_loss:
                gloss = loss
                selection_loss = loss*total_mul
                matrix = possible_abcs
                valid_occu_dct = occupation_dicts

    else:
        matrix = list(map(int, config['supercell_mode'].lower().split('x')))
        base_mul = np.prod(matrix)
        matrix = list(itertools.permutations(matrix))
        valid_occu_dct = {slabel: {} for slabel in eq_site_dct.keys()}
        for site_label, eq_sites in eq_site_dct.items():
            eq_site = eq_sites[0]
            remain_mul = len(eq_sites)
            for label, species in eq_site.species.items():
                if label not in valid_occu_dct[site_label]:
                    valid_occu_dct[site_label][label] = {}
                for elem in species:
                    occu = species[elem]
                    valid_occu_dct[site_label][label][elem] = \
                        round(occu*base_mul*remain_mul)


    if matrix is not None:  # find some supercell within crit
        # chage na, nb, nc order to make cubic-like structure
        #matrix = _pick_nanbnc(matrix, stct.lattice)
        matrix = _pick_nanbnc_by_cubicity(matrix, stct.lattice)
        crystal = supercell_from_occupation_dicts(stct, config, matrix, valid_occu_dct)

        comp = stct.composition.get_el_amt_dict()
        rformula = "".join(f"{el}{round(amount, 2)}" for el, amount in comp.items())
        crystal.info = {'rformula': rformula, 'matrix': matrix}
        return crystal

    return final_err


def _pick_nanbnc_by_cubicity(supercells, lattice):
    L = supercells @ lattice.matrix
    cubicity = calc_cubicity(L).tolist()
    idx = cubicity.index(max(cubicity))
    return supercells[idx].tolist()


def _pick_nanbnc(matrices, lattice):
    min_var = float('inf')
    for matrix in matrices:
        a = lattice.a*matrix[0]
        b = lattice.b*matrix[1]
        c = lattice.c*matrix[2]
        var = np.var([a, b, c])
        if var < min_var:
            na, nb, nc = matrix
            min_var = var
    return na, nb, nc


def crystal_from_pymatgen(stct, config):
    sites = sites_from_pymatgen(stct, config)
    eq_site_dct = group_eq_sites(sites)

    # sanitize if given
    tot_chg = stct.charge
    if sanitize_param:=config.get('sanitize_cif_neutral', {}):
        if abs(tot_chg) > sanitize_param.pop('attempt_tolerance', 0):
            sanitize_to_neutral(
                eq_site_dct=eq_site_dct,
                **sanitize_param
            )
    #breakpoint()
    tolerance = config.get('tolerance', 0.)
    crit = config.get(
        'supercell_criterion', {'latt': {'min': 1, 'max': MAX_SUPERCELL_MUL}}
    )
    unit = config.get('supercell_unit', 'orbit')

    matrix = None
    gloss = float('inf')

    if config['supercell_mode'].lower() == 'auto':
        min_n = crit.get('latt', {}).get('min', 1)
        max_n = crit.get('latt', {}).get('max', MAX_SUPERCELL_MUL)
        matrices = list(
            itertools.combinations_with_replacement(
                range(min_n, max_n+1), 3
            )
        )
        matrices.sort(key=max)
        for na, nb, nc in matrices:
            total_mul = na*nb*nc
            if 'mul' in crit:
                min_m = crit['mul'].get('min', 1)
                max_m = crit['mul'].get('max', float('inf'))
                if total_mul < min_m or total_mul > max_m:
                    continue

            occupation_dicts, loss = get_occupation_with_charge_balance(
                eq_site_dct, total_mul, tolerance, unit
            )
            if occupation_dicts is None:  # fail to generate
                continue
            sup = supercell_from_occupation_dicts(stct, config, [na, nb, nc], occupation_dicts)
            if any([orb.liquid_like for orb in sup.orbits]):
                #print(sup)
                #continue
                return None

            if 'natom' in crit:
                min_nat = crit['natom'].get('min', 1)
                max_nat = crit['natom'].get('max', float('inf'))
                if len(sup) < min_nat or len(sup) > max_nat:
                    continue

            if 'entropy' in crit:
                min_ent = crit['entropy'].get('min', 0)
                max_ent = crit['entropy'].get('max', float('inf'))
                entropy = sup.calculate_entropy()
                if entropy < min_ent or entropy > max_ent:
                    continue

            if 'permutation' in crit:
                min_perm = crit['permutation'].get('min', 1)
                max_perm = crit['permutation'].get('max', float('inf'))
                perm = sup.get_num_every_combination()
                if perm < min_perm or perm > max_perm:
                    continue
            loss_crit = config.get('supercell_selection', 'size')
            if loss_crit == 'size' or loss == 0:
                matrix = [na, nb, nc]
                valid_occu_dct = occupation_dicts
                break

            elif loss_crit == 'error' and  loss < gloss:
                gloss = loss
                matrix = [na, nb, nc]
                valid_occu_dct = occupation_dicts
            elif loss_crit == 'product' and loss*total_mul < gloss:
                gloss = loss*total_mul
                matrix = [na, nb, nc]
                valid_occu_dct = occupation_dicts

    else:
        matrix = list(map(int, config['supercell_mode'].lower().split('x')))
        base_mul = np.prod(matrix)
        valid_occu_dct = {slabel: {} for slabel in eq_site_dct.keys()}
        for site_label, eq_sites in eq_site_dct.items():
            eq_site = eq_sites[0]
            remain_mul = len(eq_sites)
            for label, species in eq_site.species.items():
                if label not in valid_occu_dct[site_label]:
                    valid_occu_dct[site_label][label] = {}
                for elem in species:
                    occu = species[elem]
                    valid_occu_dct[site_label][label][elem] = \
                        round(occu*base_mul*remain_mul)


    if matrix is not None:  # find some supercell within crit
        # chage na, nb, nc order to make cubic-like structure
        latt = [stct.lattice.a, stct.lattice.b, stct.lattice.c]
        latt_indices = np.argsort(latt)
        matrix = sorted(matrix, reverse=True)
        new_matrix = [0]*3
        for i, j in enumerate(latt_indices):
            new_matrix[j] = matrix[i]
        crystal = supercell_from_occupation_dicts(stct, config, new_matrix, valid_occu_dct)

        comp = stct.composition.get_el_amt_dict()
        rformula = "".join(f"{el}{round(amount, 2)}" for el, amount in comp.items())
        crystal.info = {'rformula': rformula, 'matrix': new_matrix}
        return crystal

    return None


if __name__=='__main__':
    import sys
    """
    config = {
        'sanitize_cif_neutral':
            {
                'attempt_tolerance': 0.01,
                'relative_error': False,
                'weight_atomic_number': True,
                'multiplicity_power': 2,
                'clip': None,
            },
        'supercell_mode': 'auto',
        'supercell_unit': 'site',
        'supercell_criterion': {
            'latt':{
                'min': 2,
                'max': 7,
            }
        },
        'supercell_selection': 'size',
        'tolerance': 0.005,
        'positional_disorder': {
            'hard_cutoff': 0.5,
            'element_cutoff': '../../example/elem_radii.yaml',
            'multiplier': 0.5,
        },
    }
    ox_tol=0.05,
    unit='orbit',
    loss_type='l1',
    weight_type='scattering',

    """
    config = {
        'supercell_mode': 'auto',
        'supercell_unit': 'orbit',
        'supercell_criterion': {
            'mul':{
                'min': 8,
                'max': 125,
            },
            'natom':{
                'max': 3000,
            },
        },
        'oxidation_state_tol': 0.05,
        'max_occ_tol': 1.05,
        'supercell_selection': 'error',
        'supercell_loss_type': 'l1',
        'supercell_weight_type': 'scattering',
        'positional_disorder': {
            'hard_cutoff': 1.0,
            'hydrogen_hard_cutoff': 0.5,
            'element_cutoff': '../../example/elem_radii.yaml',
            'multiplier': 0.5,
        },
    }
    def _update_radius_dict(config):
        radii = config['positional_disorder'].get('element_cutoff', None)
        if not isinstance(radii, str):
            return config

        if radii.endswith('.yaml'):
            import yaml
            with open(radii, 'r') as f:
                radii_dict = yaml.load(f, Loader=yaml.FullLoader)
        elif radii.endswith('.json'):
            import json
            with open(radii, 'r') as f:
                radii_dict = json.load(f)
        elif radii.endswith('.pkl') or radii.endswith('.pickle'):
            import pickle
            with open(radii, 'rb') as f:
                radii_dict = pickle.load(f)
        else:
            raise NotImplementedError(f'Failed to read file: {radii}')

        config['positional_disorder']['element_cutoff'] = radii_dict
        return config

    config = _update_radius_dict(config)
    #config['sanitize_cif_neutral'] = {}
    #stct = Structure.from_file(sys.argv[1])
    parser = CifParser(sys.argv[1])
    stct = parser.parse_structures(primitive=False)[0]
    occu_dct, transforms, final_err = \
        get_valid_occu_dct_and_lattice_transform(parser, config)
    matrix = transforms[0]
    crystal = supercell_from_occupation_dicts(
        stct, config, matrix, occu_dct
    )
    crystal.random_generate_structure()
    #print(crystal)
    for _ in tqdm(range(10000)):
        crystal.random_exchange_random_idx()
    """
    from collections import Counter
    counter = dict(Counter(atoms.get_chemical_symbols()))
    div  = counter['O'] / 3
    for elem, count in counter.items():
        print(elem, count/div)
    """
    """
    from copy import deepcopy
    import time
    t1 = time.time()
    for _ in range(100):
        b = deepcopy(crystal)
    t2 = time.time()
    for _ in range(100):
        b = crystal.copy()
    t3 = time.time()
    print(t2-t1, t3-t2)
    breakpoint()
    """
    """
    from ase.io import write
    write('a1.extxyz', atoms)
    crystal.random_permute_random_idx()
    atoms = crystal.to_ase_atoms()
    write('a2.extxyz', atoms)
    crystal.random_exchange_random_idx()
    atoms = crystal.to_ase_atoms()
    print(crystal)
    write('a3.extxyz', atoms)
    breakpoint()
    """
