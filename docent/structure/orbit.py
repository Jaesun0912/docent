import numpy as np
import math
import itertools
from typing import Union, List, Dict
from tqdm import tqdm
from numba import jit

from docent.structure.site import (
    VirtualSite,
    CombinedSite,
    group_eq_sites,
    sites_from_pymatgen
)
from docent.util.utils import (
    combination,
    get_possible_supercell_matrix,
    get_occupation_dict,
)
from docent.util.const import FULLY_OCCUPY

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pymatgen.core import Structure

MAX_SUPERCELL_MUL = 10

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
        if i == j:  # positional disorder exchange
            site = self.eq_sites[i]
            elem = site.occ_element
            label = site.occ_label
            orig_pos_idx = site.occ_position_idx
            pos_indices = np.arange(len(site.allowed_positions[label][elem]))
            mask = ~np.isin(pos_indices, orig_pos_idx)
            pos_idx = np.random.choice(pos_indices[mask], 1)[0]
            site.occupy_site(site.occ_element, label, pos_idx)

        else:
            site1 = self.eq_sites[i]
            site2 = self.eq_sites[j]

            if not site2.is_occ:  # exchange with vacancy
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
            entropy = orbit.calculate_entropy()
            self.entropy += entropy
            if orbit.is_disordered:
                self.disordered_orbit_idx.append(idx)
            else:
                self.ordered_orbit_idx.append(idx)


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
        return natoms


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


    def random_permute_random_idx(self):
        idx = np.random.choice(self.disordered_orbit_idx, 1)[0]
        self.random_permute_deterministic_idx(idx)


    def random_exchange_random_idx(self):
        idx = np.random.choice(self.disordered_orbit_idx, 1)[0]
        self.orbits[idx].random_exchange()


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


    def to_ase_atoms(self):
        assert self.is_occupied
        elems = []
        positions = []
        #for idx in self.disordered_orbit_idx:
        for orbit in self.orbits:
            #orbit = self.orbits[idx]
            for site in orbit.eq_sites:
                if not site.is_occ:
                    continue
                elems.append(site.occ_element)
                positions.append(site.occ_position)

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
        atoms.info = self.info

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
def get_valid_occupation_number(
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

    print(multiplier, loss)
    return occupation_dicts, loss


def sanitize_to_neutral(
    eq_site_dct,
    relative_error=False,
    multiplicity_power=0,
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
                coeff_matrix[0][0] += (chg*mul)**2 / (mul**multiplicity_power) * rel
                coeff_matrix[idx+1][0] += chg*mul / (mul**multiplicity_power) * rel
                coeff_matrix[0][idx+1] += chg*mul / (mul**multiplicity_power) * rel
                coeff_matrix[idx+1][idx+1] += 1 / (mul**multiplicity_power) * rel

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
        if clip is not None:
            delta = float(np.clip(delta, -abs(clip), abs(clip))) 
        for eq_site in eq_site_dct[slabel]:
            eq_site.species[label][elem] += delta


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
                continue

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


def crystal_from_pymatgen_v1(stct, config):
    sites = sites_from_pymatgen(stct, config)
    eq_site_dct = group_eq_sites(sites)
    precision = config.get('precision', 0.)
    matrices = get_possible_supercell_matrix(eq_site_dct.values(), precision)
    supercell_crit_info = config.get('supercell_criterion')
    crit = supercell_crit_info.get('criterion').lower()
    min_val = supercell_crit_info.get('min')
    max_val = supercell_crit_info.get('max')
    crystal = None
    for matrix in matrices:
        total_mul = np.prod(matrix)
        if crit == 'latt':
            if min_val > min(matrix) or max_val < max(matrix):
                continue
        if crit == 'mul':
            if min_val > total_mul or max_val < total_mul:
                continue
        crystal = supercell_from_pymatgen(stct, matrix, config)
        if crit == 'natom':
            if min_val > len(crystal) or max_val < len(crystal):
                continue
        if crit == 'entropy':
            entropy = crystal.calculate_entropy()
            if min_val > entropy or max_val < entropy:
                continue
        if crit == 'permutation':
            perm = crystal.get_num_every_combination()
            if min_val > perm or max_val < perm:
                continue
        break

    if crystal is not None:
        latt = [stct.lattice.a, stct.lattice.b, stct.lattice.c]
        latt_indices = np.argsort(latt)
        matrix = sorted(matrix, reverse=True)
        new_matrix = [0]*3
        for i, j in enumerate(latt_indices):
            new_matrix[j] = matrix[i]
        crystal = supercell_from_pymatgen(stct, new_matrix, config)

        comp = stct.composition.get_el_amt_dict()
        rformula = "".join(f"{el}{round(amount, 2)}" for el, amount in comp.items())
        crystal.info = {'rformula': rformula, 'matrix': new_matrix}
        
    return crystal


if __name__=='__main__':
    import sys
    config = {
        'sanitize_cif_neutral':
            {
                'attempt_tolerance': 0.01,
                'relative_error': True,
                'multiplicity_power': 2,
                'clip': None,
            },
        'supercell_mode': 'auto',
        'supercell_unit': 'orbit',
        'supercell_criterion': {
            'latt':{
                'min': 2,
                'max': 7,
            }
        },
        'supercell_selection': 'size',
        'tolerance': 0.005,
        'positional_disorder': {
            'hard_cutoff': 1.,
            'element_cutoff': '../../example/elem_radii.yaml',
            'multiplier': 0.5,
        },
    }
    #config['sanitize_cif_neutral'] = {}
    stct = Structure.from_file(sys.argv[1])
    crystal = crystal_from_pymatgen(stct, config)
    #print(crystal)
    crystal.random_generate_structure()
    atoms = crystal.to_ase_atoms()
    print(atoms)
    """
    from collections import Counter
    counter = dict(Counter(atoms.get_chemical_symbols()))
    div  = counter['O'] / 3
    for elem, count in counter.items():
        print(elem, count/div)
    """
    print(crystal.info)
    #breakpoint()
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
