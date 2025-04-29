import numpy as np
from typing import Union, List, Dict
from docent.structure.site import (
    VirtualSite,
    CombinedSite,
    calc_site_entropy,
    group_eq_sites,
    sites_from_pymatgen
)
from docent.util.utils import (
    combination,
    get_possible_supercell_matrix,
    get_occupation_dict,
)

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pymatgen.core import Structure


class Orbit:
    def __init__(
        self,
        eq_sites: Union[List[VirtualSite], List[CombinedSite]],
        occupation_dict: Dict[str, int],
    ):
        self.eq_sites = eq_sites
        self.occupation_dict = occupation_dict
        total_occ = sum(occupation_dict.values())
        #total_sites = sum([len(site) for site in eq_sites])
        total_sites = len(eq_sites)
        self.has_substitutional = len(occupation_dict) > 1
        self.has_vacancy = total_occ < total_sites
        self.is_combined = isinstance(eq_sites[0], CombinedSite)
        self.is_disordered = self.has_substitutional or self.has_vacancy or self.is_combined
        self._allowed_exchanges = None
        self.entropy = calc_site_entropy(eq_sites[0])*len(eq_sites)


    def __str__(self):
        name = self.get_disorder_symbol()
        site = str(self.eq_sites[0])
        return f'{len(self.eq_sites)}x{site} {name} {self.occupation_dict}'


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
        site_combination = combination(
            sum([len(site) for site in self.eq_sites]),
            self.occupation_dict.values()
        )
        pos_combination = 1
        for elem, pos in self.eq_sites[0].allowed_positions.items():
            pos_combination *= (len(pos)**self.occupation_dict[elem])

        return site_combination * pos_combination


    def random_permute(self):
        remain_idx = np.arange(len(self.eq_sites))
        for eq_site in self.eq_sites:
            eq_site.vacant_site()
        for element, number in self.occupation_dict.items():
            selected_idx = np.random.choice(remain_idx, number, replace=False)
            mask = ~np.isin(remain_idx, selected_idx)
            remain_idx = remain_idx[mask]

            for idx in selected_idx:
                self.eq_sites[idx].occupy_site(element)


    def _get_possible_exchanges(self):
        if self._allowed_exchanges is not None:  # already calculated
            return
        total_idx = np.arange(len(self.eq_sites))
        allowed_exchanges = []
        for i in total_idx:
            if not self.eq_sites[i].is_occ:
                continue  # avoid exchange btw vacancies.
            for j in total_idx:
                elem_i = self.eq_sites[i].occ_element
                if j < i and self.eq_sites[j].is_occ:
                    # (j, i) pair is already considered, skip (i, j)
                    continue
                if i == j and len(self.eq_sites[i].allowed_positions[elem_i]) == 1:
                    # same site & no position to exchange
                    continue

                elem_j = self.eq_sites[j].occ_element
                if i != j and elem_i == elem_j:
                    # case for exchange btw the same elements
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
            orig_pos_idx = site.occ_position_idx
            pos_indices = np.arange(len(site.allowed_positions[elem]))
            mask = ~np.isin(pos_indices, orig_pos_idx)
            pos_idx = np.random.choice(pos_indices[mask], 1)[0]
            site.occupy_site(site.occ_element, pos_idx)

        else:
            site1 = self.eq_sites[i]
            site2 = self.eq_sites[j]

            if not site2.is_occ:  # exchange with vacancy
                site2.occupy_site(site1.occ_element, site1.occ_position_idx)
                site1.vacant_site()

            else:  # exchange btw elements
                elem1 = site1.occ_element
                pos_idx_1 = site1.occ_position_idx
                elem2 = site2.occ_element
                pos_idx_2 = site2.occ_position_idx

                site1.occupy_site(elem2, pos_idx_2)
                site2.occupy_site(elem1, pos_idx_1)

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
            self.entropy += orbit.entropy
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
            natoms += sum(orbit.occupation_dict.values())
        return natoms

    def random_generate_structure(self):
        self.is_occupied = True
        for idx in self.disordered_orbit_idx:
            self.random_permute_deterministic_idx(idx)
        for idx in self.ordered_orbit_idx:
            orbit = self.orbits[idx]
            for site in orbit.eq_sites:
                elem = list(site.allowed_elements)[0]
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
            elif orbit.has_vacancy:
                name.append('V')
            elif orbit.is_combined:
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


def crystal_from_pymatgen(stct, config):
    sites = sites_from_pymatgen(stct, config)
    eq_site_dct = group_eq_sites(sites)
    precision = config.get('precision', 0.01)
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
            if min_val > crystal.entropy or max_val < crystal.entropy:
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
        'supercell_mode': 'auto',
        'supercell_criterion': {
            'criterion': 'latt',
            'min': 2,
            'max': 10,
        },
        'percision': 0.01,
        'positional_disorder': {
            'hard_cutoff': 1.,
            'element_cutoff': '../../example/elem_radii.yaml',
            'multiplier': 0.5,
        },
    }
    stct = Structure.from_file(sys.argv[1])
    crystal = crystal_from_pymatgen(stct, config)
    print(crystal)
    crystal.random_generate_structure()
    atoms = crystal.to_ase_atoms()
    breakpoint()
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
