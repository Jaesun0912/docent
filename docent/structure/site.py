from typing import Union, List, Optional
from pymatgen.core.sites import PeriodicSite
import numpy as np
from docent.util.utils import get_intersection


class VirtualSite:
    def __init__(
        self,
        site: PeriodicSite,
        idx: Optional[int] = None
    ):
        self.site = site
        species = {}
        oxidation = {}
        self.label = site.label
        # similar to PeriodicSite.species, but ignore oxidation number
        # since in practice, dft and mlp does not distinguish them
        for specie in site.as_dict()['species']:
            element = specie['element']
            occu = specie['occu']
            if element not in species:
                species[element] = 0.
            species[element] += occu
            oxidation[element] = specie['oxidation_state']

        self.species = {self.label: species}
        self.oxidation = {self.label: oxidation}
        self.is_occ = False
        self.occ_label = None
        self.occ_element = None
        self.occ_position = None
        self.occ_position_idx = None

        self.allowed_labels = {self.label}
        self.allowed_elements = {self.label: site.species.chemical_system_set}
        self.allowed_positions = {
            self.label: {k: [site.coords] for k in site.species.chemical_system_set}
        }
        self.idx = idx


    def __len__(self):
        return 1


    def __str__(self):
        return f'VSite {self.label} {self.species}'


    def occupy_site(
        self,
        element: str,
        label: Optional[str] = None,
        position_idx: int = 0,
    ):
        assert element in self.allowed_elements[self.label]
        self.is_occ = True
        self.occ_label = self.label
        self.occ_element = element
        self.occ_position = self.allowed_positions[self.label][element][position_idx]
        self.occ_position_idx = position_idx
        

    def vacant_site(self):
        self.is_occ = False
        self.occ_label = None
        self.occ_element = None
        self.occ_position = None
        self.occ_position_idx = None


class CombinedSite:
    def __init__(
        self,
        sites: List[VirtualSite],
    ):
        labels_sorted = sorted(list(set([site.label for site in sites])))
        """
        self.sites = []
        for label in labels_sorted:
            for site in sites:
                if label != site.label:
                    continue
                self.sites.append(site)  # sort by label to preserve position order
        """
        self.sites = sites
        self.label = ''.join(labels_sorted)
        self.species = {}
        self.oxidation = {}
        self.allowed_labels = set(labels_sorted)
        self.allowed_positions = {}
        self.allowed_elements = {}
        for site in self.sites:
            self.oxidation.update(site.oxidation)
            self.allowed_elements.update(site.allowed_elements)
            if site.label not in self.species:
                self.species[site.label] = {}
                self.allowed_positions[site.label] =\
                    {e: [] for e in site.allowed_elements[site.label]}

            for occ_dict in site.species.values():
                for elem, occu in occ_dict.items():
                    if elem not in self.species[site.label]:
                        self.species[site.label][elem] = 0.
                    self.species[site.label][elem] += occu

            for elem, pos in site.allowed_positions[site.label].items():
                self.allowed_positions[site.label][elem] += pos

        self.is_occ = False
        self.occ_label = None
        self.occ_element = None
        self.occ_position = None
        self.occ_position_idx = None
        self.idx = [site.idx for site in self.sites]

        """
        all_elems = []
        for site in self.sites:
            all_elems += list(site.allowed_elements)
        self.allowed_elements = set(all_elems)
        self.allowed_positions = {e: [] for e in self.allowed_elements}
        for site in self.sites:
            all_pos = site.allowed_positions
            for elem, pos in all_pos.items():
                self.allowed_positions[elem] += pos
        """


    def __len__(self):
        return len(self.sites)


    def __str__(self):
        return f'CSite {self.label} {self.species}'


    def occupy_site(
        self,
        element: str,
        label: str,
        position_idx: Optional[int] = None,
    ):
        assert label in self.allowed_labels
        assert element in self.allowed_elements[label]
        self.is_occ = True
        self.occ_label = label
        self.occ_element = element
        pos_candidates = self.allowed_positions[label][element]
        if position_idx is None:
            position_idx = np.random.choice(np.arange(len(pos_candidates)), 1)[0]
        self.occ_position = pos_candidates[position_idx]
        self.occ_position_idx = position_idx
        

    def vacant_site(self):
        self.is_occ = False
        self.occ_label = None
        self.occ_element = None
        self.occ_position = None
        self.occ_position_idx = None


def get_radii_from_dict(pmg_sites, radii_dict):
    # TODO: change pmg_sites to VSite
    radii = []
    for site in pmg_sites:
        species = site.as_dict()['species']
        radius = []
        for specie in species:
            elem = specie['element']
            ox = int(round(specie['oxidation_state'], 0))
            if ox in radii_dict[elem]['ionic']:
                radius.append(radii_dict[elem]['ionic'][ox])
            else:
                radius.append(radii_dict[elem]['empirical'])
        radii.append(sum(radius)/len(radius))

    return radii


def group_eq_sites(sites):
    eq_site_dct = {}
    for site in sites:
        label = site.label
        if label not in eq_site_dct:
            eq_site_dct[label] = []
        eq_site_dct[label].append(site)
    return eq_site_dct


def sites_from_pymatgen(stct, config):
    pmg_sites = stct.sites
    cell = stct.lattice.matrix
    positions = []
    for site in pmg_sites:
        pos = site.coords
        positions.append(pos)
    radii = config['positional_disorder'].get('element_cutoff', None)
    multiplier = config['positional_disorder'].get('multiplier', 1.)
    hard_cutoff = config['positional_disorder'].get('hard_cutoff', None)
    if isinstance(radii, str):
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
        radii = get_radii_from_dict(pmg_sites, radii_dict)
        radii = [r*multiplier for r in radii]

    all_sites = []
    combined_sites = get_intersection(positions, cell, radii, hard_cutoff)
    idx_in_csite = []
    for combined_site_idx in combined_sites:
        idx_in_csite += list(combined_site_idx)
        site_list = []
        for idx in combined_site_idx:
            vsite = VirtualSite(pmg_sites[idx], idx)
            site_list.append(vsite)
        csite = CombinedSite(site_list)
        all_sites.append(csite)

    for idx, site in enumerate(pmg_sites):
        if idx in idx_in_csite:
            continue
        vsite = VirtualSite(site, idx)
        all_sites.append(vsite)
    return all_sites

