import numpy as np
from collections import defaultdict
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
from scipy.special import comb
from itertools import combinations_with_replacement
MAX_SUPERCELL_MUL = 10


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, u):
        if u not in self.parent:
            self.parent[u] = u
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pu] = pv

    def groups(self):
        components = defaultdict(set)
        for node in self.parent:
            root = self.find(node)
            components[root].add(node)
        return list(components.values())


def get_statistics_of_list(lst):
    mean = np.mean(lst)
    std = np.std(lst)
    mini = min(lst)
    q1 = np.quantile(lst, 0.25)
    mid = np.quantile(lst, 0.5)
    q3 = np.quantile(lst, 0.75)
    maxi = max(lst)

    return {
        'mean (eV)': mean,
        'std (eV)': std,
        'min (eV)': mini,
        'Q1 (eV)': q1,
        'mid (eV)': mid,
        'Q3 (eV)': q3,
        'max (eV)': maxi
    }


def combination(total, samples):
    num = 1
    for sample in samples:
        num *= comb(total, sample)
        total -= sample

    return int(num)


def diff_to_integer(num):
    integer = round(num)
    return abs(integer-num)


def get_eqsite_total_occu(eq_site):
    eq_site_occus = eq_site[0].species.copy()
    for site in eq_site[1:]:
        for specie, occu in  site.species.items():
            eq_site_occus[specie] += occu
    return eq_site_occus


def get_possible_supercell_matrix(eq_sites, precision=0.01):
    all_occus = []
    for eq_site in eq_sites:
        eq_site_occus = get_eqsite_total_occu(eq_site)
        all_occus += list(eq_site_occus.values())

    possible = []
    combs = list(combinations_with_replacement(range(1, MAX_SUPERCELL_MUL+1), 3))
    combs.sort(key=max)
    for i, j, k in combs:
        all_nums = [occu*i*j*k for occu in all_occus]
        if all([diff_to_integer(num) < precision for num in all_nums]):
            possible.append((i, j, k))
    return possible


def get_occupation_dict(eq_site, na, nb, nc):
    mul = na*nb*nc
    eq_site_occus = get_eqsite_total_occu(eq_site)
    return {k: round(mul*v) for k, v in eq_site_occus.items()}


def get_intersection(pos, cell, radii=None, hard_cutoff=None):
    # radii: multiplier must be pre-processed.
    assert radii is not None or hard_cutoff is not None

    cutoff_max = []
    if radii is None:
        radii = [0.]*len(pos)
    if hard_cutoff is None:
        hard_cutoff = 0.
    cutoff_max = max(hard_cutoff, max(radii)*2)

    edge_src, edge_dst, edge_vec = primitive_neighbor_list(
        'ijD', [True, True, True], cell, np.array(pos), cutoff_max, self_interaction=False
    )

    inter_src, inter_dst = [], []
    for src, dst, vec in zip(edge_src, edge_dst, edge_vec):
        dist = np.linalg.norm(vec)
        if dist < max(radii[src]+radii[dst], hard_cutoff):
            inter_src.append(src)
            inter_dst.append(dst)

    if len(inter_src) == 0:
        return []

    uf = UnionFind()
    for src, dst in zip(inter_src, inter_dst):
        uf.union(src, dst)

    return uf.groups()


def add_if_is_not_adaptive(tconfig, key, t_kwargs):
    if isinstance(tconfig[key], str) and tconfig[key].lower() == 'adaptive':
        return
    t_kwargs.update({key: tconfig[key]})


if __name__=='__main__':
    list1 = [0, 1,]
    list2 = [1, 0,]

    uf = UnionFind()
    for u, v in zip(list1, list2):
        uf.union(u, v)

    print(uf.groups())
