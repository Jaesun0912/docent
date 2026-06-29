import numpy as np
import spglib


def generate_hermite_normal_form(det):
    hnfs = []
    for a in range(1, det + 1):
        if det % a != 0:
            continue
        md = det // a

        for d in range(1, md + 1):
            if md % d != 0:
                continue
            f = md // d

            for b in range(d):
                for c in range(f):
                    for e in range(f):
                        H = (
                            (a, b, c),
                            (0, d, e),
                            (0, 0, f)
                        )
                        hnfs.append(H)
    return np.array(hnfs)


def calc_abc(L):
    # L = transforms @ lattice
    return np.linalg.norm(L, axis=2)


def calc_cubicity(L):
    # L = transforms @ lattice
    G = L @ np.transpose(L, (0, 2, 1))
    alpha = np.trace(G, axis1=1, axis2=2) / 3.0
    I = np.eye(3)
    G_cubic = alpha[:, None, None] * I
    diff_norm = np.linalg.norm(G - G_cubic, axis=(1, 2))
    ref_norm = np.linalg.norm(G_cubic, axis=(1, 2))
    cubicity = 1.0 - diff_norm / ref_norm

    return cubicity


def get_possible_lattice_transforms(stct, det):
    lattice = stct.lattice.matrix
    hnfs = generate_hermite_normal_form(det)
    sup_lattices = hnfs @ lattice
    niggli_lattices = []
    for HL in sup_lattices:
        nrl = spglib.niggli_reduce(HL)  # faster than pymatgen
        if nrl is None:
            niggli_lattices.append(HL)
        else:
            niggli_lattices.append(nrl)

    niggli_transforms = niggli_lattices @ np.linalg.inv(sup_lattices)
    niggli_transforms = np.rint(niggli_transforms).astype(int)
    transforms = niggli_transforms @ hnfs

    mask = np.rint(np.linalg.det(transforms)).astype(int) != det  # something wrong in niggli
    transforms[mask] = hnfs[mask]

    return transforms


if __name__=='__main__':
    from pymatgen.core import Structure
    import sys
    stct = Structure.from_file(sys.argv[1])
    #hnfs = generate_hermite_normal_form(2)
    transforms = get_possible_lattice_transforms(stct, 2)
    breakpoint()

