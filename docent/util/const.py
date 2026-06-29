class Essential:
    pass

kB = 8.617333262e-5  # in eV/K
T_MIN = 1  # in K
MAX_SUPERCELL_MUL = 1000
LOG_ORDER = ['T (K)', 'mean (eV)', 'std (eV)', 'min (eV)', 'Q1 (eV)', 'mid (eV)', 'Q3 (eV)', 'max (eV)', 'MC accept', 'RX accept']
RES_INTERVAL = 3
FULLY_OCCUPY = 0.99

DEFAULT_DATA_CONFIG = {
    'input_path': Essential(),
    'output_path': Essential(),
}


DEFAULT_CALC_CONFIG = {
    'calc_type': 'sevennet',
    'calc_path': Essential(),
    'calc_args': {},
    'disorder_only': False,
    'batch_size': None,
    'avg_atom_num': None,
}


DEFAULT_FREE_CALC_CONFIG = {
    'free_calc_type': None,
    'free_calc_path': '',
    'free_calc_args': {},
    'free_calc_batch_size': None,
    'free_calc_avg_atom_num': None,
}


DEFAULT_SUPERCELL_CONFIG = {
    'supercell_mode': 'auto',
    'supercell_unit': 'orbit',
    'supercell_criterion': {
        'mul': {
            'min': 2,
            'max': 100,
        },
    },
    'oxidation_state_tol': 0.05,
    'max_occ_tol': 1.05,
    'attempt_occ_tol': 0.01,
    'supercell_loss_type': 'l1',
    'supercell_weight_type': None,
    'supercell_selection': 'error',
    'tolerance': 0.005,
    'positional_disorder': {
        'hard_cutoff': 1.0,
        'hydrogen_hard_cutoff': 0.5,
    }
}

DEFAULT_MC_CONFIG = {
    'seed': 777,
    'mc_method': 'pt',
    'mc_params': Essential(),
    'save': {
        'per_mc_step': 1,
        'per_cycle': 1,
        'final_minima': 1,
    }
}

# MAX_CYCLE = 100
