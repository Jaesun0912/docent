import sys
import numpy as np
import yaml
import os
from copy import deepcopy

from pymatgen.core import Structure

from docent.structure.orbit import crystal_from_pymatgen
from docent.util.logger import Logger
from docent.util.calc import calc_from_config
from docent.util.parse_input import parse_config
from docent.scripts.process_mc import process_mc_for_replica

def _get_all_cifs(fpath):
    abs_path = os.path.abspath(fpath)
    if os.path.exists(abs_path):
        if os.path.isfile(abs_path):
            return [abs_path]
        elif os.path.isdir(abs_path):
            return [
                f'{abs_path}/{d}' for d in os.listdir(abs_path) 
                if d.endswith('.cif')
            ]
        else:
            raise ValueError (f'{fpath} is neither a regular file nor directory.')
    else:
        raise ValueError(f'{fpath} does not exist.')


def _update_config_for_mc(config):
    mc_method = config.get('mc_method', 'pt')
    if '_' in mc_method:
        mc_method = ''.join([s[0].lower() for s in mc_method.split('_')])
    mc_method = 'pt' if mc_method == 'rxmc' else mc_method
    config['mc_method'] = mc_method

    mc_params = config['mc_params']
    if mc_method == 'pt':
        mc_params['n_temperatures'] = mc_params['n_replicas']
    elif mc_method == 'pa':
        mc_params['n_cycles'] = mc_params['n_temperatures']

    return config


def main():
    logger = Logger('log.docent')
    logger.greetings()
    logger.writeline('\nstarting docent\n')

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.writeline(f'Reading config file {config_path} for docent.')
    config_all = parse_config(config)
    logger.writeline('Reading config successful!')
    logger.writeline('\nConfigs for docent')

    config = {}
    for key, conf in config_all.items():
        logger.writeline(f'--------------  {key}  -------------')
        logger.log_config(conf)
        config.update(conf)
        logger.writeline('')
    config = _update_config_for_mc(config)

    np.random.seed(config['seed'])
    calc = calc_from_config(config)
    cifs = _get_all_cifs(config['input_path'])
    cifs.sort()
    logger.init_recorder(config['mc_params']['n_replicas'])

    for idx, cif in enumerate(cifs):
        logger.log_bar()
        logger.writeline(f'Crystal {idx+1}/{len(cifs)}')
        stct = Structure.from_file(cif)
        supercell = crystal_from_pymatgen(stct, config)
        try:
            assert supercell is not None
            crystals = []
            for _ in range(config['mc_params']['n_replicas']):
                supercell.random_generate_structure()
                crystals.append(deepcopy(supercell))
            rformula = supercell.info['rformula']
        except:  # case when occupying supercell failed within criterion
            logger.log_bar()
            logger.writeline(f'WARNING: failed to generate supercell for {cif}!')
            logger.writeline('This might means that criteria is too tight for this cif.')
            logger.writeline('We will skip this for Monte Carlo')
            logger.log_bar()
            continue

        save_path = os.path.basename(cif).replace('.cif', '')
        save_path = f'{config["output_path"]}/{save_path}/'
        os.makedirs(save_path, exist_ok=True)
        info = {
            'CIF_path': cif,
            'Formula': supercell.info['rformula'],
            'Supercell': supercell.info['matrix'],
            'Entropy': supercell.entropy,
            'n_permutation': f"{supercell.get_num_every_combination():.2e}",
            'disorder': supercell.get_disorder_symbol(),
            'orbit_disorder': set(supercell.get_disorder_orbit_symbols()),
        }
        logger.log_bar()
        logger.log_config(info)
        process_mc_for_replica(config, crystals, calc, save_path)

    logger.log_terminate()
