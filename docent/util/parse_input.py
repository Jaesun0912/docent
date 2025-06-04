import os
from docent.util.const import (
    Essential,
    DEFAULT_DATA_CONFIG,
    DEFAULT_CALC_CONFIG,
    DEFAULT_SUPERCELL_CONFIG,
    DEFAULT_MC_CONFIG,
)


def update_config_with_defaults(config):
    key_parse_pair = {
        'data': DEFAULT_DATA_CONFIG,
        'calculator': DEFAULT_CALC_CONFIG,
        'supercell': DEFAULT_SUPERCELL_CONFIG,
        'monte_carlo': DEFAULT_MC_CONFIG,
    }

    for key, default_config in key_parse_pair.items():
        config_parse = default_config.copy()
        config_parse.update(config[key])

        for k, v in config_parse.items():
            if not isinstance(v, Essential):
                continue
            raise ValueError(f'{key}: {k} must be given')
        config[key] = config_parse

    return config


def _isinstance_in_list(inp, insts):
    return any([isinstance(inp, inst) for inst in insts])


def _islistinstance(inps, insts):
    return all([_isinstance_in_list(inp, insts) for inp in inps])


def check_calc_config(config):
    config_calc = config['calculator']
    assert config_calc['calc_type'].lower() in ['sevennet', 'sevennet-energy', 'sevennet-batch', 'custom']
    assert isinstance(config_calc['calc_path'], str)
    assert _isinstance_in_list(config_calc['batch_size'], [int, type(None)])
    assert _isinstance_in_list(config_calc['avg_atom_num'], [int, type(None)])


def check_supercell_config(config):
    config_sup = config['supercell']
    assert isinstance(config_sup['sanitize_cif_neutral'], dict)
    assert all(
        [
            k.lower() in ['attempt_tolerance', 'relative_error', 'multiplicity_power', 'clip']
            for k in config_sup['sanitize_cif_neutral'].keys()
        ]
    )
    assert isinstance(config_sup['supercell_mode'], str)
    assert 'x' in config_sup['supercell_mode'].lower() or config_sup['supercell_mode'].lower() == 'auto'
    assert isinstance(config_sup['supercell_unit'], str)
    assert config_sup['supercell_unit'].lower() in ['orbit', 'site']
    assert isinstance(config_sup['supercell_criterion'], dict)
    assert all(
        [
            k.lower() in ['mul', 'latt', 'natom', 'entropy', 'permutation']
            for k in config_sup['supercell_criterion'].keys()
        ]
    )
    assert all(
        [
            _isinstance_in_list(v['min'], [int, float]) and _isinstance_in_list(v['max'], [int, float])
            for v in config_sup['supercell_criterion'].values()
        ]
    )
    assert config_sup['supercell_selection'].lower() in ['size', 'error', 'product']
    assert isinstance(config_sup['tolerance'], float)
    assert config_sup['tolerance'] < 1
    assert isinstance(config_sup['positional_disorder'], dict)
    assert _isinstance_in_list(config_sup['positional_disorder'].get('hard_cutoff', 1.), [int, float])
    assert isinstance(config_sup['positional_disorder'].get('element_cutoff', '.yaml'), str)
    assert _isinstance_in_list(config_sup['positional_disorder'].get('multiplier', 1), [int, float])


def check_mc_config(config):
    config_mc = config['monte_carlo']
    assert isinstance(config_mc['seed'], int)
    assert isinstance(config_mc['mc_method'], str)
    assert (mc := config_mc['mc_method'].lower()) in ['pt', 'parallel_tempering', 'rxmc', 'pa', 'population_annealing']
    assert isinstance((param := config_mc['mc_params']), dict)
    assert isinstance(param['n_replicas'], int)
    assert isinstance(param['n_temperatures'], int) or param['n_temperatures'].lower() == 'adaptive'
    assert isinstance(param['n_cycles'], int) or param['n_cycles'].lower() == 'adaptive'
    assert isinstance(param['n_mc_steps'], int)
    assert _isinstance_in_list(param['t_low'], [int, float]) or param['t_low'].lower() == 'adaptive'
    assert _isinstance_in_list(param['t_high'], [int, float]) or param['t_high'].lower() in ['inf', 'adaptive']
    assert isinstance(param['t_schedule_mode'], str) and param['t_schedule_mode'].lower() in ['temperature', 'beta', 'adaptive']
    assert param['mc_mode'].lower() in ['exchange', 'permute']
    assert isinstance(ov:=param.get('pa_overlap', 0.5), float) and 0. < ov and ov < 1.
    assert isinstance(config_mc['save'], dict)
    for v in config_mc.get('save', {}).values():
        assert isinstance(v, int)

    ad_keys = []
    for key, val in param.items():
        if isinstance(val, str) and val.lower() == 'adaptive':
            ad_keys.append(key)
    if mc == 'pt' and len(ad_keys) > 0:
        raise NotImplementedError('Adaptive temperature scheduling not implemented for parallel tempering')
    check_adaptive(ad_keys)


def check_adaptive(ad_keys):
    if ('n_cycles' in ad_keys or 'n_temperatures' in ad_keys) and 't_schedule_mode' not in ad_keys:
        raise ValueError('n_cycles set as adaptive but t_schedule_mode is deterministic!')

    if 't_low' in ad_keys and ('n_cycles' in ad_keys or 't_schedule_mode' not in ad_keys):
        raise ValueError('either n_cycles or t_low should be determined in adaptive scheduling!')


def parse_config(config):
    config = update_config_with_defaults(config)
    check_calc_config(config)
    check_supercell_config(config)
    check_mc_config(config)

    return config

