import numpy as np

from docent.util.logger import Logger
from docent.monte_carlo.replica import Replica
from docent.monte_carlo.temperature import AdaptiveTemperatureScheduler


def process_mc_for_replica(config, crystals, calc, save_path):
    logger = Logger()
    mc_method = config['mc_method']
    mc_params = config['mc_params']
    n_cycles = mc_params['n_cycles']
    n_mc_steps = mc_params['n_mc_steps']
    total_iter = ''
    if isinstance(n_cycles, int):  # not adaptive
        total_iter += f'{n_cycles}'

    save = config.get('save', {})
    save_minima_num = save.get('final_minima', 10)
    save_per_mc_step = save.get('per_mc_step', n_mc_steps + 1)
    save_per_cycle = save.get('per_cycle', 1)

    replica = Replica(crystals, save_minima_num)
    replica.calc_energy_of_replica(calc)
    replica.init_t_scheduler_from_config(config)

    if (
        isinstance(replica.t_scheduler, AdaptiveTemperatureScheduler)
        and not isinstance(mc_params['t_low'], str)
    ):
        # adaptive annealing ends when reach target temperature.
        total_iter += '?'

    logger.log_bar()
    logger.writeline(f'Cycle 0/{total_iter}')
    stat_dct = replica.get_statistics()
    for idx in range(len(replica.crystals)):
        stat_dct[f'R{idx+1}']['T (K)'] = None
    logger.recorder.update_recorder(stat_dct)
    logger.log_mc_cycle()
    replica.save_replica_ase_atoms(f'{save_path}/init.extxyz')
    replica.prepare_next_cycle()

    # for cycle in range(n_cycles):
    while True:
        cycle = replica.cycle
        for mc_step in range(n_mc_steps):
            logger.log_progress_bar(
                mc_step,
                mc_params['n_mc_steps'],
                f'Cycle {cycle}/{total_iter}'
            )
            replica.process_single_mc_step(calc, mode=mc_params['mc_mode'])
            if (mc_step + 1) % save_per_mc_step == 0:
                replica.save_replica_ase_atoms(f'{save_path}/C{cycle}M{mc_step}.extxyz')
        logger.finalize_progress_bar()
        if cycle % save_per_cycle == 0:
            replica.save_replica_ase_atoms(f'{save_path}/C{cycle}.extxyz')

        t_before_update = replica.t_scheduler.temperatures
        if stop := replica.t_scheduler.is_stop_iter():
            pass
        elif mc_method == 'pt':
            replica.process_parallel_tempering()
        else:
            replica.process_population_annealing()

        stat_dct = replica.get_statistics()
        for idx, t in enumerate(t_before_update):
            stat_dct[f'R{idx+1}']['T (K)'] = t
            if stop and mc_method == 'pa':
                stat_dct[f'R{idx+1}']['RX accept'] = '----'
        logger.recorder.update_recorder(stat_dct)
        logger.log_mc_cycle()
        if stop:
            break
        replica.prepare_next_cycle()

    replica.save_minima_ase_atoms(f'{save_path}/minima.extxyz')
    replica.save_replica_ase_atoms(f'{save_path}/final.extxyz')

    #return replica
