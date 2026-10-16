import numpy as np

from docent.util.logger import Logger
from docent.monte_carlo.replica import Replica
from docent.monte_carlo.temperature import AdaptiveTemperatureScheduler
from docent.monte_carlo.lambda_switch import AdaptiveLambdaSampler


def check_t_scheduler_is_adaptive(scheduler):
    check = scheduler if len(scheduler) == 1 else scheduler.scheduler_list[0]
    return isinstance(check, AdaptiveTemperatureScheduler)


def process_mc_for_replica(config, crystals, calc, free_calc, save_path):
    logger = Logger()
    mc_method = config['mc_method']
    mc_params = config['mc_params']
    n_cycles = mc_params['n_cycles']
    n_mc_steps = mc_params['n_mc_steps']

    save = config.get('save', {})
    save_minima_num = save.get('final_minima', 10)
    save_per_mc_step = save.get('per_mc_step', n_mc_steps + 1)
    save_per_cycle = save.get('per_cycle', 1)

    replica = Replica(crystals, save_minima_num)
    replica.init_replica(calc, free_calc, disorder_only=config.get('disorder_only', False))
    replica.init_t_scheduler_from_config(config)
    replica.init_lambda_sampler_from_config(config)

    total_iter = ''
    if isinstance(n_cycles, int):  # not adaptive
        total_iter += f'{n_cycles * len(replica.t_scheduler)}'
    if (
        check_t_scheduler_is_adaptive(replica.t_scheduler)
        and not isinstance(mc_params['t_low'], str)
    ):
        # adaptive annealing ends when reach target temperature.
        total_iter += '?'

    logger.log_bar()
    logger.writeline(f'Cycle 0/{total_iter}')
    stat_dct = replica.get_statistics(mc_method)
    logger.recorder.update_recorder(stat_dct)
    logger.log_mc_cycle()
    replica.save_replica_ase_atoms(f'{save_path}/init.extxyz')
    #replica.prepare_next_cycle()

    # for cycle in range(n_cycles):
    while True:
        if mc_method == 'pt':
            replica.process_parallel_tempering()
        else:
            replica.process_population_annealing(free_calc)
        replica.t_scheduler.update_cycle()
        cycle = replica.cycle

        for mc_step in range(n_mc_steps):
            logger.log_progress_bar(
                mc_step,
                mc_params['n_mc_steps'],
                f'Cycle {cycle}/{total_iter}'
            )
            replica.process_single_mc_step(
                calc,
                free_calc,
                disorder_only=config.get('disorder_only', False),
                mode=mc_params['mc_mode']
            )
            if (mc_step + 1) % save_per_mc_step == 0:
                replica.save_replica_ase_atoms(f'{save_path}/C{cycle}M{mc_step}.extxyz')

        
        if (
            mc_method == 'pa'
            and free_calc is not None 
            and replica.get_current_lambda() == 0
        ):
            # If free energy evaluation with lambda = 0,
            # need free energy evaluation for calculate overlap & save
            # if large overlap, no switching and use inference value as vib free energy
            # else, utilize to integral over lambda switching process
            replica.infer_free_energy(
                free_calc,
                disorder_only=config.get('disorder_only', False)
            )

        if cycle % save_per_cycle == 0:
            replica.save_replica_ase_atoms(f'{save_path}/C{cycle}.extxyz')
        logger.finalize_progress_bar()

        replica.update_mc_result()
        if cycle % save_per_cycle == 0:
            replica.save_mc_result(f'{save_path}/mc_result.pkl')

        stat_dct = replica.get_statistics(mc_method)
        logger.recorder.update_recorder(stat_dct)
        logger.log_mc_cycle()
        if replica.t_scheduler.is_stop_iter():
            break
        replica.prepare_next_cycle()

        #process lambda switching

        if mc_method == 'pt':
            # No switching avail.
            continue


        req_mc_lambda = replica.require_mc_for_lambda_switching(
            mc_params.get('skip_lambda_mc_overlap', 0.95)
        )

        if req_mc_lambda is None:
            # No switching available
            continue

        if req_mc_lambda:
            if replica.get_current_lambda() == 0.:
                # require switching to 1
                replica = process_lambda_switching(
                    config, replica, calc, free_calc, save_path,
                )
                if cycle % save_per_cycle == 0:
                    replica.save_replica_ase_atoms(f'{save_path}/C{cycle}_switched.extxyz')
            # cf) lambda = 1, no need to go back, keep the lane

        else:
            # no mc cost -> make in lambda = 0 lane
            replica.lambda_sampler.reset_sampler()

    replica.save_minima_ase_atoms(f'{save_path}/minima.extxyz')
    replica.save_replica_ase_atoms(f'{save_path}/final.extxyz')
    replica.save_mc_result(f'{save_path}/mc_result.pkl')

    #return replica


def process_lambda_switching(config, replica, calc, free_calc, save_path):
    logger = Logger()
    mc_params = config['mc_params']
    mc_method = config['mc_method']
    n_mc_steps = mc_params['n_mc_steps']
    total_iter = (
        '?' if isinstance(replica.lambda_sampler, AdaptiveLambdaSampler)
        else len(replica.lambda_sampler.lambda_list) - 1
    )
    cycle = 1

    while True:
        replica.process_lambda_resampling()  # TODO
        replica.lambda_sampler.update_lambda()  # TODO

        for mc_step in range(n_mc_steps):
            logger.log_progress_bar(
                mc_step,
                mc_params['n_mc_steps'],
                f'Switching lambda {cycle}/{total_iter}'
            )
            replica.process_single_mc_step(
                calc,
                free_calc,
                disorder_only=config.get('disorder_only', False),
                mode=mc_params['mc_mode']
            )

        logger.finalize_progress_bar()

        replica.update_mc_result()
        replica.save_mc_result(f'{save_path}/mc_result.pkl')

        stat_dct = replica.get_statistics(mc_method)
        for idx, t in enumerate(replica.t_scheduler.temperatures):
            stat_dct[f'R{idx+1}']['T (K)'] = t
            # TODO stat_dct[f'R{idx+1}']['lambda'] = 0

        stat_dct['Final']['T (K)'] = replica.t_scheduler.temperatures[0]
        # TODO stat_dct[f'Final']['lambda'] = 0
        logger.recorder.update_recorder(stat_dct)
        logger.log_mc_cycle()
        replica.prepare_next_cycle()
        if replica.lambda_sampler.is_stop_iter():  # TODO
            break

        cycle += 1

    return replica
