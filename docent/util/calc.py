import numpy as np
from tqdm import tqdm

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

try:
    import torch
    from torch_geometric.loader import DataLoader
    import sevenn._keys as KEY
    from sevenn.atom_graph_data import AtomGraphData
    from sevenn.train.atoms_dataset import SevenNetAtomsDataset
    from sevenn.train.modal_dataset import SevenNetMultiModalDataset
    import sevenn.train.dataload as dataload
    from sevenn.train.dataload import _set_atoms_y
    from sevenn.calculator import SevenNetCalculator
except:
    # Dummy class to avoid error
    class SevenNetCalculator:
        pass


class SevenNetEnergyCalculator(SevenNetCalculator):
    def __init__(
        self,
        model='7net-0',
        file_type='checkpoint',
        device='auto',
        modal=None,
        use_avg_model=False,
        compile=False,
        compile_kwargs=None,
        enable_cueq=False,
        sevennet_config=None,
        batch_size=None,
        avg_atom_num=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            file_type=file_type,
            device=device,
            modal=modal,
            use_avg_model=use_avg_model,
            compile=compile,
            compile_kwargs=compile_kwargs,
            enable_cueq=enable_cueq,
            sevennet_config=sevennet_config,
            **kwargs
        )
        if 'force_output' in self.model._modules:
            del self.model.force_output
        self.implemented_properties = ['energy']

    def output_to_results(self, output):
        energy = output[KEY.PRED_TOTAL_ENERGY].detach().cpu().item()
        return {'energy': energy}


    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call parent class to set necessary atom attributes
        Calculator.calculate(self, atoms, properties, system_changes)

        if atoms is None:
            raise ValueError('No atoms to evaluate')

        data = AtomGraphData.from_numpy_dict(
            unlabeled_atoms_to_graph(atoms, self.cutoff)
        )

        if self.modal:
            data[KEY.DATA_MODALITY] = self.modal

        data.to(self.device)  # type: ignore

        if isinstance(self.model, torch_script_type):
            data = self._preprocess_data_for_torchscript(data)
        elif self.is_compiled():
            if self._pad is None:
                self.set_padding_nums_for_compile(example_graph=data)  # type: ignore
            data = self.pad(data)

        with torch.no_grad():
            output = self.model(data).detach().cpu()
        self.results = self.output_to_results(output)


class SevenNetBatchCalculator(SevenNetEnergyCalculator):
    # TODO: implement this in original sevennet
    # To avoid dependency issues with pyte
    def __init__(
        self,
        model='7net-0',
        file_type='checkpoint',
        device='auto',
        modal=None,
        use_avg_model=False,
        compile=False,
        compile_kwargs=None,
        enable_cueq=False,
        sevennet_config=None,
        batch_size=None,
        avg_atom_num=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            file_type=file_type,
            device=device,
            modal=modal,
            use_avg_model=use_avg_model,
            compile=compile,
            compile_kwargs=compile_kwargs,
            enable_cueq=enable_cueq,
            sevennet_config=sevennet_config,
            **kwargs
        )
        if batch_size is None and avg_atom_num is None:
            raise ValueError('one of batch size or avg_atom_num should be given')

        self.batch_size = batch_size
        self.avg_atom_num = avg_atom_num


    def batch_calculate(self, atoms_list, desc=None):
        self.model.set_is_batch_data(True)
        dataset = SevenNetAtomsDataset(self.cutoff, [])
        def _unlabeled_graph_build(self, atoms):
            #return dataload.atoms_to_graph(
            return dataload.unlabeled_atoms_to_graph(
            atoms,
            self.cutoff,
            # transfer_info=False,
            # y_from_calc=False,
            # allow_unlabeled=True,
        )
        SevenNetAtomsDataset._graph_build = _unlabeled_graph_build  # TODO: avoid monkey patching
        # atoms_list = _set_atoms_y(atoms_list)
        dataset._atoms_list = atoms_list

        if self.modal is not None:
            dataset = SevenNetMultiModalDataset({self.modal: dataset})

        if self.batch_size is None:
            total_atom_num = sum([len(atoms) for atoms in atoms_list])
            batch_size = int(self.avg_atom_num * len(atoms_list) / total_atom_num)
        else:
            batch_size = self.batch_size

        batch_size = max(1, batch_size)
        loader = DataLoader(dataset, batch_size, shuffle=False)
        result_dict_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = batch.to(self.device)
                output_list = self.model(batch).detach().cpu()
                for energy in torch.unbind(output_list[KEY.PRED_TOTAL_ENERGY]):
                    result_dict = {'energy': energy.item()}
                    result_dict_list.append(result_dict)

        result = []
        for atoms, result_dict in zip(atoms_list, result_dict_list):
            single_calc = SinglePointCalculator(atoms, **result_dict)
            result.append(single_calc.get_atoms())

        return result


class ThreeNetBatchCalculator(SevenNetBatchCalculator):
    # TODO: implement this in original sevennet
    # To avoid dependency issues with pyte
    def __init__(
        self,
        model='7net-0',
        file_type='checkpoint',
        device='auto',
        modal=None,
        use_avg_model=False,
        compile=False,
        compile_kwargs=None,
        enable_cueq=False,
        sevennet_config=None,
        batch_size=None,
        avg_atom_num=None,
        converged_temperature=3000,
        **kwargs
    ):
        super().__init__(
            model=model,
            file_type=file_type,
            device=device,
            modal=modal,
            use_avg_model=use_avg_model,
            compile=compile,
            compile_kwargs=compile_kwargs,
            enable_cueq=enable_cueq,
            sevennet_config=sevennet_config,
            batch_size=batch_size,
            avg_atom_num=avg_atom_num,
            **kwargs
        )
        self.converged_temperature = converged_temperature
        self.model.infer_heat_capacity(False)


    def _calculate_lt(self, loader, desc):
        result_dict_list = []
        self.model.infer_grad_entropy(True)
        self.model.infer_debye(False)
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(self.device)
            output_list = self.model(batch).detach().cpu()
            for internal_energy, entropy, free_energy in zip(
                torch.unbind(output_list[KEY.PRED_TOTAL_INTERNAL_ENERGY]),
                torch.unbind(output_list[KEY.PRED_TOTAL_ENTROPY]),
                torch.unbind(output_list[KEY.PRED_TOTAL_FREE_ENERGY]),
            ):
                result_dict = {
                    'internal_energy': internal_energy.item(),
                    'vib_entropy': entropy.item(),
                    'vib_free_energy': free_energy.item(),
                }
                result_dict_list.append(result_dict)
        return result_dict_list


    def _calculate_ht(self, loader, desc):
        result_dict_list = []
        self.model.infer_grad_entropy(False)
        self.model.infer_debye(False)
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = batch.to(self.device)
                output_list = self.model(batch).detach().cpu()
                for temperature, head_asymptot, head_entropy, free_energy in zip(
                    torch.unbind(output_list[KEY.TEMPERATURE]),
                    torch.unbind(output_list[KEY.PRED_TOTAL_HEAD_ASYMPTOT]),
                    torch.unbind(output_list[KEY.PRED_TOTAL_HEAD_ENTROPY]),
                    torch.unbind(output_list[KEY.PRED_TOTAL_FREE_ENERGY]),
                ):
                    A, T = head_asymptot.item(), temperature.item()
                    result_dict = {
                        'internal_energy': -2 * A / T,
                        'vib_entropy': head_entropy.item() - A / (T**2),
                        'vib_free_energy': free_energy.item(),
                    }
                    result_dict_list.append(result_dict)
        return result_dict_list


    def _calculate_debye(self, loader, desc):
        result_dict_list = []
        self.model.infer_debye(True)
        debye_block = self.model.debye_block
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = batch.to(self.device)
                output_list = debye_block(batch).detach().cpu()
                for internal_energy, entropy, free_energy in zip(
                    torch.unbind(output_list[KEY.DEBYE_INTERNAL_ENERGY]),
                    torch.unbind(output_list[KEY.DEBYE_ENTROPY]),
                    torch.unbind(output_list[KEY.DEBYE_FREE_ENERGY]),
                ):
                    result_dict = {
                        'debye_internal_energy': internal_energy.item(),
                        'debye_entropy': entropy.item(),
                        'debye_free_energy': free_energy.item(),
                    }
                    result_dict_list.append(result_dict)
        return result_dict_list        


    def batch_calculate(self, atoms_list, temperature_list, infer_debye=False, desc=None):
        self.model.set_is_batch_data(True)

        if isinstance(temperature_list, (int, float)):
            temperature_list = [temperature_list]*len(atoms_list)
        temperature_list = list(map(float, temperature_list))

        lt_atoms_list, ht_atoms_list = [], []
        lt_ht_indices, is_ht_list = [], []

        for idx, (atoms, temperature) in enumerate(zip(atoms_list, temperature_list)):
            atoms.info['temperature'] = temperature
            if temperature > self.converged_temperature:
                lt_ht_indices.append(len(ht_atoms_list))
                is_ht_list.append(True)
                ht_atoms_list.append(atoms)
            else:
                lt_ht_indices.append(len(lt_atoms_list))
                is_ht_list.append(False)
                lt_atoms_list.append(atoms)

        lt_dataset = SevenNetAtomsDataset(self.cutoff, [])
        ht_dataset = SevenNetAtomsDataset(self.cutoff, [])
        debye_dataset = SevenNetAtomsDataset(self.cutoff, [])
        def _unlabeled_graph_build(self, atoms):
            #return dataload.atoms_to_graph(
            return dataload.unlabeled_atoms_to_graph(
            atoms,
            self.cutoff,
            # transfer_info=False,
            # y_from_calc=False,
            # allow_unlabeled=True,
        )
        SevenNetAtomsDataset._graph_build = _unlabeled_graph_build  # TODO: avoid monkey patching
        # atoms_list = _set_atoms_y(atoms_list)
        lt_dataset._atoms_list = lt_atoms_list
        ht_dataset._atoms_list = ht_atoms_list

        if infer_debye:
            unique_T, atoms_idx, temperature_idx = np.unique(temperature_list, return_index=True, return_inverse=True)
            debye_atoms_list = [atoms_list[idx] for idx in atoms_idx]
            debye_dataset._atoms_list = debye_atoms_list

        if self.modal is not None:
            lt_dataset = SevenNetMultiModalDataset({self.modal: lt_dataset})
            ht_dataset = SevenNetMultiModalDataset({self.modal: ht_dataset})

        if self.batch_size is None:
            total_atom_num = sum([len(atoms) for atoms in atoms_list])
            batch_size = int(self.avg_atom_num * len(atoms_list) / total_atom_num)
        else:
            batch_size = self.batch_size

        batch_size = max(1, batch_size)
        lt_loader = DataLoader(lt_dataset, batch_size, shuffle=False)
        ht_loader = DataLoader(ht_dataset, batch_size, shuffle=False)
        result_dict_list = []
        lt_result_dict_list = [] if len(lt_atoms_list) == 0 else self._calculate_lt(lt_loader, desc)
        ht_result_dict_list = [] if len(ht_atoms_list) == 0 else self._calculate_ht(ht_loader, desc)

        if infer_debye:
            debye_loader = DataLoader(debye_dataset, batch_size=len(debye_dataset), shuffle=False)
            unique_debye_result_dict_list = self._calculate_debye(debye_loader, desc)
            debye_result_dict_list = []
            for atoms, t_idx in zip(atoms_list, temperature_idx):
                debye_result_dict_list.append({k: v*len(atoms) for k, v in unique_debye_result_dict_list[t_idx].items()})
        else:
            debye_result_dict_list = [
                {'debye_internal_energy': None, 'debye_entropy': None, 'debye_free_energy': None}
            ] * len(atoms_list)
            

        result = []
        for idx, (atoms, lt_ht_idx, is_ht) in enumerate(zip(atoms_list, lt_ht_indices, is_ht_list)):
            result_dict = ht_result_dict_list[lt_ht_idx] if is_ht else lt_result_dict_list[lt_ht_idx]
            atoms.info.update(result_dict)
            atoms.info.update(debye_result_dict_list[idx])
            result.append(atoms)

        return result


def calc_from_py(script):
    import importlib.util
    from pathlib import Path

    file_path = Path(script).resolve()
    spec = importlib.util.spec_from_file_location('generate_calc', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calc = module.generate_calc()
    return calc


def calc_from_config(config):
    calc_type = config['calc_type'].lower()
    calc_args = config.get('calc_args', {})

    if calc_type == 'sevennet':
        return SevenNetCalculator(model=config['calc_path'], **calc_args)

    elif calc_type == 'sevennet-energy':
        return SevenNetEnergyCalculator(model=config['calc_path'], **calc_args)

    elif calc_type == 'sevennet-batch':
        batch_size = config.get('batch_size', None)
        avg_atom_num = config.get('avg_atom_num', None)
        return SevenNetBatchCalculator(
            model=config['calc_path'],
            batch_size=batch_size,
            avg_atom_num=avg_atom_num,
            **calc_args,
        )
    elif calc_type == 'custom':
        script = config['calc_path']
        return calc_from_py(script)

    else:
        raise NotImplementedError


def free_calc_from_config(config):
    calc_type = config.get('free_calc_type', None)
    calc_args = config.get('free_calc_args', {})
    if calc_type is None:
        return None

    if calc_type == 'threenet-batch':
        batch_size = config.get('free_calc_batch_size', None)
        avg_atom_num = config.get('free_calc_avg_atom_num', None)
        return ThreeNetBatchCalculator(
            model=config['free_calc_path'],
            batch_size=batch_size,
            avg_atom_num=avg_atom_num,
            **calc_args,
        )

    elif calc_type == 'custom':
        script = config['free_calc_path']
        return calc_from_py(script)

    else:
        raise NotImplementedError


def single_point_calculate(atoms, calc):
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    calc_results = {"energy": energy}
    calculator = SinglePointCalculator(atoms, **calc_results)
    new_atoms = calculator.get_atoms()

    return new_atoms


def single_point_calculate_list(atoms_list, calc, desc=None):
    calculated = []
    for atoms in tqdm(atoms_list, desc=desc, leave=False):
        calculated.append(single_point_calculate(atoms, calc))

    return calculated


def get_energy_of_atoms_list(atoms_list, calc, desc='atoms oneshot'):
    if isinstance(calc, SevenNetBatchCalculator):
        result = calc.batch_calculate(atoms_list, desc=desc)
    else:
        result = single_point_calculate_list(atoms_list, calc, desc=desc)

    energy_list = [atoms.get_potential_energy() for atoms in result]
    return energy_list


def get_free_energy_of_atoms_list(atoms_list, temperature_list, infer_debye, calc, desc='atoms free oneshot'):
    if isinstance(calc, ThreeNetBatchCalculator):
        result = calc.batch_calculate(atoms_list, temperature_list, infer_debye, desc=desc)
    else:
        raise NotImplementedError

    free_keys = ['vib_free_energy', 'vib_entropy', 'internal_energy']
    free_keys += ['debye_free_energy', 'debye_entropy', 'debye_internal_energy']
    free_energy_list = [{k: atoms.info[k] for k in free_keys} for atoms in result]
    return free_energy_list
