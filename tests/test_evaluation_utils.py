import torch
import pytest
import os
import openmm.unit as u
from itertools import islice
from timewarp.utils.evaluation_utils import (
    sample_on_batches,
    sample_on_single_conditional,
    sample_with_model,
)

from timewarp.utils.dataset_utils import get_dataset
from pathlib import Path
from timewarp.datasets import RawMolDynDataset
from simulation.md import get_simulation_environment_integrator, get_simulation_environment
from timewarp.dataloader import (
    moldyn_dense_collate_fn,
)
from timewarp.utils.openmm import OpenmmPotentialEnergyTorch
from timewarp.tests.assets import get_model_config


def get_data_and_simulation(n_samples):
    # load the data
    step_width = 1000
    train_data, _ = get_dataset(
        "AD-1", cache_dir=Path(".data"), step_width=step_width  # , splits=["train"]
    )
    data_dir = train_data.data_dir
    if train_data.downloader is not None:
        train_data.download_all()
    protein = "ad1"
    raw_dataset = RawMolDynDataset(data_dir=data_dir, step_width=step_width)
    pdb_names = [protein]
    raw_iterator = raw_dataset.make_iterator(pdb_names)
    batches = (moldyn_dense_collate_fn([datapoint]) for datapoint in raw_iterator)
    batches = list(islice(batches, n_samples))

    state0pdbpath = os.path.join(data_dir, f"{protein}-traj-state0.pdb")
    parameters = "alanine-dipeptide"
    simulation = get_simulation_environment(state0pdbpath, parameters)
    integrator = get_simulation_environment_integrator(parameters)
    system = simulation.system
    openmm_potential_energy_torch = OpenmmPotentialEnergyTorch(
        system, integrator, platform_name="CPU"
    )
    num_atoms = system.getNumParticles()
    masses = [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(num_atoms)]
    masses = torch.tensor(masses)
    return batches, openmm_potential_energy_torch, simulation, masses


@pytest.mark.parametrize("random_velocities", [True, False])
def test_sample_on_batches(random_velocities, device):
    batches, openmm_potential_energy_torch, _, masses = get_data_and_simulation(100)
    model, config = get_model_config("transformer_nvp")
    model = model.to(device)
    (
        y_coords_model,
        y_velocs_model,
        traj_coords,
        traj_velocs,
        traj_coords_conditioning,
        _,
        ll_reverse,
        ll_forward,
        ll_reverse_training,
        ll_forward_training,
        acceptance,
    ) = sample_on_batches(
        batches,
        model,
        device,
        openmm_potential_energy_torch,
        True,
        masses.to(device),
        random_velocs=random_velocities,
    )
    assert y_coords_model.shape == y_velocs_model.shape
    assert y_coords_model.shape == traj_coords.shape
    assert ll_reverse.shape == ll_forward.shape


@pytest.mark.parametrize("random_velocities", [True, False])
def test_sample_on_single_conditional(random_velocities, device):
    batches, _, simulation, _ = get_data_and_simulation(100)
    model, _ = get_model_config("transformer_nvp")
    model = model.to(device)
    (
        y_coords_model_conditional,
        y_velocs_model_conditional,
        traj_coords_conditional,
        traj_velocs_conditional,
        traj_coords_conditioning_conditional,
    ) = sample_on_single_conditional(
        batches[2],
        model,
        num_samples=10,
        sim=simulation,
        step_width=1000,
        random_velocs=random_velocities,
        device=device,
    )
    assert y_coords_model_conditional.shape == y_velocs_model_conditional.shape
    assert y_coords_model_conditional.shape == traj_coords_conditional.shape


@pytest.mark.parametrize("random_velocities", [True, False])
@pytest.mark.parametrize("accept", [True, False])
@pytest.mark.parametrize("num_proposal_steps", [1, 10])
@pytest.mark.parametrize("openmm_on_current", [True, False])
def test_sample_with_model(
    random_velocities, accept, num_proposal_steps, openmm_on_current, device
):
    batches, openmm_potential_energy_torch, simulation, masses = get_data_and_simulation(100)
    model, _ = get_model_config("transformer_nvp")
    model = model.to(device)
    sampled_coords, _, _, chain_stats = sample_with_model(
        batches[2],
        model,
        device=device,
        openmm_potential_energy_torch=openmm_potential_energy_torch,
        masses=masses.to(device),
        num_samples=10,
        accept=accept,
        random_velocs=random_velocities,
        resample_velocs=False,
        initialize_randomly=False,
        sim=simulation if openmm_on_current else None,
        openmm_on_current=openmm_on_current,
        openmm_on_proposal=False,
        num_openmm_steps=10 if openmm_on_current else 0,
        num_proposal_steps=num_proposal_steps
        if accept
        else 1,  # can't do parallel proposals if we accept everything
    )
    # `sampled_coords` also include the initial state.
    assert len(sampled_coords) == len(chain_stats.acceptance) + 1
