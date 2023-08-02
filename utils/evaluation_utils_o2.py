import torch
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

from tqdm.auto import tqdm

# O2 constants
# Spring constant in kJ/(mol nm^2)
k = 248940
# Minimal bondlength in nm
r0 = 0.1016
# Oxygen mass in Dalton
oxygen_mass = 15.999
# Boltzmann constant times temperature in kJ/mol
kbT = 2.577483411627504


def create_o2_system():
    system = mm.System()
    system.addParticle(oxygen_mass * u.amu)
    system.addParticle(oxygen_mass * u.amu)
    topology = app.Topology()
    topology.addChain("O2")
    O2chain = list(topology.chains())[0]
    topology.addResidue("O2", O2chain)
    O2residue = list(topology.residues())[-1]
    OA = topology.addAtom("OA", app.Element.getBySymbol("O"), O2residue)
    OB = topology.addAtom("OB", app.Element.getBySymbol("O"), O2residue)
    topology.addBond(OA, OB)
    # harmonic bond force
    harm_force = mm.CustomBondForce("0.5*k*(r-r0)^2")

    harm_force.addGlobalParameter("k", k * u.kilojoule_per_mole / u.nanometer**2)
    harm_force.addGlobalParameter("r0", r0 * u.nanometer)
    harm_force.addBond(0, 1, [])
    system.addForce(harm_force)
    return system, topology


def harm_osci_prob(x):
    return np.exp(-0.5 * k / kbT * (x - r0) ** 2) * (2 * np.pi * kbT / k) ** (-0.5)


def compute_bond_length(coords):
    return np.linalg.norm(coords[:, 0, :] - coords[:, 1, :], axis=-1)


# sampling
def sample_with_model(n_samples, model, initial_batch, target_energy, system, device):
    assert initial_batch.atom_coords.shape[0] == 1, "Batch with single initial sample required."

    x_coords = initial_batch.atom_coords.to(device, non_blocking=True)
    x_velocs = initial_batch.atom_velocs.to(device, non_blocking=True)
    edge_batch_idx = initial_batch.edge_batch_idx.to(device, non_blocking=True)
    masked_elements = initial_batch.masked_elements.to(device, non_blocking=True)
    adj_list = (initial_batch.adj_list.to(device, non_blocking=True),)
    atom_types = initial_batch.atom_types.to(device, non_blocking=True)
    sampled_coords = [x_coords]
    sampled_velocs = [x_velocs]
    num_particles = target_energy.num_particles
    masses = [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(num_particles)]
    masses = torch.tensor(masses).to(device)
    accepted = 0
    with torch.no_grad():
        for _ in tqdm(range(n_samples)):

            y_coords, y_velocs = model.conditional_sample(
                atom_types=atom_types,
                x_coords=x_coords,
                x_velocs=x_velocs,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                num_samples=1,
            )

            y_coords = y_coords.squeeze(0)
            y_velocs = y_velocs.squeeze(0)

            p_xy = model.log_likelihood(
                atom_types=atom_types,
                x_coords=x_coords,
                x_velocs=x_velocs,
                y_coords=y_coords,
                y_velocs=y_velocs,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
            )

            assert y_coords.shape == x_coords.shape
            e_kin = 0.5 * (masses * (y_velocs**2.0).sum(-1)).sum(-1) - 0.5 * (
                masses * (x_velocs**2.0).sum(-1)
            ).sum(-1)
            e_pot = target_energy(y_coords) - target_energy(x_coords)
            e_pot = e_pot.view(-1)
            assert e_kin.shape == e_pot.shape
            energy = e_pot + e_kin

            p_yx = model.log_likelihood(
                atom_types=atom_types,
                y_coords=x_coords,
                y_velocs=-x_velocs,
                x_coords=y_coords,
                x_velocs=-y_velocs,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
            )
            assert energy.shape == p_xy.shape
            assert p_yx.shape == p_xy.shape
            exp = energy / kbT + p_xy - p_yx
            p_acc = torch.min(torch.tensor(1), torch.exp(-exp))
            if torch.rand(1).to(p_acc) < p_acc:
                x_coords = y_coords
                x_velocs = y_velocs
                accepted += 1
            sampled_coords.append(x_coords)
            sampled_velocs.append(x_velocs)

    acceptance_rate = accepted / n_samples
    sampled_coords = torch.cat(sampled_coords, dim=0)
    sampled_velocs = torch.cat(sampled_velocs, dim=0)
    return acceptance_rate, sampled_coords, sampled_velocs
