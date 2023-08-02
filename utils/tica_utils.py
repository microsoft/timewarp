import numpy as np
import deeptime as dt
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SELECTION = "symbol == C or symbol == N or symbol == S"


def distances(xyz):
    distance_matrix_ca = np.linalg.norm(xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1)
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


def wrap(array):
    return (np.sin(array), np.cos(array))


def tica_features(trajectory, use_dihedrals=True, use_distances=True, selection=SELECTION):
    trajectory = trajectory.atom_slice(trajectory.top.select(selection))
    # n_atoms = trajectory.xyz.shape[1]
    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_phi(trajectory)
        _, omega = md.compute_omega(trajectory)
        dihedrals = np.concatenate([*wrap(phi), *wrap(psi), *wrap(omega)], axis=-1)
    if use_distances:
        ca_distances = distances(trajectory.xyz)
    if use_distances and use_dihedrals:
        return np.concatenate([ca_distances, dihedrals], axis=-1)
    elif use_distances:
        return ca_distances
    else:
        return []


def run_tica(trajectory, lagtime=500, dim=40):
    ca_features = tica_features(trajectory)
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(ca_features).fetch_model()
    tica_model = tica.fit(ca_features, reweighting_model).fetch_model()
    return tica_model


def plot_tic01(ax, tics, name, tics_lims, cmap="viridis"):
    _ = ax.hist2d(tics[:, 0], tics[:, 1], bins=100, norm=LogNorm(), cmap=cmap)
    ax.set_xlabel("tic0")
    ax.set_ylabel("tic1")
    ax.set_ylim(tics_lims[:, 1].min(), tics_lims[:, 1].max())
    ax.set_xlim(tics_lims[:, 0].min(), tics_lims[:, 0].max())

    plt.title(f"TICA - {name}")


def plot_free_energy(ax, tics, xlabel, label=None, axis=0):
    hist, edges = np.histogram(tics[:, axis], bins=100, density=True)
    free_energy = -np.log(hist / hist.max())
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, free_energy, linewidth=5, label=label)
    ax.set_xlabel(xlabel)
    plt.title("Free energy")
