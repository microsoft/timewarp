import pytest

import os
from pathlib import Path

import torch

from timewarp.modules.model_wrappers.density_model_base import ConditionalDensityModel
from timewarp.tests.assets import get_model_config
from timewarp.utils.torsion_utils import get_all_torsions, get_all_torsions_model


DATA_DIR = (
    Path(os.path.join(os.path.dirname(__file__), "../testdata/output/")).expanduser().resolve()
)


@pytest.mark.parametrize("targets", [True, False])
@pytest.mark.parametrize("protein", ["1hgv"])
def test_get_all_torsions(targets, protein):
    # Test if the dihedral angles are computed for a test dataset
    n_initial_samples = 10
    dihedrals = get_all_torsions(
        protein, DATA_DIR, step_width=1000, n_initial_samples=n_initial_samples, targets=targets
    )
    assert len(dihedrals.__dataclass_fields__) == 14
    assert dihedrals.phi.shape == (n_initial_samples, 1, 45)
    assert dihedrals.phi_indices.shape == (45, 4)


@pytest.mark.parametrize("n_initial_samples", [1, 10])
@pytest.mark.parametrize("n_samples_model", [1, 10])
@pytest.mark.parametrize("model_name", ["transformer_nvp"])
@pytest.mark.parametrize("protein", ["1hgv"])
def test_get_all_torsions_model(
    n_initial_samples, n_samples_model, model_name, protein, device: torch.device
):
    # construct a simple model
    model, _ = get_model_config(model_name)
    model = model.to(device)

    # Test if the dihedral angles are computed for a test dataset
    n_initial_samples = 10

    assert isinstance(model, ConditionalDensityModel)
    dihedrals = get_all_torsions_model(
        protein,
        DATA_DIR,
        step_width=1000,
        model=model,
        n_initial_samples=n_initial_samples,
        n_samples_model=n_samples_model,
        device=device,
    )
    assert len(dihedrals.__dataclass_fields__) == 14
    assert dihedrals.phi.shape == (n_initial_samples, n_samples_model, 45)
    assert dihedrals.phi_indices.shape == (45, 4)
