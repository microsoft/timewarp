import abc
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class OpenMMProviderConfig:
    pdb_dirs: str
    device: str = "cpu"
    cache_size: int = 8


class AbstractLossConfig(abc.ABC):
    pass


@dataclass
class NLLConfig(AbstractLossConfig):
    random_velocs: bool = True


@dataclass
class NLLAndEnergyLossConfig(AbstractLossConfig):
    openmm_provider: OpenMMProviderConfig

    # Used by both losses.
    random_velocs: bool = True

    # Used by `EnergyLoss` only.
    num_samples: int = 1

    # Combination.
    weights: Optional[List[float]] = None
    pre_softmax_weights: Optional[List[float]] = None

    def __post_init__(self):
        assert (
            self.weights is not None or self.pre_softmax_weights is not None
        ), "either weights or pre_softmax_weights has to be specified"


@dataclass
class NLLAndAcceptanceLossConfig(AbstractLossConfig):
    openmm_provider: OpenMMProviderConfig

    # Used for both losses.
    random_velocs: bool = True

    # Used by `AcceptanceLoss` only.
    beta: float = 0.2
    clamp: bool = False
    num_samples: int = 1
    high_energy_threshold: float = -1

    # Combination.
    weights: Optional[List[float]] = None
    pre_softmax_weights: Optional[List[float]] = None

    def __post_init__(self):
        assert (
            self.weights is not None or self.pre_softmax_weights is not None
        ), "either weights or pre_softmax_weights has to be specified"


@dataclass
class LossConfig:
    nll: Optional[NLLConfig] = None
    nll_and_energy: Optional[NLLAndEnergyLossConfig] = None
    nll_and_acceptance: Optional[NLLAndAcceptanceLossConfig] = None


@dataclass
class LossScheduleConfig:
    factor: List[float]
    every: int
