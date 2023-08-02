import abc
from typing import Optional
import torch.nn as nn
from torch import Tensor, BoolTensor

from utilities.logger import TrainingLogger


class BaseModelWrapper(nn.Module, abc.ABC):
    """
    Defines the interface that all model wrappers need to adhere to.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        y_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        y_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:
        pass  # [1] float tensor, loss


class BaseModelWrapperWithForce(nn.Module, abc.ABC):
    """
    Defines the interface that all model wrappers that take forces as inputs need to
    adhere to.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        x_forces: Tensor,  # [batch_size, num_points, 3] float tensor
        y_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        y_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:
        pass  # [1] float tensor, loss
