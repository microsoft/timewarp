import math
import torch.nn as nn
import torch

from torch import Tensor, BoolTensor

from typing import Sequence, Optional, Tuple
from timewarp.dataloader import ELEMENT_VOCAB
from timewarp.modules.transformer_gaussian_density import FlatTransformerMeanLogScaleModel

from timewarp.modules.layers.transformer_block import TransformerConfig
from timewarp.modules.model_wrappers.density_model_base import ConditionalDensityModel
from utilities.logger import TrainingLogger
from timewarp.utils.molecule_utils import get_centre_of_mass


class ConditionalVAEModel(ConditionalDensityModel):
    """Conditional VAE model of [Sohn et al., NIPS 2015]."""

    def __init__(
        self,
        atom_embedding_dim: int,
        latent_cvae_dim: int,
        num_elbo_samples: int,
        elbo_estimator: str,
        transformer_hidden_dim: int,
        num_transformer_layers: int,
        latent_mlp_hidden_dims: Sequence[int],
        transformer_config: TransformerConfig,
    ):
        super().__init__()
        self.num_elbo_samples = num_elbo_samples
        self.elbo_estimator = elbo_estimator

        self.atom_embedder = nn.Embedding(
            num_embeddings=len(ELEMENT_VOCAB),
            embedding_dim=atom_embedding_dim,
        )

        # CVAE Prior model p(z|x)
        self.prior = FlatTransformerMeanLogScaleModel(
            input_dim=atom_embedding_dim + 6,
            output_dim=latent_cvae_dim,
            hidden_dim=transformer_hidden_dim,
            num_transformer_layers=num_transformer_layers,
            mlp_hidden_layers_dims=latent_mlp_hidden_dims,
            transformer_config=transformer_config,
        )

        # CVAE Generator model p(y|x,z)
        self.generator = FlatTransformerMeanLogScaleModel(
            input_dim=atom_embedding_dim + 6 + latent_cvae_dim,
            output_dim=6,  # coords and velocs
            hidden_dim=transformer_hidden_dim,
            num_transformer_layers=num_transformer_layers,
            mlp_hidden_layers_dims=latent_mlp_hidden_dims,
            transformer_config=transformer_config,
        )

        # CVAE Recognition model q(z|x,y)
        self.recognizer = FlatTransformerMeanLogScaleModel(
            input_dim=atom_embedding_dim + 6 + 6,
            output_dim=latent_cvae_dim,
            hidden_dim=transformer_hidden_dim,
            num_transformer_layers=num_transformer_layers,
            mlp_hidden_layers_dims=latent_mlp_hidden_dims,
            transformer_config=transformer_config,
        )

    def log_likelihood(
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
    ) -> Tensor:  # [batch_size] float tensor
        """
        Returns the ELBO lower bound to the log-likelihood for each element in
        the batch.
        """
        atom_features = self.atom_embedder(atom_types)  # [B, num_points, atom_embedding_dim]

        # Predict the change (residual) rather than absolute values
        y_coords_residual = y_coords - x_coords
        y_velocs_residual = y_velocs

        # Canonicalise input position:
        centre_of_mass = get_centre_of_mass(x_coords, masked_elements=masked_elements)
        x_coords_centered = x_coords - centre_of_mass  # [B, num_points, 3]

        # 1. Sample z ~ q(z|y,x) and evaluate log q(z|y,x)
        xy_flat_features = torch.cat(
            (atom_features, x_coords_centered, x_velocs, y_coords_residual, y_velocs_residual),
            dim=-1,
        )  # [B, num_points, atom_embedding_dim + 12]
        z_recog_mean, z_recog_log_scale = self.recognizer(
            input=xy_flat_features,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            logger=logger,
        )
        z_recog_scale = torch.exp(z_recog_log_scale)
        z_recog_dist = torch.distributions.Normal(z_recog_mean, z_recog_scale)

        z_sample = z_recog_dist.rsample(
            (self.num_elbo_samples,)
        )  # [S, B, num_points, latent_cvae_dim]
        log_prob_recognizer_z = z_recog_dist.log_prob(
            z_sample
        )  # [S, B, num_points, latent_cvae_dim]
        log_prob_recognizer_z = log_prob_recognizer_z.sum(dim=(-1, -2))  # [S, B]

        # 2. Evaluate log p(z|x)
        x_flat_features = torch.cat(
            (atom_features, x_coords_centered, x_velocs),
            dim=-1,
        )  # [B, num_points, atom_embedding_dim + 6]
        z_prior_mean, z_prior_log_scale = self.prior(
            input=x_flat_features,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            logger=logger,
        )
        z_prior_scale = torch.exp(z_prior_log_scale)
        z_prior_dist = torch.distributions.Normal(z_prior_mean, z_prior_scale)
        log_prob_prior_z = z_prior_dist.log_prob(z_sample)  # [S, B, num_points, latent_cvae_dim]
        log_prob_prior_z = log_prob_prior_z.sum(dim=(-1, -2))  # [S, B]

        # 3. Evaluate log p(y|x,z)

        # reshape z_sample linearly, [S, B] to [S*B]
        z_sample_flat = z_sample.reshape(
            -1, z_sample.shape[-2], z_sample.shape[-1]
        )  # [S*B, num_points, latent_cvae_dim]
        xz_flat_features = torch.cat(
            (
                atom_features.repeat(
                    self.num_elbo_samples, 1, 1
                ),  # [S*B, num_points, atom_embedding_dim]
                x_coords_centered.repeat(self.num_elbo_samples, 1, 1),  # [S*B, num_points, 3]
                x_velocs.repeat(self.num_elbo_samples, 1, 1),  # [S*B, num_points, 3]
                z_sample_flat,
            ),  # [S*B, num_points, latent_cvae_dim]
            dim=-1,
        )  # [S*B, num_points, atom_embedding_dim + 6 + latent_cvae_dim]

        y_mean, y_log_scale = self.generator(
            input=xz_flat_features,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements.repeat(self.num_elbo_samples, 1),  # [S*B, num_points]
            logger=logger,
        )  # [S*B, num_points, 6], [S*B, num_points, 6]
        y_scale = torch.exp(y_log_scale)
        y_dist = torch.distributions.Normal(y_mean, y_scale)
        y_flat_features = torch.cat((y_coords_residual, y_velocs_residual), dim=-1)
        log_prob_generator_y = y_dist.log_prob(
            y_flat_features.repeat(self.num_elbo_samples, 1, 1),
        )  # [S*B, num_points, 6]
        log_prob_generator_y = log_prob_generator_y.sum(dim=(-1, -2))  # [S*B]
        log_prob_generator_y = log_prob_generator_y.reshape(self.num_elbo_samples, -1)  # [S,B]

        # ELBO = E_{z~q(z|x,y)}[ log p(y|x,z) - log q(z|y,x) + log p(z|x) ]
        #      = E_{z~q(z|x,y)}[ log p(y|x,z) ] - D_{KL}(q(z|y,x) | p(z|x))
        #
        # We have IWAE_1 = ELBO, and
        #
        # IWAE_k = log sum_k exp( log_weights ) - log K,
        # where log_weights = log p(y|x,z) - log q(z|y,x) + log p(z|x)
        log_weights = log_prob_generator_y - log_prob_recognizer_z + log_prob_prior_z  # [S, B]
        if self.elbo_estimator == "elbo":
            # ELBO with reduced variance (1/num_elbo_samples)
            elbo = torch.mean(log_weights, dim=0)  # [B]
        elif self.elbo_estimator == "iwae":
            # IWAE
            elbo = torch.logsumexp(log_weights, dim=0) - math.log(self.num_elbo_samples)  # [B]

        obj = elbo

        # Note: KL difference has a different meaning in IWAE
        kl = log_prob_recognizer_z - log_prob_prior_z  # [S,B]
        kl = kl.mean(dim=0)  # [B]

        # Compute and monitor various quantities
        if logger is not None:
            logger.log_scalar_async("kl", kl.mean())
            logger.log_scalar_async("obj", obj.mean())
            logger.log_scalar_async("elbo", elbo.mean())
            logger.log_scalar_async("log_prob_generator_y", log_prob_generator_y.mean())
            logger.log_scalar_async("log_prob_recognizer_z", log_prob_recognizer_z.mean())
            logger.log_scalar_async("log_prob_prior_z", log_prob_prior_z.mean())

        return obj

    def conditional_sample(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        num_samples: int,
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = x_coords.shape[0]
        atom_features = self.atom_embedder(atom_types)  # [B, num_points, atom_embedding_dim]

        # Canonicalise input position:
        centre_of_mass = get_centre_of_mass(x_coords, masked_elements=masked_elements)
        x_coords_centered = x_coords - centre_of_mass

        # 1. Sample z ~ p(z|x)
        x_flat_features = torch.cat(
            (atom_features, x_coords_centered, x_velocs),
            dim=-1,
        )  # [B, num_points, atom_embedding_dim + 6]
        z_prior_mean, z_prior_log_scale = self.prior(
            input=x_flat_features,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            logger=logger,
        )
        z_prior_scale = torch.exp(z_prior_log_scale)
        z_prior_dist = torch.distributions.Normal(z_prior_mean, z_prior_scale)
        z_sample = z_prior_dist.sample((num_samples,))  # [S, B, num_points, latent_cvae_dim]
        z_sample = z_sample.reshape(
            -1, z_sample.shape[-2], z_sample.shape[-1]
        )  # [S*B, num_points, latent_cvae_dim]

        # Sample y ~ p(y|x,z)
        xz_flat_features = torch.cat(
            (
                atom_features.repeat(num_samples, 1, 1),  # [S*B, num_points, D]
                x_coords_centered.repeat(num_samples, 1, 1),  # [S*B, num_points, 3]
                x_velocs.repeat(num_samples, 1, 1),  # [S*B, num_points, 3]
                z_sample,
            ),  # [S*B, num_points, latent_cvae_dim]
            dim=-1,
        )  # [S*B, num_points, atom_embedding_dim + 6 + latent_cvae_dim]
        y_mean, y_log_scale = self.generator(
            input=xz_flat_features,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements.repeat(num_samples, 1),  # [S*B, num_points]
            logger=logger,
        )  # [S*B, num_points, 6], [S*B, num_points, 6]
        y_scale = torch.exp(y_log_scale)
        y_dist = torch.distributions.Normal(y_mean, y_scale)
        y_flat_sample = y_dist.sample()  # [S*B, num_points, 6]

        y_coords_residual, y_velocs_residual = torch.tensor_split(y_flat_sample, 2, dim=-1)

        # Model predicts the change (residual) rather than absolute values
        y_coords = x_coords.repeat(num_samples, 1, 1) + y_coords_residual  # [S * B, V, 3]
        y_velocs = y_velocs_residual  # [S * B, V, 3]

        y_coords = y_coords.reshape(
            num_samples, batch_size, y_coords.shape[-2], y_coords.shape[-1]
        )  # [S, B, V, 3]
        y_velocs = y_velocs.reshape(
            num_samples, batch_size, y_velocs.shape[-2], y_velocs.shape[-1]
        )  # [S, B, V, 3]

        return y_coords, y_velocs
