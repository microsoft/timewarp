# Timewarp - Investigating conditional density models for molecular dynamics prediction

The code in this repository accompanies the preprint ``[Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics](https://arxiv.org/abs/2302.01170)'', by Leon Klein, Andrew Y. K. Foong, Tor Erlend Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noé, and Ryota Tomioka.

This directory contains code for various density models that model $p(y|x)$, where $y$ is the final state of a molecule and $x$ is the initial state.


## Training

Various experiment configs can be found in `configs/`, each corresponding to different models. Please see `models.md` for a description.

For example, to train the kernel transformerNVP model on the AD-1 alanine-dipeptide dataset (this should take less than 12 hours on a single machine with a Tesla P100 GPU), run 
```
python train.py configs/kernel_transformer_nvp.yaml
``` 
Arguments in the yaml file can be overridden on the command line, e.g. 
```
python train.py configs/kernel_transformer_nvp.yaml learning_rate=0.01
```

## Monitoring and Reproducibility

Metrics such as the training and validation losses are automatically logged to TensorBoard, and are saved in the output directory (default is `outputs/`) along with the experiment config and the best model.


## Plotting Samples, Energies, Internal coordinate Distributions, and More, locally

After the model is trained, you can run:
```
python sample.py \
    --savefile path-to-model-output-directory/best_model.pt \
    --data_dir path-to-directory-containing-protein-data/ \
    --protein protein-pdb-name \
    --num_samples 20 \
    --output_dir path-to-directory-to-save-figures/
```
This will generate figures and GIFs showing conditional samples from the model, and also the ground truth initial and final states of the model. A plot of the potential, kinetic and total energies along the speficied trajectory is also generated. 

For some datasets there are additional evaluation scripts. The evaluation script `evaluate_o2.py` for the oxygen molecule data sets (currently `O2` and `O2-CoM`) can be run with 
```
python evaluate_o2.py \
    --savefile path-to-model-output-directory/best_model.pt \
    --data_dir path-to-directory-containing-protein-data/ \
    --num_samples 20 \
    --output_dir path-to-directory-to-save-figures/ \
    --sample True
```
This will generate a figures for the conditional distribution for a single conditioning sample as well as plots for samples generates with the model using the Metropolis–Hastings algorithm.

A more general evaluation script `evaluate.py` is designed for Di- and Tetra-peptides, such as Alanine Dipeptide. It can be run with
```
python evaluate.py \
    path-to-model-output-directory/best_model.pt \
    --data_dir path-to-directory-containing-protein-data/ \
    --num_samples 1000 \
    --output_dir path-to-directory-to-save-figures/ 
```
but has many more options. This command will sample each 1000 samples conditioned on samples from the Boltzmann distribution, a single conditioning state, as well sampling with the model by accepting all proposals. 

To sample with the model using the Metropolis Hastings algorithm, generating and accepting proposals for a single conditioning state in parallel, can we run with
```
python evaluate.py \
    path-to-model-output-directory/best_model.pt \
    --data_dir path-to-directory-containing-protein-data/ \
    --num_samples 1000 \
    --output_dir path-to-directory-to-save-figures/ \
    --mh  \
    --sample--num-proposal-steps 1000
```
In the sampling process this will propose 1000 samples for each conditioning state and add all samples until the first accepted one to the Markov chain. This parallelization speeds up the sampling by a lot depending on the acceptance probabilities.


## Multi-GPU training

We use [deepspeed](https://www.deepspeed.ai/) for multi-GPU training. To run data-parallel training on 2 GPUs,
```
deepspeed --num_gpus=2 train.py configs/kernel_transformer_nvp.yaml
```
Note that the batch size per GPU is `batch_size` divided by the number of GPUs. In other words, you need to multiply `batch_size` by the number of available GPUs to keep the GPU utilization constant.


## Config arguments

See [`training_config.py`](training_config.py).


## Model Variants

The current model of interest is the model_constructor.custom_transformer_nvp_constructor with settings from
`configs/kernel_transformer_nvp.yaml`.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
