
# Simulation - Classic Molecular Dynamics trajectory data generation using [OpenMM](https://openmm.org/)

Date: 24th September 2021, last updated 11th January 2022

This directory contains several scripts to run simulations of protein dynamics
using classical molecular dynamics simulations.  The basic functionality is:

* Input: a protein specified as a PDB file
* Method: a choice of MD system parameters, MD force field, and simulation
  parameters.
* Output: a NumPy/NPZ file containing trajectory data.

## Preset Parameters

Here a _preset_ refers to the entire configurable part of the simulation before the computation is carried out.  These system parameters are: `--force-field`, `--waterbox-pad`, `--temperature`, `--timestep`, `--friction`, and `--old-integrator`.

Because the choice of these parameters requires more detailed knowledge we define presets of parameter combinations.  We recommend the use of these presets.  The following presets are available via the `--preset` option:

* `--preset=amber14-implicit`: AMBER14 with implicit OBC1 water.
* `--preset=amber14-explicit`: AMBER14 with explicit TIP3 PFB water in a 1nm waterbox.
* (deprecated: `--preset=amber99-implicit`: AMBER99 force field with implicit OBC water.)

All the above choices are for a temperature of 310K and use conservative choices for the remaining parameters.

If you do not want to use presets, please adjust the default parameter values suitably through the respective options.  The default option choice corresponds
to the same values as chosen by `--preset=amber14-implicit`.

### Implicit versus Explicit Water

There are two types of systems that can be simulated, distinguished by whether
the water solvent is handled explicitly or implicitly.  Explicit water means
that we add water molecules to the system and simulate their movement.  This
takes additional simulation time and requires periodic boundary conditions.
Implicit water means that the average effect of water on the protein is
simulated without explicitly representing water molecules.  Treating water
implicitly is a more severe approximation but has the benefit of not requiring
the simulation of individual water molecule trajectories.

For the two choices we use the following force fields and system conditions:

1. _Explicit water system_: a system with periodic boundary conditions and
   explicit water molecules added.  For explicit water we use the AMBER14 force
   fields and the AMBER14 TIP3PFB water model (`--forcefield=amber14-explicit`).
2. _Implicit water system_: an infinite system with implicit water.  We
   currently support both the older AMBER99 SBILDN force field with the
   AMBER99 OBC water model (`--forcefield=amber99-implicit`) as well as the AMBER14 force field with
   [AMBER14 OBC](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.131.8941&rep=rep1&type=pdf)
   water model (`--forcefield=amber14-implicit`).  The force field to use can be selected using the
   `--force-field` option.  The default is to use the AMBER14 force field with implicit water.

## System Preparation

The protein from the PDB file contains an initial configuration in terms of 3D
coordinates of each atom.  This 3D configuration is typically obtained from
crystallography [(RCSB protein data bank)](https://www.rcsb.org/) or from
another machine learning system (AlphaFold2 or similar).  PDB files from
databases often contain additional molecules such as water which we are not
interested in, or miss hydrogen atoms in the amino acids.  In addition,
proteins obtained from any other source often have atom positions that produce
very large initial energies and resulting forces.  One reason is that
crystallography requires low temperatures so the protein at low temperatures is
slightly different from the same protein at body temperature.  Another reason
is that a system like AlphaFold2 may have its own biases yielding
configurations which correspond to non-physical positions.  A third reason is
that force fields that are used for protein simulations,
[AMBER](http://ambermd.org/) in our case, are not perfect and only model the
true physical interactions between atoms in an approximate manner.

If we were to start simulating from the 3D state provided in the PDB file the
large initial forces will lead to numerical issues and ultimately inaccurate
simulation.

Therefore, by default we perform the following standard protocol:

1. _Complete protein_: we add all missing hydrogen atoms to the amino acids of
   any protein.
2. _Removing water_: we remove any water molecules from the system.  This step
   can be skipped using the `--keep-water` option and in that case the movement
   of remaining water molecules is also simulated.
3. (_Waterbox creation_): if the system is to be simulated using explicit
   water, we center the protein and determine a box around it using a provided
   padding parameter (default 1nm, but given by the `--waterbox-pad` option).
   We then fill this box with water molecules.  The box has periodic boundary
   conditions.
4. _Minimization_: we minimize the force field energy over the 3D atom
   coordinates, starting with the 3D coordinates provided by the PDB file.
   This can be seen as simulating a system at zero temperature using the force
   field.  The optimization is performed until a high degree of accuracy is
   reached, by default 2.0 kJ/mol, but the required accuracy can be provided by
   the `--min-tol` option.  The minimization can be skipped by giving the
   option `--do-not-minimize`.
5. _Equilibration / Burn-in_: we then equilibrate the system for a few million
   steps to the target temperature.  This step can be skipped by specifying the
   option `--burn-in=0`.

After this setup process we write the obtained configuration to a new PDB file,
the so called `state0.pdb` file.

## Simulation Parameters

We simulate non-physical [Langevin
dynamics](http://docs.openmm.org/latest/userguide/theory/04_integrators.html#langevinintegator).
Each such Langevin dynamics simulation is governed by the following parameters:

* _Temperature_ (`--temperature` option): the system temperature (in Kelvin).
  By default we use 310K as the human body temperature.  In the simulation,
  temperature enters
* _Timestep_ (`--timestep` option): the time discretization step length.  This
  is typically in the range of 0.5fs to 5fs.
* _Friction_ (`--friction` option): Langevin dynamics use an
  artificial/non-physical coupling to a virtual heat bath and the strength of
  this coupling is defined by a friction parameter.  All choices of friction
  parameters sample from the right equilibrium distribution but some choices
  are more efficient.  We typically use a value of 0.3(ps)^{-1}.

The above parameters determine the system simulation.  In addition the
following parameters determine the duration and sampling of the simulation:

* _Total integration steps_ (`--sampling` parameter): the total number of time
  steps to run the integrator for.  Each time step integrates for the duration
  given by the `--timestep` parameter.
* _Thinning steps between samples_ (`--spacing` parameter): a MD simulation may
  have millions or even billions of steps so writing out all the data is
  infeasible.  Instead frames are only stored at regular intervals, for example
  every one million steps.

In addition to regular spacing samples that are written (approximately
(sampling/spacing) number of samples), we also store samples on a logarithmic
spacing scale.  For example, we would store the following simulation steps:

```
       [ 10000, 10001, 10010, 10100, 11000,
         20000, 20001, 20010, 20100, 21000, ...]
```

As you can see the large step here (`--spacing`) is 10,000 steps, but we also
store +1, +10, +100, +1000 sub-steps.  Storing this data is useful for looking
at changes at different time-increments during the simulation, as used in the
`timewarp` project.

There are also other spacing available as used for other projects.

## Output Data

All output files are written to the same directory, specified using the
`--output-dir` option.  Two files are generated by default:

* `state0.pdb` a PDB file with the initial coordinates (after equilibration)
  and containing the molecule topology as well as explicit water if used.
* `arrays.npz`: a NumPy/NPZ file storing trajectory data.

The created NPZ file will contain the following arrays:

* `time`: `(T,)` array, simulation time in picoseconds.
* `energies`: `(T,2)` array, each row containing [potential, kinetic] energies
  in kJ/mol.
* `positions`: `(T,num_atoms,3)` array, positions in nm.
* `velocities`: `(T,num_atoms,3)` array, velocities in nm/ps.
* `forces`: `(T,num_atoms,3)` array, forces in kJ/(mol nm).

The choice of units above is to agree with the standard [OpenMM unit
system](http://docs.openmm.org/latest/userguide/theory/01_introduction.html#units).

## Checkpointing

To support pre-emptible computation backends we also provide a checkpointing
functionality.  It is enabled by default, but can be disabled using the
`--no-checkpointing` option.

When enabled we write checkpointing information regular every given number of
minutes, as specified by the `--cpmins` option.  Because checkpointing can
write a large amount of data we recommend to use a larger interval, say 5 or 15
minutes.

When checkpointing is enabled and the program is started it attempts to find
checkpointing data in the output directory (`--output-dir` option).  If the
program finds checkpointing information it resumes from the last checkpointed
state.  If no checkpointing information is found it starts the simulation from
scratch.

## Example usage
An example command to simulate a molecule with the corresponding `.pdb` file `pdb-file-name.pdb` for 100 Million steps is:

```
python simulate_trajectory.py \
  --preset=amber14-implicit \
  --spacing=10000 \
  --sampling=100_000_000 \
  --cpmins=120 \
  path-to-input-directory/pdb-file-name.pdb \
  path-to-output-directory/output-file-name.pdb
```

All outputs will be save in the `<path-to-output-directory>`
- The `.pdb` file of the initial state `<output-file-name>-state0.pdb`
- The trajectory `<output-file-name>-arrays.npz`
- A checkpoint every 120 minutes to continue the simulation from
- A `.npy` file that stores the wall-clock-time of the simulation `<output-file-name>-time.npy`

`<output-file-name>` should be replaced with a name that corresponds to the sampled protein.

## Creating new multi-peptide datasets
The initial configuration of the two small peptide dataset 2AA and 4AA were generated with the [create_random_peptides.py](create_random_peptides.py) script.
To generate the `.pdb` files for 10 unique dipeptides (two amino acids) run
```
python simulation/create_random_peptides.py 2 10 --no-duplicates <path-to-output-directory>
```
The generation is not totally random, 
as the frequency of the aminoacids for the generated peptides is given by the their relative frequencies in nature. 
If a different distribution is desired, the amino_acid_frequencies have to be changed in the [create_random_peptides.py](create_random_peptides.py) file accordingly. 
The so generated `.pdb` files can then be used as initial states for MD trajectories as described above.  

## Authors

* Sebastian Nowozin
* Leon Klein
