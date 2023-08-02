"""Create random peptides in linear extended form.

Usage:
  create_random_peptides.py [options] <aa-length> <count> <output-directory/>

Options :
  -h --help             Show this screen.
  --no-duplicates       Filter out any duplicates.
                        May generate less than <count> sequences.

Randomly sample <count> number of peptides of length <aa-length> amino acids.
The sampling is done with the empirically observed amino acid distribution.
The extension of <input.ext> needs to be '.pdb' for processed PDB files that
will be directly simulated.
"""

import os
import sys
import subprocess
import numpy as np
from docopt import docopt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simulation.pdbfix import fix_pdb

# From: http://www.tiem.utk.edu/~gross/bioed/webmodules/aminoacid.htm
amino_acid_frequencies = {
    "ALA": 7.4,  # Alanine
    "ARG": 4.2,  # Arginine
    "ASN": 4.4,  # Asparagine
    "ASP": 5.9,  # Aspartic Acid
    "CYS": 3.3,  # Cysteine
    "GLU": 5.8,  # Glutamic Acid
    "GLN": 3.7,  # Glutamine
    "GLY": 7.4,  # Glycine
    "HIS": 2.9,  # Histidine
    "ILE": 3.8,  # Isoleucine
    "LEU": 7.6,  # Leucine
    "LYS": 7.2,  # Lysine
    "MET": 1.8,  # Methionine
    "PHE": 4.0,  # Phenylalanine
    "PRO": 5.0,  # Proline
    "SER": 8.1,  # Serine
    "THR": 6.2,  # Threonine
    "TRP": 1.3,  # Tryptophan
    "TYR": 3.3,  # Tyrosine
    "VAL": 6.8,  # Valine
}
# One letter amino acid abbreviations
amino_acid_abbreviations = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# amino_acid_longform["Y"] = "TYR"
amino_acid_longform = {
    residue_char: residue_tla for (residue_tla, residue_char) in amino_acid_abbreviations.items()
}


def residue_seq_to_longform(seq):
    """Map a residue character sequence to a longform sequence string."""
    prot_seq_str = " ".join(amino_acid_longform[residue_char] for residue_char in seq)
    return prot_seq_str


def create_extended_protein_from_amino_acids(prot_seq_str, pdb_filepath):
    """Use AMBER's tleap program to create a protein in extended form.

    We use the canonical residue coordinates as described in the
    AMBER14 ff14SBonlysc force field.

    Arguments
    ---------
    prot_seq_str : str
        Protein amino acid sequence, e.g. "SER ALA SER ASP".
    pdb_filepath : str
        Filepath of the new PDB file that is created.
    """
    tleap_command_string = """
source leaprc.protein.ff14SBonlysc
set default PBradii mbondi3
prot = sequence { %s }
savepdb prot %s
quit
    """ % (
        prot_seq_str,
        pdb_filepath,
    )

    # Run AMBER tleap program to construct 3D protein
    tleap = subprocess.Popen(
        ["tleap", "-s", "-f", "-"],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    outs, errs = tleap.communicate(input=tleap_command_string, timeout=20)
    fix_pdb(pdb_filepath, pdb_filepath)


if __name__ == "__main__":
    args = docopt(__doc__, version="create_random_peptides 0.1")
    print(args)

    aalength = int(args["<aa-length>"])
    count = int(args["<count>"])
    output_dir = args["<output-directory/>"]

    aalist = list(amino_acid_frequencies.keys())
    aafreq = np.array([amino_acid_frequencies[aa] for aa in aalist])
    aafreq /= np.sum(aafreq)

    aa_ids = np.random.choice(aalist, size=(count, aalength), p=aafreq)
    aa_ids_abbrev = {
        " ".join(list(aa_ids[i, :])): "".join(
            [amino_acid_abbreviations[aa] for aa in list(aa_ids[i, :])]
        )
        for i in range(count)
    }
    aa_ids = [" ".join(list(aa_ids[i, :])) for i in range(count)]

    print("Created %d sequences." % count)
    if args["--no-duplicates"]:
        # Remove duplicate sequences
        aa_ids = list(set(aa_ids))
        print(
            "Removed %d duplicates, %d of %d remaining." % (count - len(aa_ids), len(aa_ids), count)
        )

    for i in range(len(aa_ids)):
        prot_seq_str = aa_ids[i]  # e.g. "SER ALA SER ASP"
        pdb_filepath = os.path.join(
            args["<output-directory/>"], aa_ids_abbrev[prot_seq_str] + ".pdb"
        )
        print(prot_seq_str)
        print(aa_ids_abbrev[prot_seq_str])
        print(pdb_filepath)

        create_extended_protein_from_amino_acids(prot_seq_str, pdb_filepath)
