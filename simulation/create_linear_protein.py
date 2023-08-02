"""Create linear extended form of protein from given PDB file.

Usage:
  create_linear_protein.py [options] <input.pdb> <output.pdb>

Options :
  -h --help     Show this screen.
  --chain=<id>  ID of chain to select if there are multiple.
"""

import os
import sys
from docopt import docopt

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simulation.create_random_peptides import (
    residue_seq_to_longform,
    create_extended_protein_from_amino_acids,
)


def get_sequences_from_pdb(pdbfile):
    """Return amino acid sequences for all proteins in the PDB file.

    Arguments
    ---------
    pdbfile : str
        Filename of a PDB file to load.

    Returns
    -------
    chains : dict
        Dictionary with keys being structures and values being the residues.
    """
    pdbparser = PDBParser()
    structure = pdbparser.get_structure("", pdbfile)
    chains = {
        chain.id: seq1("".join(res.resname for res in chain)) for chain in structure.get_chains()
    }
    return chains


if __name__ == "__main__":
    args = docopt(__doc__, version="create_linear_protein 0.1")
    print(args)

    selected_chain = args["--chain"]
    chains = get_sequences_from_pdb(args["<input.pdb>"])
    if len(chains) == 0:
        print("No chains found in PDB.")
        sys.exit(1)
    elif len(chains) > 1 and not selected_chain:
        print("There are %d chains present in the PDB, please select one using the --chain option.")
        print("The following chains are present:")
        for id, res in chains.items():
            print("%s: %s" % (id, res))
        sys.exit(1)
    elif len(chains) == 1 and not selected_chain:
        selected_chain = next(iter(chains.keys()))

    chain = chains[selected_chain]
    pdboutfile = args["<output.pdb>"]
    print("Creating linear extended protein in PDB file '%s'." % pdboutfile)
    print("Short form: %s" % chain)
    chain_long = residue_seq_to_longform(chain)
    print("Long form: %s" % chain_long)
    create_extended_protein_from_amino_acids(chain_long, pdboutfile)
    print("Done.")
