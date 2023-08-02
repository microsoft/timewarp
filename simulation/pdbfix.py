"""Fix PDB file.

Usage:
  pdbfix.py <input.pdb> <output.pdb>

"""

from pdbfixer import PDBFixer
from openmm.app import PDBFile
from docopt import docopt


def fix_pdb(pdbinput, pdboutput):
    print("Processing '%s'" % pdbinput)

    # Fix PDB
    fixer = PDBFixer(pdbinput)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # pH for protonation state
    # Remove water, see
    # https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html
    fixer.removeHeterogens(False)

    PDBFile.writeFile(fixer.topology, fixer.positions, open(pdboutput, "w"))
    print("   wrote '%s'." % pdboutput)


if __name__ == "__main__":
    args = docopt(__doc__, version="pdbfix 0.1")
    print(args)

    pdbinput = args["<input.pdb>"]
    pdboutput = args["<output.pdb>"]

    fix_pdb(pdbinput, pdboutput)
