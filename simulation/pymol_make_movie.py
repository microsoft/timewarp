"""Create a movie from an MD trajectory PDB file using PyMol.

Usage:
  make_movie.py [options] <trajectory.pdb> <movie.gif>

Options :
  -h --help        Show this screen.
  --keep-water     Do not remove water molecules.
  --sticks         Create a sticks rendering, colored by atom type.
  --hq             Create a high-quality rendering (takes long).
  --res=<res>      Resolution for rendering in pixels [default: 960].

"""

import __main__

__main__.pymol_argv = ["pymol", "-qc"]  # Quiet and no GUI

import time
import pymol2
import mdtraj
from docopt import docopt
import tempfile
from PIL import Image


if __name__ == "__main__":
    # Load args.
    args = docopt(__doc__)
    trajpath = args["<trajectory.pdb>"]
    traj = mdtraj.load(trajpath)
    num_frames = len(traj)

    pymol = pymol2.PyMOL()
    pymol.start()

    # Directory for PDB frames
    frame_dir_prefix = "moroni-vis-frames.tmp."
    with tempfile.TemporaryDirectory(prefix=frame_dir_prefix) as tmpdir:
        frame_dir = tmpdir
        pymol.finish_launching()

        print("Processing %d frames..." % num_frames)
        for frame in range(num_frames):
            # Split off a single frame
            traj_frame = traj[frame]
            traj_frame_path = frame_dir + f"/frame_{frame}.pdb"
            traj_frame.save(traj_frame_path)

            # Load Structures
            name = f"frame_{frame}"
            # The following re-init is needed to preserve the rainbow
            # color scheme across frames.
            pymol.cmd.reinitialize()
            pymol.cmd.load(traj_frame_path, name)
            pymol.cmd.load(traj_frame_path)
            pymol.cmd.disable("all")
            pymol.cmd.enable(name)
            if not args["--keep-water"]:
                pymol.cmd.remove("solvent")

            # TODO(senowoz): set ray_opaque_background, on/off
            # for creating transparent GIFs

            if args["--sticks"]:
                # Sticks representation, colored by atom type
                pymol.cmd.hide("cartoon")
                pymol.cmd.show("sticks")
                pymol.cmd.color("atomic")
            else:
                # Cartoon representation, colored by residue
                pymol.cmd.spectrum("count")

            res = args["--res"]
            if args["--hq"]:
                pymol.cmd.ray(res, res)
            else:
                pymol.cmd.draw(res, res, 1)

            pymol.cmd.png(frame_dir + f"/pymol_image_{frame}.png")

        # PyMol seems to have internal asynchronous operations, where
        # pymol.cmd.png will return before the image file has been fully written.
        # There seems to be no PyMol API to wait until completion, so this
        # hack hopefully allows the last image file to be written.
        time.sleep(2.0)
        pymol.cmd.quit()

        # Collate into GIF.
        fp_in = [frame_dir + f"/pymol_image_{frame}.png" for frame in range(num_frames)]
        fp_out = args["<movie.gif>"]
        img, *imgs = [Image.open(f) for f in fp_in]
        img.save(fp=fp_out, format="GIF", append_images=imgs, save_all=True, duration=200, loop=0)
