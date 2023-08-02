# Run "source setenv.sh"
ulimit -s unlimited
export OMP_STACKSIZE=16G
export OMP_NUM_THREADS=6
export OMP_MAX_ACTIVE_LEVELS=1
export XTBPATH=${CONDA_PREFIX}/share/xtb/

