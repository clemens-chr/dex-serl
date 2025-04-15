export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python3 ../../train_rlpd_sim.py "$@" \
    --exp_name=orca_pick_cube_sim \
    --checkpoint_path=orca3 \
    --actor \
