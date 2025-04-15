export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python3 ../../train_rlpd_sim.py "$@" \
    --exp_name=orca_pick_cube_sim \
    --checkpoint_path=orca3 \
    --demo_path=../../../demo_data/orca_pick_cube_sim_25_demos_2025-04-14_15-47-24.pkl\
    --learner \
