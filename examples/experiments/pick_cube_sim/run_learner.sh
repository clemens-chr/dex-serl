export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python3 ../../train_rlpd_sim.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=test1 \
    --demo_path=../../../demo_data/pick_cube_sim_30_demos_2025-04-10_18-17-13.pkl\
    --learner \
