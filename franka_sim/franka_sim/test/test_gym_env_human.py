import time

import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs

env = envs.PandaPickCubeGymEnv(render_mode="human", action_scale=(0.1, 1))
action_spec = env.action_space


def sample():
    a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    return a.astype(action_spec.dtype)


m = env.model
d = env.data

reset = False
KEY_SPACE = 32


def key_callback(keycode):
    if keycode == KEY_SPACE:
        global reset
        reset = True


env.reset()
with mujoco.viewer.launch_passive(model=m, data=d, show_left_ui=False, show_right_ui=False, key_callback=key_callback) as viewer_1, \
     mujoco.viewer.launch_passive(model=m, data=d, show_left_ui=False, show_right_ui=False, key_callback=key_callback) as viewer_2:

    start = time.time()
    while viewer_1.is_running() and viewer_2.is_running():
        if reset:
            env.reset()
            reset = False
        else:
            step_start = time.time()
            env.step(sample())
            viewer_1.sync()
            viewer_2.sync()
            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
