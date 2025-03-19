import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs
import gymnasium as gym

# import joystick wrapper
from franka_env.envs.wrappers import AVPIntervention

from franka_sim.utils.viewer_utils import MujocoViewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avp_ip", type=str, default="10.93.181.127", help="Controller type. xbox|ps5")

    args = parser.parse_args()

# env = envs.PandaPickCubeGymEnv(render_mode="human", image_obs=True)
env = gym.make("OrcaPickCubeVision-v0", render_mode="human", image_obs=True)
env = AVPIntervention(env, avp_ip=args.avp_ip)

env.reset()
m = env.unwrapped.model
d = env.unwrapped.data

# Create the dual viewer
dual_viewer = MujocoViewer(env.unwrapped.model, env.unwrapped.data)

# intervene on position control
with dual_viewer as viewer:
    for i in range(100000):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        viewer.sync()
