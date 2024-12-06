import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None, image_obs=True):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        self.image_obs = image_obs
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"] if self.image_obs else {}),
            }
        )

    def observation(self, obs):
        obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"] if self.image_obs else {}),
        }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs