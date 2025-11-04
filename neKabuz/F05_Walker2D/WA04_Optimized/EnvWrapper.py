import gymnasium as gym
import numpy as np

class WalkerWithCommand(gym.Wrapper):
    def __init__(self, env: gym.Env, speed_function, n_speeds: int,
                 penalty: float = 1.0, speed_name: str = "x_velocity"):
        super().__init__(env)
        self.speed_function = speed_function
        self.n_speeds = n_speeds
        self.penalty = penalty
        self.speed_name = speed_name
        self.t = 0

        old_low  = self.observation_space.low
        old_high = self.observation_space.high
        self.observation_space = gym.spaces.Box(
            low = np.concatenate([old_low, [-np.inf]]),
            high = np.concatenate([old_high, [np.inf]]),
            dtype = np.float32
        )

    def reset(self, *, seed = None, options = None, **kwargs):
        if options is not None:
            sf = options.get("speed_function", None)
            ns = options.get("n_speeds", None)
            if sf is not None:
                self.speed_function = sf
            if ns is not None:
                self.n_speeds = ns

        self.t = 0
        obs, info = self.env.reset(seed = seed, **kwargs)
        speed = self.speed_function(0.0)
        obs = np.concatenate([obs, [speed]]).astype(np.float32)
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.t += 1

        time = (self.t / self.env._max_episode_steps) * self.n_speeds
        v_desired = self.speed_function(time)
        v_real = info.get(self.speed_name, None)

        if v_real is not None:
            track_penalty = self.penalty * (v_real - v_desired) ** 2
            rew = rew - track_penalty

        obs = np.concatenate([obs, [v_desired]]).astype(np.float32)
        return obs, rew, terminated, truncated, info
