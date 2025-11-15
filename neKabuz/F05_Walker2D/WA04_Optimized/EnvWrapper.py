import gymnasium as gym
import numpy as np
from Config import Config

def make_interp_function(t: np.ndarray, y: np.ndarray, clamp: bool = True):
    t = np.asarray(t, dtype = float)
    y = np.asarray(y, dtype = float)

    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]

    left = y[0] if clamp else np.nan
    right = y[-1] if clamp else np.nan

    def f(tq):
        tq_arr = np.asarray(tq, dtype = float)
        vals = np.interp(tq_arr, t, y, left = left, right = right)
        return float(vals) if np.isscalar(tq) else vals

    return f


class WalkerWithCommand(gym.Wrapper):
    def __init__(self, env: gym.Env, config: Config):
        super().__init__(env)

        self.desired_torso_height = config.torso_height
        self.weight_torso = config.weight_torso

        self.speed_function = None
        self.n_speeds = None
        self.weight_speed = config.weight_speed
        self.speed_name = config.speed_name
        self.sigma = config.sigma_speed
        self.sigma_torso = config.sigma_torso
        self.t = 0

        old_low  = self.observation_space.low
        old_high = self.observation_space.high
        self.observation_space = gym.spaces.Box(
            low = np.concatenate([old_low, [-np.inf, -np.inf]]),
            high = np.concatenate([old_high, [np.inf, np.inf]]),
            dtype = np.float64
        )

    def reset(self, *, seed: int | None = None, options: dict, **kwargs):
        t_arr = options.get("speed_t", None)
        y_arr = options.get("speed_y", None)
        if t_arr is None or y_arr is None:
            t_arr = np.array([0.0], dtype = float)
            y_arr = np.array([0.0], dtype = float)

        self.n_speeds = len(t_arr)

        self.speed_function = make_interp_function(t_arr, y_arr)

        self.t = 0
        obs, info = self.env.reset(seed = seed, **kwargs)

        speed = self.speed_function(0.0)
        obs = np.concatenate([obs, [speed, 0.0]]).astype(np.float32)
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.t += 1

        # Get speed
        time = (self.t / self.env._max_episode_steps) * self.n_speeds
        v_desired = self.speed_function(time)
        v_real = info.get(self.speed_name, 0.0)
        v_error = v_real - v_desired
        track_score = np.exp(-(v_error ** 2) / (2 * self.sigma ** 2))

        # Torso
        torso_height = obs[0]
        torso_error = torso_height - self.desired_torso_height
        height_Score = np.exp(-(torso_error ** 2) / (2 * self.sigma_torso ** 2))


        rew += self.weight_speed * track_score + self.weight_torso * height_Score

        obs = np.concatenate([obs, [v_desired, v_error]]).astype(np.float32)
        return obs, rew, terminated, truncated, info
