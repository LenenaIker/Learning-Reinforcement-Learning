import gymnasium as gym
import numpy as np

def _make_interp_function(t: np.ndarray, y: np.ndarray, clamp: bool = True):
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
            # antes: options["speed_function"] traía una función (no picklable)
            # ahora: puede traer arrays
            t_arr = options.get("speed_t", None)
            y_arr = options.get("speed_y", None)
            ns = options.get("n_speeds", None)

            if t_arr is not None and y_arr is not None:
                self.speed_function = _make_interp_function(t_arr, y_arr)

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
