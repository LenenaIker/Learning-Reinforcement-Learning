import numpy as np

# Replay bufferrren berdina, baino kasu hontan objetu hau erabiliz, efizienteagoa izangoa numpy erabiltzeulako
class RolloutBufferBatched:
    def __init__(self, rollout_length, obs_shape, n_envs):
        self.LENGTH = rollout_length
        self.N_ENVS = n_envs
        self.states = np.zeros((self.LENGTH, self.N_ENVS) + obs_shape, dtype = np.uint8)
        self.actions = np.zeros((self.LENGTH, self.N_ENVS), dtype = np.int32)
        self.rewards = np.zeros((self.LENGTH, self.N_ENVS), dtype = np.float32)
        self.dones = np.zeros((self.LENGTH, self.N_ENVS), dtype = np.bool_)
        self.logprobs = np.zeros((self.LENGTH, self.N_ENVS), dtype = np.float32)
        self.values = np.zeros((self.LENGTH, self.N_ENVS), dtype = np.float32)
        self.index = 0

    
    def store_batch(self, state, action, reward, done, logprob, value):
        if self.is_full(): raise IndexError("RolloutBuffer lleno.")
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.logprobs[self.index] = logprob
        self.values[self.index] = value
        self.index +=  1

    def getXPs(self):
        end = self.index
        return (self.states[:end], self.actions[:end], self.rewards[:end], self.dones[:end], self.logprobs[:end], self.values[:end])

    def is_full(self):
        return self.index >= self.LENGTH
    
    def size(self):
        return self.index

    def reset(self):
        self.index = 0
        # Eztia beste atributo guztik birsortu behar, ze berriz idaztean baloreak zapaldukoitugu
