import numpy as np

# Replay bufferrren berdina, baino kasu hontan objetu hau erabiliz, efizienteagoa izangoa numpy erabiltzeulako
class RolloutBuffer:
    def __init__(self, rollout_length, obs_shape):
        self.max = rollout_length
        self.states = np.zeros((rollout_length,) + obs_shape, dtype = np.uint8)
        self.actions = np.zeros((rollout_length,), dtype = np.int32)
        self.rewards = np.zeros((rollout_length,), dtype = np.float32)
        self.terminateds = np.zeros((rollout_length,), dtype = np.bool_)
        self.truncateds = np.zeros((rollout_length,), dtype = np.bool_)
        self.logprobs = np.zeros((rollout_length,), dtype = np.float32)
        self.values = np.zeros((rollout_length,), dtype = np.float32)
        self.ptr = 0
    
    def store(self, state, action, reward, terminated, truncated, logprob, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminateds[self.ptr] = terminated
        self.truncateds[self.ptr] = truncated
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr] = value
        self.ptr +=  1
    
    def full(self):
        return self.ptr >= self.max
    
    def reset(self):
        self.ptr = 0
