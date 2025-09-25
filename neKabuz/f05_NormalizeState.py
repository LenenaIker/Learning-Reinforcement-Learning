import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode = "human")

obs_low  = env.observation_space.low.astype(np.float32)
obs_high = env.observation_space.high.astype(np.float32)

env.close()

# Gure ingurumenaren balore maximoak ta minimoak ezagutu
print("Obs_low: ", obs_low, " | Obs_high: ", obs_high)

def normalize(state):
    scale = np.where((obs_high - obs_low) == 0, 1.0, (obs_high - obs_low))
    return ((state - obs_low) / scale * 2.0 - 1.0).astype(np.float32)

# Probatu ze iteun
print(normalize([-1.2, -0.07]))
print(normalize([0.6, 0.07]))
print(normalize([0.0, 0.0]))
print(normalize([0.02, 0.02]))


# Azkenian honelin lortzedeuna
# Gure ingurumenak 2 datu itzultzeitu estatu modun: [Posizioa, Abiadura]
# Posizioak -1.2tik 0.6rako baloreak ituliko dizkigu
# Abiadurak -0.07tik 0.07rako baloreak itzuliko dizkigu
# 
# Hau jakinda, |posizioa| askoz zenbaki altuagoa izangoa |abiadura|kin konparatuz.
# 
# Horrenbeste, normalizatu in behar deu, biak eskala berdinan eoteko
# Bestela gure sare neuronalak abiadura gutxietsiko du.


