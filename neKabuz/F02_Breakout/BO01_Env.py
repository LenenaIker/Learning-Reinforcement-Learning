import gymnasium as gym
import ale_py
from time import sleep
import numpy as np

gym.register_envs(ale_py)

MAX_EPISODE_STEPS = 100

env = gym.make("ALE/Breakout-v5", render_mode = "human", max_episode_steps = MAX_EPISODE_STEPS + 1)

print("Action space: ", env.action_space.n, " Action names:\n", env.unwrapped.get_action_meanings())
print("Observations/State: ", env.observation_space.shape)


obs, info = env.reset()

print("Obs: ", obs, " | Info: ",  info)

print("Bizitzak: ", info.get("lives"),"\n")

OBS_LOWS = env.observation_space.low.astype(np.float32)
OBS_HIGHS = env.observation_space.high.astype(np.float32)

print(OBS_LOWS, OBS_HIGHS)

for i in range(MAX_EPISODE_STEPS):
    action = i % env.action_space.n
    
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {i}\nAction {action}\nReward {reward}\nInfo: {info}\n\n")

    if terminated or truncated:
        print(f"Terminado {terminated}\nTruncado {truncated}")
        break

    sleep(1.5)


env.close()



# Oaingoz ikusitenagatik, 4 akzio ditu:
# 0 == Ezer ez egin
# 1 == Pelota atea
# 2 == Pelota muitzen dun paleta ezkerreta muitu
# 3 == Pelota muitzen dun paleta eskubira muitu


# Obserbazio askoz gehio daude entorno hontan, MountainCarrekin konparatuz: Observations/State:  (210, 160, 3)
# Ustet estatuak pelota nun daon esangoiala
# Emateu 0-255 bitarteko baloreak itzultzeitula obserbazio bezala.
# Honek esateigu, gure sarrera 210x160 pixel-eko irudi bat izangoala, eta irudi horretako pixel bakoitzak 3 balore izango ditula (R, G, B) koloreak adierazteko. 

# Infok oain bai zeoze ekartzeiala: Info: {'lives': 5, 'episode_frame_number': 0, 'frame_number': 0}


# Teja bat puskatzeak reward 1 emateu:
# Step 89
# Action 1
# Reward 1.0
# Info: {'lives': 3, 'episode_frame_number': 360, 'frame_number': 360}


# Kontuan izan behar deu, palak pelota jotzeunetik, pelotak teja puskatu harte denboa bat pasatzeal.
# Honek esan nahi du, orainean lortutako irabaziek, iraganean pasatako akzioak saritu beharkoitula
# Beraz, gure modeloak epe luzeko ikuspegia izan behar du. Hau lortzeko "deskontu" sistema bat aplikatu behakoeu "discount factor".

