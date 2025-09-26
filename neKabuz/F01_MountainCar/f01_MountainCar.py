
import gymnasium as gym
import numpy as np
import keras

env = gym.make("MountainCar-v0", render_mode = "human")


# Helburua: Banderiña ikutzea


print(env.action_space.n)
print(env.observation_space.shape)


obs, info = env.reset()

print(obs, info)

# Bakitenez 3 akzio ditula bakarrik, bakoitzak ze iteun ikusikot:

action = 0
for i in range(200): # 200 Bakarrik ze max_episode_steps defektuz hoi da.

    if i == 70:
        action = 1
    elif i == 140:
        action = 2

    # print(f"I {i}   Action {action}")

    obs, reward, terminated, truncated, info = env.step(action)

    if i % 25 == 0:
        # print(obs, info)
        pass

    if terminated or truncated:
        print(f"Terminado {terminated}   Truncado {truncated}")
        break

# 0 == Ezker
# 1 == Ezer
# 2 == Eskubi

env.close()


model = keras.models.Sequential([
    keras.Input(shape = env.observation_space.shape),
    keras.layers.Dense(32, activation = "elu"),
    keras.layers.Dense(32, activation = "elu"),
    keras.layers.Dense(env.action_space.n)
])

print(model.summary())
