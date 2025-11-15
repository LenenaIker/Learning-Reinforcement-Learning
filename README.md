# Learning RL

I bought the book "Hands-on Machine Learning with Scikit-Learn, Keras & Tensorflow" by Aurélien Géron, published by O'Reilly. I have the second edition.

In the “book” folder, there are different codes that I have copied from the book to learn.


In the folder “neKabuz,” which means “on my own”, I save the different attempts I have made in order.

Inside the ne kabuz folder are the different environments that I have attempted to/managed to resolve.

### **MountainCar-v0**

  - f08 was the first to solve MountainCar.
  - f07 came close but didn't have enough episodes.
  - In f07, I used a single model with Reward Shaping, and in f08, I used Double DQN without Reward Shaping.



### **ALE/Breakout-v5**

  - Attempt BO02 is almost the same as f08_MountainCar.
  - In BO03, I understand that PPO uses an actor and a model.
  - In BP04, I understand that I need to change the way experiences are stored --> RolloutBuffer.py. I am also beginning to understand how PPO works thanks to Gepeto's codes. Furthermore, I discover that I won't be able to run that on my laptop.
  - In BO05_Lightweight_1 and BO05_Lightweight_2, I try to make the code as efficient as possible. I finally understand that it's still too much for my laptop, and that I'll have to resort to Wrappers or even computing services like RunPod.
  - In BO06_Preproc_FrameStack, I implement AtariPreprocessing and FrameStack. The time for the first game is still around three minutes and a bit.
  - In BO07_AsyncVectorEnv, I manage to launch several environments at once, eight in my case. This reduces the time of the first game from a little over 3 minutes to 1 minute and 58 seconds. Keep in mind that the learning phase is just as heavy; what I reduce is the time of the steps. I consider it solved. With 50 episodes, it is already capable of hitting the ball. It is not capable of solving it completely, but I don't want to burn out my laptop training a model for Atari.
  - BO08 is the same as BO07, but it uses the already trained BO07 model (which has 50 episodes) and adds a reward system to subtract points for each step taken without removing a ball or for each life lost (25 episodes). It is interesting to see how the model not only fails to improve, but actually gets worse. I believe this is due to the reward shaping of subtracting reward if the ball is not removed, as I compare the current observations with the previous ones, causing the AI to choose to move constantly in any direction to avoid being punished instead of removing the ball. You can view the files ./videos/ BO07_50e.mp4 and BO08_75e.mp4 to check it out for yourself. You can also view my professional configuration BO07_Training.jpg.


### **Pendulum-v1**

It has been carried out mainly with GPT, with the objetive of understanding the idea of DDPG.


### **Reacher-v5**

It implements TD3, which is an improvement on DDPG. Once I understood DDPG thanks to Pendulum, I adapted the code to solve Reacher.

### **Walker2D-v5**

I have resolved it with SAC. It is an interesting algorithm because instead of predicting actions, it predicts means and standard deviations, which are used to generate the actions.

  - WA02: I started using Runpod seriously. PyTorch has caused me fewer problems than TensorFlow. The model with the best score seems to have learned to run on one leg, but even so, it's fast and smooth, so I'm happy.
  - WA03: This is my first attempt at controlling the agent. The goal is to steer it like a radio-controlled toy. The code was never executed because I wanted to optimize it before launching it on Runpod. I soon realized that it needed more (a lot more) than just optimization.
  - WA04: I hope to achieve the goal.


Until now, I have spent 50€ training models on Runpod.
