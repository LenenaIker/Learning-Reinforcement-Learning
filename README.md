# Learning RL

I bought the book "Hands-on Machine Learning with Scikit-Learn, Keras & Tensorflow" by Aurélien Géron, published by O'Reilly. I have the second edition.

In the “book” folder, there are different codes that I have copied from the book to learn.


In the folder “neKabuz,” which means “on my own”, I save the different attempts I have made in order.

Inside the ne kabuz folder are the different environments that I have attempted to/managed to resolve.

MountainCar-v0

  - f08 was the first to solve MountainCar.
  - f07 came close but didn't have enough episodes.
  - In f07, I used a single model with Reward Shaping, and in f08, I used Double DQN without Reward Shaping.



ALE/Breakout-v5

  - Attempt BO02 is almost the same as f08_MountainCar.
  - In BO03, I understand that PPO uses an actor and a model.
  - In BP04, I understand that I need to change the way experiences are stored --> RolloutBuffer.py. I am also beginning to understand how PPO works thanks to Gepeto's codes. Furthermore, I discover that I won't be able to run that on my laptop.
  - In BO05_Lightweight_1 and BO05_Lightweight_2, I try to make the code as efficient as possible. I finally understand that it's still too much for my laptop, and that I'll have to resort to Wrappers or even computing services like RunPod.
  - In BO06_Preproc_FrameStack, I implement AtariPreprocessing and FrameStack. The time for the first game is still around three minutes and a bit.
  - In BO07_AsyncVectorEnv, I manage to launch several environments at once, eight in my case. This reduces the time of the first game from a little over 3 minutes to 1 minute and 58 seconds. Keep in mind that the learning phase is just as heavy; what I reduce is the time of the steps.
  - BO08 is the same as BO07 but uses the already trained BO07 model (which has 50 episodes) and adds reward shaping to deduct points for each step taken without removing a ball or for each life lost. It is interesting to see how not only does the model not improve, but it actually gets worse. You can watch the ./videos/ BO07_50e.mp4 and BO08_75e.mp4 to see for yourself. You can also view my professional setup BO07_Training.jpg.
