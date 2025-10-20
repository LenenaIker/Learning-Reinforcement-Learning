from dataclasses import dataclass

@dataclass
class Config:
    
    seed: int = 42

    # Env
    env_id: str = "Reacher-v5"
    total_episodes: int = 2000
    max_steps_per_episode: int = 500

    # Discount factor
    gamma: float = 0.99
    
    # Soft update
    tau: float = 0.005

    # Optimizers
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Memory related
    buffer_size: int = int(1e6)
    batch_size: int = 256
    warmup_steps: int = 5_000

    # Noise
    use_ou_noise: bool = False # True = Ornstein-Uhlenbeck | False = Gaussian
    noise_sigma: float = 0.2 # Standard deviation of exploration noise
    noise_decay: float = 0.98 # Per-episode decay factor for exploration noise
    noise_min_sigma: float = 0.03

    # Policy smothing
    policy_noise: float = 0.2 # Standard deviation of smoothing noise
    noise_clip: float = 0.5

    # Evaluation & update frequencies
    eval_every: int = 10 # eval sin ruido cada N episodios
    eval_episodes: int = 5
    policy_freq: int = 2 # número de pasos de críticos por cada paso del actor

    # Checkpoints / best models 
    ckpt_dir: str = "neKabuz/F04_Reacher/RE02/checkpoints_ddpg_reacher"
