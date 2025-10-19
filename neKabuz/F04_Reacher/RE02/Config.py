from dataclasses import dataclass

@dataclass
class Config:
    
    seed: int = 42

    # Env interactions
    env_id: str = "Reacher-v5"
    total_episodes: int = 5000
    max_steps_per_episode: int = 250

    # 
    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Memory related
    buffer_size: int = int(1e6)
    batch_size: int = 256
    warmup_steps: int = 25_000

    # Noise
    use_ou_noise: bool = False # ou_noise if use_ou_noise else gauss
    noise_sigma: float = 0.2
    noise_decay: float = 0.98
    noise_min_sigma: float = 0.03
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    # Wait X for something to happen
    eval_every: int = 10 # eval sin ruido cada N episodios
    eval_episodes: int = 5
    policy_freq: int = 2 # número de pasos de críticos por cada paso del actor


    ckpt_dir: str = "neKabuz/F04_Reacher/RE02/checkpoints_ddpg_reacher"
