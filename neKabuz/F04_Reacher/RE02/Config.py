from dataclasses import dataclass

@dataclass
class Config:
    env_id: str = "Reacher-v5"
    seed: int = 42
    total_episodes: int = 300
    max_steps_per_episode: int = 200

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3

    buffer_size: int = int(1e6)
    batch_size: int = 256
    warmup_steps: int = 15_000

    # Ruido
    use_ou_noise: bool = False # If false == Gauss
    noise_sigma: float = 0.2
    noise_decay: float = 0.995
    noise_min_sigma: float = 0.05
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    eval_every: int = 10 # eval sin ruido cada N episodios
    eval_episodes: int = 5
    policy_freq: int = 2 # número de pasos de críticos por cada paso del actor


    save_every: int = 50
    ckpt_dir: str = "neKabuz/F04_Reacher/RE02/checkpoints_ddpg_reacher"
