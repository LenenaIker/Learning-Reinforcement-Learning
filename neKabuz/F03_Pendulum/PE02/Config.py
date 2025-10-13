from dataclasses import dataclass

@dataclass
class Config:
    env_id: str = "Pendulum-v1"
    seed: int = 42
    total_episodes: int = 300
    max_steps_per_episode: int = 200

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3

    buffer_size: int = int(1e6)
    batch_size: int = 256
    warmup_steps: int = 1000

    # Ruido
    use_ou_noise: bool = True
    noise_sigma: float = 0.2
    noise_decay: float = 0.995
    noise_min_sigma: float = 0.05

    eval_every: int = 10 # eval sin ruido cada N episodios
    eval_episodes: int = 5

    save_every: int = 50
    ckpt_dir: str = "checkpoints_ddpg_pendulum"
