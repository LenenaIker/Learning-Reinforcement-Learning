from dataclasses import dataclass

@dataclass
class Config:
    
    seed: int = 42

    # Env
    env_id: str = "Walker2d-v5"
    total_episodes: int = 3500
    max_steps_per_episode: int = 1000

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

    # SAC – entropy
    auto_tune_alpha: bool = True
    alpha: float = 0.2 # only if auto_tune_alpha = False
    alpha_lr: float = 3e-4
    target_entropy: float = -6.0 # -action_dim para Walker2D


    # Evaluation & update frequencies
    # updates_per_step: int = 1
    policy_freq: int = 2 # número de pasos de críticos por cada paso del actor
    eval_every: int = 10 # eval sin ruido cada N episodios
    eval_episodes: int = 5

    # Checkpoints / best models 
    ckpt_dir: str = "neKabuz/F05_Walker2D/WA02/models_sac_walker2d"
