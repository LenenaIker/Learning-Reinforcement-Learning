from dataclasses import dataclass

@dataclass
class Config:
    
    seed: int = 42
    use_compile = False # Me ha dado muchos problemas. Recomiendo mantenerlo en False.

    # Env
    env_id: str = "Walker2d-v5"
    total_episodes: int = 3500
    max_steps_per_episode: int = 1000

    num_train_envs = 16
    num_eval_envs = 4
    
    # Discount factor
    gamma: float = 0.99

    # Reward function
    sigma_speed: float = 0.7
    sigma_torso: float = 0.5

    weight_speed: float = 4.0
    weight_torso: float = 1.5
    
    speed_name: str = "x_velocity"
    torso_height: float = 1.2 # A ojo, berez ustet 1.25ekin hasteala, baño nahiagoet belaunak flexionatuak izatea

    # Soft update
    tau: float = 0.005

    # Optimizers
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Memory related
    buffer_size: int = int(1e6)
    batch_size: int = 1024
    warmup_steps: int = 5_000

    # SAC – entropy
    alpha: float = 0.2
    alpha_lr: float = 3e-4
    target_entropy: float | None = None# -6.0 # -action_dim para Walker2D


    # Evaluation & update frequencies
    updates_per_step: int = 2
    # policy_freq: int = 2 # número de pasos de críticos por cada paso del actor, en SAC no se usa esto.
    eval_every: int = 35
    eval_episodes: int = 5
    target_update_every = 1
    save_every = 500

    # Checkpoints / best models 
    ckpt_dir: str = "neKabuz\F05_Walker2D\WA04_Optimized\models_sac_walker2d"
