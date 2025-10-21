====================================
CHECKLIST SIMPLE: TD3 → SAC
====================================

    1) Actor
    - Cambiar Actor determinista → Actor estocástico:
        * output: mean, log_std
        * sample acción con reparametrización (mean + std * eps)
        * aplicar tanh → acción final
        * calcular log_pi con corrección tanh
    - Devolver acción y log_pi

    2) Critic
    - Mantener 2 críticos como en TD3 (no cambia arquitectura)

    3) Alpha
    - Añadir log_alpha como parámetro
    - Optimizar log_alpha si se usa alpha automático
    - target_entropy = -act_dim

    4) act()
    - Eliminar ruido externo
    - Entrenamiento: usar acción muestreada
    - Evaluación: usar tanh(mean) sin muestreo

    5) _target()
    - Reemplazar:
        TD3: r + gamma * min(Q1,Q2)
        SAC: r + gamma * ( min(Q1,Q2) - alpha * log_pi_next )

    6) train_step()
    - Eliminar policy_delay
    - Actualizar actor y críticos cada paso
    - actor_loss = (alpha * log_pi - Q1).mean()
    - alpha_loss si alpha es automático
    - soft update siempre

    7) Quitar de TD3:
    - policy_noise, noise_clip, OU/Gaussian noise
    - policy_freq
    - policy smoothing
