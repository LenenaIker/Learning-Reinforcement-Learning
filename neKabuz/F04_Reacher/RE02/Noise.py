import numpy as np


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta = 0.15, dt = 1e-2, initial_noise = None):
        self.mean = mean # valor medio al que tiende el proceso
        self.std_deviation = std_deviation # intensidad del ruido
        self.theta = theta # Tasa de reversión hacia la media: ¿Qué tan rápido vuelve la media?. Si es alto, hace que el ruido se estabilice antes.
        self.dt = dt # sqrt(dt): escala del ruido / volatilidad
        self.initial_noise = initial_noise
        self.reset()

    def __call__(self):
        noise = (self.theta * (self.mean - self.prev_noise) * self.dt + self.std_deviation * np.sqrt(self.dt) * np.random.normal(size = self.mean.shape))
        self.prev_noise += noise
        return self.prev_noise

    def reset(self):
        self.prev_noise = self.initial_noise if self.initial_noise is not None else np.zeros_like(self.mean)


# Más adecuado para TD3
# Como Gaussiano no está "pegado" a valores previos, explora más libremente al principio.
# El paper original de TD3 muestra que el OU no da ventajas frente al gaussiano en sus benchmarks (Mujoco)
class GaussianActionNoise:
    def __init__(self, mean, std):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __call__(self):
        return np.random.normal(loc = self.mean, scale = self.std).astype(np.float32)
    
    def reset(self):
        pass 