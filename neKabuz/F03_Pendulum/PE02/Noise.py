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


class OUNoise:
    def __init__(self, dim, mu = 0.0, theta = 0.15, sigma = 0.2):
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.dim) * self.mu

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.dim)
        self.state += dx
        return self.state
