import math
import torch
import torch.nn as nn



def fanin_init(layer: nn.Linear):
    bound = 1.0 / math.sqrt(layer.weight.data.size()[1]) # in_features
    nn.init.uniform_(layer.weight.data, -bound, bound)
    nn.init.uniform_(layer.bias.data, -bound, bound)

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden = (1024, 1024), activation_fn = nn.ReLU, out_act = None):
        super().__init__()
        layers = []
        
        last = input_dim
        for h in hidden:
            lin = nn.Linear(last, h)
            fanin_init(lin)
            layers += [lin, activation_fn()]
            last = h
        
        out = nn.Linear(last, output_dim)

        nn.init.uniform_(out.weight, -3e-3, 3e-3)
        nn.init.uniform_(out.bias, -3e-3, 3e-3)
        
        if out_act:
            layers += [out, out_act()]
        else:
            layers += [out]
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StochasticActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden = (1024, 1024), min_log_std: float = -20.0, max_log_std: float = 2.0, act_limit: float = 1.0):
        super().__init__()
        self.body = MLP(
            obs_dim,
            2 * act_dim, # salida: [mean, log_std]
            hidden = hidden
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.act_limit = act_limit

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True):
        """
        El cuerpo (body) del actor va a devolver medias y desviaciones estandar.
        Con una media y una desviación estandar, tenemos una ditribución normal.
        Se selecciona un número aleatorio dentro de la distribución. El número se llama z.
        Se escala z a [-1, 1] usando tanh() y el resultado será nuestra acción.

        Args:
            obs: Observaciones del entorno.
            deterministic (bool): if True: z = mu else: z = mu + std * rand.
            with_logprob (bool): if False: omite el cálculo del log-prob (ahorro de cómputo).

        Returns:
            action: Acción
            log_pi: Probabilidad de que haya salido action teniendo en cuenta mu y std.

        """

        mu_logstd = self.body(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim = -1)

        # clamp de log_std para estabilidad
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std) # Asegurar que std sea siempre positivo (std negativo no tiene sentido estadistico), ya que log_std es el resultado de la red que puede ser negativo

        # reparametrización: z = mu + std * eps (o determinista en eval)
        if deterministic:
            z = mu
        else:
            eps = torch.randn_like(mu) # No usa mu para nada, solo para saber el tamaño que tiene que tener el output
            # eps en este punto es puro ruido. Un tensor del tamaño de mu, pero lleno de ruido.
            z = mu + std * eps # Transformar el tensor de ruido en una distribución gaussiana centrada en mu con dispersión std

        # squash: a = tanh(z) -> acción final (opcional reescala)
        a = torch.tanh(z)
        action = a * self.act_limit


        # Calcular la probabilidad de que haya salido action teniendo en cuenta mu y std.
        log_pi = None
        if with_logprob:
            # log N(z | mu, std): suma sobre dimensiones de la acción
            # -0.5 * [ ((z - mu)/std)^2 + 2*log_std + log(2*pi) ]
            gaussian_log_prob = -0.5 * (
                ((z - mu) / (std + 1e-8))**2 + 2.0 * log_std + math.log(2.0 * math.pi)
            )

            gaussian_log_prob = gaussian_log_prob.sum(dim = -1, keepdim = True)

            # corrección por el cambio de variable de tanh:
            # log |det(d a / d z)| = sum log(1 - tanh(z)^2)
            # -> se RESTA porque p_a(a) = p_z(z) * |det(Jacobian)|^{-1}
            correction = torch.log(1.0 - a.pow(2) + 1e-6).sum(dim = -1, keepdim = True)
            log_pi = gaussian_log_prob - correction

        return action, log_pi


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = MLP(
            obs_dim + act_dim, # Input shape == Obserbations + Actions shape
            1, # Output shape == Value
            hidden = (1024, 1024)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim = -1)
        return self.q(x)


