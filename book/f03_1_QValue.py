import numpy as np
from f03_0_Data import transition_probabilities, rewards, possible_actions


# Azkenian Bellmanek esateuna da, Markoven erabaki prozesu baten estatu, akzio ta probabilitate guztik bazkau.
# Politika hobena kalkula dezakeula.
# Q_star[s, a] = E[ R[s, a] + gamma * max_a_prime Q_star[s_next, a_prime] ]

# Q_star(s, a) = Valor óptimo de tomar acción a en el estado s.
# s = Estado actual.
# a = Acción actual.
# s_next = Estado siguiente (al que se llega después de tomar acción a en s).
# a' (a prima) = Cualquier acción posible en el siguiente estado.
# R(s, a) = Recompensa inmediata obtenida al hacer acción a en estado s.
# gamma (γ) = Factor de descuento (número entre 0 y 1 que mide cuánto importa el futuro).
# max_a' Q_star(s_next, a') = El valor máximo de las acciones posibles en el próximo estado.
# E[ ... ] = Valor esperado (promedio ponderado si el entorno es estocástico).

Q_values = np.full(shape = (3, 3), fill_value = -np.inf)

for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0

print(Q_values) # Akzio imposibleak infinitoakin geatzeia, bestiak 0.0akin



gamma = 0.95 # 90 jartzebaet [0, 0, 1] akzioak diala hobenak esateu, baino 95 ekin [0, 2, 1]

# Rellenamos nuestra Q_table usando la formula Q_value
for i in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                transition_probabilities[s][a][sp] * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp])) for sp in range(3)
            ])

print(Q_values)

# Ta oain, estado bakoitzeko erabaki onena ikusi nahi baeu, probabilitate maximoa dakan argumentua erakustedeu argmax erabiliz.
print(np.argmax(Q_values, axis = 1))
