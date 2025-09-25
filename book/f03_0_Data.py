
# Escribir las probabilidades para cada acción en cada estado de la imágen:
# .\images\markov-decision-process-machine-learning-reinforcement-learning.png

# Hay 3 estados, y cada estado tiene sus acciones, con sus probabilidades de viajar a estados.

transition_probabilities = [ # Shape = [s, a, s']
    [ # Status 0
        [ # Action 0
            0.7, # 70% de posibilidades de volver al estado 0 (al estado de donde ha salido)
            0.3, # 30% para S1
            0.0 # 0% para S2
        ],
        [ # A1
            1.0, # Si o si vuelve al S0
            0.0,
            0.0
        ],
        [ # A2
            0.8,
            0.2,
            0.0
        ]
    ],
    [ # S1
        [ # A0
            0.0,
            1.0,
            0.0
        ],
        None, # No hay A1
        [ # A2
            0.0,
            0.0,
            1.0
        ]
    ],
    [ # S2
        None, # No hay A0
        [ # A1
            0.8,
            0.1,
            0.1
        ],
        None # No hay A2
    ]
]

# Goikoan berdina baino irabazi/galduak gordetzeko
rewards = [
    [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
    [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]
]

# Hau berez 'transition_probabilities'etik atea daiteke, bainoweno...
possible_actions = [
    [0, 1, 2], # S0
    [0, 2], # S1
    [1] # S2
]
