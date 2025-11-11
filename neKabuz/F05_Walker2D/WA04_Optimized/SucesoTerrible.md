He cometido varios errores a la hora de entrenar WA04_Optimized en Runpod, lo que me ha llevado a perder aproximadamente 10 €. Los dos más relevantes son los siguientes:

    1. Al conectarme con SSH a Runpod, no usé herramientas como tmux, lo que me hizo que la ejecución quedase enlazada a mi terminal, perdiendo la ejecución al perder la conexión. En este caso, la solución ha sido fácil: usar tmux.

    2. A la hora de modificar la función de reward, he quitado forward_reward y he añadido una penalización por tener una velocidad diferente a la que yo externamente haya seleccionado. El forward_reward era relevante, ya que daba puntos por avanzar; al no tenerlo, mi reward no crece tan fácilmente. Esto es crucial, porque mi script guardaba la mejor versión, considerando la mejor versión, aquella con mayor reward. Esto ha hecho que no se guarde ningún modelo, por lo que he estado entrenando en vano.

Reward function default:
`reward = healthy_reward + forward_reward - ctrl_cost`

Reward function modificada:
`reward = healthy_reward - ctrl_cost - speed_reward`


Ejemplo: estos son logs de algunas evaluaciones:

``` bash
[Eval] Episodios 1881-1900: Retorno medio = -18.38
[Eval] Episodios 1901-1920: Retorno medio = -9.75
[Eval] Episodios 1921-1940: Retorno medio = -23.37
[Eval] Episodios 1941-1960: Retorno medio = -58.34
[Eval] Episodios 1961-1980: Retorno medio = -127.22
```

Para compararlo, podemos observar evaluaciones guardadas con la reward function por defecto, donde:

``` bash
Episodio 0980: Retorno medio = 3565
Episodio 1420: Retorno medio = 4007
Episodio 2670: Retorno medio = 4777
```

Por lo que, teniendo en cuenta qué:
    healthy_reward = punto por no morirse por step
    ctrl_cost = penalización por tomar acciones demasiado grandes
    speed_reward = penalización por tener una velocidad alejada de la seleccionada por mí

me encuentro en una situación donde la única posibilidad de obtener un modelo entrenado, es que mi modelo siga vivo mucho tiempo, seleccionando acciones pequeñas que satisfagan las velocidades.

En el entrenamiento la función de velocidad es aleatoria, por lo que es posible que le toqué una muy fácil, permitiéndole tener un reward suficientemente alto.

Ya van 11h 22m de ejecución, el reward más alto por ahora ha sido 73 y ha ocurrido en el episodio 20.
01:02 y tengo pocas expectativas de que vaya a conseguir un modelo.

[Eval] Episodios 2061-2080: Retorno medio = 18.83

Ojito, tremendo intento de aproximarse al record. Ha sucedido a las 01:16.

No sé si dejarlo hasta mañana. Acabo de añadir 10€ de saldo por si hay suerte (Gambling).



Por alguna razón no avanza. Miro las telemetrías y están a 0. Me rindo, lo intentaré mañana.


Además del problema del reward_function, aumentar la cantidad de perceptrones por capa no ha sido buena idea. Ha resultado en overfitting. He vuelto a bajar los perceptrones y parece ser que ya está obteniendo resultados. 

``` bash
Episodio 101 | Retorno medio:    81.07
Episodio 102 | Retorno medio:   656.05
Episodio 103 | Retorno medio:    86.19
Episodio 104 | Retorno medio:    69.16
Episodio 105 | Retorno medio:   203.52
Episodio 106 | Retorno medio:   383.57
Episodio 107 | Retorno medio:   101.87
Episodio 108 | Retorno medio:   200.29
Episodio 109 | Retorno medio:   372.42
Episodio 110 | Retorno medio:   319.69
Episodio 111 | Retorno medio:   279.44
Episodio 112 | Retorno medio:   235.00
Episodio 113 | Retorno medio:   400.50
Episodio 114 | Retorno medio:   171.98
Episodio 115 | Retorno medio:   147.61
Episodio 116 | Retorno medio:   131.92
Episodio 117 | Retorno medio:   194.00
Episodio 118 | Retorno medio:   149.68
Episodio 119 | Retorno medio:   197.61
Episodio 120 | Retorno medio:   354.99
[Eval] Episodios 101-120: Retorno medio = 134.54
```

Me da un poco de pena porque estoy alquilando una tarjeta gráfica para hacer uso de solo el 20%, cuando antes me acercaba al 80%. Pero de nada sirve aumentar el uso si mi modelo no llega a aprender.

Aun así he añadido lo siguiente para guardar modelos cada 250 episodios:

``` python
if ep % config.save_every == 0:
    path = os.path.join(config.ckpt_dir, f"latest_ep{ep}.pt")
    agent.save(path)
```


# Parte 2

He adaptado el reward function para que, en vez de ser una distribución normal en el rango [-1, 1], esté en [-6, 6].
Esto debería de ayudar a generar más peticiones de velocidades alcanzables como 3 m/s.
Además, he sesgado la distribución para que sea un poco positivista. Ya que al Walker2D, le cuesta andar hacia atrás.
Por lo que pasaría de [-6, 6] a [-4.1, 5.1]

Recomiendo ejecutar WA04_Optimized\comparation_random_speed_arrays.py para visualizar la distribución normal.

El agente lleva 19h 25m entrenando y estos son los últimos logs:

``` bash
[Eval] Episodios 1041-1060: Retorno medio = -490.09
Episodio 1061 | Retorno medio:   206.77
Episodio 1062 | Retorno medio:    62.08
Episodio 1063 | Retorno medio:    61.88
Episodio 1064 | Retorno medio:    71.41
Episodio 1065 | Retorno medio:   138.23
Episodio 1066 | Retorno medio:   -22.83
Episodio 1067 | Retorno medio:    37.38
Episodio 1068 | Retorno medio:   198.51
Episodio 1069 | Retorno medio:   227.82
Episodio 1070 | Retorno medio:    37.18
Episodio 1071 | Retorno medio: -1363.37
Episodio 1072 | Retorno medio:    41.71
Episodio 1073 | Retorno medio:   693.12
Episodio 1074 | Retorno medio:   174.03
Episodio 1075 | Retorno medio:   114.85
Episodio 1076 | Retorno medio:   153.39
Episodio 1077 | Retorno medio:   356.20
Episodio 1078 | Retorno medio:   118.98
Episodio 1079 | Retorno medio:  -206.60
Episodio 1080 | Retorno medio:   485.94
[Eval] Episodios 1061-1080: Retorno medio = -1041.03
Episodio 1081 | Retorno medio:  -522.00
Episodio 1082 | Retorno medio:    31.45
Episodio 1083 | Retorno medio:   108.12
Episodio 1084 | Retorno medio:     0.08
Episodio 1085 | Retorno medio:   234.33
Episodio 1086 | Retorno medio:    81.30
Episodio 1087 | Retorno medio:   110.46
Episodio 1088 | Retorno medio:   217.20
Episodio 1089 | Retorno medio:   168.93
Episodio 1090 | Retorno medio:   -79.95
Episodio 1091 | Retorno medio:   -51.55
Episodio 1092 | Retorno medio:    51.65
Episodio 1093 | Retorno medio:     4.76
Episodio 1094 | Retorno medio:    55.01
Episodio 1095 | Retorno medio: -8224.31   <-- ¿Cómo ha llegado a un reward medio tan tan malo? JAJAJAJ, lo consideraré un logro.
Episodio 1096 | Retorno medio:   -22.13
Episodio 1097 | Retorno medio:  -186.41
Episodio 1098 | Retorno medio:   340.60
Episodio 1099 | Retorno medio:   269.70
Episodio 1100 | Retorno medio:    99.61
[Eval] Episodios 1081-1100: Retorno medio = -852.44
```

Son malas noticias. Por ahora no conozco la razón por la cual haya ocurrido esto. Quizás va siendo hora de añadir un sistema de logs/telemetrías más avanzado, estilo tensorboard.

Es posible que tenga que reducir el peso de la penalización por no tener la velocidad adecuada.
Supongo que al ser un número [-7, 7] al cuadrado (^2), se vuelve un número demasiado grande, que rompe mi función de reward. También podría aplicar abs(), para no usar ^2, o sqrt(+1), para limitar la influencia de diferencias enormes.


He parado la ejecución. Voy a realizar cambios y lanzar el entrenamiento de nuevo.


# Parte 3

Tras un día y diecisiete horas de ejecución, he entrenado a un inutil. No sirve para nada.

Tengo que hacer cambios importantes en la reward function. Y gastar otros 10 € en entrenar una cosa qué no se si funcionará. XD

Por ahora voy a reducir la demanda de velocidades negativas en el generador de velocidades aleatorias.
``` python
# InputController.py

# Antes
Y_MIN, Y_MAX = -2.5, 4.0

# Después
Y_MIN, Y_MAX = -1.0, 4.0
```
Además voy a quitar el castigo por alejarse demasiado del objetivo y voy a ampliar el rango de premio.
``` python
# EnvWrapper.py

# Antes
rew = rew + w1 * track_reward - w2 * (error**2)

# Después
rew = rew + w1 * track_reward
```

Recomiendo abrir GeoGebra y abrir el siguiente archivo para ver el cambio al quitar el castigo:
``` bash
neKabuz\F05_Walker2D\WA04_Optimized\GeoGebra_RewardRepresentation.ggb
```

Voy a aumentar el sigma de 0.5 a 2, para ampliar el rango donde hace efecto el reward.