He cometido varios errores a la hora de entrenar WA04_Optimized en Runpod, lo que me ha llevado a perder aproximadamente 10 €. Los dos más relevantes son los siguientes:
    1. Al conectarme con SSH a Runpod, no usé herramientas como tmux, lo que me hizo que la ejecución quedase enlazada a mi terminal, perdiendo la ejecución al perder la conexión. En este caso, la solución ha sido fácil: usar tmux.
    2. A la hora de modificar la función de reward, he quitado forward_reward y he añadido una penalización por tener una velocidad diferente a la que yo externamente haya seleccionado. El forward_reward era relevante, ya que daba puntos por avanzar; al no tenerlo, mi reward no crece tan fácilmente. Esto es muy relevante, porque mi script guardaba la mejor versión, considerando la mejor versión, aquella con mayor reward. Esto ha hecho que no se guarde ningún modelo, por lo que he estado entrenando en bano.

Reward function default:
`reward = healthy_reward + forward_reward - ctrl_cost`

Reward function modificada:
`reward = healthy_reward - ctrl_cost - speed_reward`


Ejemplo: estos son logs de algunas evaluaciones:
[Eval] Episodios 1881-1900: Retorno medio = -18.38
[Eval] Episodios 1901-1920: Retorno medio = -9.75
[Eval] Episodios 1921-1940: Retorno medio = -23.37
[Eval] Episodios 1941-1960: Retorno medio = -58.34
[Eval] Episodios 1961-1980: Retorno medio = -127.22

Para compararlo, podemos observar evaluaciones guardadas con la reward function por defecto, donde:
Episodio 0980: Retorno medio = 3565
Episodio 1420: Retorno medio = 4007
Episodio 2670: Retorno medio = 4777


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