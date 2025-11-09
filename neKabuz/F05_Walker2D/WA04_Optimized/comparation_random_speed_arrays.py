from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from InputController import random_smooth_speed_arrays, random_speed_arrays

n_samples = 1000
n_speeds = 10 # Asko aldatzea emaitza n_speed aldatuz

y1s = []
y2s = []

for _ in range(n_samples):
    _, y1 = random_smooth_speed_arrays(n_speeds)
    _, y2 = random_speed_arrays(n_speeds)
    y1s.append(y1)
    y2s.append(y2)

y1s = np.array(y1s)
y2s = np.array(y2s)


df1 = pd.DataFrame(y1s, columns=[f"V{i}" for i in range(n_speeds)])
df2 = pd.DataFrame(y2s, columns=[f"V{i}" for i in range(n_speeds)])


plt.figure(figsize=(10,5))
plt.hist(df1.values.flatten(), bins = 30, alpha = 0.6, label = "Smooth speeds")
plt.hist(df2.values.flatten(), bins = 30, alpha = 0.6, label = "Random speeds")
plt.title("Distribución de velocidades")
plt.xlabel("Valor de velocidad")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# Eztia emaitza onak netzat. Distribuzio normalakin generatzeakoan, eztet kontuan izan abiadura maximoak eztiala askotan ateako
# Emateunez, zutik mantentzen expertoa dan tipo bat entrenatuet. 

# Kontuatunaiz eztaola hain gaizki, ze abiadura bajuak izango dia gehien erabili nahi ditutenak, abiadura altuakin kontrolaezina bihurtukoa.
# Oaintxe abiadurak [-1, 1] iguruan sortzeitut. Interneten bilatuz [-7, 7]n sortu beharko nituzkelataz kontuatu naiz.

# Distribuzio normalakin seitukot, baino aldaketa batzuk aplikatuz