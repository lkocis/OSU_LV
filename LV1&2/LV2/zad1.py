'''Pomocu funkcija numpy.array i matplotlib.pyplot pokušajte nacrtati sliku
2.3 u okviru skripte zadatak_1.py. Igrajte se sa slikom, promijenite boju linija, debljinu linije i sl.'''

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 3, 1])
y = np.array([1, 2, 2, 1, 1])

plt.plot(x, y,  'b', linewidth=2, marker="*", markersize=12)
plt.axis([0, 4, 0, 4])
plt.xlabel('x-os')
plt.ylabel('y-os')
plt.title('ZAD 2.4.1')
plt.show()