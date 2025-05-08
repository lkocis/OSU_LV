''' Napišite program koji ´ ce kreirati sliku koja sadrži ˇ cetiri kvadrata crne odnosno
 bijele boje (vidi primjer slike 2.4 ispod). Za kreiranje ove funkcije koristite numpy funkcije
 zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. Kako biste ih složili
 u odgovaraju´ ci oblik koristite numpy funkcije hstack i vstack.'''

import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50, 50))
white = np.ones((50, 50)) * 255

top_row = np.hstack((black, white))
bottom_row = np.hstack((white, black))
final_img = np.vstack((top_row, bottom_row))

plt.xticks(np.arange(0, 100, 20))  
plt.yticks(np.arange(0, 100, 20))
plt.gca().invert_yaxis()
plt.imshow(final_img, cmap='gray', vmin=0, vmax=255)
plt.show()