'''Skripta zadatak_3.py uˇ citava sliku ’road.jpg’. Manipulacijom odgovaraju´ ce
 numpy matrice pokušajte:
 a) posvijetliti sliku,
 b) prikazati samo drugu ˇ cetvrtinu slike po širini,
 c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
 d) zrcaliti sliku.'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = r'D:\OSU_LV\src\road.jpg' 
image = plt.imread(image_path)
plt.figure()

# a) Posvijetliti sliku
image_1 = image.copy()
plt.subplot(2, 2, 1)
plt.title("a) Posvijetljena slika")
plt.imshow(image_1, alpha=0.5)
plt.axis('off')

#b) prikazati samo drugu ˇ cetvrtinu slike po širini,
image_2 = Image.open(image_path)
height, width = image_2.size
image_cropped = image_2.crop((0, height // 4, width, height//2))
plt.subplot(2, 2, 2)  
plt.imshow(image_cropped)
plt.title("b) Druga četvrtina slike po širini")
plt.axis('off')

#c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
image_3 = image_2.copy()
image_rotated = image_3.rotate(-90)
plt.subplot(2, 2, 3)  
plt.imshow(image_rotated)
plt.title("b) Zarotirana slika za 90 stupnjeva")
plt.axis('off')

#d) zrcaliti sliku.
image_4 = image_2.copy()
image_mirrored = np.fliplr(image_4)
plt.subplot(2, 2, 4)  
plt.imshow(image_mirrored)
plt.title("b) Zrcaljena slika")
plt.axis('off')
plt.show()



