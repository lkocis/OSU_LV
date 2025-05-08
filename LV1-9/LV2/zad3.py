'''Skripta zadatak_3.py uˇ citava sliku ’road.jpg’. Manipulacijom odgovaraju´ ce
 numpy matrice pokušajte:'''

import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('src/road.jpg')
img = img[:,:,0].copy()
print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img, cmap="gray")
plt.title("Originalna slika")
plt.show()

'''a) posvijetliti sliku,'''
img1 = img.copy()
lightened_img = np.clip(img1 + 50, 0, 255)
plt.figure()
plt.title('Posvijetljena slika')
plt.imshow(lightened_img, cmap = 'gray')
plt.show()

'''b) prikazati samo drugu ˇ cetvrtinu slike po širini,'''
height, width = img.shape
second_quarter_img = img[:, width // 4: width // 2]
plt.figure()
plt.title('Croppana slika')
plt.imshow(second_quarter_img, cmap='gray')
plt.show()

''' c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,'''
img2 = img.copy()
rotated_img = np.rot90(img2, k=-1)
plt.figure()
plt.title('Zarotirana slika')
plt.imshow(rotated_img, cmap='gray')
plt.show()

''' d) zrcaliti sliku.'''
img3 = img.copy()
mirrored_img = np.fliplr(img3)
plt.figure()
plt.title('Zarotirana slika')
plt.imshow(mirrored_img, cmap='gray')
plt.show()