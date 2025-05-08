import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

''' Kvantizacija boje je proces smanjivanja broja razliˇ citih boja u digitalnoj slici, ali
 uzimaju´ci u obzir da rezultantna slika vizualno bude što sliˇcnija originalnoj slici. Jednostavan
 naˇcin kvantizacije boje može se posti´ci primjenom algoritma K srednjih vrijednosti na RGB
 vrijednosti elemenata originalne slike. Kvantizacija se tada postiže zamjenom vrijednosti svakog
 elementa originalne slike s njemu najbližim centrom. Na slici 7.3a dan je primjer originalne
 slike koja sadrži ukupno 106,276 boja, dok je na slici 7.3b prikazana rezultantna slika nakon
 kvantizacije i koja sadrži samo 5 boja koje su odre¯ dene algoritmom K srednjih vrijednosti.'''

''' 1. Otvorite skriptu zadatak_2.py. Ova skripta uˇcitava originalnu RGB sliku test_1.jpg
 te ju transformira u podatkovni skup koji dimenzijama odgovara izrazu (7.2) pri ˇ cemu je n
 broj elemenata slike, a m je jednak 3. Koliko je razliˇ citih boja prisutno u ovoj slici?'''

unique_colors = np.unique(img_array, axis=0)
print("Broj različitih boja u slici:", unique_colors.shape[0])

''' 2. Primijenite algoritam K srednjih vrijednosti koji ´ce prona´ci grupe u RGB vrijednostima
 elemenata originalne slike.'''
k = 7
km = KMeans(n_clusters=k, init='random',n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(img_array[:,0],img_array[:,1], c=labels, cmap='viridis', s=30, label='Podaci')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title(f"podatkovni primjeri, k={k}")
plt.show()

''' 3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadaju´ cim centrom.'''

centroids = km.cluster_centers_

img_array_aprox = centroids[labels]  

img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title(f"Kvantizirana slika (K={k})")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

''' 4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene
 rezultate.'''

''' 5. Primijenite postupak i na ostale dostupne slike.'''
# ucitaj sliku 2
img = Image.imread("imgs\\test_2.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

k = 8
km = KMeans(n_clusters=k, init='random',n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)

'''prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(img_array[:,0],img_array[:,1], c=labels, cmap='viridis', s=30, label='Podaci')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title(f"podatkovni primjeri, k={k}")
plt.show()'''

centroids = km.cluster_centers_

img_array_aprox = centroids[labels]  

img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title(f"Kvantizirana slika (K={k})")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------
# ucitaj sliku 3
img = Image.imread("imgs\\test_3.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

k = 5
km = KMeans(n_clusters=k, init='random',n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)

'''prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(img_array[:,0],img_array[:,1], c=labels, cmap='viridis', s=30, label='Podaci')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title(f"podatkovni primjeri, k={k}")
plt.show()'''

centroids = km.cluster_centers_

img_array_aprox = centroids[labels]  

img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title(f"Kvantizirana slika (K={k})")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------
# ucitaj sliku 4
img = Image.imread("imgs\\test_4.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

k = 4
km = KMeans(n_clusters=k, init='random',n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)

'''prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(img_array[:,0],img_array[:,1], c=labels, cmap='viridis', s=30, label='Podaci')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title(f"podatkovni primjeri, k={k}")
plt.show()'''

centroids = km.cluster_centers_

img_array_aprox = centroids[labels]  

img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title(f"Kvantizirana slika (K={k})")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------
# ucitaj sliku 5
img = Image.imread("imgs\\test_5.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

k = 7
km = KMeans(n_clusters=k, init='random',n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)

'''prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(img_array[:,0],img_array[:,1], c=labels, cmap='viridis', s=30, label='Podaci')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title(f"podatkovni primjeri, k={k}")
plt.show()'''

centroids = km.cluster_centers_

img_array_aprox = centroids[labels]  

img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title(f"Kvantizirana slika (K={k})")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------------
# ucitaj sliku 6
img = Image.imread("imgs\\test_6.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

k = 5
km = KMeans(n_clusters=k, init='random',n_init=5, random_state=0)
km.fit(img_array)
labels = km.predict(img_array)

'''prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(img_array[:,0],img_array[:,1], c=labels, cmap='viridis', s=30, label='Podaci')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title(f"podatkovni primjeri, k={k}")
plt.show()'''

centroids = km.cluster_centers_

img_array_aprox = centroids[labels]  

img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title(f"Kvantizirana slika (K={k})")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

'''6. Grafiˇcki prikažite ovisnost J o broju grupa K. Koristite atribut inertia objekta klase
 KMeans. Možete li uoˇ citi lakat koji upu´ cuje na optimalni broj grupa?'''

inertias = []
K_range = range(1, 16)  

for k in K_range:
    km = KMeans(n_clusters=k, init='random', n_init=5, random_state=0)
    km.fit(img_array)  
    inertias.append(km.inertia_)  

plt.figure()
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Broj grupa K')
plt.ylabel('Funkcija troška J (inertia)')
plt.title('Elbow metoda za određivanje optimalnog K')
plt.grid(True)
plt.tight_layout()
plt.show()

'''7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što
 primje´ cujete?'''

k = 5
for i in range(k):
    
    mask = (labels == i).astype(np.uint8)

    mask_image = np.reshape(mask, (w, h))

    plt.figure()
    plt.title(f'Grupa {i+1}')
    plt.imshow(mask_image, cmap='gray')  
    plt.axis('off')
    plt.tight_layout()
    plt.show()