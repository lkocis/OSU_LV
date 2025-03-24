''' Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
 ženama. Skripta zadatak_2.py uˇ citava dane podatke u obliku numpy polja data pri ˇ cemu je u
 prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci
 stupac polja je masa u kg.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

# a) Na temelju veliˇ cine numpy polja data, na koliko osoba su izvršena mjerenja?
peopleNum = len(data)
print(f"\na) Broj osoba je: {peopleNum}")

# b) Prikažite odnos visine i mase osobe pomo´ cu naredbe matplotlib.pyplot.scatter.
'''
x = data[:, 1] 
y = data[:, 2]
plt.scatter(x, y, color='blue')
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height-weight ratio")
plt.show()
'''

# c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
'''
x50 = data[0::50, 1]
y50 = data[0::50, 2]
plt.scatter(x50, y50, color='blue')
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height-weight ratio")
plt.show()
'''

# d) Izraˇ cunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom podatkovnom skupu.
print(f"\nd) Vrijednosti:")
print(f"Minimalna vrijednost visine: {data[:, 1].min()}")
print(f"Maksimalna vrijednost visine: {data[:, 1].max()}")
print(f"Srednja vrijednost visine: {data[:, 1].mean()}")

#e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka. ind = (data[:,0] == 1)
heights = data[:, 1]
ind_men = (data[:, 0] == 1)  
heights_men = heights[ind_men]

min_height_men = np.min(heights_men)
max_height_men = np.max(heights_men)
mean_height_men = np.mean(heights_men)

print(f"\ne) Za muškarce:")
print(f"Minimalna visina: {min_height_men} cm")
print(f"Maksimalna visina: {max_height_men} cm")
print(f"Srednja visina: {mean_height_men} cm")

ind_women = (data[:, 0] == 0)  
heights_women = heights[ind_women]

min_height_women = np.min(heights_women)
max_height_women = np.max(heights_women)
mean_height_women = np.mean(heights_women)

print(f"\ne) Za žene:")
print(f"Minimalna visina: {min_height_women} cm")
print(f"Maksimalna visina: {max_height_women} cm")
print(f"Srednja visina: {mean_height_women} cm")
