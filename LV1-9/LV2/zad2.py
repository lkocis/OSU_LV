'''Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja ˇ data pri cemu je u ˇ
prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci´
stupac polja je masa u kg.'''

import pandas as pd
import matplotlib.pyplot as plt

'''a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja?'''
data = pd.read_csv('src\data.csv')
people_num = len(data)
print(f"Na {people_num} osoba su izvrsena mjerenja.")

'''b) Prikažite odnos visine i mase osobe pomo´ cu naredbe matplotlib.pyplot.scatter.'''
plt.figure()
plt.scatter(data.Height, data.Weight)
plt.xlabel("Visina")
plt.ylabel("Težina")
plt.title("Odnos visine i težine")
plt.show()

'''c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.'''
every_50_data = data[::50]
plt.figure()
plt.scatter(every_50_data.Height, every_50_data.Weight)
plt.xlabel("Visina")
plt.ylabel("Težina")
plt.title("Odnos visine i težine za svaku 50. osobu")
plt.show()

''' d) Izraˇ cunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom
 podatkovnom skupu.'''

min_h = data.Height.min()
max_h = data.Height.max()
mean_h = data.Height.mean()

print(f"Minimalna visina: {min_h} cm")
print(f"Maksimalna visina: {max_h} cm")
print(f"Srednja visina: {mean_h:.2f} cm")

''' e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
 muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
 ind = (data[:,0] == 1)'''

data_men = data[data['Gender'] == 1]
data_wmn = data[data['Gender'] == 0]

print(f"Minimalna visina muskaraca: {data_men.Height.min()} cm")
print(f"Maksimalna visina muskaraca: {data_men.Height.max()} cm")
print(f"Srednja visina muskaraca: {data_men.Height.mean()} cm")

print(f"Minimalna visina zena: {data_wmn.Height.min()} cm")
print(f"Maksimalna visina zena: {data_wmn.Height.max()} cm")
print(f"Srednja visina zena: {data_wmn.Height.mean()} cm")
