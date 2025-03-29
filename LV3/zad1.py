'''Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljede ´ ca pitanja:'''

'''a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili ˇ
duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke veli ˇ cine konvertirajte u tip ˇ
category'''
import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')
print (len(data))
print('\n')

print(data.dtypes)
print('\n')

print(f'Postoje duplicirane vrijednosti: {data.duplicated().sum()}')
data = data.drop_duplicates()
print('\n')

print(f'\nPostoje izostale vrijednosti: {data.isnull().sum()}')
data = data.dropna()

'''b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: ´
ime proizvoda¯ ca, model vozila i kolika je gradska potrošnja.'''
max_consumption = data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].nlargest(3, 'Fuel Consumption City (L/100km)')
print(f'\nTri automobila s najvecom potrosnjom: {max_consumption}')

min_consumption = data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].nsmallest(3, 'Fuel Consumption City (L/100km)')
print(f'\nTri automobila s najmanjom potrosnjom: {min_consumption}')

'''c) Koliko vozila ima velicinu motora izme ˇ du 2.5 i 3.5 L? Kolika je prosje ¯ cna C02 emisija ˇ
plinova za ova vozila?'''
motor_size = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print(f'\nBroj vozila s motorom koji je između 2.5L i 3.5L: {len(motor_size)}')

avg_CO2_emission = data['Engine Size (L)'].mean()
print(f'\nSrednja vrijednost C02 emisije plinova: {avg_CO2_emission}')

'''d) Koliko mjerenja se odnosi na vozila proizvoda¯ ca Audi? Kolika je prosje ˇ cna emisija C02 ˇ
plinova automobila proizvoda¯ ca Audi koji imaju 4 cilindara?'''
audi_num = (data['Make'] == 'Audi').sum()
print(f'\nBroj Audi vozila u mjerenjima: {audi_num}')

audi_4cylinders = data[(data['Make'] == 'Audi') & (data['Cylinders'] == 4)]
CO2_emission_audi_4cylinders = audi_4cylinders['CO2 Emissions (g/km)'].mean()
print(f'\nProsjecna emisija C02 plinova automobila proizvodaca Audi koji imaju 4 cilindra: {CO2_emission_audi_4cylinders}')

'''e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na ˇ
broj cilindara?'''
cylinders_4_6_8 = len(data[(data['Cylinders'] == 4) | (data['Cylinders'] == 6) | (data['Cylinders'] == 8)])
print(f'\nBroj vozila s 4,6,8... cilindara: {cylinders_4_6_8}')

average_co2_by_cylinders = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print(f'\nProsjecna emisija C02 plinova s obzirom na broj cilindara: {average_co2_by_cylinders}')

'''f) Kolika je prosjecna gradska potrošnja u slu ˇ caju vozila koja koriste dizel, a kolika za vozila ˇ
koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?'''
regular_gasoline_vehicles = data[data['Fuel Type'] == 'X']
average_city_fuel_consumption_d = regular_gasoline_vehicles['Fuel Consumption City (L/100km)'].mean()
print(f'\nProsječna gradska potrošnja goriva za vozila koja koriste regularni benzin je: {average_city_fuel_consumption_d} L/100km')

diesel_vehicles = data[data['Fuel Type'] == 'D']
average_city_fuel_consumption_ag = diesel_vehicles['Fuel Consumption City (L/100km)'].mean()
print(f'\nProsječna gradska potrošnja goriva za vozila koja koriste dizel je: {average_city_fuel_consumption_ag} L/100km')

'''g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?'''
diesel_4_cylinders = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
max_fuel_consumption_index = diesel_4_cylinders['Fuel Consumption City (L/100km)'].idxmax()
max_fuel_consumption_vehicle = data.loc[max_fuel_consumption_index]
print(f'\nPodaci za vozilo s najvećom gradskom potrošnjom goriva: {max_fuel_consumption_vehicle}')

'''h) Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)?'''
manual_transmission_vehicles = len(data[data['Transmission'] == 'M'])
print(f'\nBroj vozila s ručnim mjenjačem: {manual_transmission_vehicles}')

'''i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat.'''
numeric_columns = data.select_dtypes(include=['number']).columns
correlation_matrix = data[numeric_columns].corr()
print(f'\nKorelacija između numeričkih veličina: {correlation_matrix}')



    




