''' Skripta zadatak_1.py uˇcitava podatkovni skup iz data_C02_emission.csv.
 Dodajte programski kod u skriptu pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:'''

import pandas as pd

data = pd.read_csv('src/data_C02_emission.csv')

'''a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka veliˇcina? Postoje li izostale ili
 duplicirane vrijednosti? Obrišite ih ako postoje. Kategoriˇcke veliˇcine konvertirajte u tip
 category.'''

data_num = len(data)
print(f'Broj mjerenja: {data_num}')
print(f'Tipovi velicina: \n{data.dtypes}')
print(f'Postoje duplicirane vrijednosti: {data.duplicated().sum()}')
print(f'\nPostoje izostale vrijednosti: {data.isnull().sum()}')

data.drop_duplicates()
data.dropna(axis = 0)
data = data.reset_index(drop = True)

for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype('category')

''' b) Koja tri automobila ima najve´ cu odnosno najmanju gradsku potrošnju? Ispišite u terminal:
 ime proizvo¯ daˇ ca, model vozila i kolika je gradska potrošnja.'''
max_consumption = data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].nlargest(3, 'Fuel Consumption City (L/100km)')
print(f'Tri auta s najvecom gradskom potrosnjom: \n{max_consumption}')

min_consumption = data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].nsmallest(3, 'Fuel Consumption City (L/100km)')
print(f'Tri auta s najmanjom gradskom potrosnjom: \n{min_consumption}')

'''c) Koliko vozila ima veliˇcinu motora izme¯ du 2.5 i 3.5 L? Kolika je prosjeˇcna C02 emisija
 plinova za ova vozila?'''
cars_num_motor = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print(f'Broj vozila koja imaju velicinu motora izmedu 2.5 i 3.5 L: {len(cars_num_motor)}')

avg_CO2_emission = cars_num_motor[['CO2 Emissions (g/km)']]
print(f'Prosjecna C02 emisija plinova za ova vozila: {avg_CO2_emission.mean()}')

''' d) Koliko mjerenja se odnosi na vozila proizvo¯ daˇca Audi? Kolika je prosjeˇcna emisija C02
 plinova automobila proizvo¯ daˇ ca Audi koji imaju 4 cilindara?'''
audi_num = (data['Make'] == 'Audi')
print(f'Broj Audi auta: {audi_num.sum()}')

audi_4cylinders = data[(data['Make'] == 'Audi') & (data['Cylinders'] == 4)]
CO2_emission_audi_4cylinders = audi_4cylinders['CO2 Emissions (g/km)']
print(f'Prosjecna emisija C02 plinova automobila proizvodaca Audi koji imaju 4 cilindara: {CO2_emission_audi_4cylinders.mean()}')

'''e) Koliko je vozila s 4,6,8... cilindara? Kolika je prosjeˇ cna emisija C02 plinova s obzirom na
 broj cilindara?'''

cars_4andmorecylinders = data[data['Cylinders'] >= 4]
print(f'Broj vozila s 4,6,8... cilindara: {len(cars_4andmorecylinders)}')
avg_CO2_cylinders = cars_4andmorecylinders['CO2 Emissions (g/km)'].mean()
print(f'Prosjecna emisija C02 plinova s obzirom na broj cilindara: {avg_CO2_cylinders}')

'''f) Kolika je prosjeˇ cna gradska potrošnja u sluˇ caju vozila koja koriste dizel, a kolika za vozila
 koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?'''
diesel_cars = data[data['Fuel Type'] == 'D']
regular_cars = data[data['Fuel Type'] == 'X']

avg_city_diesel = diesel_cars['Fuel Consumption City (L/100km)'].mean()
avg_city_regular = regular_cars['Fuel Consumption City (L/100km)'].mean()

median_city_diesel = diesel_cars['Fuel Consumption City (L/100km)'].median()
median_city_regular = regular_cars['Fuel Consumption City (L/100km)'].median()

print(f'Prosjecna gradska potrošnja za dizel: {avg_city_diesel}')
print(f'Prosjecna gradska potrošnja za regularni benzin: {avg_city_regular}')
print(f'Medijalna gradska potrošnja za dizel: {median_city_diesel}')
print(f'Medijalna gradska potrošnja za regularni benzin: {median_city_regular}')

''' g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najve´ cu gradsku potrošnju goriva?'''
car_4cy_D = data[(data['Fuel Type'] == 'D') & (data['Cylinders'] == 4)]
car_4cy_D_cityfuelconsum = car_4cy_D.nlargest(1, 'Fuel Consumption City (L/100km)')
print(f'Vozilo s 4 cilindra koje koristi dizelski motor i ima najve´ cu gradsku potrošnju goriva: \n{car_4cy_D_cityfuelconsum}')

'''h) Koliko ima vozila ima ruˇ cni tip mjenjaˇ ca (bez obzira na broj brzina)?'''
manual_car_num = data[data['Transmission'].str.startswith('M')].shape[0]
print(f'Broj vozila s rucnim mjenjacem: {manual_car_num}')

''' i) Izraˇ cunajte korelaciju izme¯ du numeriˇ ckih veliˇ cina. Komentirajte dobiveni rezultat.'''
print(f'Korelacija izmedu numerickih podataka:\n{data.corr(numeric_only=True)}')