''' Napišite programski kod koji ´ ce prikazati sljede´ ce vizualizacije:'''

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('src/data_C02_emission.csv')

'''a) Pomo´ cu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.'''
plt.figure()
plt.hist(data['CO2 Emissions (g/km)'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('CO₂ emisije (g/km)')
plt.ylabel('Broj vozila')
plt.title('Histogram emisije CO₂ (g/km)')
plt.grid(True)
plt.show()

''' b) Pomo´cu dijagrama raspršenja prikažite odnos izme¯ du gradske potrošnje goriva i emisije
 C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izme¯ du
 veliˇ cina, obojite toˇ ckice na dijagramu raspršenja s obzirom na tip goriva.'''
plt.figure()
plt.scatter(x=data['Fuel Consumption City (L/100km)'], y=data['CO2 Emissions (g/km)'], color='blue', edgecolors='black')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Odnos izmedu gradske potrošnje goriva i emisije C02 plinova')
plt.grid(True)
plt.show()

'''c) Pomo´ cu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip
 goriva. Primje´ cujete li grubu mjernu pogrešku u podacima?'''
plt.figure()
data.boxplot(column =['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.xlabel('Fuel Consumption Hwy (L/100km)')
plt.ylabel('Fuel Type')
plt.title('Razdioba izvangradske potrošnje s obzirom na tip goriva')
plt.grid(True)
plt.show()

'''d) Pomo´cu stupˇcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu
 groupby.'''
fuel_counts = data.groupby('Fuel Type').size()
plt.figure()
plt.bar(fuel_counts.index, fuel_counts.values, color='blue')
plt.xlabel('Fuel Type')
plt.ylabel('Number of vehicles')
plt.title('Broj vozila po tipu goriva')
plt.grid(axis='y')
plt.show()

'''e) Pomo´ cu stupˇ castog grafa prikažite na istoj slici prosjeˇ cnu C02 emisiju vozila s obzirom na
 broj cilindara.'''
avg_emission_per_cylinder = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
plt.figure()
plt.bar(avg_emission_per_cylinder.index.astype(str), avg_emission_per_cylinder.values, color='blue')
plt.xlabel('Cylinders')
plt.ylabel('Average CO2 Emissions (g/km)')
plt.title('Prosječna emisija CO₂ prema broju cilindara')
plt.grid(axis='y')
plt.tight_layout()
plt.show()