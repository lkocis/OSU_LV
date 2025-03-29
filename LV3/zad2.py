#Napišite programski kod koji ce prikazati sljede ´ ce vizualizacije:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv ('data_C02_emission.csv')
                      
'''a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz. ´'''
plt.figure ()
data['CO2 Emissions (g/km)'].plot( kind ='hist', bins = 20)
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Frequency')
plt.show()

'''b) Pomocu dijagrama raspršenja prikažite odnos izme ´ du gradske potrošnje goriva i emisije ¯
C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu¯
velicina, obojite to ˇ ckice na dijagramu raspršenja s obzirom na tip goriva. ˇ'''
plt.figure()
sns.scatterplot(x=data['Fuel Consumption City (L/100km)'], 
                y=data['CO2 Emissions (g/km)'], 
                hue=data['Fuel Type'], 
                palette='viridis', 
                s=100, 
                edgecolor='black')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend(title='Fuel Type')
plt.show()

'''c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip ´
goriva. Primjecujete li grubu mjernu pogrešku u podacima? ´'''
plt.figure()
data.boxplot(column =['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.show()

'''d) Pomocu stup ´ castog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
groupby.'''
grouped_by_fuel_type = data.groupby('Fuel Type').size()
plt.figure()
sns.barplot(x=grouped_by_fuel_type.index, y=grouped_by_fuel_type.values, palette='muted')
plt.xlabel('Fuel Type')
plt.ylabel('Vehicle Number')
plt.show()

'''e) Pomocu stup ´ castog grafa prikažite na istoj slici prosje ˇ cnu C02 emisiju vozila s obzirom na ˇ
broj cilindara.'''
grouped_by_cylinders = data.groupby('Cylinders').size()
plt.figure()
sns.barplot(x=grouped_by_cylinders.index, y=grouped_by_cylinders.values, palette='muted')
plt.xlabel('Number of cylinders')
plt.ylabel('Vehicle Number')
plt.show()