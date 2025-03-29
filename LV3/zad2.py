#Napišite programski kod koji ce prikazati sljede ´ ce vizualizacije:
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv ('data_C02_emission.csv')
                      
'''a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz. ´'''
plt.figure ()
data ['CO2 Emissions (g/km)'].plot( kind ='hist', bins = 20)
#plt.show()

'''b) Pomocu dijagrama raspršenja prikažite odnos izme ´ du gradske potrošnje goriva i emisije ¯
C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu¯
velicina, obojite to ˇ ckice na dijagramu raspršenja s obzirom na tip goriva. ˇ'''
colors = data['Fuel Type'].map({'X': 'red', 'D': 'green', 'Z': 'blue', 'E': 'purple', 'N': 'pink'})
data.plot.scatter(x='Fuel Consumption City (L/100km)',
                        y='CO2 Emissions (g/km)',
                        c=colors,  
                        s=100,  
                        edgecolor='black')
plt.show()

'''c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip ´
goriva. Primjecujete li grubu mjernu pogrešku u podacima? ´'''

'''d) Pomocu stup ´ castog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
groupby.'''

'''e) Pomocu stup ´ castog grafa prikažite na istoj slici prosje ˇ cnu C02 emisiju vozila s obzirom na ˇ
broj cilindara.'''