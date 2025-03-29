'''Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih veli ˇ cina. Detalje oko ovog podatkovnog skupa mogu se prona ˇ ci u 3. ´
laboratorijskoj vježbi.'''
'''a) Odaberite željene numericke veli ˇ cine speci ˇ ficiranjem liste s nazivima stupaca. Podijelite
podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%.'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

data = pd.read_csv('data_C02_emission.csv')

column_names = ['Make','Model','Vehicle Class','Engine Size (L)','Cylinders','Transmission','Fuel Type','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
x = data[column_names]
y = data['CO2 Emissions (g/km)']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

'''b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
o jednoj numerickoj veli ˇ cini. Pri tome podatke koji pripadaju skupu za u ˇ cenje ozna ˇ cite ˇ
plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.'''
plt.scatter(X_train['Fuel Consumption City (L/100km)'], y_train, color='blue', label='Trening')
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, color = 'red', label='Testni')
plt.figlegend()
plt.show()

'''c) Izvršite standardizaciju ulaznih velicina skupa za u ˇ cenje. Prikažite histogram vrijednosti ˇ
jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja ˇ
transformirajte ulazne velicine skupa podataka za testiranje. '''
#sc = StandardScaler()
sc = MinMaxScaler()
X_train_fuel = X_train[['Fuel Consumption City (L/100km)']]
X_train_sc = sc.fit_transform (X_train_fuel)
plt.hist(X_train_fuel['Fuel Consumption City (L/100km)'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram prije skaliranja ul. vel. - ucenje')
plt.show()

plt.hist(X_train_sc[:, 0], bins=20, color='red', alpha=0.7)
plt.title('Histogram nakon skaliranja ul.vel. - ucenje')
plt.show()

X_test_fuel = X_test[['Fuel Consumption City (L/100km)']]
X_test_sc = sc.transform(X_test_fuel)
plt.hist(X_test_sc[:, 0], bins=20, color='green', alpha=0.7)
plt.title('Histogram nakon transformiranja skalirane ul. vel. - testiranje')
plt.show()

'''d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
povežite ih s izrazom 4.6.'''
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X_test_fuel)

linearModel = lm.LinearRegression ()
linearModel.fit(X_train_sc , y_train )

print(f'Parameters: intercept-> {linearModel.intercept_}, slope-> {linearModel.coef_}')

'''e) Izvršite procjenu izlazne velicine na temelju ulaznih veli ˇ cina skupa za testiranje. Prikažite ˇ
pomocu dijagrama raspršenja odnos izme ´ du stvarnih vrijednosti izlazne veli ¯ cine i procjene ˇ
dobivene modelom.'''
y_test_p = linearModel.predict(X_test_sc)
plt.scatter(y_test, y_test_p, color='blue', label='Stvarne vs Predviđene vrijednosti')
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predviđene vrijednosti")
plt.figlegend()
plt.show()

'''f) Izvršite vrednovanje modela na nacin da izra ˇ cunate vrijednosti regresijskih metrika na ˇ
skupu podataka za testiranje.'''
MAE = mean_absolute_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
RMSE = root_mean_squared_error(y_test , y_test_p) 
MAPE = mean_absolute_percentage_error(y_test , y_test_p)

print(f'Mean absolute error: {MAE}')
print(f'Mean squared error: {MSE}')
print(f'Root mean squared error: {RMSE}')
print(f'Mean absolute percentage error: {MAPE}')

