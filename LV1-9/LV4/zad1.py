''' Skripta zadatak_1.py uˇcitava podatkovni skup iz data_C02_emission.csv.
 Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju os
talih numeriˇckih ulaznih veliˇcina. Detalje oko ovog podatkovnog skupa mogu se prona´ci u 3.
 laboratorijskoj vježbi.'''

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

data = pd.read_csv('src/data_C02_emission.csv')

''' a) Odaberite željene numeriˇ cke veliˇ cine specificiranjem liste s nazivima stupaca. Podijelite
 podatke na skup za uˇ cenje i skup za testiranje u omjeru 80%-20%.'''
column_names = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
X = data[column_names]
y = data['CO2 Emissions (g/km)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

'''b) Pomo´ cu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova
 o jednoj numeriˇckoj veliˇcini. Pri tome podatke koji pripadaju skupu za uˇcenje oznaˇcite
 plavom bojom, a podatke koji pripadaju skupu za testiranje oznaˇ cite crvenom bojom.'''
plt.figure()
plt.scatter(X_train['Fuel Consumption City (L/100km)'], y_train, color='blue', label='Trening skup', edgecolors='black')
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, color='red', label='Test skup', edgecolors='black')
plt.xlabel('Gradska potrošnja goriva (L/100km)')
plt.ylabel('CO₂ emisije (g/km)')
plt.title('Ovisnost CO₂ emisije o gradskoj potrošnji goriva')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

''' c) Izvršite standardizaciju ulaznih veliˇ cina skupa za uˇ cenje. Prikažite histogram vrijednosti
 jedne ulazne veliˇ cine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja
 transformirajte ulazne veliˇ cine skupa podataka za testiranje.'''
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

X_train_n_df = pd.DataFrame(X_train_n, columns=X_train.columns)
X_test_n_df = pd.DataFrame(X_test_n, columns=X_test.columns)

plt.hist(X_train['Fuel Consumption City (L/100km)'], color='blue')
plt.grid(True)
plt.title('Train skup prije skaliranja: Fuel Consumption City (L/100km)')
plt.show()

plt.hist(X_train_n_df['Fuel Consumption City (L/100km)'], color='blue')
plt.grid(True)
plt.title('Train skup nakon skaliranja: Fuel Consumption City (L/100km)')
plt.show()

plt.hist(X_test['Fuel Consumption City (L/100km)'], color='blue')
plt.grid(True)
plt.title('Test skup prije skaliranja: Fuel Consumption City (L/100km)')
plt.show()

plt.hist(X_test_n_df['Fuel Consumption City (L/100km)'], color='blue')
plt.grid(True)
plt.title('Test skup nakon skaliranja: Fuel Consumption City (L/100km)')
plt.show()

'''d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
 povežite ih s izrazom 4.6.'''
model = lm.LinearRegression()
model.fit(X_train_n, y_train)

print(f'Koeficijent/i: {model.coef_}')
print(f'Intercept: {model.intercept_}')

''' e) Izvršite procjenu izlazne veliˇ cine na temelju ulaznih veliˇ cina skupa za testiranje. Prikažite
 pomo´ cu dijagrama raspršenja odnos izme¯ du stvarnih vrijednosti izlazne veliˇ cine i procjene
 dobivene modelom.'''
y_test_p = model.predict(X_test_n)
plt.figure()
plt.scatter(y_test, y_test_p, color='purple', edgecolors='black')
plt.xlabel('Stvarna vrijednost CO₂ emisije (g/km)')
plt.ylabel('Predviđena vrijednost CO₂ emisije (g/km)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='gray', linestyle='--', label='Savršeno predviđanje (y=x)')
plt.title('Stvarne vs. Predviđene vrijednosti CO₂ emisije')
plt.grid(True)
plt.show()

''' f) Izvršite vrednovanje modela na naˇcin da izraˇcunate vrijednosti regresijskih metrika na
 skupu podataka za testiranje.'''
MAE = mean_absolute_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
RMSE = root_mean_squared_error(y_test , y_test_p) 
MAPE = mean_absolute_percentage_error(y_test , y_test_p)
r2 = r2_score(y_test, y_test_p)
print(f'Koeficijent determinacije R²: {r2}')

print(f'Mean absolute error: {MAE}')
print(f'Mean squared error: {MSE}')
print(f'Root mean squared error: {RMSE}')
print(f'Mean absolute percentage error: {MAPE}')
print(f'Koeficijent determinacije R²: {r2}')

'''g) Što se doga¯ da s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj
 ulaznih veliˇ cina?'''
