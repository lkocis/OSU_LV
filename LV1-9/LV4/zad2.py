'''Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoriˇ cku
 varijable „Fuel Type“ kao ulaznu veliˇcinu. Pri tome koristite 1-od-K kodiranje kategoriˇckih
 veliˇ cina. Radi jednostavnosti nemojte skalirati ulazne veliˇ cine. Komentirajte dobivene rezultate.
 Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
 vozila radi?'''

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('src/data_C02_emission.csv')

numeric_features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)',
                    'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)',
                    'Fuel Consumption Comb (mpg)']
categorical_feature = ['Fuel Type']

X_num = data[numeric_features]
X_cat = data[categorical_feature]
y = data['CO2 Emissions (g/km)']

encoder = OneHotEncoder(drop='first', sparse_output=False)  
X_cat_encoded = encoder.fit_transform(X_cat)

X_full = pd.concat([X_num, pd.DataFrame(X_cat_encoded, columns=encoder.get_feature_names_out(categorical_feature))], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=1)

model = lm.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = MSE ** 0.5
R2 = r2_score(y_test, y_pred)

print(f"MAE: {MAE:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"RMSE: {RMSE:.2f}")
print(f"R²: {R2:.4f}")

errors = abs(y_test - y_pred)
max_error_index = errors.idxmax()
max_error_value = errors[max_error_index]
worst_model = data.loc[max_error_index]

print(f"\nMaksimalna pogreška u predikciji CO₂ emisije: {max_error_value} g/km")
print("Podaci modela vozila s najvećom pogreškom:")
print(worst_model[['Make', 'Model', 'Fuel Type', 'CO2 Emissions (g/km)']])
