from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

species_colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("src/penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

'''Skripta zadatak_2.py uˇcitava podatkovni skup Palmer Penguins [1]. Ovaj
 podatkovni skup sadrži mjerenja provedena na tri razliˇcite vrste pingvina (’Adelie’, ’Chins
trap’, ’Gentoo’) na tri razliˇcita otoka u podruˇcju Palmer Station, Antarktika. Vrsta pingvina
 odabrana je kao izlazna veliˇcina i pri tome su klase oznaˇcene s cjelobrojnim vrijednostima
 0, 1 i 2. Ulazne veliˇcine su duljina kljuna (’bill_length_mm’) i duljina peraje u mm (’flip
per_length_mm’). Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna
 funkcija plot_decision_region.'''

'''a) Pomo´cu stupˇcastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu
 pingvina) u skupu podataka za uˇcenje i skupu podataka za testiranje. Koristite numpy
 funkciju unique.'''

y_test_unique, count_train = np.unique(y_train, return_counts=True)
y_train_unique, count_test = np.unique(y_test, return_counts=True)

plt.figure()
plt.bar(y_train_unique, count_train, color=species_colors[:len(y_train_unique)])
plt.title("Broj pingvina - ucenje")
plt.xlabel("Vrste pingvnina")
plt.ylabel("Broj pingvina")
plt.xticks(y_train_unique, [labels[i] for i in y_train_unique])
plt.show()

plt.figure()
plt.bar(y_test_unique, count_test, color=species_colors[:len(y_test_unique)])
plt.title("Broj pingvina - testiranje")
plt.xlabel("Vrste pingvnina")
plt.ylabel("Broj pingvina")
plt.xticks(y_test_unique, [labels[i] for i in y_test_unique])
plt.show()

'''b) Izgradite model logistiˇ cke regresije pomo´ cu scikit-learn biblioteke na temelju skupa poda
taka za uˇ cenje.'''

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

y_test_p = LogRegression_model.predict(X_test)

'''c) Prona¯ dite u atributima izgra¯ denog modela parametre modela. Koja je razlika u odnosu na
 binarni klasifikacijski problem iz prvog zadatka?'''

theta0 = LogRegression_model.intercept_[0]
theta1 = LogRegression_model.coef_[0][0]
theta2 = LogRegression_model.coef_[0][1]
print(f'Theta 0: {theta0}\nTheta 1: {theta1}\nTheta2: {theta2}')

''' d) Pozovite funkciju plot_decision_region pri ˇcemu joj predajte podatke za uˇcenje i
 izgra ¯ deni model logistiˇ cke regresije. Kako komentirate dobivene rezultate?

plot_decision_regions(X_train[:, 0], y_train[:, 0], classifier=LogRegression_model)
plt.show()'''

''' e) Provedite klasifikaciju skupa podataka za testiranje pomo´ cu izgra¯ denog modela logistiˇ cke
 regresije. Izraˇ cunajte i prikažite matricu zabune na testnim podacima. Izraˇ cunajte toˇ cnost.
 Pomo´cu classification_report funkcije izraˇcunajte vrijednost ˇcetiri glavne metrikena skupu podataka za testiranje.
'''

cm = confusion_matrix(y_test, y_test_p)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print(f"Točnost modela: {accuracy_score(y_test, y_test_p)}")

print(classification_report(y_test, y_test_p))

''' f) Dodajte u model još ulaznih veliˇcina. Što se doga¯ da s rezultatima klasifikacije na skupu
 podataka za testiranje?'''

new_input_variables = ['bill_length_mm',
                        'flipper_length_mm',
                        'bill_depth_mm',
                        'body_mass_g']

X_new = df[new_input_variables].to_numpy()

X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=123)

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train_new, y_train)

cm = confusion_matrix(y_test, y_test_p)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print(f"Točnost modela: {accuracy_score(y_test, y_test_p)}")

print(classification_report(y_test, y_test_p))