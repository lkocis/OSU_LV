import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

''' Skripta zadatak_1.py uˇcitava Social_Network_Ads.csv skup podataka [2].
 Ovaj skup sadrži podatke o korisnicima koji jesu ili nisu napravili kupovinu za prikazani oglas.
 Podaci o korisnicima su spol, dob i procijenjena pla´ca. Razmatra se binarni klasifikacijski
 problem gdje su dob i procijenjena pla´ca ulazne veliˇcine, dok je kupovina (0 ili 1) izlazna
 veliˇcina. Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija
 plot_decision_region [1]. Podaci su podijeljeni na skup za uˇ cenje i skup za testiranje modela
 u omjeru 80%-20% te su standardizirani. Izgra¯ den je model logistiˇ cke regresije te je izraˇ cunata
 njegova toˇ cnost na skupu podataka za uˇ cenje i skupu podataka za testiranje. Potrebno je:'''

''' 1. Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Izraˇcunajte toˇcnost
 klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje. Usporedite
 dobivene rezultate s rezultatima logistiˇ cke regresije. Što primje´ cujete vezano uz dobivenu
 granicu odluke KNN modela?'''

# inicijalizacija i ucenje KNN modela
KNN_model = KNeighborsClassifier(5)
KNN_model.fit(X_train_n, y_train)

# predikcija na skupu podataka za testiranje i ucenje
y_test_p_KNN = KNN_model.predict(X_test_n)
y_train_p_KNN = KNN_model.predict(X_train_n)

print("Tocnost klasifikacije: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

# granica odluke KNN modela - overfitting
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

''' 2. Kako izgleda granica odluke kada je K =1 i kada je K = 100?'''
# inicijalizacija i ucenje KNN modela
KNN_model = KNeighborsClassifier(1)
KNN_model.fit(X_train_n, y_train)

# predikcija na skupu podataka za testiranje i ucenje
y_test_p_KNN = KNN_model.predict(X_test_n)
y_train_p_KNN = KNN_model.predict(X_train_n)

print("Tocnost klasifikacije K=1: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

# granica odluke KNN modela - overfitting
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()


# inicijalizacija i ucenje KNN modela
KNN_model = KNeighborsClassifier(100)
KNN_model.fit(X_train_n, y_train)

# predikcija na skupu podataka za testiranje i ucenje
y_test_p_KNN = KNN_model.predict(X_test_n)
y_train_p_KNN = KNN_model.predict(X_train_n)

print("Tocnost klasifikacije K=100: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

# granica odluke KNN modela - overfitting
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

'''Pomo´cu unakrsne validacije odredite optimalnu vrijednost hiperparametra K
 algoritma KNN za podatke iz Zadatka 1.'''

param_grid = {'n_neighbors': np.arange(1, 101)}

grid_search = GridSearchCV(estimator=KNN_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_n, y_train)

print(f"Optimalni K: {grid_search.best_params_['n_neighbors']}")

''' Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
 te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
 ovih hiperparametara utjeˇce na granicu odluke te pogrešku na skupu podataka za testiranje?
 Mijenjajte tip kernela koji se koristi. Što primje´ cujete?'''

pipe = pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf'))

param_grid = {'svc__C': [10, 100, 1000], 'svc__gamma': [10, 1, 0.1, 0.01]}

svm_gscv = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_gscv.fit(X_train, y_train)

print(svm_gscv.best_params_)

''' Pomo´ cu unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ
 algoritma SVM za problem iz Zadatka 1.'''

SVM_model = svm.SVC(kernel='rbf', gamma = 1, C=0.1)
SVM_model.fit(X_train_n, y_train)

y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train_n, y_train)

print(f"Optimalni hiperparametri (C, gamma): {grid_search.best_params_}")
print(f"Najbolja točnost na skupu za ucenje: {grid_search.best_score_:.3f}")

optimal_SVM_model = grid_search.best_estimator_
y_train_optimal = optimal_SVM_model.predict(X_train_n)
y_test_optimal = optimal_SVM_model.predict(X_test_n)

print("Optimalni SVM model (RBF Kernel): ")
print("Tocnost train: " + "{:0.4f}".format(accuracy_score(y_train, y_train_optimal)))
print("Tocnost test: " + "{:0.4f}".format(accuracy_score(y_test, y_test_optimal)))
