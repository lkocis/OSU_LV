import numpy as np
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import load_model


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
img = x_train[1]
plt.subplot(2, 2, 1)
plt.imshow(img)

img = x_train[2]
plt.subplot(2, 2, 2)
plt.imshow(img)

img = x_train[3]
plt.subplot(2, 2, 3)
plt.imshow(img)

img = x_train[4]
plt.subplot(2, 2, 4)
plt.imshow(img)
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape = input_shape))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = "relu"))
model.add(layers.Dense(num_classes, activation = "softmax"))
model.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy",])

# TODO: provedi ucenje mreze
batch_size = 32
epochs = 1
history = model.fit(x_train_s, y_train_s, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
predictions = model.predict(x_test_s)
score = model.evaluate(x_test_s, y_test_s, verbose = 0)

# TODO: Prikazi test accuracy i matricu zabune
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')

y_pred_classes = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test_s, axis=1)
confusion_matrix = metrics.confusion_matrix(y_true, y_pred_classes)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot()
plt.show()

# TODO: spremi model
keras.model.save ("FCN/")
del model
model = load_model('FCN/')
model.summary()
