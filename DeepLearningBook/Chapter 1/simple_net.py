import tensorflow as tf
from tensorflow import keras

# Network and training

epochs = 200
batch_size = 128
verbose = 1
nb_classes = 10 # number of outputs = number of digits
n_hidden = 128
validation_split = 0.2 # how much train is reserved for validation

mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


#print(len(X_train[0][0][0]))



X_train = X_train.reshape((60000, 784))
X_train = X_train.astype('float32')

X_test = X_test.reshape((10000, 784))
X_test = X_test.astype('float32')



X_train, X_test = X_train / 255.0, X_test / 255.0

Y_train = keras.utils.to_categorical(Y_train, nb_classes)
Y_test = keras.utils.to_categorical(Y_test, nb_classes)

model = keras.models.Sequential()

model.add(keras.layers.Dense(128, input_shape=(784,), name="Dense1", activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, name="Dense2", activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10, name="Dense3", activation='softmax'))

model.summary()

model.compile(optimizer = 'SGD', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split)

test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"test accuracy: {test_accuracy}, test loss: {test_loss}")
