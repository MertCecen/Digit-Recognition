import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D
from tensorflow.keras.utils import to_categorical


mnist = tf.keras.datasets.mnist # dataset
number_of_classes = 10  # ten digits
input_shape = (28, 28, 1)

(X_train, y_train), (X_test, y_test) = mnist.load_data() # returns tuple of numpy arrays


#X_train = tf.keras.utils.normalize(X_train, axis=1)
#X_test = tf.keras.utils.normalize(X_test, axis=1)

# Reshape
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) # 1 for grayscale image
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) 

# Converting the labels into one hot encoding therefore use loss = categorical_crossentropy
y_train = to_categorical(y_train, number_of_classes)
y_test = to_categorical(y_test, number_of_classes)

# Normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'x train samples')
print(X_test.shape[0], 'x test samples')
print(y_train.shape, 'y train samples')
print(y_test.shape, 'y test samples')

# Shape control
assert X_train.shape == (60000, 28, 28, 1)
assert X_test.shape == (10000, 28, 28, 1)
assert y_train.shape == (60000, 10)
assert y_test.shape == (10000, 10)

# create model

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = "relu", input_shape = input_shape))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Conv2D(64, 3, activation = "relu"))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.25))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(rate = 0.5))
model.add(Dense(number_of_classes , activation = "softmax"))

"""
model = tensorflow.keras.Sequential(
    [
        tensorflow.keras.Input(shape=input_shape),
        tensorflow.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tensorflow.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dropout(0.5),
        tensorflow.keras.layers.Dense(number_of_classes, activation="softmax"),
    ]
)
"""
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()
history = model.fit(X_train, y_train, batch_size = 128, epochs = 15, validation_split = 0.2)
print("Successfully trained the model")
model.save("mnist.h5")
print("Saved the model as mnist.h5")
score = model.evaluate(X_test, y_test)
print("Test Accuracy: ", score[1], "- Test Loss: ", score[0])














