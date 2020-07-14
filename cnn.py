import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import listdir
import numpy as np
import tensorflow.keras as keras
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

target_size = (128, 128)


def load_dataset() -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    train_path = "Dataset/Train/"
    test_path = "Dataset/Test/"

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    load_set_from_directory(train_path, x_train_list, y_train_list)
    load_set_from_directory(test_path, x_test_list, y_test_list)

    return (np.array(x_train_list), np.array(y_train_list)), (np.array(x_test_list), np.array(y_test_list))


def load_set_from_directory(train_path, x_train_list, y_train_list):
    load_images_from_directory(np.array([1, 0, 0]), f'{train_path}ClassD/', x_train_list, y_train_list)
    load_images_from_directory(np.array([0, 1, 0]), f'{train_path}ClassE/', x_train_list, y_train_list)
    load_images_from_directory(np.array([0, 0, 1]), f'{train_path}ClassF/', x_train_list, y_train_list)


def load_images_from_directory(label, path, x_train_list, y_train_list):
    for img_name in listdir(path):
        # x_train_list.append(np.array(Image.open(f'{path}{img_name}').convert('L').resize(target_size)) / 255.0) # Grayscale
        x_train_list.append(
            np.array(Image.open(f'{path}{img_name}').convert('RGB').resize(target_size)) / 255.0)  # color
        y_train_list.append(label)


def create_model():
    m = keras.models.Sequential()

    m.add(keras.layers.Conv2D(8, kernel_size=(3, 3), activation=keras.activations.relu, padding='same'))
    m.add(keras.layers.MaxPool2D((2, 2)))

    m.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation=keras.activations.relu, padding='same'))
    m.add(keras.layers.MaxPool2D((2, 2)))

    m.add(keras.layers.Dropout(0.2))

    m.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation=keras.activations.relu, padding='same'))
    m.add(keras.layers.MaxPool2D((2, 2)))

    m.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation=keras.activations.relu, padding='same'))
    m.add(keras.layers.MaxPool2D((2, 2)))

    m.add(keras.layers.Dropout(0.2))

    m.add(keras.layers.Flatten())

    m.add(keras.layers.Dense(64, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(3, activation=keras.activations.sigmoid))
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m

def show_confusion_matrix(m, x, y, show_errors: bool = False):
    predicted_values = m.predict(x)
    predicted_labels = np.argmax(predicted_values, axis=1)
    true_labels = np.argmax(y, axis=1)

    print(confusion_matrix(true_labels, predicted_labels))

    if show_errors:
        for i in range(len(predicted_labels)):
            if predicted_labels[i] != true_labels[i]:
                plt.imshow(x[i])
                plt.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model = create_model()

    show_confusion_matrix(model, x_test, y_test)

    logs = model.fit(x_train, y_train, epochs=50,
                     batch_size=4,
                     validation_data=(x_test, y_test), verbose=0)

    # Affichage ddes courbes de loss et d'accuracy de l'apprentissage
    plt.plot(logs.history['loss'])
    plt.plot(logs.history['val_loss'])
    plt.show()

    # Affichage ddes courbes de loss et d'accuracy de l'apprentissage
    plt.plot(logs.history['accuracy'])
    plt.plot(logs.history['val_accuracy'])
    plt.show()

    show_confusion_matrix(model, x_test, y_test, show_errors=True)

    model.save("cnn.keras")

