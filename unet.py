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

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # second layer
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x

def create_model():
    input_tensor = keras.layers.Input((target_size[0], target_size[1], 3))
    n_filters = 16
    dropout = 0.05
    batchnorm = True

    c1 = conv2d_block(input_tensor, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = keras.layers.Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = keras.layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = keras.layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = keras.layers.Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    u6 = keras.layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    u7 = keras.layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    u8 = keras.layers.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1], axis=3)
    u9 = keras.layers.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(c9)
    x = keras.layers.Dense(3, activation='softmax', name='fc1000')(x)

    m = keras.models.Model(input_tensor, x, name='customUNet')
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.binary_crossentropy, metrics=["accuracy"])
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

    logs = model.fit(x_train, y_train, epochs=150,
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

    model.save("unet.keras")

