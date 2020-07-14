import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import listdir
import numpy as np
import tensorflow.keras as keras
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

target_size = (64, 64)


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

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):

    filters1, filters2, filters3 = filters
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = keras.layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def create_model():
    input_tensor = keras.layers.Input((target_size[0], target_size[1], 3))

    x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
    x = keras.layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = keras.layers.BatchNormalization(axis=1, name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = keras.layers.Dense(3, activation='softmax', name='fc1000')(x)

    m = keras.models.Model(input_tensor, x, name='customResnet')
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.categorical_crossentropy,
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

    model.save("rnn.keras")

