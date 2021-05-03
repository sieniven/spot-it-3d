import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers


# function will get data from csv, and store in train_data /test_data, train_label/test_label
def get_data(filename):
    df = pd.read_csv(filename, sep=',')
    df.head()
    train_data = df.iloc[:, 3].values
    train_labels = df.iloc[:, 4].values

    num_validation_samples = 100
    test_samples = 100

    val_data = np.array(train_data[:num_validation_samples])
    test_data = np.array(train_data[num_validation_samples:(num_validation_samples + test_samples)])
    train_data = np.array(train_data[(num_validation_samples + test_samples):])

    val_labels = np.array(train_labels[:num_validation_samples])
    test_labels = np.array(train_labels[num_validation_samples:(num_validation_samples + test_samples)])
    train_labels = np.array(train_labels[(num_validation_samples + test_samples):])

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# create binary classification model
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(2, 150)))    # 5 seconds of feature signal (30 FPS)
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, avtivation='sigmoid'))    # sigmoid function used for binary classification

    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])    # configure hyperparameters

    return model


# predict model results
def predict(model, input, correct_label):
    prediction = model.predict(input)
    print("The predicted value is %d and the actual value is %d." % (prediction, correct_label))


def get_losses(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    return loss_values, val_loss_values


# plot graph of Training/Validation loss vs Epochs
def get_loss_plot(loss_values, val_loss_values, epoch):
    plt.plot(epoch, loss_values, 'bo', label='Training loss')   # 'bo' stands for blue dot
    plt.plot(epoch, val_loss_values, 'b', label='Validation loss')   # 'b' stands for solid blue line
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def get_acc(history):
    history_dict = history.history
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    return acc_values, val_acc_values


# plot graph of Training/Validation accuracy vs Epochs
def get_acc_plot(acc_values, val_acc_values, epoch):
    plt.plot(epoch, acc_values, 'bo', label='Training acc')  # 'bo' stands for blue dot
    plt.plot(epoch, val_acc_values, 'b', 'Validation acc')  # 'b' stands for solid blue line
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def train_model():
    filepath = "~.csv"  # input filepath
    data = pd.read_csv(filepath)  # to be edited
    train_data, train_labels, val_data, val_labels, test_data, test_labels = get_data(data)

    model = build_model()
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=1)

    history = model.fit(train_data, train_labels, epochs=10, callbacks=cp_callback, batch_size=200,
                        validation_data=(val_data, val_labels))  # subjected to change regarding epochs and batch size
    loss_values, val_loss_values = get_losses(history)
    acc_values, val_acc_values = get_acc(history)

    model.save_weights('./checkpoints/my_checkpoint')
    model.save('re-identification_model.h5')

    epoch = range(1, 11)  # add 1 to epochs number
    plt.figure()
    get_loss_plot(loss_values, val_loss_values, epoch)
    plt.figure()
    get_acc_plot(acc_values, val_acc_values, epoch)


if __name__ == '__main__':
    train_model()
