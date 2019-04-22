from __future__ import print_function

import os.path

import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.utils import to_categorical
from PIL import Image
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime
import evaluate_image
import random
import inception_v4

batch_size = 64
nb_classes = 203094  # for class index shift from 0
nb_samples = 4132914 #// 1000  # use part of training data (4132 images) to test the model learning capability 
print(nb_samples)
nb_epoch = 650
nb_val_samples=500

img_rows, img_cols = 299, 299
img_channels = 3
img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)

weights_file = "checkpoints/lyz-with-val.h5"
log_dir = "tensorboard_dir"
image_path_prefix = "/mnt/lyz/Landmark-19"
csv_path = "/mnt/lyz/Landmark-19/train.csv"
model = ""

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def buildmodel():
    model = inception_v4.create_model(num_classes=nb_classes, include_top=True)
    print("Model created")

#    model.summary()
    optimizer = Adam(lr=0.0001) # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiled.")
    return model


def csv_data_generator(csv_path, image_path_prefix, batch_size): 
    with open(csv_path, 'r') as csv: 
        data = csv.readlines()

        x_imgs = []
        y_ids = []
        idx = 1
        while True:
            # data.shuffle()
            x_id, url, y_id = data[idx].strip().split(',')
            img_path = os.path.join(image_path_prefix, x_id[0], x_id[1], x_id[2], x_id + '.jpg')
            img = evaluate_image.get_processed_image(img_path)
            if img is not None:
                x_imgs.append(img)
                y_ids.append(y_id)

            if len(x_imgs) == batch_size:
                y_ids = to_categorical(y_ids, num_classes=nb_classes)
                yield (np.array(x_imgs), np.array(y_ids))
                x_imgs = []
                y_ids = []
            if idx == nb_samples - 1:
                idx = 0
            idx += 1


def load_eval_data(csv_path, image_path_prefix, nb_val_samples):
    with open(csv_path, 'r') as csv: 
        data = csv.readlines()
        x_imgs = []
        y_ids = []
        start = random.randint(1, nb_samples - nb_val_samples - 1)
        for idx in range(start, start + nb_val_samples):
            x_id, url, y_id = data[idx].strip().split(',')
            img_path = os.path.join(image_path_prefix, x_id[0], x_id[1], x_id[2], x_id + '.jpg')
            img = evaluate_image.get_processed_image(img_path)
            if img is not None:
                x_imgs.append(img)
                y_ids.append(y_id)

        y_ids = to_categorical(y_ids, num_classes=nb_classes)
        return (np.array(x_imgs), np.array(y_ids))


def train():
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        cooldown=0, patience=5, min_lr=2e-5)
    model_checkpoint= ModelCheckpoint(weights_file, monitor="val_loss", save_best_only=False,
                                      save_weights_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks=[lr_reducer, model_checkpoint, tensorboard]

    print("loading validation data...")
    val_data = load_eval_data(csv_path, image_path_prefix, nb_val_samples)

    print('loading pretrained weights...')
    model.load_weights(weights_file, by_name=True)
#    print('loading imageNet pretrained weights...')
#    model.load_weights('imagenet-weights/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

    print('training...')
    model.fit_generator(csv_data_generator(csv_path, image_path_prefix, batch_size),
                        steps_per_epoch= (nb_samples // 50) // batch_size, epochs=nb_epoch,  # about 8w samples making a checkpoint
                        callbacks=callbacks,
                        validation_data=val_data,
                        validation_steps=len(val_data) // batch_size, verbose=1,
                        initial_epoch=151)

    print('saving weights...')
    model.save_weights(weights_file, overwrite=True)

'''
def test(testX, testY, classnames):
    model = load_weights(model)
    print('testing...')
    predY = model.predict(testX)
    predY = np.argmax(predY, axis=1)
    testY = np.argmax(testY, axis=1)

    accuracy = metrics.accuracy_score(testY, predY) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)
    confusion_mat = confusion_matrix(testY, predY)
    print(class_names)
    print(confusion_mat)


def predict(model, testX, classnames):
    print('predicting...')
    testX = np.expand_dims(testX, axis=0) 
    predY = model.predict(testX)[0]
    print(predY)
    Y = np.argmax(predY, axis=0)
    print(str(Y) + '---' + classnames[Y] + ': ' + str(predY[Y]) )
    return Y, predY[Y], classnames[Y]


def load_weights(model):
    # Load model
    if os.path.exists(weights_file):
        model.load_weights(weights_file, by_name=True)
        print("Model loaded")
        return model
    else:
        print("Model " + weights_file + " not found, Exited!")
        return None



def plot_sonfusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    thresh = cm.max()/2.0
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment='center',color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
'''


if __name__ == '__main__':
    model = buildmodel()
    train()
#    test()

# ssh -L 6006:127.0.0.1:6006 root@192.168.8.2 -p2222
# for tensorboard


# predicts one-hot
# too many indices
