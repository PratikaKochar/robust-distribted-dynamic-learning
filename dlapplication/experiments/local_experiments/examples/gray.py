import _pickle as pickle
from keras.utils import np_utils
import numpy as np
import pandas as pd

img_size = 32
import os
home = os.path.expanduser('~')
data_path = os.path.join(home, "Downloads/cifar-100-python/cifar-100-python")
img_channels = 3
nb_classes = 100
# length of the image after we flatten the image into a 1-D array
img_size_flat = img_size * img_size * img_channels
nb_files_train = 1
images_per_file = 50000
# number of all the images in the training dataset
nb_images_train = nb_files_train * images_per_file

def load_data(file_name):
    file_path = os.path.join(data_path, "cifar-100/", file_name)

    print('Loading ' + file_name)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    print(data)
    for key in data:
        print(key)
    raw_images = data[b'data']
    cls = np.array(data[b'fine_labels'])

    images = raw_images.reshape([-1, img_channels, img_size, img_size])
    # move the channel dimension to the last
    images = np.rollaxis(images, 1, 4)

    return images, cls


def load_training_data():
    # pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[nb_images_train, img_size, img_size, img_channels],
                      dtype=int)
    cls = np.zeros(shape=[nb_images_train], dtype=int)

    begin = 0
    for i in range(nb_files_train):
        images_batch, cls_batch = load_data(file_name="train")
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end

    #return images, np_utils.to_categorical(cls, nb_classes)
    return images, cls


def load_test_data():
    images, cls = load_data(file_name="test")

    return images, cls


def load_cifar():
    X_train, Y_train = load_training_data()
    X_test, Y_test = load_test_data()

    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_cifar()

print('\nX_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst

X_train_gray = grayscale(X_train)
X_test_gray = grayscale(X_test)

print('\nX_train shape:', X_train_gray.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test_gray.shape)
print('Y_test shape:', Y_test.shape)


df= pd.DataFrame(data=Y_train)
df.to_csv('labels.csv', index=False, header= False)
img = []
images=[]
for i in range(50000):
    images.append(X_train_gray[i].flatten())


print(len(images))
df = pd.DataFrame(data=images)
df.to_csv('data.csv', index=False, header= False)
df_SN7577i_c = pd.read_csv("labels.csv")
df_SN7577i_d = pd.read_csv("data.csv")
df_all_cols = pd.concat([df_SN7577i_c, df_SN7577i_d], axis=1)
df_all_cols.to_csv('cifar_trainfinal.csv', index=False)
print(df_all_cols.shape)

#df1 = pd.read_csv('cifar_train.csv')
#df2 = pd.read_csv('cifar_train2.csv')
#print(df1.shape)
#print(df2.shape)
#df = pd.concat([df1, df2])
#print(df.shape)