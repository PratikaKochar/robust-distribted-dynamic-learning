import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.initializers import glorot_uniform
import pandas as pd
import pickle
import sys
sys.path.append("C:/Users/PRATIKA/Downloads/dlplatform-dev/dlapplication-dev")
sys.path.append("C:/Users/PRATIKA/Downloads/dlplatform-dev/dlplatform-dev")

model_lo = pickle.load(open(r'C:\Users\PRATIKA\Downloads\dlplatform-dev\MNISTkerasCNN_2020-03-13_05-08-44\coordinator\currentAveragedState','rb'))

df = pd.read_csv(r'C:\Users\PRATIKA\Downloads\dlplatform-dev\dlapplication-dev\data\cnn_mnist\mnist_test.csv' )

labels = df.iloc[:,0]
test_data = df.iloc[:,1:]
print(test_data)
reshaped_array = test_data.values.reshape(9999, 28, 28,1)
print(reshaped_array.shape)
print(model_lo)
weights = np.asarray(model_lo.get())
print(type(weights))

import tensorflow as tf




numClasses = 10
imgRows = 28
imgCols = 28
inputShape = (imgRows, imgCols, 1)
np.random.seed(42)
tf.set_random_seed(42)
static_initializer = glorot_uniform(seed=42)

inp = Input(shape=inputShape)
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu',
            kernel_initializer=static_initializer)(inp)
conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=static_initializer)(conv1)
pool = MaxPooling2D(pool_size=(2, 2))(conv2)
dp1 = Dropout(0.25, seed=42)(pool)
fl = Flatten()(dp1)
ds = Dense(128, activation='relu', kernel_initializer=static_initializer)(fl)
dp2 = Dropout(0.5, seed=42)(ds)
outp = Dense(numClasses, activation='softmax', kernel_initializer=static_initializer)(dp2)
network = Model(inputs=inp, outputs=outp)

print(network)
network.set_weights(weights)


predictions = network.predict(reshaped_array)
predictions = np.argmax(predictions, axis=1)
print(predictions.shape)
comp_df = pd.DataFrame()
comp_df['true'] = labels
print(comp_df)
comp_df['pred'] = predictions
correct = (comp_df['true'] == comp_df['pred']).sum()
print(correct)
print('accuracy = ', correct/ comp_df.shape[0] * 100)


