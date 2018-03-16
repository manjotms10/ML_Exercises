import glob
import numpy as np
from sklearn.model_selection import train_test_split

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pylab as plt
from matplotlib.pyplot import imshow

def load_dataset():
    X_data = []
    Y_data = []
    classes = {'Faces':1, 'Panda':2, 'Rooster':3}
    files = glob.glob ("data/Faces/*")
    for myFile in files:
        print(myFile)
        image = plt.imread (myFile)
        image = plt.resize(image, (32,32,3))
        X_data.append(image)
        Y_data.append(1)

    files = glob.glob ("data/panda/*")
    for myFile in files:
        print(myFile)
        image = plt.imread (myFile)
        image = plt.resize(image, (32,32,3))
        X_data.append(image)
        Y_data.append(2)

    files = glob.glob ("data/rooster/*")
    for myFile in files:
        print(myFile)
        image = plt.imread (myFile)
        image = plt.resize(image, (32,32,3))
        X_data.append(image)
        Y_data.append(3)

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    Y_data = Y_data.reshape((Y_data.shape[0],1))

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    return X_train, X_test, Y_train, Y_test


def createModel(input_shape):
    #input layer
    X_input = Input(input_shape)

    #padding
    X = ZeroPadding2D((3,3))(X_input)

    #layers
    X = Conv2D(16, (3,3), strides = (1,1), name='conv_1')(X_input)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name='max_pool_1')(X)

    X = Conv2D(128, (3,3), strides = (1,1), name='conv_2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name='max_pool_2')(X)

    #Flatten and FC layer
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='CNN')
    return model
    
def main():
    X_train, X_test, Y_train, Y_test = load_dataset()
    
    model = createModel(X_train.shape[1:])
    model.compile(optimizer='Adam', loss='mean_squared_logarithmic_error', metrics = ["accuracy"])
    model.fit( x = X_train, y = Y_train, epochs = 20, batch_size = 20)
    predictions = model.evaluate( x = X_test, y = Y_test)
    print()
    print("Loss: ", predictions[0])
    print("Accuracy: ", predictions[1])
    model.summary()
    imshow(X_test[1])

if __name__=='__main__':
    main()
