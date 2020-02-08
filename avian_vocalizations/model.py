from collections import namedtuple
from keras.models import Model, Sequential
from keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D, \
                         Dropout, Dense, Input, Concatenate, Flatten

# ModelParams = namedtuple("ModelParams",['n_frames', 'dropout_rate'])

def ModelFactory(n_classes):
    model = Sequential()
    model.add(Conv2D(64,3,input_shape=(128,128,1),padding='valid',activation="relu"))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(rate=.2))
    model.add(Conv2D(64,3,padding='valid',activation="relu"))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(rate=.2))
    model.add(Conv2D(64,3,padding='valid',activation="relu"))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(rate=.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation="softmax"))
    return model