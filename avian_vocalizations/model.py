from collections import namedtuple
from keras.models import Model, Sequential
from keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D, \
                         Dropout, Dense, Input, Concatenate, Flatten

# ModelParams = namedtuple("ModelParams",['n_frames', 'dropout_rate'])

def ModelFactory(n_classes, n_frames=128, dropout_rate=.2):
#     melsg_input = Input(shape=(128,n_frames,1),name='melsg')
#     melsg_pathway = Conv2D(16,3,name='melsg_conv2d_1',
#                          padding='same',activation="relu")(melsg_input)
#     melsg_pathway = MaxPooling2D(pool_size=3,name='melsg_maxpooling_1')(melsg_pathway)
#     melsg_pathway = Dropout(rate=dropout_rate,name='melsg_dropout_1')(melsg_pathway)
#     melsg_pathway = Conv2D(32,3,padding='same',activation="relu", 
#                          name='melsg_conv2d_2')(melsg_pathway)
#     melsg_pathway = MaxPooling2D(pool_size=3,name='melsg_maxpooling_2')(melsg_pathway)
#     melsg_pathway = Dropout(rate=dropout_rate, name='melsg_dropout_2')(melsg_pathway)
#     melsg_pathway = Conv2D(64,3,padding='same',activation="relu", 
#                          name='melsg_conv2d_3')(melsg_pathway)
#     melsg_pathway = MaxPooling2D(pool_size=2,name='melsg_maxpooling_3')(melsg_pathway)
#     melsg_pathway = Dropout(rate=dropout_rate,name='melsg_dropout_3')(melsg_pathway)
#     # melsg_pathway = GlobalAveragePooling2D(name='melsg_globalaverage2d')(melsg_pathway)
#     melsg_pathway = Flatten()(melsg_pathway)

#     mfcc_input = Input(shape=(20,n_frames,1),name='mfcc')
#     # Each filter is 20 high, for each of the 20 coefficients (MFCCs)
#     mfcc_pathway = Conv2D(64,[20,2],padding='same',
#                           name='mfcc_conv2d_1',activation="relu")(mfcc_input)
#     mfcc_pathway = MaxPooling2D(pool_size=[1,3],name='mfcc_maxpooling_1')(mfcc_pathway) 
#                             # output shape (20, 42, 16)
#     mfcc_pathway = Dropout(rate=dropout_rate,name='mfcc_dropout_1')(mfcc_pathway)
#     mfcc_pathway = Conv2D(32,[20,1],padding='same',activation="relu",
#                           name='mfcc_conv2d_2')(mfcc_pathway)
#     mfcc_pathway = MaxPooling2D(pool_size=[1,3],name='mfcc_maxpooling_2')(mfcc_pathway) 
#                             # output shape ( 20, 14, 32 )
#     mfcc_pathway = Dropout(rate=dropout_rate,name='mfcc_drouout_2')(mfcc_pathway)
#     mfcc_pathway = Conv2D(64,[20,1],padding='same',name='mfcc_conv2d_3',
#                           activation="relu")(mfcc_pathway)
#     mfcc_pathway = MaxPooling2D(pool_size=[1,2],name='mfcc_maxpooling_3')(mfcc_pathway)
#                             # output shape( 20, 7, 32 )
#     mfcc_pathway = Dropout(rate=dropout_rate,name='mfcc_dropout_3')(mfcc_pathway)
#     # mfcc_pathway = GlobalAveragePooling2D(name='mfcc_globalaverage2d',)(mfcc_pathway)
#     mfcc_pathway = Flatten()(mfcc_pathway)

#     joined = Concatenate(axis=-1,name='joint1')([melsg_pathway, mfcc_pathway])
#     joined = Dense(16, name='joined_dense1', activation='relu')(joined)
#     joined = Dense(32, name='joined_dense2', activation='relu')(joined)
#     joined = Dense(n_classes, activation='softmax', name='joined_output')(joined)

#     model = Model([melsg_input, mfcc_input], joined)
    model = Sequential()
    model.add(Conv2D(64,3,input_shape=(128,n_frames,1),padding='valid',activation="relu"))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(64,3,padding='valid',activation="relu"))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(64,3,padding='valid',activation="relu"))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(rate=dropout_rate))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation="softmax"))
    return model