from collections import namedtuple
from keras.models import Model, Sequential
from keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D, \
                         Dropout, Dense, Input, Concatenate, Flatten

# ModelParams = namedtuple("ModelParams",['n_frames', 'dropout_rate'])

def SgModel(n_classes, n_frames=128, dropout_rate=.2, include_top=True):
    sg_input = Input(shape=(128,n_frames,1),name='melsg')
    sg_pathway = Conv2D(16,3,name='sg_conv2d_1',
                         padding='valid',activation="relu")(sg_input)
    sg_pathway = MaxPooling2D(pool_size=2,name='sg_maxpooling_1')(sg_pathway)
    sg_pathway = Dropout(rate=dropout_rate,name='sg_dropout_1')(sg_pathway)
    sg_pathway = Conv2D(32,3,padding='valid',activation="relu",
                         name='sg_conv2d_2')(sg_pathway)
    sg_pathway = MaxPooling2D(pool_size=2,name='sg_maxpooling_2')(sg_pathway)
    sg_pathway = Dropout(rate=dropout_rate, name='sg_dropout_2')(sg_pathway)
    sg_pathway = Conv2D(64,(30,1),padding='valid',activation="relu",
                         name='sg_conv2d_3')(sg_pathway)
    sg_pathway = MaxPooling2D(pool_size=(1,2),name='sg_maxpooling_3')(sg_pathway)
    sg_pathway = Dropout(rate=dropout_rate,name='sg_dropout_3')(sg_pathway)
    if include_top:
        sg_pathway = GlobalAveragePooling2D(name='sg_globalaverage2d')(sg_pathway)
        sg_pathway = Dense(n_classes, activation='softmax', name='output')(sg_pathway)
    else:
        return sg_input, sg_pathway

    model = Model(sg_input, sg_pathway, name='ExpSgModel')
    return model


def MFCCModel(n_classes, n_frames=128, dropout_rate=.2, include_top=True):
    mfcc_input = Input(shape=(20,n_frames,1),name='mfcc')
    mfcc_pathway = Conv2D(16,(1,3),name='mfcc_conv2d_1',
                         padding='valid',activation="relu")(mfcc_input)
    mfcc_pathway = MaxPooling2D(pool_size=(1,2),name='mfcc_maxpooling_1')(mfcc_pathway)
    mfcc_pathway = Dropout(rate=dropout_rate,name='mfcc_dropout_1')(mfcc_pathway)
    mfcc_pathway = Conv2D(32,(1,3),padding='valid',activation="relu",
                         name='mfcc_conv2d_2')(mfcc_pathway)
    mfcc_pathway = MaxPooling2D(pool_size=(1,2),name='mfcc_maxpooling_2')(mfcc_pathway)
    mfcc_pathway = Dropout(rate=dropout_rate, name='mfcc_dropout_2')(mfcc_pathway)
    mfcc_pathway = Conv2D(64,(20,1),padding='valid',activation="relu",
                         name='mfcc_conv2d_3')(mfcc_pathway)
    mfcc_pathway = MaxPooling2D(pool_size=(1,2),name='mfcc_maxpooling_3')(mfcc_pathway)
    mfcc_pathway = Dropout(rate=dropout_rate,name='mfcc_dropout_3')(mfcc_pathway)
    if include_top:
        mfcc_pathway = GlobalAveragePooling2D(name='mfcc_globalaverage2d')(mfcc_pathway)
        mfcc_pathway = Dense(n_classes, activation='softmax', name='output')(mfcc_pathway)
    else:
        return mfcc_input, mfcc_pathway

    model = Model(mfcc_input,mfcc_pathway)
    return model


def CombinedModel(n_classes, n_frames=128, dropout_rate=.2):
    sg_input, sg_pathway = SgModel(n_classes, n_frames, dropout_rate, include_top=False)

    mfcc_input, mfcc_pathway = MFCCModel(n_classes, n_frames, dropout_rate, include_top=False)

    joined = Concatenate(axis=-1, name='joined_sg_mfc')([sg_pathway, mfcc_pathway])
    joined = Dense(128, name='joined_dense1', activation='relu')(joined)
    joined = Dense(128, name='joined_dense2', activation='relu')(joined)
    joined = GlobalAveragePooling2D(name='joined_globalaverage2d')(joined)
    joined = Dense(n_classes, activation='softmax', name='joined_output')(joined)

    model = Model([sg_input, mfcc_input], joined)
    return model


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