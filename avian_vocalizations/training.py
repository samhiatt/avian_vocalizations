from avian_vocalizations.data import AudioFeatureGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
import os
import pandas as pd
import numpy as np


def train_model(model, X_train, y_train,
                n_epochs=10,
                n_splits=1,
                n_frames=1024,
                batch_size=24,
                validation_size=.25,
                seed=37,
                output_dir='training_example_output',
                tensorboard_dir=None,
                ):
    sss = StratifiedShuffleSplit(n_splits=n_splits,
                                 test_size=validation_size,
                                 random_state=seed)
    n_classes = len(np.unique(y_train))
    scores = []
    for cv_train_index, cv_val_index in sss.split(X_train, y_train):
        training_generator = AudioFeatureGenerator(
            [X_train[i] for i in cv_train_index],
            [y_train[i] for i in cv_train_index],
            data_dir='../data', scale=True, batch_size=batch_size, n_classes=n_classes,
            shuffle=True, seed=37, n_frames=n_frames)
        validation_generator = AudioFeatureGenerator(
            [X_train[i] for i in cv_val_index],
            [y_train[i] for i in cv_val_index],
            data_dir='../data', scale=True, batch_size=batch_size, n_classes=n_classes,
            seed=37, n_frames=n_frames)

        partial_filename = "split%02i" % len(scores)
        checkpointer = ModelCheckpoint(verbose=1, save_best_only=True,
                                       filepath=os.path.join(output_dir, 'weights.best.%s.hdf5' % partial_filename))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        callbacks = [checkpointer]

        if tensorboard_dir:
            callbacks.append(TensorBoard("%s/split%i_%s" % (
                tensorboard_dir,
                len(scores),
                datetime.now().strftime("%Y_%m_%dT%H_%M_%S"))))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        learning = model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            epochs=n_epochs,
            steps_per_epoch=training_generator.n_batches,
            validation_steps=validation_generator.n_batches,
            callbacks=callbacks,
            # use_multiprocessing=True, workers=4,
            verbose=1, )
        history_output_file = os.path.join(output_dir, 'training_history_split%i.csv' % len(scores))
        pd.DataFrame(learning.history).to_csv(history_output_file, index_label='epoch')
        acc_at_min_loss = learning.history['val_accuracy'][np.argmin(learning.history['val_loss'])]
        scores.append(acc_at_min_loss)
        print("Split %i: min loss: %.5f, accuracy at min loss: %.5f" % (
            len(scores), np.min(learning.history['val_loss']), acc_at_min_loss))
    print("Cross Validation Accuracy: mean(val_acc[argmin(val_loss)]): %.4f" % (np.mean(scores)))
