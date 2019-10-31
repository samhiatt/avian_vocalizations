from avian_vocalizations.model import ModelFactory
from avian_vocalizations import data
from collections import namedtuple
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
# from keras.callbacks import ModelCheckpoint
from hyperopt.fmin import fmin_pass_expr_memo_ctrl
from hyperopt import STATUS_OK
from hyperopt import pyll, STATUS_OK, STATUS_RUNNING
from io import StringIO
import numpy as np
import json
import time

import warnings; warnings.simplefilter('ignore', FutureWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ParamSpace = namedtuple("ParamSpace",['n_frames','dropout_rate','batch_size'])
Scores = namedtuple("Scores",['loss','accuracy','argmin_loss'])

def EvaluatorFactory(n_splits=3, n_epochs=10, data_dir='data'):
    
    @fmin_pass_expr_memo_ctrl
    def ModelEvaluator(expr, memo, ctrl):
        def get_model():
            return ModelFactory(n_classes, 
                                n_frames=hp.n_frames, 
                                dropout_rate=hp.dropout_rate)
        index_df, shapes_df, train_df, test_df = data.load_data(data_dir)

        label_encoder = LabelEncoder().fit(index_df['english_cname'] )
        n_classes = len(label_encoder.classes_)

        X_train = index_df.loc[index_df['test']==False].index.values
        y_train = label_encoder.transform(index_df.loc[index_df['test']==False,"english_cname"].values)

        pyll_rval = pyll.rec_eval(
            expr,
            memo=memo,
            print_node_on_error=True)
        hp = ParamSpace(*pyll_rval)
        print("Running training trial with %i splits, %i epochs, with hyperparams: %s"%(
                n_splits, n_epochs, hp))
        
        model = get_model()
        out=StringIO(newline='\n')
        model.summary(print_fn=lambda x: out.write(x+'\n'))
        ctrl.checkpoint({'status':STATUS_RUNNING, 
                         'attachments':{
                             'model_summary':out.getvalue(),
                         }})
        
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/4, random_state=37)
        scores = []
        t0 = time.time()
        for cv_train_index, cv_val_index in sss.split(X_train, y_train):
            print("Split %i/%i"%(len(scores)+1, n_splits))
            training_generator = data.AudioFeatureGenerator(
                [X_train[i] for i in cv_train_index], 
                [y_train[i] for i in cv_train_index], 
                batch_size=hp.batch_size, shuffle=True, seed=37 )
            validation_generator = data.AudioFeatureGenerator(
                [X_train[i] for i in cv_val_index], 
                [y_train[i] for i in cv_val_index], 
                batch_size=hp.batch_size )
            # Get a new (untrained) model
            model = get_model()
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            result = model.fit_generator(
                        training_generator, 
                        validation_data=validation_generator,
                        epochs=n_epochs, 
                        steps_per_epoch=training_generator.n_batches,
                        validation_steps=validation_generator.n_batches,
                        #callbacks=[checkpointer], 
                        #use_multiprocessing=True, workers=4,
                        verbose=0, )
            min_loss = np.min(result.history['val_loss'])
            argmin_loss = np.argmin(result.history['val_loss'])
            acc_at_min_loss = result.history['val_acc'][argmin_loss]
            # Scores are tuples of ( min_loss, acc_at_min_loss, argmin(min_loss) )
            score = Scores(min_loss, acc_at_min_loss, argmin_loss)
            scores.append(score)
            print("Split %i: min loss: %.5f, accuracy at min loss: %.5f, min loss at epoch %i"%(
                len(scores), score.loss, score.accuracy, score.argmin_loss ))
        mean_loss = np.mean([score.loss for score in scores])
        var_loss = np.var([score.loss for score in scores])
        mean_acc = np.mean([score.accuracy for score in scores])
        var_acc = np.var([score.accuracy for score in scores])
        print("Cross Validation Accuracy: mean(val_acc[argmin(val_loss)]): %.4f +/-%.4f, mean loss: %.4f +/-%.4f"%(
                mean_acc, np.sqrt(var_acc), mean_loss, np.sqrt(var_loss) ))
            
        return {'status':STATUS_OK,
                'loss': mean_loss,
                'loss_variance': var_loss,
                'accuracy': mean_acc,
                'accuracy_variance': var_acc,
                'attachments':{
                    'history': json.dumps(result.history).encode('utf-8'),
                    },
                'scores':[dict(score._asdict()) for score in scores],
                'trial_time_s': int(time.time()-t0),
                }

    return ModelEvaluator