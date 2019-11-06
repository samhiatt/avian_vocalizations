from avian_vocalizations.model import ModelFactory
from avian_vocalizations import data
from collections import namedtuple, defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
# from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback, TensorBoard
from hyperopt.fmin import fmin_pass_expr_memo_ctrl
from hyperopt import STATUS_OK
from hyperopt import pyll, STATUS_OK, STATUS_RUNNING
from io import StringIO
import numpy as np
# import pickle
import json
import time

import warnings; warnings.simplefilter('ignore', FutureWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ParamSpace = namedtuple("ParamSpace",['n_frames','dropout_rate','batch_size'])
Scores = namedtuple("Scores",['loss','accuracy','argmin_loss'])
    
class StatusReporter(Callback):
    def __init__(self, ctrl, split_i):
        super().__init__()
        self.ctrl = ctrl
        self.split_i = split_i

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        atmts = self.ctrl.attachments
        metrics_key = 'metrics.split%i'%self.split_i
        metrics = json.loads(atmts[metrics_key]) if metrics_key in atmts.keys() \
                else {k:[] for k in logs.keys()}
        #print("epoch %i: metrics: %s"%(epoch, metrics))
        assert len(metrics['loss'])==epoch, \
                "%i['loss'] should have %i elements."%(metrics_key,epoch)
    
        for k, v in logs.items(): # accuracy, val_accuracy, loss, val_loss
            if isinstance(v, (np.ndarray, np.generic)):
                metrics[k].append(v.item())
            else:
                metrics[k].append(v)
        
        self.ctrl.attachments[metrics_key] = bytes(json.dumps(metrics), encoding='utf-8')
        
        status = {
            'status': STATUS_RUNNING,
            'split': self.split_i,
            'epoch': epoch }
        #print("Reporting status: %s"%status)
        print("split %i epoch %i: %s"%(self.split_i, epoch, logs))
        self.ctrl.checkpoint(status)

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
        ctrl.attachments['params'] = bytes(json.dumps(dict(hp._asdict())), encoding='utf-8')
#         ctrl.checkpoint(dict(status=STATUS_RUNNING, params=dict(hp._asdict())))
        print(dir(ctrl.trials))
        print(ctrl.trials._exp_key)
        
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
            status_reporter = StatusReporter(ctrl, split_i=len(scores))
            tensorboard_callback = TensorBoard(log_dir="./tensorboard/%s/dropout%f/split%i"%(
                    ctrl.trials._exp_key, hp.dropout_rate, len(scores)))
            result = model.fit_generator(
                        training_generator, 
                        validation_data=validation_generator,
                        epochs=n_epochs, 
                        steps_per_epoch=training_generator.n_batches,
                        validation_steps=validation_generator.n_batches,
                        callbacks=[status_reporter, tensorboard_callback],
                        use_multiprocessing=True, workers=4,
                        verbose=0, )
            min_loss = np.min(result.history['val_loss'])
            argmin_loss = np.argmin(result.history['val_loss'])
            acc_at_min_loss = result.history['val_accuracy'][argmin_loss]
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
            
        out=StringIO(newline='\n')
        model.summary(print_fn=lambda x: out.write(x+'\n'))
        
        return {'status':STATUS_OK,
                'loss': mean_loss,
                'loss_variance': var_loss,
                'accuracy': mean_acc,
                'accuracy_variance': var_acc,
                'attachments':{
                    'summary': bytes(out.getvalue(), encoding='utf-8'),
                    },
                'scores':[dict(score._asdict()) for score in scores],
                'trial_time_s': int(time.time()-t0),
                }

    return ModelEvaluator