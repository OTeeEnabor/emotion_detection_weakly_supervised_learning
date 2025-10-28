# Libraries Needed
import os
import re
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Any, List, Tuple, Union
import sklearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import pickle
import json
from tqdm import tqdm
# Functions needed

# create output directory start
def generate_output_dir(outdir, run_desc):
  prev_run_dirs = []
  if os.path.isdir(outdir):
    prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir( \
      os.path.join(outdir, x))]
  prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
  prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
  cur_run_id = max(prev_run_ids, default=-1) + 1
  run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
  assert not os.path.exists(run_dir)
  os.makedirs(run_dir)
  return run_dir
# create output directory end

# create logger start
class Logger(object):

  """Redirect stderr to stdout, optionally print stdout to a file, and
  optionally force flushing on both stdout and the file."""

  def __init__(self, file_name: str = None, file_mode: str = "w",should_flush: bool = True):

    self.file = None

    if file_name is not None:
      self.file = open(file_name, file_mode)

    self.should_flush = should_flush
    self.stdout = sys.stdout
    self.stderr = sys.stderr

    sys.stdout = self
    sys.stderr = self

  def __enter__(self) -> "Logger":
    return self

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    self.close()

  def write(self, text: str) -> None:
    """Write text to stdout (and a file) and optionally flush."""
    if len(text) == 0:
      return

    if self.file is not None:
      self.file.write(text)

    self.stdout.write(text)

    if self.should_flush:
      self.flush()

  def flush(self) -> None:
    """Flush written text to both stdout and a file, if open."""
    if self.file is not None:
      self.file.flush()

    self.stdout.flush()

  def close(self) -> None:
    """Flush, close possible files, and remove
        stdout/stderr mirroring."""
    self.flush()

    # if using multiple loggers, prevent closing in wrong order
    if sys.stdout is self:
      sys.stdout = self.stdout
    if sys.stderr is self:
      sys.stderr = self.stderr

    if self.file is not None:
      self.file.close()
# create logger end

# create ModelCheckpoint start
class MyModelCheckpoint(ModelCheckpoint):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def on_epoch_end(self, epoch, logs=None):
    super().on_epoch_end(epoch,logs)\

    # Also save the optimizer state
    filepath = self._get_file_path(epoch=epoch,
        logs=logs, batch=None)
    filepath = filepath.rsplit( ".", 1 )[ 0 ]
    filepath += ".pkl"

    with open(filepath, 'wb') as fp:
      pickle.dump(
        {
          'opt': model.optimizer.get_config(),
          'epoch': epoch+1
         # Add additional keys if you need to store more values
        }, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('\nEpoch %05d: saving optimizer to %s' % (epoch + 1, filepath))
# create ModelCheckpoint end
def load_model_data(model_path, opt_path):
  model = load_model(model_path)
  return model

def my_metrics(y_true, y_pred):
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted')
  f1Score = f1_score(y_true, y_pred, average='weighted')
  print("Accuracy  : {}".format(accuracy))
  print("Precision : {}".format(precision))
  print("f1Score : {}".format(f1Score))
  cm = confusion_matrix(y_true, y_pred)
  print(cm)
  return accuracy, precision, f1Score

# create run directory
now = datetime.now()
now = now.strftime("%d%m%Y")

# load model and make predictions on test data
# define the model and optimizer path
MODEL_PATH = 'C:\\Users\\user\\Documents\\MSc Computer\\Cluster\\model_logs\\ER_Model_cluster_training\\er_00002_21102022_096\\model-41-0.05.hdf5'#'/home-mscluster/oenabor/home/er_models/saved_models/00002-test-train-21102022/model-41-0.05.hdf5'

OPT_PATH = 'C:\\Users\\user\\Documents\\MSc Computer\\Cluster\\model_logs\\ER_Model_cluster_training\\er_00002_21102022_096\\model-41-0.05.pkl'#'/home-mscluster/oenabor/home/er_models/saved_models/00002-test-train-21102022/model-41-0.05.pkl'

# load model using model and optimizer path
model  = load_model_data(MODEL_PATH, OPT_PATH)

# load the test numpy binary files

# define x and y test binary file path
X_test_path = 'C:\\Users\\user\\Documents\\MSc Computer\\EmotionDetectionUsingCNN\\KDEF_and_AKDEF\\X_kdef.npy'#'/home-mscluster/oenabor/home/er_models/data/test_split/X_test.npy'
y_test_path = 'C:\\Users\\user\\Documents\\MSc Computer\\EmotionDetectionUsingCNN\\KDEF_and_AKDEF\\y_kdef.npy'#'/home-mscluster/oenabor/home/er_models/data/test_split/y_test.npy'

# load test numpy files
X_test = np.load(X_test_path,allow_pickle=True)
y_test = np.load(y_test_path,allow_pickle=True)
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_test)
encoded_y = encoder.transform(y_test)
# convert integers to one-hot encoding
true_y  = to_categorical(encoded_y)
print(encoded_y)
print(true_y)

# # evaluate model on test data
test_loss, test_acc = model.evaluate(x=X_test,y=true_y, verbose=1)
print(f'Model Accuracy on test: {test_acc*100:6.2f}')

# predict on test data
predictions = model.predict(X_test)
yPredictions = np.argmax(predictions, axis=1)
true_y = np.argmax(true_y, axis=1)
testAcc,testPrec, testFScore = my_metrics(true_y, yPredictions)

result_dict  = {
  'test_loss': test_loss,
  'test_accuracy_keras': test_acc,
  'test_accuracy_sklearn': testAcc,
  'test_precision': testPrec,
  'test_F1Score': testFScore

}

with open('C:\\Users\\user\\Documents\\MSc Computer\\Cluster\\model_logs\\ER_Model_cluster_training\\er_00002_21102022_096\\eval_test_kdef.json','w') as output:#'/home-mscluster/oenabor/home/er_models/saved_models/00002-test-train-21102022/eval_test.json'
  json.dump(result_dict, output)
# json_object = json.dumps(result_dict)
# pred_df.to_csv(f'prediction_results_{now}.csv')
# prediction_results.tofile('prediction_results.csv',sep=',')
print('Done')


