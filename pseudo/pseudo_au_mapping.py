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
from sklearn.metrics import multilabel_confusion_matrix, classification_report\
  ,accuracy_score, f1_score, precision_score, confusion_matrix
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
def subset_accuracy(y_true,y_pred):
  y_pred = tensorflow.py_function(label_binarizer,(y_pred,0.5), tensorflow.double)
  return tensorflow.py_function(accuracy_score,(y_true,y_pred),tensorflow.double)

def load_model_data(model_path, opt_path):
  model = load_model(model_path,custom_objects={"subset_accuracy":subset_accuracy})
  return model
# Multiclassification class start

class multiLabelClassificationMetrics():

  def __init__(self, actual, predictions,k):
    self.actual = actual
    self.pred = predictions
    self.k = k

  def patk (self, actual, pred, k):
    if k == 0:
      return 0
    # taking only the top k predictions in a class
    k_pred = pred[:k]

    # taking the set of the actual values
    actual_set = set(actual)
    # taking the set of the predicted values
    pred_set = set(k_pred)

    # taking the intersection of the actual set and the pred set to find the common values
    common_values = actual_set.intersection(pred_set)

    return len(common_values)/ len(k_pred)

def create_dataset(df):
  X_dataset = []
  img_list = df['Path'].tolist()
  for i in tqdm(range(df.shape[0])):
    img = image.load_img(img_list[i])
    img = image.img_to_array(img)
    img = img / 255.
    X_dataset.append(img)
  X = np.array(X_dataset)
  return X
# Multiclassification class end
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

def multi_label_metric(y_true, y_pred):

  '''Build a text report showing the main classification metrics.
  '''

  report = classification_report(y_true, y_pred )

  '''In multilabel classification, this function computes subset accuracy:
  the set of labels predicted for a sample must exactly match the corresponding
  set of labels in y_true.
  '''
  subset_accuracy = accuracy_score(y_true, y_pred)
  return report, subset_accuracy
def label_binarizer(y_intensity,threshold_val):
  return np.where(y_intensity<threshold_val, 0, 1)

# create run directory
now = datetime.now()
now = now.strftime("%d%m%Y")

# load model and make predictions on test data
# define the model and optimizer path
MODEL_PATH = 'C:\\Users\\user\\Documents\\MSc Computer\\Cluster\\pseudo\\model_logs\\au_models\\00074-test-train-26022023\\model-01-0.36.hdf5'
OPT_PATH = 'empty'

# load model using model and optimizer path
model = load_model_data(MODEL_PATH, OPT_PATH)
# LOAD THE DATASET
# create dataset path
# data_path = '/home-mscluster/oenabor/home/exp1/data/AU_img_data_norm.csv'
# create dataframe
# data_df = pd.read_csv(data_path)
# drop unnecessary columns

HOME_PATH = 'emotion_mapping'
file_list = os.listdir(HOME_PATH)
for file in file_list:
  print(file)
  emotion = file.strip('_true.csv')
  emotion_df = pd.read_csv(f'{HOME_PATH}\\{file}')
  path_list = emotion_df['Path'].tolist()
  X = create_dataset(emotion_df)
  y_pred = model.predict(X)
  y_pred = label_binarizer(y_pred,0.5)
  pred_df = pd.DataFrame(y_pred,
                         columns=['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15', 'AU17', 'AU20', 'AU25',
                                  'AU26'])
  pred_df['Emotion'] = emotion
  pred_df['Path'] = path_list
  # save to csv
  path = f'C:\\Users\\user\\Documents\\MSc Computer\\Cluster\\pseudo\\model_logs\\au_models\\00074-test-train-26022023\\emotion_mapping\\{emotion}_pred.csv'
  pred_df.to_csv(path)
  print(f'{emotion} csv saved')
# # loop through  dataset using emotions list
# emotions_list = ['anger','disgust','fear','happy','neutral','sad','surprise']
#
# for emotion in emotions_list:
#   #filter the dataset
#   emotion_df = data_df[data_df['Emotion'] == emotion]
#   # print(emotion_df)
#   # select n=100 samples from dataset
#   emotion_df = emotion_df.sample(n=100)
#   # save to csv as true values
#   path_true = f'/home-mscluster/oenabor/home/pseudo/exp1/saved_models/00033-test-train-13122022/' \
#          f'emotionAUmappings/{emotion}_true.csv'
#   emotion_df.to_csv(path_true)
#   print(f'{emotion} true csv saved')
#   path_list = emotion_df['Path'].tolist()
#   # print(path_list)
#   # create x input for model
#   X = create_dataset(emotion_df)
#   # feed input to model and make prediction
#   y_pred = model.predict(X)
#   # threshold the results
#   y_pred = label_binarizer(y_pred,0.5)
#   # create dataframe
#   pred_df = pd.DataFrame(y_pred, columns=['AU1','AU2','AU4','AU5','AU6','AU9','AU12','AU15','AU17','AU20','AU25','AU26'])
#   pred_df['Emotion'] = emotion
#   pred_df['Path'] = path_list
#   # save to csv
#   path = f'/home-mscluster/oenabor/home/wsml/exp1/saved_models/00033-test-train-13122022/emotionAUmappings/{emotion}_pred.csv'
#   pred_df.to_csv(path)
#   print(f'{emotion} csv saved')
# print(f'Mappings done')
#
#
