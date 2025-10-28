# Libraries Needed
import os
import re
import sys
import time
import json
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Any, List, Tuple, Union
import sklearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, average_precision_score,\
  precision_score, confusion_matrix
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
from tqdm import tqdm
import math
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

# create step decay schedule start
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):

  def schedule(epoch):

    return initial_lr * (decay_factor ** np.floor(epoch / step_size))

  return LearningRateScheduler(schedule)
# create step decay schedule end

# create dataset start
def create_dataset(df):
  X_dataset = []
  img_list = df['Path'].tolist()
  for i in tqdm(range(df.shape[0])):
    img = image.load_img(img_list[i])
    img = image.img_to_array(img)
    img = img / 255.
    X_dataset.append(img)

  X = np.array(X_dataset)
  # drop unnecessary columns not used for training
  y = np.array(df.drop(['Index', 'Filename', 'Path', 'Emotion'], axis=1))

  return X, y


# create dataset end
def label_binarizer(y_intensity, threshold_val):
  return np.where(y_intensity < threshold_val, 0, 1)


def subset_accuracy(y_true, y_pred):
  #     y_true = tensorflow.py_function(label_binarizer,(y_true,0.5), tensorflow.double)
  #     y_pred = tensorflow.py_function(label_binarizer,(y_pred,0.5), tensorflow.double)
  return tensorflow.py_function(accuracy_score, (y_true, y_pred), tensorflow.double)


def load_model_data(model_path, opt_path):
  model = load_model(model_path, custom_objects={"subset_accuracy": subset_accuracy})
  return model


# create model start
def create_model(width=224, height=224):
  # load pretrained model 'VGG16'
  base_model = keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(width, height, 3))
  base_model.trainable = False
  model = Sequential()
  model.add(base_model)
  model.add(Flatten())
  model.add(Dense(12, activation='sigmoid', name='final_au', kernel_initializer='glorot_normal'))
  return model
# create model end

def loss_an(logits, noisy_labels, P):
  """
  This function calculates the loss matrix of assumed negative labels and
  the corrected negative loss matrix.
  Args:
      logits(tensor): keras model predictions - tensor
      noisy_labels(numpy.ndarray): label vector - numpy.ndarray
      P (dict): parameter dictionary

  Returns:
      loss_matirx, corrected loss matrix

  """
  # initialize KerasBinaryCrossentropy loss
  BCE = tensorflow.keras.losses.BinaryCrossentropy(
    #     from_logits = True,
    reduction="none"
  )
  # compute assumed negative loss binary cross entropy matrix
  loss_matrix = BCE(noisy_labels, logits)
  #     print(f'first loss matrix.shape from first definition of loss_an: {loss_matrix.shape}')

  # change noisy labels to boolean vector
  noisy_labels = (noisy_labels > 0.5)
  # compute logical not of noisy labels boolean vector
  noisy_labels = tensorflow.logical_not(noisy_labels)
  # change boolean tensor to binary tensor
  noisy_labels = tensorflow.where(noisy_labels, 1, 0)
  # change noisy labels to float32 dtype
  noisy_labels = tensorflow.cast(noisy_labels, dtype=tensorflow.float32)
  # calculate corrected loss matrix
  corrected_loss_matrix = BCE(noisy_labels, logits)
  return loss_matrix, corrected_loss_matrix


def compute_batch_loss(logits, noisy_labels, P):
  # get batch size
  batch_size = int(logits.shape[0])
  # get num classes in multilabel problem
  num_classes = int(logits.shape[1])
  # get boolean matrix for all the labels equal to
  unobserved_mask = (noisy_labels == 0)  # 32 * 12
  # calculate batch loss and corrected loss matrices
  loss_matrix, corrected_loss_matrix = loss_an(logits, noisy_labels, P)
  correction_indices = None

  if P['clean_rate'] == 1:  # epoch ==1
    final_loss_matrix = loss_matrix
  # after the first epoch
  else:
    if P['mod_scheme'] == 'LL-Cp':
      k = math.ceil(batch_size * num_classes * P['delta_rel'])
    else:
      k = math.ceil(batch_size * num_classes * (1 - P['clean_rate']))

    unobserved_mask = tensorflow.where(unobserved_mask, 1, 0)  # 32 * 12

    unobserved_mask = tensorflow.cast(unobserved_mask, dtype=tensorflow.float32)
    #         # convert loss matrix to ndarray for multiplication
    #         # change k to int32 which is reuqeired
    #         k = tensorflow.cast(k, dtype = tensorflow.int32)
    #         print(k)

    #         print(unobserved_mask)
    #         print(loss_matrix)

    unobserved_loss = tensorflow.math.multiply(unobserved_mask, loss_matrix.numpy().reshape(
      [batch_size, -1]))  # .numpy().reshape([32,-1])
    # #         print(unobserved_loss.shape)
    flatten_unobserved_loss = tensorflow.reshape(unobserved_loss, [-1])
    # #         print(flatten_unobserved_loss)
    topk = tensorflow.math.top_k(flatten_unobserved_loss, tensorflow.cast(k, dtype=tensorflow.int32))
    topk_lossvalue = topk.values[-1]

    correction_indices = tensorflow.where(unobserved_loss > topk_lossvalue)

    if P['mod_scheme'] in ['LL-Ct', 'LL-Cp']:
      # print(tensorflow.less(unobserved_loss,topk_lossvalue).shape,loss_matrix.shape,corrected_loss_matrix.shape)
      final_loss_matrix = tensorflow.where(unobserved_loss < topk_lossvalue,tensorflow.reshape(loss_matrix, [batch_size,-1]),tensorflow.reshape(corrected_loss_matrix, [batch_size,-1]))
      # final_loss_matrix = tensorflow.where(unobserved_loss < topk_lossvalue,
      #                                      loss_matrix.numpy().reshape([batch_size, -1]),
      #                                      corrected_loss_matrix.numpy().reshape([batch_size, -1]))
    #             print(final_loss_matrix)
    else:
      zero_loss_matrix = tensorflow.zeros_like(loss_matrix)
      final_loss_matrix = tensorflow.where(unobserved_loss < topk_lossvalue, tensorflow.reshape(loss_matrix, [batch_size,-1]), tensorflow.reshape(zero_loss_matrix, [batch_size,-1]))
  main_loss = tensorflow.math.reduce_mean(final_loss_matrix)

  return main_loss, correction_indices


def partial_dataset(noisy_labels):
  for i in range(len(noisy_labels)):
    rand_column_index = np.random.randint(0, 11)
    noisy_labels[i][rand_column_index] = 1
  return noisy_labels
def compute_metrics(y_actual, y_pred, save_path, epoch):
  """
  This function calculates metrics used to evaluate the performance of the model.
  accuracy, precision, recall, classification.
  :param y_actual (ndarray): Nosiy labels
  :param y_pred (ndarray):  predicted labels
  :return: accuracy, precision, recall, save classification csv to disk
  """
  f1score = f1_score(y_actual,y_pred,average='weighted')
  # calculate the classification report dictionary
  report_dictionary = classification_report(y_actual,y_pred, output_dict=True)
  # convert dictionary to JSON
  json_report_dictionary = json.dumps(report_dictionary, indent=4)
  df= pd.DataFrame(report_dictionary).transpose()
  df.to_csv(f'{save_path}/report_{epoch}.csv')
  # with open(f'{save_path}/report_{epoch}.json', 'w') as outfile:
  #   json.dump(json_report_dictionary, outfile)
  #   print('dictionary saved')
  return f1score



def load_model_data(model_path, opt_path):
  model = load_model(model_path, custom_objects={"subset_accuracy": subset_accuracy})
  return model

###############################################

# define the test data path
X_test_path = '/home-mscluster/oenabor/home/wsml/data/X_test_91222.npy'
y_test_path = '/home-mscluster/oenabor/home/wsml/data/y_test_91222.npy'
# load test path
X_test = np.load(X_test_path, allow_pickle=True)
y_test = np.load(y_test_path, allow_pickle=True)
y_test = label_binarizer(y_test, 0.5)

# define the model path
model_path = '/home-mscluster/oenabor/home/wsml/exp1/saved_models/00061-test-train-30122022/model-376-0.67_LL-R.hdf5'
model = load_model_data(model_path,"empty")

# define model parameters
# create hyperparameter dictionary
P = {
  "batch_size":32,
  "num_classes": 12,
  "lr" : 0.0001,
  "optimizer" :"adam",
  "num_epochs" : 100,
  "clean_rate": 1,
  "delta_rel": 0.01,
  "mod_scheme": "LL-R"
}

# define test loop
# calculate the number of batches required to go through the entire test set
test_batches_no = int(X_test.shape[0]/P["batch_size"])
pred_test_final = np.zeros((y_test.shape[0], P["num_classes"]))
for batch in range(0, test_batches_no):
  # determine the start and end indices to select batch
  start_batch = batch * P["batch_size"]
  end_batch = start_batch + P["batch_size"]
  # select input batch
  test_batch_input = X_test[start_batch:end_batch]
  # make a prediction
  pred_test = model(test_batch_input, training=False)
  pred_test = label_binarizer(pred_test, 0.5)
  # update pred_final with batch predictions
  pred_test_final[start_batch:end_batch] = pred_test
# after looping through the entire test set calculate evaluation metrics and save results
accuracy = accuracy_score(y_true=y_test, y_pred=pred_test_final)
f1 = f1_score(y_true=y_test, y_pred=pred_test_final,average="weighted")
ap = average_precision_score(y_true=y_test, y_score=pred_test_final, average="weighted")
print(accuracy, f1, ap)
print(f'Performance on Test: accuracy- {accuracy},  f1_score- {f1}')

# save the prediction array
np.save('/home-mscluster/oenabor/home/wsml/exp1/saved_models/00061-test-train-30122022/pred_test.npy',pred_test_final)
# prediction array saved
print('Prediction array saved in numpy format')
