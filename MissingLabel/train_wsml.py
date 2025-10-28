# Libraries Needed
import os
import re
import sys
import time
import json
import pickle
from tqdm import tqdm
import math
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
    # load image
    img = image.load_img(img_list[i])
    # convert image to numpy array
    img = image.img_to_array(img)
    # normalize the image
    img = img / 255.
    # add image to list
    X_dataset.append(img)
  # convert list to numpy array
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
  base_model.trainable = True
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
# create run directory
now = datetime.now()
now = now.strftime("%d%m%Y")
outdir = '/home-mscluster/oenabor/home//wsml/exp1/saved_models/'
run_desc = f"test-train-{now}"
run_dir = generate_output_dir(outdir, run_desc)
print(f"Results saved to: {run_dir}")

# load noisy dataset
# define x and y train binary file path
X_path = '/home-mscluster/oenabor/home/wsml/data/X_train_unlabeled_13112022.npy'
y_path = '/home-mscluster/oenabor/home/wsml/data/y_train_unlabeled_13112022.npy'
# load data numpy files
X = np.load(X_path)
# load true data
y = np.load(y_path, allow_pickle=True)
# print(y)
# change y to binary
y_train = label_binarizer(y, 0.5)
print(y_train, y_train.dtype, y[0:10])
# load validation data
X_val_path = '/home-mscluster/oenabor/home/wsml/data/X_val_91222.npy'
X_val = np.load(X_val_path)
y_val_path = '/home-mscluster/oenabor/home/wsml/data/y_val_91222.npy'
y_val = np.load(y_val_path)
y_val = label_binarizer(y_val, 0.5)

# create AU model
model = create_model()
# create hyperparameter dictionary
P = {
  "batch_size":32,
  "num_classes": 12,
  "lr" : 0.1,
  "optimizer" :"adam",
  "num_epochs" : 100,
  "clean_rate": 1,
  "delta_rel": 0.01,
  "mod_scheme": "LL-R"
}
clean_rate_list = []
epoch_list = []
f1_list = []
accuracy_list  =[]
average_precision_list = []
loss_list = []
# instantiate an optimizer
if P['optimizer'] == 'adam':
    opt = tensorflow.keras.optimizers.Adam(learning_rate=P['lr'])
# calculate the number of batch updates per epoch
numUpdates = int(X.shape[0]/P["batch_size"])
# define best accuracy metric
best_accuracy = 0
# loop over the number of epochs
for epoch in range(0, P["num_epochs"]):
  print(P["clean_rate"])
  # define dummy arrayto calulate  prediction and actual
  # pred_epoch_final = np.zeros((y.shape[0], P["num_classes"]))
  # y_actual = np.zeros((y.shape[0], P["num_classes"]))
  # loop over the data in batch size increments
  # TRAINING LOOP
  for i in range(0, numUpdates):
    # determine the start and slice indeces for the current batch
    start = i * P["batch_size"]
    end = start + P["batch_size"]
    # take a step
    with tensorflow.GradientTape() as tape:
      print(f'\n Epoch {epoch}: make prediction: {i}')
      x = X[start:end]
      pred = model(x, training=True)
      y = y_train[start:end]
      loss, correction_indices = compute_batch_loss(pred, y, P)
    # calculate the gradients using our tape and then update the model weights
    grads = tape.gradient(loss, model.trainable_variables)#,unconnected_gradients=tensorflow.UnconnectedGradients.ZERO)
    # print(grads)
    opt.apply_gradients(zip(grads,model.trainable_variables))
    # if we are using large-loss correction permanent mode
    if P["mod_scheme"] == "LL-Cp" and correction_indices is not None:
      # change correction indices from tensorflow tensor to numpy array
      correction_indices_np = correction_indices.numpy()
      # extract the row indices
      rows = correction_indices_np.shape[0]
      # extract the column indices
      columns = correction_indices_np.shape[1]
      # loop through the correction indices and change y actual value depending on current value
      for row in range(0,rows):
        for column in range(0,columns):
          if y[row][column] == 1:
            y[row][column] = 0
          else:
            y[row][column] = 1
      # change the y_actual values
      # y[correction_indices[:,0], correction_indices[:,1]] = 1
      # update y_true label with permanent correction batch
      y_train[start:end] = tensorflow.cast(y, dtype=tensorflow.int32).numpy()
  # VALIDATION LOOP
  numUpdates_val = int(X_val.shape[0]/P["batch_size"])
  pred_val_final = np.zeros((y_val.shape[0], P["num_classes"]))
  for j in range(0, numUpdates_val):
    # determine start and end indices
    start_val = j * P["batch_size"]
    end_val = start_val+ P["batch_size"]
    val_input = X_val[start_val:end_val]
    pred_val = model(val_input, training=False)
    pred_val = label_binarizer(pred_val,0.5)
    # update final pred array with batch results
    pred_val_final[start_val:end_val] = pred_val
  # calculate validation accuracy
  accuracy = accuracy_score(y_true=y_val, y_pred=pred_val_final)
  f1 = f1_score(y_true=y_val, y_pred=pred_val_final,average="weighted")
  ap = average_precision_score(y_true=y_val, y_score=pred_val_final, average="weighted")

  print(accuracy, f1, ap)
    # pred_val = model.predict(X_val)
    # pred_val = label_binarizer(pred_val, 0.5)
    # accuracy = accuracy_score(y_val,pred_val)
    # f1 = compute_metrics(y_val, pred_val, run_dir, epoch)
  print(f'Epoch {epoch} performance on validation: accuracy- {accuracy},  f1_score- {f1}')
  if best_accuracy < f1:
    best_accuracy = f1
    best_accuracy_epoch = epoch
    print(f'saving model weight for best metric model.....')
    path = os.path.join(run_dir, f'model-{epoch:02d}-{best_accuracy:.2f}_{P["mod_scheme"]}.hdf5')
    model.save(path)
    print('model_saved')
  clean_rate_list.append(P["clean_rate"])
  epoch_list.append(epoch)
  loss_list.append(loss.numpy())
  accuracy_list.append(accuracy)
  f1_list.append(f1)
  average_precision_list.append(ap)
  P["clean_rate"] -= P["delta_rel"]

train_dictionary = {
  "epoch": epoch_list,
  "clean-rate": clean_rate_list,
  "loss": loss_list,
  "accuracy": accuracy_list,
  "f1": f1_list,
  "average_precision": average_precision_list
}
path = os.path.join(run_dir, f'model-{epoch:02d}-{accuracy:.2f}_{P["mod_scheme"]}.hdf5')
model.save(path)
train_df = pd.DataFrame(train_dictionary)
train_df.to_csv(f'{run_dir}/train_info.csv')






