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
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, \
  LearningRateScheduler, ModelCheckpoint, CSVLogger
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
import pickle
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

# create step decay schedule start
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):

  def schedule(epoch):

    return initial_lr * (decay_factor ** np.floor(epoch / step_size))

  return LearningRateScheduler(schedule)
# create step decay schedule end

# create dataset start
def create_dataset(df,format):
  X_dataset = []
  img_list = df['Path'].tolist()
  for i in tqdm(range(df.shape[0])):
    img = image.load_img(img_list[i])
    img = image.img_to_array(img)
    img = img / 255.
    X_dataset.append(img)
  X = np.array(X_dataset)

  # drop unnecessary columns not used for training
  if format == 'emotion':
    y = np.array(df['Emotion'])
  if format == 'AU':
    y = np.array(df[['AU1',"AU2","AU4","AU5","AU6","AU9","AU12","AU15","AU17","AU20","AU25","AU26"]])
  return X,y
# create dataset end

# load emotion recognition model start
def load_model_data(model_path, opt_path):
  model = load_model(model_path)
  return model
  # with open(opt_path, 'rb') as fp:
  #   d = pickle.load(fp)
  #   epoch = d['epoch']
  #   opt = d['opt']
  #   return epoch, model, opt
# load emotion recognition model end

# create AU model start
def create_AU_model(er_model):
  # create base model
  base_model = er_model
  base_model.pop()
  # freeze base model
  base_model.trainable = True

  # add new model on top
  au_inputs = tensorflow.keras.Input(shape=(224, 224, 3))
  x = base_model(au_inputs, training=False)
  x = tensorflow.keras.layers.Flatten()(x)
  outputs = tensorflow.keras.layers.Dense(12, activation='sigmoid', name='final_au',
                                          kernel_initializer='glorot_normal')(x)

  au_model = tensorflow.keras.Model(au_inputs, outputs)

  au_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  return au_model

# def create_model(width=224, height=224):
#   # load pretrained model 'VGG16'
#   base_model = keras.applications.VGG16(
#     include_top=False,
#     weights="imagenet",
#     input_shape=(width, height, 3))
#   base_model.trainable = False
#   model = Sequential()
#   model.add(base_model)
#   model.add(BatchNormalization())  # batch normalization
#   model.add(Dropout(0.4))  # dropout for preventing overfitting
#   model.add(Flatten())
#   model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
#   model.add(Dropout(0.4))
#   model.add(Dense(12, activation='sigmoid', kernel_initializer='glorot_normal'))  # softmax classification for 7 labels
#
#   model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
#   return model
# # create model end

# create train_AU_model
def train_AU_model(X_train, y_train, model):
  start_time = time.time()
  EPOCHS = 50
  kfold = KFold(2, shuffle=True, random_state=123)
  for f, (trn_ind, val_ind) in enumerate(kfold.split(X_train)):
    print();
    print('#' * 50)
    print('Fold: ', f + 1)
    print('#' * 50)

    # Training Data
    x_train = X_train[trn_ind]
    y_trn = y_train[trn_ind]

    # Validation Data
    x_val = X_train[val_ind]
    y_val = y_train[val_ind]

    # Define start and end epoch for each folds
    fold_start_epoch = f * EPOCHS
    fold_end_epoch = EPOCHS * (f + 1)
    # Create callbacks
    # checkpoint callback
    checkpoint_cb = MyModelCheckpoint(
      os.path.join(run_dir, 'model-{epoch:02d}-{val_loss:.2f}.hdf5'),
      monitor='val_accuracy', verbose=1)
    # learning rate callback
    lr_sched_cb = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, \
                                      step_size=2)
    # early stopping callback
    early = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5, verbose=1, mode='auto')
    # csv log callback
    log_csv = CSVLogger(f'{run_dir}/model_{f + 1}_{now}log.csv', separator=',', append=False)

    # callbacks list
    cb = [checkpoint_cb, lr_sched_cb, early, log_csv]

    model.fit(x_train, y_trn, initial_epoch=fold_start_epoch, epochs=fold_end_epoch,
              callbacks=cb,
              validation_data=(x_val, y_val))

    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(elapsed_time))

##########################################################################
# read in datasets
# train_df = pd.read_csv('/home-mscluster/oenabor/home/pseudo/exp2/datasets/train.csv')
# val_df = pd.read_csv('/home-mscluster/oenabor/home/pseudo/exp1/datasets/val.csv')
unlabeled_df = pd.read_csv('/home-mscluster/oenabor/home/pseudo/exp2/datasets/unlabeled.csv')
# train_unlabeled_df = pd.read_csv('/home-mscluster/oenabor/home/pseudo/exp/datasets/pseudo_train_unlabeled_13112022.csv')
# create numpy representation of datasets
# X_train, y_train = create_dataset(train_df,'AU')
# X_val, y_val = create_dataset(val_df,'AU')
X_unlabeled, y_unlabeled = create_dataset(unlabeled_df,'AU')
# X_train_unlabeled, y_train_unlabeled = create_dataset(train_unlabeled_df,'AU')

#save np
path = '/home-mscluster/oenabor/home/pseudo/exp2/datasets/AU_data'
np.save(f'{path}/X_unlabelled', X_unlabeled)
np.save(f'{path}/y_unlabelled', y_unlabeled)

print('numpy save done')

