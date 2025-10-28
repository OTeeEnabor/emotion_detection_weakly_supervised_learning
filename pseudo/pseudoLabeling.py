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

# load emotion recognition model start
def load_model_data(model_path, opt_path):
  model = load_model(model_path)
  return model
def label_binarizer(y_intensity,threshold_val):
  return np.where(y_intensity<threshold_val, 0, 1)
# load emotion recognition model end
## custom accuracy metric
def subset_accuracy(y_true, y_pred):
  # y_true = tensorflow.make_ndarray(y_true)
  # y_pred = tensorflow.make_ndarray(y_pred)
  # y_pred = label_binarizer(y_pred, 0.5)
  return accuracy_score(y_true, y_pred)
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
  # convert features of shape 'base_model.ouput_shape[1:]' to vectors - 1 x 7
  # x = tensorflow.keras.layers.GlobalAveragePooling1D()(x)
  x = tensorflow.keras.layers.Flatten()(x)
  outputs = tensorflow.keras.layers.Dense(12, activation='sigmoid', name='final_au',
                                          kernel_initializer='glorot_normal')(x)
  au_model = tensorflow.keras.Model(au_inputs, outputs)
  au_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  return au_model

# create train_AU_model
def train_AU_model(X_train, y_train, model):
  start_time = time.time()
  EPOCHS = 35
  kfold = KFold(3, shuffle=True, random_state=123)
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
    lr_sched_cb = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
    # early stopping callback
    early = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=15, verbose=1, mode='auto')
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
# Load the the main csv
main_csv = pd.read_csv('/home-mscluster/oenabor/home/au_models/data/img_data_norm.csv')
# get the emotions present in the data set
emotion_list = main_csv['Emotion'].unique()
train_df_list = []
val_df_list = []
unlabeled_df_list = []
for emotion in emotion_list:
  #filter the dataset by emotion
  emotion_df = main_csv[main_csv['Emotion'] == emotion]
  # calculate the how many samples in the emotion_df
  emotion_df_len = len(emotion_df)
  # calculate 20%
  train_label_split = round(0.2*emotion_df_len)
  # calculate 10%
  val_label_split = round(0.1* emotion_df_len)

  train_df_emotion = emotion_df.iloc[:train_label_split]# 20/%
  val_df_emotion = emotion_df.iloc[train_label_split:(train_label_split+val_label_split)] #10%
  unlabel_df_emotion = emotion_df.iloc[(train_label_split+val_label_split):] # 70%

  train_df_list.append(train_df_emotion)
  print(f'train_{emotion} added to train_df_list')
  val_df_list.append(val_df_emotion)
  print(f'val_{emotion} added to train_df_list')
  unlabeled_df_list.append(unlabel_df_emotion)
  print(f'unlabeled_{emotion} added to train_df_list')

# concatenate dfs and save as csv
train_df = pd.concat(train_df_list)
train_df.to_csv(f'/home-mscluster/oenabor/home/pseudo/exp2/datasets/train.csv')
val_df = pd.concat(val_df_list)
val_df.to_csv(f'/home-mscluster/oenabor/home/pseudo/exp2/datasets/val.csv')
unlabeled_df = pd.concat(unlabeled_df_list)
unlabeled_df.to_csv(f'/home-mscluster/oenabor/home/pseudo/exp2/datasets/unlabeled.csv')
print('Done')


