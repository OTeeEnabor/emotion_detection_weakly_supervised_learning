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
  y = np.array(df.drop(['Index', 'Filename','Path','AU1',"AU2","AU4","AU5","AU6","AU9","AU12","AU15","AU17","AU20","AU25","AU26"], axis=1))

  return X, y
# create dataset end
def subset_accuracy(y_true,y_pred):
  y_pred = tensorflow.py_function(label_binarizer,(y_pred,0.5), tensorflow.double)
  return tensorflow.py_function(accuracy_score,(y_true,y_pred),tensorflow.double)
def load_model_data(model_path, opt_path):
  model = load_model(model_path,custom_objects={"subset_accuracy":subset_accuracy})
  #,"loss":pseudo_loss(epoch) })#custom_objects={'loss': asymmetric_loss(alpha)}
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
  # sigmoid classification for 12 au labels

  model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[subset_accuracy])
  return model
# create model end
def pseudo_model(width=224,height=224):
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
def alpha_weight(step):

    """
    :param T1:
    :param T2:
    :param af:
    :param step:
    :return:
    """
    T1 = 10
    T2 = 50
    af = 3
    if step < T1:
      return 0.0
    elif step > T2:
      return af
    else:
      return ((step - T1) / (T2 - T1)) * af
def pseudo_loss(epoch):
  T1 = 10
  T2 = 50
  af = 3
  if epoch < T1:
    alpha = 0.0
  elif epoch > T2:
    alpha = af
  else:
    alpha = ((epoch - T1) / (T2 - T1)) * af
  def loss(y_true, y_pred):
    loss_fn = keras.losses.BinaryCrossentropy()
    return alpha * loss_fn(y_true, y_pred)
  return loss

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
  # sigmoid classification for 12 au labels

  # model.compile(optimizer='adam', loss=pseudo_loss(epoch),metrics=[subset_accuracy])
  return model


def label_binarizer(y_intensity,threshold_val):
  return np.where(y_intensity<threshold_val, 0, 1)
def pseduo_label_binarizer(pseudo_label):
  conf_label = np.where(pseudo_label >0.9, pseudo_label, 1)
  conf_label = np.where(conf_label < 0.1, conf_label, 0)
  return conf_label
# create train_ER_model
def train_pseudo(model, X_labelled, y_labelled,X_unlabelled, X_val, y_val,num_epochs):
  for epoch in range(0,num_epochs):
    model.fit(X_labelled,y_labelled, intial_epoch=epoch, epochs=epoch+1)
    pseudo_y = model.predict(X_unlabelled)
    model.fit(X_unlabelled,pseudo_y,intial_epoch=epoch, epochs=epoch+1
              )

# create run directory
now = datetime.now()
now = now.strftime("%d%m%Y")
outdir = '/home-mscluster/oenabor/home/pseudo/exp1/saved_models/au_models'
run_desc = f"test-train-{now}"
run_dir = generate_output_dir(outdir, run_desc)
print(f"Results saved to: {run_dir}")

# define the model and optimizer path
MODEL_PATH = '/home-mscluster/oenabor/home/pseudo/exp1/saved_models/au_models/00004-test-train-12112022/model-60-0.01.hdf5'
OPT_PATH = '/home-mscluster/oenabor/home/pseudo/exp1/saved_models/au_models/00004-test-train-12112022/model-60-0.01.pkl'

# load model using model and optimizer path
labelled_model = load_model_data(MODEL_PATH, OPT_PATH)
# load unlabelled model
unlabelled_model = create_model()
# define labelled dataset
X_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/X_train.npy'
y_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/y_train.npy'
# load data numpy files
X_labelled = np.load(X_path)
X_labelled_tensor = tensorflow.convert_to_tensor(X_labelled, dtype=tensorflow.float32)
# load true labels
y = np.load(y_path, allow_pickle=True)
y_labelled = label_binarizer(y,0.5)
y_labelled_tensor = tensorflow.convert_to_tensor(y_labelled, dtype=tensorflow.float32)

# define unlabelled dataset
X_unlabelled_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/X_unlabeled.npy'
X_unlabelled = np.load(X_unlabelled_path,allow_pickle=True)
X_unlabelled_tensor = tensorflow.convert_to_tensor(X_unlabelled, dtype=tensorflow.float32)
# load test data
X_test_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/X_val_pseudo.npy'
X_val = np.load(X_test_path,allow_pickle=True)
X_val = tensorflow.convert_to_tensor(X_val, dtype=tensorflow.float32)
Y_test_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/y_val_pseudo.npy'
Y_test = np.load(Y_test_path, allow_pickle=True)
y_val = label_binarizer(Y_test,0.5)
y_val = tensorflow.convert_to_tensor(y_val, dtype=tensorflow.float32)

numEpochs = 100
best_metric = 0
list_history = []
list_epoch = []
labelled_list_accuracy = []
labelled_list_loss = []
labelled_list_accuracy_val = []
labelled_list_loss_val = []
list_accuracy = []
list_loss = []
list_accuracy_val = []
list_loss_val = []
for epoch in range(0,numEpochs):
  # change loss function
  labelled_loss = True
  if labelled_loss:
    unlabelled_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[subset_accuracy])
  # train with labelled data
  labelled_history = unlabelled_model.fit(X_labelled_tensor, y_labelled_tensor, initial_epoch=epoch, epochs=epoch + 1,batch_size=100,validation_data=(X_val,y_val))
  # predict pseudo labels with labelled model
  # create pseudo_labels_zero tensor with size of pseudo_label data
  # pseudo_label_matrix = np.zeros((X_unlabelled.shape[0], 12))
  # create number of updates based on batch size
  # numUpdates = int(X_unlabelled.shape[0]/32)
  # for i in range(0,numUpdates):
  #   start = i * 32
  #   end = start + 32
  #   pseudo_labels = labelled_model.predict(X_unlabelled[start:end])
  #   pseudo_label_matrix[start:end] = pseudo_labels
  # get confident labels

  pseuddo_labels = labelled_model.predict(X_unlabelled_tensor)
  pseudo_labels = pseduo_label_binarizer(pseuddo_labels)
  unlabelled_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=pseudo_loss(epoch),metrics=[subset_accuracy])
  # train unlabelled model with pseudo labels
  # create callbacks
  # checkpoint_cb_pseudo = MyModelCheckpoint(
  #   os.path.join(run_dir,'model-{epoch:02d}-{val_loss:.2f}_pseudo.hdf5')
  # )
  history = unlabelled_model.fit(X_unlabelled_tensor, pseudo_labels, initial_epoch=epoch, epochs=epoch+1, batch_size=100, validation_data=(X_val,y_val))
  current_accuracy = history.history['val_subset_accuracy'][0]
  if best_metric < current_accuracy:
    best_metric = current_accuracy
    print(f'saving model weight for best metric model...')
    path = os.path.join(run_dir, f'model-{epoch:02d}-{best_metric:.2f}.hdf5')
    unlabelled_model.save(path)
    print('model_saved')
  print(labelled_history.history.keys())
  print(history.history.keys())
  labelled_list_accuracy.append(labelled_history.history['subset_accuracy'][0])
  labelled_list_loss.append(labelled_history.history['loss'][0])
  labelled_list_accuracy_val.append(labelled_history.history['val_subset_accuracy'][0])
  labelled_list_loss_val.append(labelled_history.history['val_loss'][0])
  # unlablled
  list_epoch.append(epoch)
  list_accuracy.append(history.history['subset_accuracy'][0])
  list_loss.append(history.history['loss'][0])
  list_accuracy_val.append(history.history['val_subset_accuracy'][0])
  list_loss_val.append(history.history['val_loss'][0])
  list_history.append(history)

train_dictionary = {
  "epoch": list_epoch,
  "accuracy_labelled": labelled_list_accuracy,
  "val_accuracy_labelled":labelled_list_accuracy_val,
  "loss_labelled": labelled_list_loss,
  "val_loss_labelled": labelled_list_loss_val,
  "accuracy_pseudo": list_accuracy,
  "val_accuracy_pseudo": list_accuracy_val,
  "loss_pseudo": list_loss,
  "val_loss_pseudo":list_loss_val
}
train_df = pd.DataFrame(train_dictionary)
train_df.to_csv(f'{run_dir}/train_pseudo_fit_info.csv')
# for i in range(0,len(list_history)):
#   print(list_history[i].history.keys())



