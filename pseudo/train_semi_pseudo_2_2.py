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
# from pseudo.train_semi_pseudo import alpha_weight


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
def pseduo_label_binarizer(pseudo_label):
  conf_label = np.where(pseudo_label >0.9, pseudo_label, 1)
  conf_label = np.where(conf_label < 0.1, conf_label, 0)
  return conf_label


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

def supervised_train(X_train,Y_train,X_val,Y_val,P,run_dir):
  """
  Train a model with limited labelled data
  X: trainining data features
  Y: training data ground truth
  P: training parameters
  :return: a trained model
  """
  epoch_list = []
  train_accuracy_list = []
  train_loss_list = []
  val_accuracy_list = []
  val_loss_list = []
  model = create_model()
  # define adam optimizer
  opt = tensorflow.keras.optimizers.Adam(learning_rate = P['lr'])
  # calculate the number of batch updates per epoch
  numUpdates = int(X_train.shape[0]/P["batch_size"])
  pred_train_final = np.zeros((Y_train.shape[0],P["num_classes"]))
  best_accuracy = 0
  # loop over the range for number of epochs
  for epoch in range(0, P["num_epochs"]):
    # Training mini batch loops
    for i in range(0, numUpdates): # loop over the data in batch size increments
      start = i * P["batch_size"] # determine start indice for current batch
      end = start + P["batch_size"]
      with tensorflow.GradientTape() as tape:
        print(f'\n Epoch {epoch}: make prediction {i}')
        batch_x = X_train[start:end] # mini-batch
        batch_pred = model(batch_x,training=True)
        batch_y = Y_train[start:end]
        BCE = tensorflow.keras.losses.BinaryCrossentropy()
        loss = BCE(batch_y,batch_pred)
      # calculate the gradients using tape and update model weights
      grads = tape.gradient(loss,model.trainable_variables)
      opt.apply_gradients(zip(grads,model.trainable_variables))
      pred_train_final[start:end] = batch_pred
    # VALIDATION LOOP
    numUpdates_val = int(X_val.shape[0]/P["batch_size"])
    pred_val_final = np.zeros((Y_val.shape[0],P["num_classes"]))
    # validation mini batch loops
    for j in range(0, numUpdates_val):
      start_val = j* P["batch_size"]
      end_val = start_val + P["batch_size"]
      batch_val = X_val[start_val:end_val]
      pred_val = model(batch_val, training=False)
      pred_val = label_binarizer(pred_val,0.5)
      pred_val_final[start_val:end_val] = pred_val
    # end of an epoch activities
    # calculate train accuracy and loss
    pred_train_final = label_binarizer(pred_train_final,0.5)
    pred_train_final_loss = tensorflow.cast(pred_train_final, dtype=tensorflow.float32)
    train_accuracy = accuracy_score(y_true=Y_train,y_pred=pred_train_final)
    Y_train = label_binarizer(Y_train,0.5)
    Y_train = tensorflow.cast(Y_train, dtype=tensorflow.float32)
    train_loss = BCE(Y_train,pred_train_final_loss)
    # calculate val accuracy and loss
    pred_val_final = label_binarizer(pred_val_final,0.5)
    pred_val_final_loss = tensorflow.cast(pred_val_final, dtype=tensorflow.float32)
    val_accuracy = accuracy_score(y_true=Y_val,y_pred=pred_val_final)
    Y_val = label_binarizer(Y_val,0.5)
    Y_val= tensorflow.cast(Y_val, dtype=tensorflow.float32)
    val_loss = BCE(Y_val, pred_val_final_loss)
    print(f'Epoch {epoch}:training acc:{train_accuracy}-train loss:{train_loss}__val_acc:{val_accuracy} - val_loss:{val_loss}' )
    if best_accuracy < val_accuracy:
      best_accuracy = val_accuracy
      print(f'saving model weight for best metric model...')
      path  = os.path.join(run_dir,f'model-{epoch:02d}-{best_accuracy:.2f}.hdf5')
      model.save(path)
      print('model_saved')
    epoch_list.append(epoch)
    train_accuracy_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    val_accuracy_list.append(val_accuracy)
    val_loss_list.append(val_loss)

  path_last_model = os.path.join(run_dir,f'model-{epoch:02d}-{best_accuracy:.2f}-last_model.hdf5')
  model.save(path_last_model)
  train_dictionary ={
    "epoch": epoch_list,
    "train_accuracy": train_accuracy_list,
    "train_loss": train_loss_list,
    "val_accuracy": val_accuracy_list,
    "val_loss":val_loss_list
  }
  train_df = pd.DataFrame(train_dictionary)
  train_df.to_csv(f'{run_dir}/train_info.csv')
def alpha_weight(step):

  """
  :param T1:
  :param T2:
  :param af:
  :param step:
  :return:
  """
  T1 = 10
  T2 = 70
  af = 3
  if step < T1:
    return 0.0
  elif step > T2:
    return af
  else:
    return ((step - T1) / (T2 - T1)) * af
def pseudo_train(pre_model,X_labelled,Y_labelled, X_unlabelled, P, run_dir,X_test,Y_test):
  # log
  epoch_list = []
  loss_labelled_list = []
  loss_pseudo_list  = []
  loss_combined_list = []
  alpha_list = []
  test_acc_list = []
  test_loss_list = []
  test_f1_list = []
  # labelled dataset
  labelled_datasaet = tensorflow.data.Dataset.from_tensor_slices((X_labelled,Y_labelled))
  labelled_datasaet = labelled_datasaet.shuffle(1000).batch(P["batch_size"])
  pseudo_dataset = tensorflow.data.Dataset.from_tensor_slices((X_unlabelled))
  pseudo_dataset = pseudo_dataset.shuffle(1000).batch(P["batch_size"])

  avg_main_loss = tensorflow.keras.metrics.Mean(name='avg_main_loss', dtype=tensorflow.float32)
  pseudo_numUpdates = int(X_unlabelled.shape[0] / P["batch_size"])

  x_label, y_label = next(iter(labelled_datasaet))
  # pseudo images
  pseudo_x = next(iter(pseudo_dataset))
  pseudo_labels = pre_model(pseudo_x, training=False)
  pseudo_labels = label_binarizer(pseudo_labels, 0.9)

  opt = tensorflow.keras.optimizers.Adam(learning_rate=P["lr"])
  loss_fn = tensorflow.keras.losses.BinaryCrossentropy()
  best_accuracy = 0
  step = P['step']
  for epoch in range(0, P["num_epochs"]):

    for i in range(0, pseudo_numUpdates):

      print(f'\n Epoch {epoch}: make prediction: {i}')
      with tensorflow.GradientTape() as pseudo_tape:
        labelled_predict = pre_model(x_label, training=True)
        pseudo_predict = pre_model(pseudo_x, training=False)
        main_loss = loss_fn(y_label, labelled_predict)
        pseudo_loss = alpha_weight(epoch) * loss_fn(pseudo_labels,pseudo_predict)
        loss = main_loss + pseudo_loss

      x_label, y_label = next(iter(labelled_datasaet))
      pseudo_x = next(iter(pseudo_dataset))
      pseudo_labels = pre_model(pseudo_x, training=False)
      pseudo_labels = label_binarizer(pseudo_labels, 0.9)

      grads = pseudo_tape.gradient(loss, pre_model.trainable_variables)
      opt.apply_gradients(zip(grads,pre_model.trainable_variables))

      avg_main_loss.update_state(loss)
      main_loss = avg_main_loss.result()
      step += 1

    # at end of each epoch
    # evaluate model on test_data
    numUpdates_val = int(X_test.shape[0] / P["batch_size"])
    predict_test_final = np.zeros((Y_test.shape[0], int(Y_test.shape[1])))
    for j in range(0, numUpdates_val):
      start_val = j * P["batch_size"]
      end_val = start_val + P["batch_size"]
      x_test = X_test[start_val:end_val]
      predict_test = pre_model(x_test, training=False)
      predict_test = label_binarizer(predict_test,0.9)
      predict_test_final[start_val:end_val] = predict_test
    test_acc  = accuracy_score(y_true=Y_test, y_pred = predict_test_final)
    test_f1 = f1_score(y_true=Y_test, y_pred=predict_test_final, average='weighted')
    print(f'Epoch {epoch} performance on test: alpha-weight- {alpha_weight(epoch)} test_accuracy-{test_acc} - test_f1: {test_f1} ')
    if best_accuracy < test_f1:
      best_accuracy = test_f1
      print(f'saving model weight for best metric model.....')
      path = os.path.join(run_dir, f'model-{epoch:02d}-{best_accuracy:.2f}.hdf5')
      pre_model.save(path)
      print('model saved')

    epoch_list.append(epoch)
    test_acc_list.append(test_acc)
    test_f1_list.append(test_f1)
    loss_combined_list.append(main_loss.numpy())
    alpha_list.append(alpha_weight(epoch))
  #once done with training
  train_dictionary = {
    "epoch": epoch_list,
    "alpha": alpha_list,
    "f1": test_f1_list,
    "acc": test_acc_list,
    "loss": loss_combined_list
  }
  train_df = pd.DataFrame(train_dictionary)
  train_df.to_csv(f'{run_dir}/pseudo_2_2_train_info.csv')


###############################################
# create run directory
now = datetime.now()
now = now.strftime("%d%m%Y")
outdir = '/home-mscluster/oenabor/home/pseudo/exp1/saved_models/au_models'
run_desc = f"test-train-{now}"
run_dir = generate_output_dir(outdir, run_desc)
print(f"Results saved to: {run_dir}")
# define labelled dataset
X_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/X_train.npy'
y_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/y_train.npy'
# load data numpy files
X_train = np.load(X_path)
# load true labels
y = np.load(y_path, allow_pickle=True)
y_train = label_binarizer(y,0.5)
# define unlabelled dataset
X_unlabelled_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/X_unlabeled.npy'
X_unlabelled = np.load(X_unlabelled_path,allow_pickle=True)

# load test data
X_test_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/X_val_pseudo.npy'
X_test = np.load(X_test_path,allow_pickle=True)
Y_test_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/AU_data/y_val_pseudo.npy'
Y_test = np.load(Y_test_path, allow_pickle=True)
Y_test = label_binarizer(Y_test,0.5)
P = {
  "batch_size":60,
  "num_classes": 12,
  "lr" : 0.001,
  "optimizer" :"adam",
  "num_epochs" : 100,
  "step":100
}
# define the model and optimizer path
MODEL_PATH = '/home-mscluster/oenabor/home/pseudo/exp1/saved_models/au_models/00004-test-train-12112022/model-60-0.01.hdf5'
OPT_PATH = '/home-mscluster/oenabor/home/pseudo/exp1/saved_models/au_models/00004-test-train-12112022/model-60-0.01.pkl'

# load model using model and optimizer path
model = load_model_data(MODEL_PATH, OPT_PATH)
pseudo_train(model,X_train,y_train,X_unlabelled,P,run_dir,X_test,Y_test)





