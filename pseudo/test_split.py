import numpy as np
from sklearn.model_selection import train_test_split
def label_binarizer(y_intensity, threshold_val):

  return np.where(y_intensity < threshold_val, 0, 1)
#define X and y
# load test data
X_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/ER_data/X_val.npy'
X = np.load(X_path,allow_pickle=True)
Y_path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/ER_data/y_val.npy'
Y = np.load(Y_path, allow_pickle=True)
# Y = label_binarizer(Y,0.5)

X_val, X_test, y_val, y_test = train_test_split(X,Y, test_size=0.5, random_state=42)
path = '/home-mscluster/oenabor/home/pseudo/exp1/datasets/ER_data'
np.save(f'{path}/X_val_pseudo', X_val)
np.save(f'{path}/y_val_pseudo', y_val)
np.save(f'{path}/X_test_pseudo', X_test)
np.save(f'{path}/y_test_pseudo', y_test)
print('numpy save done')