import pandas as pd
from gradcam import GradCAM
import tensorflow
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def load_model_data(model_path):
    return load_model(model_path)
def get_image_paths(dataframe_path):
    df = pd.read_csv(dataframe_path)
    return df['Path'].tolist()
def preprocess_image(image_path):
    image = utils.load_img(image_path)
    image = utils.img_to_array(image)
    preprocessed_image = np.expand_dims(image, axis=0)
    return preprocessed_image

# load model
MODEL_PATH = 'C:\\Users\\user\\Documents\\MSc Computer\\Cluster\\missinglabels\\model_logs\\00033-test-train-13122022\\model-95-0.71_LL-R.hdf5'
model = load_model_data(MODEL_PATH)

# load images from disk in OpenCV format
file_path = 'C:\\Users\\user\\Documents\\MSc Computer\\EmotionDetectionUsingCNN\\KDEF_and_AKDEF\\kdef.csv'
kdef_img_list = get_image_paths(file_path)
# get image
img = preprocess_image(kdef_img_list[0])
# load image in cv2
orig = cv2.imread(kdef_img_list[0])

# make prediction with model
pred = model.predict(img)
# initialize gradient activation map and build the heatmap
cam = GradCAM(model)
heatmap = cam.compute_heatmap(img)
heatmap = cv2.resize(heatmap, (224,224))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
# print(pred, type(pred))