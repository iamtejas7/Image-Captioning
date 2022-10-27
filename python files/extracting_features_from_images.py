import string
import numpy as np
from PIL import Image
import os
import pickle
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()
import re

# 2. Extracting features from images using Xception network  CNN model

def extract_features(directory):
    # extracting features from images and creating a dictionary of image_id and feature matrix
    model = tf.keras.applications.Xception(include_top=False, pooling='avg')
    
    features = {}
    
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0

        feature = model.predict(image)
        features[img] = feature
        
    return features


features = extract_features('Flicker8k_Dataset')

pickle.dump(features, open("features.pkl","wb"))  # saving of features in pickle file for further use