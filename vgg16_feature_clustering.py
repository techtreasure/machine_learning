
import pandas as pd
import json
import os
import skimage.draw
import numpy
from PIL import Image, ImageDraw


from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = True


model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = 'croped'
targetdir = "out_file_10"
number_clusters = 10


# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.png'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())

# Clustering
kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))

# Copy images renamed by cluster
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass

print(kmeans)

print("\n")
for i, m in enumerate(kmeans.labels_):
    #     print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    #     print(m)

    
    head, tail  = os.path.split(filelist[i])
    n, file_extension = os.path.splitext(tail)
    
    try:
        os.makedirs(targetdir+'/'+str(m))
    except OSError:
        pass
    shutil.copy(filelist[i], targetdir+'/'+str(m)+'/'+ n + str(i) + "@" + str(m) + ".jpg")
