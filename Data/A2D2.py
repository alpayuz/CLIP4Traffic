'''
This notebook generates captions for the given A2D2 images based on their object-classes (38 classes with semantic segmentation)
'''

# load packages/modules 
import os
import sys
import io
import pandas as pd
import torch
import numpy as np
from numpy import zeros
from numpy import asarray
from matplotlib import pyplot as plt
from matplotlib import gridspec
%matplotlib inline 
from PIL import Image
from torchvision import transforms
import csv
import cv2 as cv
import json
import itertools
import statistics
import inflect
import fnmatch
import tarfile
from tqdm import tqdm
import pickle
import collections

from pathlib import Path
from collections import Counter
from datetime import datetime
import torchvision.transforms.functional as F
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from pprint import pprint
import shutil
from shutil import copyfile
from PIL import ImageColor
from colormap import rgb2hex

# Torch Version 
print('Torch version:', torch.__version__)

# Initialize paths [ADD PATHS!]

#images
a2d2_data_main = [xxx]
a2d2_camera_images = [xxx]
a2d2_label_images = [xxx]

#captions
a2d2_captions = ['PATH_TO_SAVE_CAPTIONS.csv']
a2d2_captions_final = ['PATH_TO_SAVE_FINAL_CAPTIONS.csv']
a2d2_objects = ['PATH_TO_SAVE_IMAGE_OBJECTS.json']


# A2D2-dataset needs to be unpacked (.tar-file) 
unpack_A2D2 = False

if unpack_A2D2 = True:
    tar = tarfile.open('PATH TO .TAR-FILE', "r") #insert path
    tar.getmembers

    file = tar.extractall(a2d2_data_main) #insert path
    tar.close()


# Read and open the class list json-file which contains the hexcodes (colors) and their corresponding object-class (label)
with open('PATH TO JSON') as json_list:
  class_list = json.load(json_list)

class_df = pd.DataFrame(class_list.items(), columns=['Color Code', 'Object Class'])
# Remove/Rename defined classes
class_df['Object Class'] = class_df['Object Class'].replace({'Obstacles / trash': 'Obstacles', 'Painted driv. instr.': 'Driving instructions', 'Traffic guide obj.': 'Traffic guide object'})


# Separate/Sort camera and label images
def sort_images(label, source_path, destination_path):
    '''move the images that contain the 'label' into the destination folder'''
  for subdirs, dirs, files in os.walk(source_path): 
    for name in files:
      if name.endswith(('.png')) and str(label) in name:
        filename = str(subdirs) + '/' + str(name)
        shutil.move(filename, destination_path)

#1. move camera images
sort_images('camera', a2d2_data_main, a2d2_camera_images)
print(len(os.listdir(a2d2_camera_images)))

#2. move label images
sort_images('label', a2d2_data_main, a2d2_label_images)
print(len(os.listdir(a2d2_label_images))) #it occured that some images are missing 


# Chek if any images left in the source folder
check_images = False

def left_images(source_path):
    '''checks for left/not-moved images in the source folder'''
  left_images_list = []
  for subdirs, dirs, files in os.walk(source_path):
    for name in files:
      if name.endswith(('.png')): 
        print(name)
        left_images_list.append(name)
  if not left_images_list:
    print('All images moved!')
  if left_images_list:
    return left_images_list

if check_images == True:
  left_images(source_path=a2d2_data_main)


# As not every camera image had a corresponding label image and vice versa, we will find the matching ones (based on the coder in their filename) and create a df
def matches(labels, originals):
    '''finds the matching images (label & camera) and stores the filenames in a dataframe '''
  match_list_label = []
  match_list_original = []
  unmatched = []
  for _, label in enumerate(labels):
    for _, original in enumerate(originals):
      if label[-25:] == original[-25:] and label[:14] == original[:14]: # these positions in their filenames are identical for camera and label
        match_list_label.append(label)
        match_list_original.append(original)
      else:
        unmatched.append(label)
        unmatched.append(original)
  
  # create df
  images_df = pd.DataFrame(data=np.column_stack([match_list_label, match_list_original]), columns=['Label Image', 'Camera Image'])
  # save df
  images_df.to_pickle('PATH_FOR_DF.pkl')

  return images_df, len(unmatched)


# Load df containing matches
images_df = pd.read_pickle('PATH_FOR_DF.pkl')
print(len(images_df)) # approx. 21,813 image-pairs


# Now we can scan the label-images and build captions based on the object-classes for the corresponding camera-images
def get_classes(image_id, object_count, dataframe, plot): #(source: https://stackoverflow.com/questions/65645044/get-hex-color-code-coordinate-of-the-pixel-in-a-image)
    '''Scan each pixel of the label image and generate a caption for the corresponding camera image based on the represented object-classes'''
  # load image
  filename = dataframe['Label Image'].iloc[image_id]
  image_path = a2d2_label_images + '/' + str(filename)
  image = Image.open(image_path)
  if plot == True:
    print(f'labeled image: {filename}')

  # get image dimensions
  pixels = image.load()
  width, height = image.size

  # get pixel colors
  pixel_colors = []
  for y in range(height):
    for x in range(width):
      r, g, b = pixels[x, y]
      pixel_colors.append(f'#{r:02x}{g:02x}{b:02x}')
  total_pixels = len(pixel_colors)
  
  # get classes
  image_classes = []
  class_share = []
  image_classes_rare = []
  class_share_rare = []

  counter = Counter(pixel_colors).most_common(object_count) #The Counter module enables an efficient and very fast way to group the pixels
  counter_rare = Counter(pixel_colors).most_common()[-16:] #We are also interested in the "small" classes with less pixels due to their size (e.g. pedestrian)

  for object_class, count in counter:
      'match the obejct_classes and also the proportion of the image'
    x = class_df[class_df['Color Code'] == object_class]['Object Class'].item() #class_df = df of json_file 
    share = (count/total_pixels)*100
    image_classes.append(x)
    class_share.append(round(share, 2))
  
  for object_class_, count_ in counter_rare:
      'same analysis for "small" classes'
    y = class_df[class_df['Color Code'] == object_class_]['Object Class'].item()
    share_ = (count_/total_pixels)*100
    if share_ >= 0.2:
      image_classes_rare.append(y)
      class_share_rare.append(round(share_, 2))

  # create caption(s)
  caption = ', '.join(image_classes)
  caption1 = caption.replace(' Irrelevant signs,', '').replace(' Irrelevant signs', '') #remove object class 'irrelevant signs'
  caption2 = caption1.replace(' RD normal street,', '').replace(' Nature object,', '').replace(' Sky,', '').replace('Buildings,', '') #remove non-automotive classes (RD normal street, Nature object, Sky)

  # get original image
  original_image = dataframe['Camera Image'].iloc[image_id]
  if plot == True:
    print(f'camera image: {original_image}')

  if plot == True:
    camera_image_path = a2d2_camera_images + '/' + str(original_image)
    fig = plt.figure(figsize=(25,22))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2,1.5,1])

    #Original Image
    im = plt.imread(camera_image_path)
    a0 = plt.subplot(gs[0])
    a0.set_title(f'Caption1: {caption1} \n Caption2: {caption2}', fontsize=12)
    a0.imshow(im)

    #Pie Chart (big)
    a1 = plt.subplot(gs[1])
    explode_list = []
    for i in range(object_count):
      explode_list.append(0.06)
    a1.pie(class_share[:count], labels=image_classes, explode=explode_list, autopct='%1.0f%%')


    #Pie Chart (small)
    a2 = plt.subplot(gs[2])
    explode_list_ = []
    for i in range(len(image_classes_rare)):
      explode_list_.append(0.08)
    a2.pie(class_share_rare[:count_], labels=image_classes_rare, explode=explode_list_, autopct='%1.0f%%')

  else:
    return image_id, image_classes, caption1, caption2, original_image

# test get_classes()
get_classes(image_id=4, object_count=12, dataframe=images_df, plot=True) #the object_count parameter controls the number of classes to consider


# GENERATE CAPTIONS
def a2d2_captions(captions_path, objects_path, dataframe):
  '''scan every image of the dataset to generate caption and store them in a csv_file'''

  # list for image objects
  training_objects = []

  for image in tqdm(range(0, len(dataframe))):
    image_id_, image_classes_, caption1_, caption2_, original_image_ = get_classes(image_id=image, object_count=11, dataframe=dataframe, plot=False)

    # save image objects
    training_objects.append(image_classes_)

    # save caption to csv
    header = ['image_id', 'camera image', 'caption']
    data1 = [image_id_, original_image_, caption1_]
    data2 = [image_id_, original_image_, caption2_]
    file_exists = os.path.isfile(captions_path)

    with open(captions_path, 'a') as csv_file:
      writer = csv.writer(csv_file)
      if file_exists == False: #write header only once
        writer.writerow(header)
      writer.writerow(data1)
      writer.writerow(data2)
  
  # save image objects to file
  with open(objects_path, 'w') as json_file:
    json.dump(training_objects, json_file, indent=2)

  print('###Captions generated!###')

# run caption-generator
generate_captions = False

if generate_captions:
    a2d2_captions(captions_path=a2d2_captions, objects_path=a2d2_objects, dataframe=images_df)


# read captions csv-file
captions_ = pd.read_csv(a2d2_captions)
# add filenames
filenames = images_df['Camera Image']
captions = captions_.join(filenames)
# save to csv
captions_exists = os.path.isfile(a2d2_captions_final)
if not captions_exists:
  captions.to_csv(a2d2_captions_final)

#print(captions)

# get the dataset-distribution
image_objects = a2d2_objects

#iterate over the dataframe and get objects (single-captions)
objects = []
for index, row in image_objects_double.iterrows():
  objects.extend([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]])

# group and count object classes
w = collections.Counter(objects)
v = pd.Series(w.values())
v = v.sort_values(ascending=False)
k = pd.Series(w.keys())
x_labels = list(w.keys())

# bar plot 
plt.figure(figsize=(25,10))
ax = v.plot(kind='bar') 
ax.set_title('Object Class Frequency')
ax.set_xlabel('Object Class')
ax.set_ylabel('Frequency')
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=13)

# save image
#plt.savefig('xxx') #add path if needed
