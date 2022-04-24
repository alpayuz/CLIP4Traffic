'''This notebook generates captions for the given TJU-DHD images based on their object-classes (5 classes: Car, Van, Truck, Pedestrian, Cyclist)'''

# import packages
import os
import io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros
from numpy import asarray
from PIL import Image
from torchvision import transforms
import torch
import csv
import cv2 as cv
import json
import itertools
import statistics
import inflect
import fnmatch
import tarfile

from pathlib import Path
from datetime import datetime
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from pprint import pprint

import skimage
import IPython.display
from collections import OrderedDict
import torchvision.transforms.functional as F

# Set date
current_datetime = datetime.now().strftime('%Y%m%d')
current_datetime = str(current_datetime)


# PATHS (ADD PATHS!)
TJU_images_path = [xxx]
TJU_annotations_path = [xxx] #JSON-file that's include in the dataset-folder
TJU_captions_path = [PATH_TO_SAVE_CAPTIONS.csv]


# Load annotations file (JSON) which contains the bounding boxes
with open(TJU_annotations_path, "r") as file:
    json_object = json.load(file)

# Show the keys in the first levels of the JSON
for key in json_object.keys():
  print(key)

x = json_object['categories'][0].keys()
y = json_object['annotations'][0].keys()
z = json_object['images'][0].keys()

print(f'categories: {x}')
print(f'annotations: {y}')
print(f'images: {z}')


# extract bounding boxes and generate image-captions
def extract_boxes(json_object, image_id):
    '''we count the occurence for each object-class on the corresponding image (if 0, it's ignored)
       in a next step, we also compare the size of the objects per class and therefore compare their distances
       this enables the generation of a variety of captions: "basic_caption", "clean_caption", "advanced_caption"
    '''
  
  # initiate lists for each category/class
  #pedestrians
  bbox_peds = list()
  near_peds = list()
  far_peds = list()

  #cyclists
  bbox_cyc = list()
  near_cyc = list()
  far_cyc = list()

  #cars
  bbox_cars = list()
  near_cars = list()
  far_cars = list()

  #trucks
  bbox_trucks = list()
  near_trucks = list()
  far_trucks = list()

  #vans
  bbox_vans = list()
  near_vans = list()
  far_vans = list()

  # load and parse the file
  annotations = json_object['annotations']
  categories = json_object['categories']
  images = json_object['images']

  boxes = list()
  categories = list()
  
  # initialize the counters for each category/class
  pedestrians = 0
  cyclists = 0
  cars = 0
  trucks = 0
  vans = 0

  # extract image dimensions:
  d = [d for i, d in enumerate(annotations) if image_id == d['image_id']]
  for object in d:
    if object['category_id'] >= 1: 
        #extract each bounding box
        xmin = object['bbox'][0] 
        ymin = object['bbox'][1]
        xmax = object['bbox'][0] + object['bbox'][2] 
        ymax = object['bbox'][1] + object['bbox'][3]
        coors = [xmin, ymin, xmax, ymax]
        boxes.append(coors)
        categories.append(object['category_id'])

        #get dimensions of bboxes
        width_bbox = xmax - xmin
        height_bbox = ymax - ymin
        area_bbox = width_bbox * height_bbox

    # count the occurance of each category/class & save their area
    if object['category_id'] == 1:
      #pedestrians = len(str(categories.count(1)))
      pedestrians += 1
      area_bbox_ped = area_bbox 
      bbox_peds.append(area_bbox_ped)
    elif object['category_id'] == 2:
      #cyclists = len(str(categories.count(2)))
      cyclists += 1
      area_bbox_cyc = area_bbox
      bbox_cyc.append(area_bbox_cyc)
    elif object['category_id'] == 3:
      #cars = len(str(categories.count(3)))
      cars += 1
      area_bbox_car = area_bbox
      bbox_cars.append(area_bbox_car)
    elif object['category_id'] == 4:
      #trucks = len(str(categories.count(4)))
      trucks += 1
      area_bbox_truck = area_bbox
      bbox_trucks.append(area_bbox_truck)
    elif object['category_id'] == 5:
      #vans = len(str(categories.count(5)))
      vans += 1
      area_bbox_van = area_bbox
      bbox_vans.append(area_bbox_van)

  # grab the image dimensions
  d = [d for i, d in enumerate(json_object['images']) if image_id in d.values()]
  width = d[0]['width']
  height = d[0]['height']

  # get the corresponding image filename
  d = [d for i, d in enumerate(images) if image_id in d.values()]
  image_filename = d[0]['file_name']

  # compare the areas inside each category to differentiate between far objects (small area of bbox) and near objects (big area of bbox)
  def distances(bbox_list, near_list, far_list, counter):
      '''if an object class occures more then once, we can compare the distance of each object to the camera based on their size
         each object-class has therefore a "near" and "far" list if occurence > 1
      '''
    if len(bbox_list) > 1:
      for a in bbox_list:
        if a >= (sum(bbox_list)/len(bbox_list)):
          near_list.append(a)
        else:
          far_list.append(a)

  '''
  # compare distances of objects and store in corresponding list
  def distances2(bbox_list, near_list, far_list, counter):
    if len(bbox_list) > 1:
      for a, b in itertools.combinations(bbox_list, 2):
      #docu: https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement 
        if a > b:
          near_list.append(a)
          far_list.append(b)
        #elif a < b:
        else:
          near_list.append(b)
          far_list.append(a)
      return a, b
    else: 
      counter += 1

  def distances3(bbox_list, near_list, far_list, counter):
    if len(bbox_list) > 1:
      for i in range(len(bbox_list)):
        for j in range(i+1, len(bbox_list)):
          if i > j:
            near_list.append(i)
            print(" i = near")
            far_list.append(j)
            print("j = far")
            return i, j
          elif i < j:
            near_list.append(j)
            print("j = near")
            far_list.append(i)
            print("i = far")
            return i, j
    else: 
      counter += 1
  '''

  # apply distance measuring for each category/class
  #pedestrians
  distances(bbox_list=bbox_peds, near_list=near_peds, far_list=far_peds, counter=pedestrians)
  #cyclists
  distances(bbox_list=bbox_cyc, near_list=near_cyc, far_list=far_cyc, counter=cyclists)
  #cars
  distances(bbox_list=bbox_cars, near_list=near_cars, far_list=far_cars, counter=cars)
  #trucks
  distances(bbox_list=bbox_trucks, near_list=near_trucks, far_list=far_trucks, counter=trucks)
  #vans
  distances(bbox_list=bbox_vans, near_list=near_vans, far_list=far_vans, counter=vans)


  # generate caption sentence
  p = inflect.engine()

  # basic caption: count occuring objects 
  basic_caption = str(f"On the image are {p.number_to_words(pedestrians)} pedestrian, {p.number_to_words(cyclists)} cyclist, {p.number_to_words(cars)} car, {p.number_to_words(trucks)} truck and {p.number_to_words(vans)} van")

  # function to compare near/far objects
  def near_far(near_list, far_list):
      '''after getting the near and far objects per class, we put them in a caption'''
    if len(near_list) and len(far_list) >= 1:
      caption = str(f"from which {p.number_to_words(len(near_list))} close and {p.number_to_words(len(far_list))} far off")
    else:
      caption = str("")
    return caption

  caption_peds = near_far(near_list=near_peds, far_list=far_peds)
  caption_cyc = near_far(near_list=near_cyc, far_list=far_cyc)
  caption_cars = near_far(near_list=near_cars, far_list=far_cars)
  caption_trucks = near_far(near_list=near_trucks, far_list=far_trucks)
  caption_vans = near_far(near_list=near_vans, far_list=far_vans)

  # advanced object: use the size of the objects per class to compare their distance
  advanced_caption = str(f"On the image are {p.number_to_words(pedestrians)} pedestrian {caption_peds}, {p.number_to_words(cyclists)} cyclist {caption_cyc}, {p.number_to_words(cars)} car {caption_cars}, {p.number_to_words(trucks)} truck {caption_trucks} and {p.number_to_words(vans)} van {caption_vans}.")
  advanced_caption = advanced_caption.replace(' ,',',').replace(' .', '.').replace('  and', ' and')

  # cleaned caption: only include occuring objects
  def cleaned(object_list, string):
      '''the 'clean caption' only lists occuring objects'''
    if len(object_list) >= 1:
      clean_cap = str(p.number_to_words(len(object_list))) + str(" ") + str(string)
    else:
      clean_cap = str("")
    return clean_cap

  clean_peds = cleaned(object_list=bbox_peds, string='pedestrian')
  clean_cyc = cleaned(object_list=bbox_cyc, string='cyclist')
  clean_cars = cleaned(object_list=bbox_cars, string='car')
  clean_trucks = cleaned(object_list=bbox_trucks, string='truck')
  clean_vans = cleaned(object_list=bbox_vans, string='van')

  cleaned_caption = str(f"There is {clean_peds}, {clean_cyc}, {clean_cars}, {clean_trucks}, {clean_vans}").replace(' ,', '')
  # check if last character is a ',' and remove it
  if cleaned_caption.endswith(', '):
    clean_caption = cleaned_caption[:-len(', ')]
  else:
    clean_caption = cleaned_caption


  ##PRINTS##
  #print(f'bbox cars: {bbox_cars}')
  print(f"image_id: {image_id}, 'filename:' {image_filename}, 'boxes:'{boxes}, 'categories:' {categories}, 'width:'{width}, 'height:'{height}")
  print('#####CAPTION#####')
  print(f'Basic: {basic_caption}')
  print(f'Advanced: {advanced_caption}')
  print(f'Cleaned: {clean_caption}')

  #return boxes, categories, width, height
  #return image_id, basic_caption, advanced_caption
  #return pedestrians, cyclists, cars, trucks, vans
  
  # save output caption and image_id to csv
  filename = TJU_captions_path #+ current_datetime
  file_exists = os.path.isfile(filename) #check if file already exists
  header = ['image_id', 'filename', 'basic caption', 'advanced caption', 'clean caption']
  data = [image_id, image_filename, basic_caption, advanced_caption, clean_caption]
  with open(filename, 'a') as outfile:
    writer = csv.writer(outfile)
    if not file_exists: #write header only once
      writer.writerow(header)
    writer.writerow(data)

    print('***SAVED TO CSV***')

# apply extract_boxes() to the TJU-dataset
generate_captions = False

if generate_captions:
    for i in range(0, len(os.listdir(TJU_images_path))):
        extract_boxes(json_object=json_object, image_id=i)



# load captions dataframe
tju_captions = pd.read_csv(TJU_captions_path)


# match captions and images
def match(image_id, plot):
  original_images = [] #not in unse
  images = []          #not in use
  '''
  given an image_id, it will find the corresponding filename (.jpg),
  open the image as PILImage and get the corresponding captions
  '''

  # split the captions dataframe into subframes per caption-type
  advanced_captions = tju_captions[['advanced caption', 'filename']]
  basic_captions = tju_captions[['basic caption', 'filename']]
  clean_captions = tju_captions[['clean caption', 'filename']]

  basic = list()
  advanced = list()
  clean = list()
  global filename

  filename = tju_captions.loc[tju_captions['image_id'] == image_id, 'filename'].item()
  if filename in images_train:
    path = TJU_images_path + '/' + filename
    image = Image.open(path).convert('RGB')
    original_images.append(image)

    #global basic_caption
    basic_caption = basic_captions.loc[basic_captions['filename'] == filename]
    basic_caption = basic_caption['basic caption'].item()
    basic.append(basic_caption)

    #global advanced_caption
    advanced_caption = advanced_captions.loc[advanced_captions['filename'] == filename]
    advanced_caption = advanced_caption['advanced caption'].item()
    advanced.append(advanced_caption)

    #global clean_caption
    clean_caption = clean_captions.loc[clean_captions['filename'] == filename]
    clean_caption = clean_caption['clean caption'].item()
    clean.append(clean_caption)

    #print(f'basic: {basic_caption}')
    #print(f'advanced: {advanced_caption}')
    #print(f'clean: {clean_caption}')

  #plot the image & corresponding captions
  if plot:
    plt.figure(figsize=(10, 8))
    plt.imshow(image, aspect='auto')
    plt.title('Basic:' + basic_caption + '\n' + 'Advanced:' + advanced_caption + '\n' + 'Clean:' + clean_caption)

  else:  
    return basic_caption, advanced_caption, clean_caption, original_images, images, image

# test match() function
test_match = False

if test_match:
    match(image_id=8000, plot=False)