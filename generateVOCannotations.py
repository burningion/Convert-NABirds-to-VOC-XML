import os
import sys

from matplotlib import pyplot as plt
import random
import math

from lxml import etree, objectify
import shutil

try:
  import networkx as nx
  has_networkx = True
except:
  has_networkx = False

# set this to -1 to generate all examples
numberExamples = 3

def load_bounding_box_annotations(dataset_path=''):
  
  bboxes = {}
  
  with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      bbox = map(int, pieces[1:])
      bboxes[image_id] = bbox
  
  return bboxes

def load_part_annotations(dataset_path=''):
  
  parts = {}
  
  with open(os.path.join(dataset_path, 'parts/part_locs.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      parts.setdefault(image_id, [0] * 11)
      part_id = int(pieces[1])
      parts[image_id][part_id] = map(int, pieces[2:])

  return parts  
  
def load_part_names(dataset_path=''):
  
  names = {}

  with open(os.path.join(dataset_path, 'parts/parts.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      part_id = int(pieces[0])
      names[part_id] = ' '.join(pieces[1:])
  
  return names  
    
def load_class_names(dataset_path=''):
  
  names = {}
  
  with open(os.path.join(dataset_path, 'classes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      class_id = pieces[0]
      names[class_id] = ' '.join(pieces[1:])
  
  return names

def load_image_labels(dataset_path=''):
  labels = {}
  
  with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      class_id = pieces[1]
      labels[image_id] = class_id
  
  return labels
        
def load_image_paths(dataset_path='', path_prefix=''):
  
  paths = {}
  
  with open(os.path.join(dataset_path, 'images.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      path = os.path.join(path_prefix, pieces[1])
      paths[image_id] = path
  
  return paths

def load_image_sizes(dataset_path=''):
  
  sizes = {}
  
  with open(os.path.join(dataset_path, 'sizes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      width, height = map(int, pieces[1:])
      sizes[image_id] = [width, height]
  
  return sizes

def load_hierarchy(dataset_path=''):
  
  parents = {}
  
  with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      child_id, parent_id = pieces
      parents[child_id] = parent_id
  
  return parents

def load_photographers(dataset_path=''):
  
  photographers = {}
  with open(os.path.join(dataset_path, 'photographers.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      photographers[image_id] = ' '.join(pieces[1:])
  
  return photographers

def load_train_test_split(dataset_path=''):
  train_images = []
  test_images = []
  
  with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      is_train = int(pieces[1])
      if is_train:
        train_images.append(image_id)
      else:
        test_images.append(image_id)
        
  return train_images, test_images 
      
if __name__ == '__main__':
  
  if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
  else:
    dataset_path = ''
  
  if len(sys.argv) > 2:
    image_path = sys.argv[2]
  else:
    image_path  = 'images'
  
  # Load in the image data
  # Assumes that the images have been extracted into a directory called "images"
  image_paths = load_image_paths(dataset_path, path_prefix=image_path)
  image_sizes = load_image_sizes(dataset_path)
  image_bboxes = load_bounding_box_annotations(dataset_path)
  image_parts = load_part_annotations(dataset_path)
  image_class_labels = load_image_labels(dataset_path)
  
  # Load in the class data
  class_names = load_class_names(dataset_path)
  class_hierarchy = load_hierarchy(dataset_path)
  
  # Load in the part data
  part_names = load_part_names(dataset_path)
  part_ids = part_names.keys()
  part_ids.sort() 
  
  # Load in the photographers
  photographers = load_photographers(dataset_path)
  
  # Load in the train / test split
  train_images, test_images = load_train_test_split(dataset_path)

  counter = 1
  if not os.path.exists('Annotations'):
    os.makedirs('Annotations')

  if not os.path.exists('VOCImages'):
    os.makedirs('VOCImages')

  
  # Visualize the images and their annotations
  image_identifiers = image_paths.keys()
  random.shuffle(image_identifiers) 
  for image_id in image_identifiers:
    if numberExamples > 0 and counter == numberExamples + 1:
      exit()
    
    image_path = image_paths[image_id]
    #image = plt.imread(image_path)
    bbox = image_bboxes[image_id]
    parts = image_parts[image_id]
    class_label = image_class_labels[image_id]
    class_name = class_names[class_label]
    bbox_x, bbox_y, bbox_width, bbox_height = bbox

    # first, copy image to new flat VOCImages directory
    shutil.copyfile(image_path, 'VOCImages/' + os.path.split(image_path)[1])

    # next, build labeled VOC xml file 
    annotation = etree.Element("annotation")
    etree.SubElement(annotation, "folder").text = os.path.dirname(image_path)
    etree.SubElement(annotation, "filename").text = os.path.split(image_path)[1]

    source = etree.SubElement(annotation, "source")
    etree.SubElement(source, "database").text = "NABirds V1"
    etree.SubElement(source, "annotation").text = "NABirds V1"
    etree.SubElement(source, "image").text = "NABirds"

    owner = etree.SubElement(annotation, "owner")
    etree.SubElement(owner, "name").text = photographers[image_id]

    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = str(image_sizes[image_id][0])
    etree.SubElement(size, "height").text = str(image_sizes[image_id][1])
    etree.SubElement(size, "depth").text = '3'

    objectN = etree.SubElement(annotation, "object")
    etree.SubElement(objectN, "name").text = class_name
    etree.SubElement(objectN, "pose").text = "Unspecified"
    etree.SubElement(objectN, "truncated").text = '0'
    etree.SubElement(objectN, "difficult").text = '0'

    bndBox = etree.SubElement(objectN, "bndbox")
    etree.SubElement(bndBox, "xmin").text = str(bbox_x)
    etree.SubElement(bndBox, "ymin").text = str(bbox_y)
    etree.SubElement(bndBox, "xmax").text = str(bbox_x + bbox_width)
    etree.SubElement(bndBox, "ymax").text = str(bbox_y + bbox_height)

    et = etree.ElementTree(annotation)
    et.write('Annotations/%06d.xml' % counter, pretty_print=True)
    counter += 1
    print class_name
