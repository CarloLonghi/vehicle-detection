import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import csv
import shutil

ann_dir = './vedai/annotations'
img_dir = './vedai/images'

train_annotations = pd.DataFrame(columns=['file','x1','y1','x2','y2','label'])
val_annotations = pd.DataFrame(columns=['file','x1','y1','x2','y2','label'])

image_files = os.listdir(img_dir)
train_dim = int(len(image_files) * 0.7)
train_files = image_files[:train_dim]
val_files = image_files[train_dim:]

for file in os.listdir(ann_dir):
    with open(os.path.join(ann_dir, file)) as f:
        reader = csv.DictReader(f, fieldnames=("label", "cx", "cy", "width", "height"), delimiter=" ")

        img_file = os.path.splitext(file)[0]+'.jpg'
        
        img = Image.open(os.path.join(img_dir, img_file))
        width, height = img.size

        for r in reader:
            cx, cy, w, h = float(r["cx"]), float(r["cy"]), float(r["width"]), float(r["height"])

            x_min = max(cx*width - w*width/2, 0)
            y_min = max(cy*height - h*height/2, 0)
            x_max = min(cx*width + w*width/2, width)
            y_max = min(cy*height + h*height/2, height)
            label = int(r["label"]) + 1
            if img_file in train_files:
                train_annotations.loc[len(train_annotations.index)] = [img_file, x_min, y_min, x_max, y_max, label]
            else:
                val_annotations.loc[len(val_annotations.index)] = [img_file, x_min, y_min, x_max, y_max, label]

train_annotations.to_csv('./vedai/train_labels.csv')
val_annotations.to_csv('./vedai/val_labels.csv')

# convert dataset to use with yolo
if not os.path.exists('yolo_vedai'):
    os.makedirs('yolo_vedai')
    for dir in ('images', 'labels'):
        os.makedirs(os.path.join('yolo_vedai', dir))
        for split in ('train', 'val'):
            os.makedirs(os.path.join('yolo_vedai', dir, split))

for split in ('train', 'val'):
    df = pd.read_csv('vedai/'+split+'_labels.csv', index_col=0)

    for img_file in df['file'].unique():
        shutil.copy(os.path.join('vedai/images', img_file), os.path.join('yolo_vedai/images', split, img_file))
        ann_file = img_file.removesuffix('jpg') + 'txt'
        shutil.copy(os.path.join('vedai/annotations', ann_file), os.path.join('yolo_vedai/labels', split, ann_file))
