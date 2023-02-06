import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import os

    
s = json.load(open('dataset/labels.json', 'r'))
train_out_file = 'dataset/train_labels.csv'
train_out = open(train_out_file, 'w')
train_out.write('id,file,x1,y1,x2,y2,label\n')
val_out_file = f'dataset/val_labels.csv'
val_out = open(val_out_file, 'w')
val_out.write('id,file,x1,y1,x2,y2,label\n')
test_out_file = f'dataset/test_labels.csv'
test_out = open(test_out_file, 'w')
test_out.write('id,file,x1,y1,x2,y2,label\n')

img_ids = []
for im in s['images']:
    img_ids.append(im['id'])

# define train, validation and test splits
ids = np.array(img_ids)
train_ids, val_ids = train_test_split(ids, train_size=70)
val_ids, test_ids = train_test_split(val_ids, train_size=10)

categories = {}
num_cat = 1
for cat in s['categories']:
    if cat['id'] != 1338239 and cat['id'] != 1338237 and cat['id'] != 1338234:
        categories[cat['id']] = num_cat
        num_cat += 1

ids_img_ann = {}
for ann in s['annotations']:
    img_id = ann['image_id']
    if img_id not in ids_img_ann.keys():
        ids_img_ann[img_id] = [ann['id'],]
    else:
        ann_ids = ids_img_ann[img_id]
        ann_ids.append(ann['id'])
        ids_img_ann[img_id] = ann_ids

total_anns = 0
for img in s['images']:
    img_id = img['id']
    if img_id in train_ids:
        out = train_out
    elif img_id in val_ids:
        out = val_out
    elif img_id in test_ids:
        out = test_out
    img_file = str(img['file_name']).replace(' ', '_')
    anns = ids_img_ann[img_id]
    total_anns += len(anns)
    for ann_id in anns:
        ann = s['annotations'][ann_id-1]
        x1 = ann['bbox'][0]
        x2 = ann['bbox'][0] + ann['bbox'][2]
        y1 = ann['bbox'][1]
        y2 = ann['bbox'][1] + ann['bbox'][3]
        if ann['category_id'] != 1338239 and ann['category_id'] != 1338237 and ann['category_id'] != 1338234:
            label = categories[ann['category_id']]
            out.write('{},{},{},{},{},{},{}\n'.format(img_id,img_file, x1, y1, x2, y2, label))

# convert the dataset to use with the YOLOv5 framework

for split in ('train', 'val', 'test'):
    df = pd.read_csv('dataset/'+split+'_labels.csv', index_col=0)

    for img_file in df['file'].unique():
        img = Image.open(os.path.join('dataset/images', img_file))
        img.save(os.path.join('yolo_dataset/images', split, img_file))

    for idx, ann in df.iterrows():
        file_name = ann['file'].removesuffix('jpg') + 'txt'
        f = open(os.path.join('yolo_dataset/labels', split, file_name), 'a')
        label = ann['label'] - 1
        img = Image.open(os.path.join('yolo_dataset/images', split, ann['file']))
        img_width = img.size[0]
        img_height = img.size[1]
        x_center = ((ann['x2'] + ann['x1']) // 2) / img_width
        y_center = ((ann['y2'] + ann['y1']) // 2) / img_height
        width = (ann['x2'] - ann['x1']) / img_width
        height = (ann['y2'] - ann['y1']) / img_height
        f.write(f'{label} {x_center} {y_center} {width} {height}\n')
