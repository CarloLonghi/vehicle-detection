import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split

dataDir='./dataset/images'
annFile='./dataset/labels.json'

def convert_coco_json_to_csv(filename):
        
    s = json.load(open(filename, 'r'))
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
            label = categories[ann['category_id']]
            out.write('{},{},{},{},{},{},{}\n'.format(img_id,img_file, x1, y1, x2, y2, label))

    print(total_anns)


convert_coco_json_to_csv('dataset/labels.json')

# Initialize the COCO api for instance annotations
# coco=COCO(annFile)

# def getClassName(classID, cats):
#     for i in range(len(cats)):
#         if cats[i]['id']==classID:
#             return cats[i]['name']
#     return "None"

# # Load the categories in a variable
# catIDs = coco.getCatIds()
# cats = coco.loadCats(catIDs)
# cat_names = [cat['name'] for cat in cats]
# print(catIDs)

# ########## ALl POSSIBLE COMBINATIONS ########
# classes = cat_names

# images = []
# if classes!=None:
#     # iterate for each individual class in the list
#     for className in classes:
#         # get all images containing given class
#         catIds = coco.getCatIds(catNms=className)
#         imgIds = coco.getImgIds(catIds=catIds)
#         images += coco.loadImgs(imgIds)
# else:
#     imgIds = coco.getImgIds()
#     images = coco.loadImgs(imgIds)
    
# # Now, filter out the repeated images    
# unique_images = []
# for i in range(len(images)):
#     if images[i] not in unique_images:
#         unique_images.append(images[i])

# dataset_size = len(unique_images)

# print("Number of images containing the filter classes:", dataset_size)

# # load and display a random image
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# I = io.imread('{}/{}'.format(dataDir,img['file_name']))/255.0

# # Load and display instance annotations
# plt.imshow(I)
# plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIDs, iscrowd=None)
# print(len(annIds))
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)

# plt.show()