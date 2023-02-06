import torch
import torchvision
from train import VDDataset, VEDAIDataset
import pandas as pd
import models
from train import execute


device = "cpu"
if torch.cuda.is_available:
  print('All good, a Gpu is available')
  device = torch.device("cuda:0")
else:
  print('Please set GPU via Edit -> Notebook Settings.')

def collate_fn(batch):
    return tuple(zip(*batch))

img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
train_img_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.3)])

data_transforms = {'train': train_img_transforms,
                    'val': img_transform,
                    'test': img_transform}

dataset_path = 'vedai/'
img_path = dataset_path + 'images'
train_labels = dataset_path + 'train_labels.csv'
val_labels = dataset_path + 'val_labels.csv'
test_labels = dataset_path + 'test_labels.csv'
df_train = pd.read_csv(train_labels)
df_val = pd.read_csv(val_labels)
#df_test = pd.read_csv(test_labels)
train_imgs = df_train['file'].unique()
val_imgs = df_val['file'].unique()
#test_imgs = df_test['file'].unique()

data_train = VEDAIDataset(img_dir=img_path,
                            img_files=train_imgs,
                            label_file=train_labels,
                            transforms=data_transforms['train'],
                            training=True)

data_val = VEDAIDataset(img_dir=img_path,
                            img_files=val_imgs,
                            label_file=val_labels,
                            transforms=data_transforms['val'])

# data_test = VDDataset(img_dir=img_path,
#                             img_files=test_imgs,
#                             label_file=test_labels,
#                             transforms=data_transforms['test'])

classes = data_train.classes
num_classes = len(classes) + 1
num_workers = 1
size_batch = 1

loader_train = torch.utils.data.DataLoader(data_train, 
                                              batch_size=size_batch, 
                                              shuffle=True, 
                                              pin_memory=True, 
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

loader_val = torch.utils.data.DataLoader(data_val,
                                            batch_size=size_batch,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn)

# loader_test = torch.utils.data.DataLoader(data_test,
#                                             batch_size=size_batch,
#                                             shuffle=False,
#                                             num_workers=num_workers,
#                                             collate_fn=collate_fn)
                                            

model = models.fasterrcnn(num_classes)
model.to(device)

name_train = "pretrain_retina_resnet50"
lr = 1e-4
num_epochs = 50
tags = ['RetinaNet','resnet50','pretrain','vedai']

execute(name_train, model, lr, num_epochs, loader_train, loader_val, None, device, tags, log_wandb=False)