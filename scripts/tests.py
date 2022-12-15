import torch
import torchvision
from train import VDDataset
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

data_transforms = {'train': torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                    'val': torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                    'test': torchvision.transforms.Compose([torchvision.transforms.ToTensor()])}

dataset_path = 'dataset/'
img_path = dataset_path + 'images'
train_labels = dataset_path + 'train_labels.csv'
val_labels = dataset_path + 'val_labels.csv'
test_labels = dataset_path + 'test_labels.csv'
df_train = pd.read_csv(train_labels)
df_val = pd.read_csv(val_labels)
df_test = pd.read_csv(test_labels)
train_imgs = df_train['file'].unique()
val_imgs = df_val['file'].unique()
test_imgs = df_test['file'].unique()

data_train = VDDataset(img_dir=img_path,
                            img_files=train_imgs,
                            label_file=train_labels,
                            transforms=data_transforms['train'])

data_val = VDDataset(img_dir=img_path,
                            img_files=val_imgs,
                            label_file=val_labels,
                            transforms=data_transforms['val'])

data_test = VDDataset(img_dir=img_path,
                            img_files=test_imgs,
                            label_file=test_labels,
                            transforms=data_transforms['test'])

classes = data_train.classes
num_classes = len(classes)
num_workers = 2
size_batch = 2

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

loader_test = torch.utils.data.DataLoader(data_test,
                                            batch_size=size_batch,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn)
                                            

model = models.retina_net(num_classes)
model.to(device)

name_train = "retina_resnet34"
lr = 1e-4
num_epochs = 1
tags = ['RetinaNet','resnet34']

execute(name_train, model, lr, num_epochs, loader_train, loader_val, loader_test, device, tags, log_wandb=False)