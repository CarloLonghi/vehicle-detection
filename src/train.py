import torch
import numpy as np
import random
import pandas as pd
import torchvision
from PIL import Image
from typing import List, Dict, Tuple
import os
import torch.nn as nn
import torch.utils as utils
from timeit import default_timer as timer
import torch.optim as optim
from torch.optim import lr_scheduler
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use. 
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

from torch.utils.data import Dataset

def get_image_annotations(img_file: str, label_file: str):
    df = pd.read_csv(label_file)
    df_img = df[df['file'] == img_file]
    return df_img['label'], df_img[['x1','y1','x2','y2']]

def get_random_crop(img: torch.tensor, size: int):
    _, h, w = img.shape
    x = torch.randint(low=0, high=h-size, size=(1,)).item()
    y = torch.randint(low=0, high=w-size, size=(1,)).item()
    return x, y

class VDDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        img_files: List[str],
        label_file: str,
        transforms: torchvision.transforms = None,
        training: bool = False,
    ) -> None:
        """Init the dataset

        Args:
            path_images: the path to the folder containing the images.
            ext_images: the extension of the images.
            ext_annotations: the extension of the annotations.
            transforms: the transformation to apply to the dataset.
        """
        self.img_dir = img_dir
        self.images = sorted(img_files)
        self.label_file = label_file
        self.transforms = transforms
        self.training = training

        self.classes = ['Bicycle', 'Motorbike', 'Car-Trailer', 'Car', 'Truck with Trailer', 'Miscellaneous', 'Truck', 'Pickup Truck', 'Van', 'Bus']

    def __getitem__(self, idx):
        img_idx = idx // 4
        crop_idx = idx % 4
        path_image = self.images[img_idx]
        image = Image.open(os.path.join(self.img_dir, path_image)).convert("RGB")
        labels, boxes = get_image_annotations(path_image, self.label_file)

        boxes = torch.as_tensor(boxes.values, dtype=torch.float32)
        labels = torch.as_tensor(labels.values, dtype=torch.int64)
        
        if self.transforms is not None:
            image = self.transforms(image)
            starting_x = image.shape[2] // 2 * (crop_idx % 2)
            starting_y = image.shape[1] // 2 * (crop_idx // 2)
            dim_x = image.shape[2] // 2 + 200
            dim_y = image.shape[1] // 2 + 200
            if crop_idx % 2 == 1:
                starting_x -= 200
            if crop_idx // 2 == 1:
                starting_y -= 200
            image = image[:, starting_y:starting_y+dim_y, starting_x:starting_x+dim_x]
            filters = [(b[0]>=starting_x and b[0]<starting_x+dim_x and b[2]>=starting_x and b[2]<starting_x+dim_x
            and b[1]>=starting_y and b[1]<starting_y+dim_y and b[3]>=starting_y and b[3]<starting_y+dim_y) for b in boxes]
            boxes = boxes[filters]
            x1 = boxes[:,0] - starting_x
            x2 = boxes[:,2] - starting_x
            y1 = boxes[:,1] - starting_y
            y2 = boxes[:,3] - starting_y
            boxes = torch.stack((x1,y1,x2,y2), dim=1)
            labels = labels[filters]

            if self.training:
                # random flip
                p_h = torch.rand(1)
                if p_h > 0.5:
                    image = torch.flip(image, dims=[1,])
                    y3 = boxes[:,3].clone()
                    boxes[:,3] = dim_y - boxes[:,1]
                    boxes[:,1] = dim_y - y3
                p_v = torch.rand(1)
                if p_v > 0.5:
                    image = torch.flip(image, dims=[2,])
                    x2 = boxes[:,2].clone()
                    boxes[:,2] = dim_x - boxes[:,0]
                    boxes[:,0] = dim_x - x2

        target = {"boxes": boxes, "labels": labels}

        return image, target

    def __len__(self):
        return len(self.images) * 4

class VEDAIDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        img_files: List[str],
        label_file: str,
        transforms: torchvision.transforms = None,
        training: bool = False,
    ) -> None:
        """Init the dataset

        Args:
            path_images: the path to the folder containing the images.
            ext_images: the extension of the images.
            ext_annotations: the extension of the annotations.
            transforms: the transformation to apply to the dataset.
        """
        self.img_dir = img_dir
        self.images = sorted(img_files)
        self.label_file = label_file
        self.transforms = transforms
        self.training = training
        self.classes = ['car', 'truck', 'pickup', 'tractor', 'camping car', 'boat', 'motorcycle', 'bus', 'van', 'other', 'small', 'large']


    def __getitem__(self, idx):
        path_image = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, path_image)).convert("RGB")
        labels, boxes = get_image_annotations(path_image, self.label_file)

        boxes = torch.as_tensor(boxes.values, dtype=torch.float32)
        labels = torch.as_tensor(labels.values, dtype=torch.int64)
        
        if self.transforms is not None:
            image = self.transforms(image)
            dim_x = image.shape[2]
            dim_y = image.shape[1]

            if self.training:
                # random flip
                p_h = torch.rand(1)
                if p_h > 0.5:
                    image = torch.flip(image, dims=[1,])
                    y3 = boxes[:,3].clone()
                    boxes[:,3] = dim_y - boxes[:,1]
                    boxes[:,1] = dim_y - y3
                p_v = torch.rand(1)
                if p_v > 0.5:
                    image = torch.flip(image, dims=[2,])
                    x2 = boxes[:,2].clone()
                    boxes[:,2] = dim_x - boxes[:,0]
                    boxes[:,0] = dim_x - x2


        target = {"boxes": boxes, "labels": labels}

        return image, target

    def __len__(self):
        return len(self.images)


def train(model: nn.Module,
          train_loader: utils.data.DataLoader,
          device: torch.device,          
          optimizer: torch.optim,          
          epoch: int) -> Dict[str, float]:
    """Trains a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        device: the device to use to train the model.        
        optimizer: the optimizer to use to train the model.        
        epoch: the number of the current epoch.

    Returns: 
        A dictionary containing:
            the sum of classification and regression loss.
            the classification loss.
            the regression loss.
    """        
    size_ds_train = len(train_loader.dataset)
    num_batches = len(train_loader)    

    losses = []
    loss_names = []
    model.train()
    for idx_batch, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        if idx_batch == 0:
            losses = [0.0] * len(loss_dict)
            loss_names = list(loss_dict.keys())
        
        loss_sum = 0
        for i, loss in enumerate(loss_dict.values()):
            loss_sum += loss
            losses[i] += loss
        loss_sum.backward()
        optimizer.step()

    dict_losses_train = {}
    for idx, name in enumerate(loss_names):
        dict_losses_train[name] = losses[idx]/num_batches

    return dict_losses_train

def evaluate(model: nn.Module,
          val_loader: utils.data.DataLoader,
          device: torch.device,          
          epoch: int) -> Dict[str, float]:

    num_batches = len(val_loader)

    losses = []
    loss_names = []
    model.train()
    with torch.no_grad():
        for idx_batch, (images, targets) in enumerate(val_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            if idx_batch == 0:
                losses = [0.0] * len(loss_dict)
                loss_names = list(loss_dict.keys())
            loss_sum = 0
            for i, loss in enumerate(loss_dict.values()):
                loss_sum += loss
                losses[i] += loss

    dict_losses_val = {}
    for idx, name in enumerate(loss_names):
        dict_losses_val[name] = losses[idx]/num_batches
    return dict_losses_val

def training_loop(name_train: str,
                  num_epochs: int,
                  optimizer: torch.optim, 
                  lr_scheduler: torch.optim.lr_scheduler,
                  model: nn.Module, 
                  loader_train: utils.data.DataLoader, 
                  loader_val: utils.data.DataLoader,
                  loader_test: utils.data.DataLoader,
                  device, 
                  verbose: bool=True,
                  log_wandb: bool=False,) -> Dict[str, List[float]]:
    """Executes the training loop.
    
        Args:
            num_epochs: the number of epochs.
            optimizer: the optimizer to use.
            lr_scheduler: the scheduler for the learning rate.
            model: the model to train.
            loader_train: the data loader containing the training data.
            loader_val: the data loader containing the validation data.
            verbose: if true print the value of loss.

        Returns:  
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch.
            the values for the train accuracy for each epoch.            
            the time of execution in seconds for the entire loop.
            the model trained 
    """    
    loop_start = timer()
    best_map = None
    increasing_loss = 0
    print("STARTING TRAINING")
    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        losses_epoch_train = train(model, loader_train, device, 
                                   optimizer, epoch)
        
        losses_epoch_val = evaluate(model, loader_val, device, epoch)

        train_loss = list(losses_epoch_train.values())
        val_loss = list(losses_epoch_val.values())
        map_test = evaluate_test(model, loader_val, device)

        if log_wandb:
            loss_names = list(losses_epoch_train.keys())
            loss_log = {}
            for i, name in enumerate(loss_names):
                loss_log['train/'+name] = train_loss[i]
                loss_log['val/'+name] = val_loss[i]

            loss_log['val/mAP'] = map_test['map']
            loss_log['val/mAP_small'] = map_test['map_small']
            loss_log['val/mAP_medium'] = map_test['map_medium']
            loss_log['val/mAP_large'] = map_test['map_large']
            loss_log['val/mAR_small'] = map_test['mar_small']
            loss_log['val/mAR_medium'] = map_test['mar_medium']
            loss_log['val/mAR_large'] = map_test['mar_large']
            loss_log['val/mAP_50'] = map_test['map_50']
            map_class = map_test['map_per_class']
            for i, m in enumerate(map_class):
              loss_log['val/mAP_class_'+str(i)] = m

            wandb.log(loss_log)

        total_map = map_test['map']
        if best_map is None:
            best_map = total_map
            if not os.path.exists('drive/MyDrive/vehicle_detection/checkpoints'):
                os.makedirs('drive/MyDrive/vehicle_detection/checkpoints')
            path_checkpoint = os.path.join('drive/MyDrive/vehicle_detection/checkpoints', f'{name_train}_checkpoint.bin')
            torch.save(model.state_dict(), path_checkpoint)
        if total_map > best_map:
            increasing_loss = 0
            best_map = total_map
            # Save model checkpoint    
            path_checkpoint = os.path.join('drive/MyDrive/vehicle_detection/checkpoints', f'{name_train}_checkpoint.bin')
            torch.save(model.state_dict(), path_checkpoint)
        else:
            increasing_loss += 1
        
        time_end = timer()

        lr = optimizer.param_groups[0]['lr']
        
        if verbose:            
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' Losses Train: {losses_epoch_train.items()} ' 
                  f' Losses Val: {losses_epoch_val.items()} '                  
                  f' Time one epoch (s): {(time_end - time_start):.4f} ')
    
        if increasing_loss >= 20:
            print('Early Stopping')
            break

    if loader_test is not None:
      model.load_state_dict(torch.load(os.path.join('drive/MyDrive/vehicle_detection/checkpoints', f'{name_train}_checkpoint.bin')))
      map_test = evaluate_test(model, loader_test, device)
      print(map_test['map_per_class'])
      if log_wandb:
          map_log = {}
          map_log['test/mAP'] = map_test['map']
          map_log['test/mAP_small'] = map_test['map_small']
          map_log['test/mAP_medium'] = map_test['map_medium']
          map_log['test/mAP_large'] = map_test['map_large']
          map_log['test/mAR_small'] = map_test['mar_small']
          map_log['test/mAR_medium'] = map_test['mar_medium']
          map_log['test/mAR_large'] = map_test['mar_large']
          map_log['test/mAP_50'] = map_test['map_50']
          map_class = map_test['map_per_class']
          for i, m in enumerate(map_class):
            map_log['test/mAP_class_'+str(i)] = m
          wandb.log(map_log)

    loop_end = timer()
    time_loop = loop_end - loop_start

    if verbose:
        print(f'Time for {epoch} epochs (s): {(time_loop):.3f}') 

def evaluate_test(model: nn.Module,
          test_loader: utils.data.DataLoader,
          device: torch.device):

    model.eval()
    with torch.no_grad():
        metric = MeanAveragePrecision(class_metrics=True)
        for idx_bathc, (images,targets) in enumerate(test_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)
            metric.update(predictions, targets)
        
        map = metric.compute()
        return map

def execute(name_train: str, 
            model: nn.Module, 
            starting_lr: float, 
            num_epochs: int, 
            data_loader_train: torch.utils.data.DataLoader,
            data_loader_val: torch.utils.data.DataLoader,
            data_loader_test: torch.utils.data.DataLoader,
            device,
            tags: List,
            log_wandb: bool = False,) -> None:
    """Executes the training loop.

    Args:
        name_train: the name for the log subfolder.
        model: the model to train.
        starting_lr: the staring learning rate.
        num_epochs: the number of epochs.
        data_loader_train: the data loader with training data.
        data_loader_val: the data loader with validation data.
    """
    if log_wandb:
        run = wandb.init(project="vehicle-detectin", tags=tags, config={
            'learning_rate': starting_lr, 'epoch': num_epochs, 'device': device})
        wandb.define_metric('test/mAP', summary='max')
        wandb.define_metric('test/mAP_small', summary='max')
        wandb.define_metric('test/mAP_medium', summary='max')
        wandb.define_metric('test/mAP_large', summary='max')
        wandb.define_metric('test/mAP_class', summary='max')
        wandb.define_metric('test/mAP_50', summary='max')
        wandb.watch(model)

    fix_random(42)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=starting_lr)    

    # Learning Rate schedule 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    training_loop(name_train, num_epochs, optimizer, scheduler, 
                               model, data_loader_train, data_loader_val, 
                               data_loader_test, device, log_wandb=log_wandb)
                               
    if log_wandb:
        run.finish()