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

        self.classes = ['Bicycle', 'Motorbike', 'Car-Trailer', 'Car', 'Truck with Trailer', 'Miscellaneous', 'Truck', 'Pickup Truck', 'Van', 'Bus']

    def __getitem__(self, idx):
        path_image = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, path_image)).convert("RGB")
        labels, boxes = get_image_annotations(path_image, self.label_file)

        boxes = torch.as_tensor(boxes.values, dtype=torch.float32)
        labels = torch.as_tensor(labels.values, dtype=torch.int64)
        
        if self.transforms is not None:
            reduced_size = 500
            image = self.transforms(image)
            x, y = get_random_crop(image, reduced_size)
            image = image[:,x:x+reduced_size, y:y+reduced_size]
            filters = [(b[0]>=x and b[0]<x+reduced_size and b[2]>=x and b[2]<x+reduced_size
            and b[1]>=y and b[1]<y+reduced_size and b[3]>=y and b[3]<y+reduced_size) for b in boxes]
            boxes = boxes[filters]
            labels = labels[filters]

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

def training_loop(num_epochs: int,
                  optimizer: torch.optim, 
                  lr_scheduler: torch.optim.lr_scheduler,
                  model: nn.Module, 
                  loader_train: utils.data.DataLoader, 
                  loader_val: utils.data.DataLoader,
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

    print("STARTING TRAINING")
    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        losses_epoch_train = train(model, loader_train, device, 
                                   optimizer, epoch)
        
        losses_epoch_val = evaluate(model, loader_val, device, epoch)

        if log_wandb:
            loss_names = list(losses_epoch_train.keys())
            loss_log = {}
            train_loss = list(losses_epoch_train.values())
            val_loss = list(losses_epoch_val.values())
            for i, name in enumerate(loss_names):
                loss_log['train/'+name] = train_loss[i]
                loss_log['val/'+name] = val_loss[i]

            wandb.log(loss_log)

        time_end = timer()

        lr = optimizer.param_groups[0]['lr']
        
        if verbose:            
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' Losses Train: {losses_epoch_train.items()} ' 
                  f' Losses Val: {losses_epoch_val.items()} '                  
                  f' Time one epoch (s): {(time_end - time_start):.4f} ')
        
        if lr_scheduler:            
            lr_scheduler.step()
    
    loop_end = timer()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}') 

def evaluate_test(model: nn.Module,
          test_loader: utils.data.DataLoader,
          device: torch.device):

    model.eval()
    with torch.no_grad():
        metric = MeanAveragePrecision()
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
        wandb.watch(model)

    fix_random(42)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=starting_lr)    

    # Learning Rate schedule 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    training_loop(num_epochs, optimizer, scheduler, 
                               model, data_loader_train, 
                               data_loader_val, device, log_wandb=log_wandb)
    
    map = evaluate_test(model, data_loader_test, device)
    if log_wandb:
        wandb.log({'test/mAP': map['map'],
                    'test/mAP_small': map['map_small'],
                    'test/mAP_medium': map['map_medium'],
                    'test/mAP_large': map['map_large'],
                    'test/mAP_class': map['map_per_class']})
        run.finish()

    # Save the model    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    path_checkpoint = os.path.join('checkpoints',
                                   f'{name_train}_{num_epochs}_epochs.bin')
    torch.save(model.state_dict(), path_checkpoint)