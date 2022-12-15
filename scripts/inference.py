import torch
from typing import List, Tuple
from PIL import Image, ImageFont, ImageDraw
import copy
from torch import nn
import numpy as np
import models
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import visualization
from matplotlib import pyplot as plt
from train import VDDataset
import pandas as pd

def detect_objects(image: Image,
                   detector: nn.Module, 
                   threshold: float, 
                   categories: List[str]) -> Tuple[List[List[int]], 
                                                   List[float], 
                                                   List[str], 
                                                   List[int]]:
    """Detects objects in the image using the provided detector. 
    This function puts the model in the eval mode.

    Args:
        image: the input image.
        detector: the object detector.
        threshold: the detection confidence score, any detections with scores below this value are discarded.
        categories: the names of the categories of the data set used to train the network.

    Returns:
        The bounding boxes of the predicted objects using the [x_min, y_min, x_max, y_max] format, 
        with values between 0 and image height and 0 and image width.
        The scores of the predicted objects.
        The categories of the predicted objects.
    """
    detector.eval()
    with torch.no_grad():
        predictions = detector([image])
        predictions = predictions[0]
    
    # Get scores, boxes, and labels
    scores = predictions['scores'].detach().cpu().numpy()
    boxes = predictions['boxes'].detach().cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    # Get all the boxes above the threshold
    mask = scores >= threshold
    boxes_filtered = boxes[mask].astype(np.int32)
    
    # Get the names of the categories above the threshold
    indices_filtered = [idx for idx, score in enumerate(list(scores)) if score >= threshold]      
    categories_filtered = [categories[labels[i]] for i in indices_filtered]
    
    # Get only the scores above the threshold
    labels_filtered = labels[mask]
    scores_filtered = scores[mask]

    return boxes_filtered, scores_filtered, categories_filtered, labels_filtered

model = models.retina_net(10)
model.load_state_dict(torch.load('checkpoints/retina_resnet34_10_epochs.bin'))

# image = Image.open('dataset/images/Screenshot_2022-09-02_182050.jpg').convert("RGB")
# totensor = torchvision.transforms.ToTensor()
# image = totensor(image)
# image = image[:,:800,:800]

img_path = 'dataset/images'
df = pd.read_csv('dataset/labels.csv')
img_files = df['file'].unique()
train_dataset, test_dataset = torch.utils.data.random_split(img_files, [70, 30])
val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [10, 20])
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
data_test = VDDataset(img_dir=img_path,
                            img_files=test_dataset,
                            transforms=transforms)

image, target = data_test[0]

threshold = 0.2

categories = ['Car', 'Motorbike', 'Truck', 'Pickup Truck', 'Van', 'Truck with Trailer', 'Bus', 'Bicycle',
            'Miscellaneous', 'Car-Trailer']

boxes_filtered, scores_filtered, categories_filtered, labels_filtered = detect_objects(image, model, threshold, categories)

colors = visualization.generate_colors(len(categories))

image_with_bb_pred = visualization.draw_boxes(
    torchvision.transforms.ToPILImage()(image),
    boxes_filtered,
    categories,
    labels_filtered,
    scores_filtered,
    colors,
    normalized_coordinates=False,
    add_text=False
)

image_with_bb_gt = visualization.draw_boxes(
    torchvision.transforms.ToPILImage()(image),
    target["boxes"], 
    categories, 
    target["labels"], 
    [1.0] * len(target["boxes"]), 
    colors, 
    normalized_coordinates=False, 
    add_text=False)   

from matplotlib import rcParams
rcParams['figure.figsize'] = 22, 16


f, axarr = plt.subplots(1, 2)
axarr[0].imshow(image_with_bb_pred)
axarr[0].set_title("Prediction")
axarr[0].axis('off')

axarr[1].imshow(image_with_bb_gt)
axarr[1].set_title("Groundtruth")
axarr[1].axis('off')
plt.show()