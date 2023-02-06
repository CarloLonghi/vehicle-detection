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
from train import VDDataset, evaluate_test
import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def collate_fn(batch):
    return tuple(zip(*batch))

def detect_objects(image: Image,
                    target,
                    device, 
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
    image = image.to(device)
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

device = "cpu"
if torch.cuda.is_available:
  print('All good, a Gpu is available')
  device = torch.device("cuda:0")
else:
  print('Please set GPU via Edit -> Notebook Settings.')

model = models.fasterrcnn(11)
model.load_state_dict(torch.load('checkpoints/fasterrcnn_resnet50_checkpoint.bin'))
model.to(device)

# image = Image.open('dataset/images/Screenshot_2022-09-02_182050.jpg').convert("RGB")
# totensor = torchvision.transforms.ToTensor()
# image = totensor(image)
# image = image[:,:800,:800]

img_path = 'dataset/images'
dataset_path = 'dataset/'
test_labels = dataset_path + 'test_labels.csv'
df_test = pd.read_csv(test_labels)
#df_test = df_test[df_test['label'] == 3]
test_imgs = df_test['file'].unique()

data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
data_test = VDDataset(img_dir=img_path,
                            img_files=test_imgs,
                            label_file=test_labels,
                            transforms=data_transforms)

loader_test = torch.utils.data.DataLoader(data_test,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            collate_fn=collate_fn)

image, target = data_test[1]
# target['boxes'] = target['boxes'][target['labels']==3]
# target['labels'] = target['labels'][target['labels']==3]

threshold = 0.5

categories = ['Background', 'Bicycle', 'Motorbike', 'Car-Trailer', 'Car', 'Truck with Trailer', 'Miscellaneous', 
                'Truck', 'Pickup Truck', 'Van', 'Bus']

#map = evaluate_test(model, loader_test, device)
#print(map['map'], map['map_per_class'])

boxes_filtered, scores_filtered, categories_filtered, labels_filtered = detect_objects(image, target, device, model, threshold, categories)

colors = visualization.generate_colors(len(categories)+1)

image_with_bb_pred = visualization.draw_boxes(
    torchvision.transforms.ToPILImage()(image),
    boxes_filtered,
    categories,
    labels_filtered,
    scores_filtered,
    colors,
    normalized_coordinates=False,
    add_text=True
)

image_with_bb_gt = visualization.draw_boxes(
    torchvision.transforms.ToPILImage()(image),
    target["boxes"], 
    categories, 
    target["labels"], 
    [1.0] * len(target["boxes"]), 
    colors, 
    normalized_coordinates=False, 
    add_text=True)   

from matplotlib import rcParams
rcParams['figure.figsize'] = 30, 25

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(image_with_bb_pred)
axarr[0].set_title("Prediction")
axarr[0].axis('off')

axarr[1].imshow(image_with_bb_gt)
axarr[1].set_title("Groundtruth")
axarr[1].axis('off')
plt.show()