import torchvision.models.detection.backbone_utils as ut
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision

def fasterrcnn (num_classes):
    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

    standard_backbone = ut._resnet_fpn_extractor(backbone=backbone, 
                                    trainable_layers=5,
                                    returned_layers=[1, 2, 3, 4],)

    sizes_anchors = ((4, 6, 8),
                    (8, 10, 12), 
                    (16, 20, 24), 
                    (32, 40, 50), 
                    (64, 80, 128),)

    # Ratio = height / width for an anchor
    ratios_anchors = ((0.5, 1.0, 2.0),) * len(sizes_anchors)

    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/anchor_utils.py    
    anchor_generator = AnchorGenerator(sizes=sizes_anchors, 
                                    aspect_ratios=ratios_anchors)

    faster_rcnn = FasterRCNN(standard_backbone,
                            num_classes=num_classes,
                            rpn_anchor_generator=anchor_generator,
                            _skip_resize = True,
                            box_detections_per_img=500,)
                            
    return faster_rcnn