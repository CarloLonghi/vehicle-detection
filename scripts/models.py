import torchvision.models.detection.backbone_utils as ut
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection import RetinaNet, FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision


def retina_net(num_classes):
    backbone = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    retina_backbone = ut._resnet_fpn_extractor(backbone=backbone, 
                                    trainable_layers=5,
                                    returned_layers=[2, 3, 4], 
                                    extra_blocks=LastLevelP6P7(256, 256))

    # RetinaNet paper(https://arxiv.org/pdf/1708.02002.pdf)-> Feature Pyramid Network Backbone
    sizes_anchors = ((8, 10, 12), 
                    (16, 20, 25), 
                    (32, 40, 50), 
                    (64, 80, 101), 
                    (128, 161, 203))

    # Ratio = height / width for an anchor
    ratios_anchors = ((0.5, 1.0, 2.0),) * len(sizes_anchors)


    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/anchor_utils.py    
    anchor_generator = AnchorGenerator(sizes=sizes_anchors, 
                                    aspect_ratios=ratios_anchors)

    retina_net = RetinaNet(retina_backbone, 
                            num_classes, 
                            anchor_generator=anchor_generator,
                            min_size=874,
                            max_size=1858,
                            _skip_resize=True,
                            detections_per_img=600)

    return retina_net

def fasterrcnn (num_classes):
    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py#L49
    # standard_backbone = ut.resnet_fpn_backbone(backbone_name='resnet34',
    #                                 weights=None,
    #                                 trainable_layers=5,
    #                                 returned_layers=[2, 3, 4])

    backbone = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights)
    standard_backbone = ut._mobilenet_extractor(backbone, 
                                    fpn=True, 
                                    trainable_layers=5)

    sizes_anchors = ((8, 10, 12), 
                    (16, 20, 25), 
                    (32, 40, 50),)

    # Ratio = height / width for an anchor
    ratios_anchors = ((0.5, 1.0, 2.0),) * len(sizes_anchors)

    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/anchor_utils.py    
    anchor_generator = AnchorGenerator(sizes=sizes_anchors, 
                                    aspect_ratios=ratios_anchors)

    faster_rcnn = FasterRCNN(standard_backbone,
                            num_classes=num_classes,
                            rpn_anchor_generator=anchor_generator,
                            min_size=874,
                            max_size=1858)

    # weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    # weights = weights.verify(weights)
    # faster_rcnn.load_state_dict(weights.get_state_dict(progress=True))

    return faster_rcnn