import torch
from torch import nn
from rpn import build_rpn
from backbone import build_model
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import build_roi_mask_head
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from utils.utils import to_bbox_targets


class EfficientMask(nn.Module):
    def __init__(self, cfg, debug=False, num_classes=80, compound_coef=0):
        super().__init__()
        self.cfg = cfg.clone()
        self.debug = debug

        self.model = build_model(num_classes=num_classes, compound_coef=compound_coef)
        # self.model = build_model(num_classes, compound_coef)
        self.rpn = build_rpn(cfg)
        self.mask = None
        if self.cfg.MODEL.MASK_ON:
            self.mask = build_roi_mask_head(cfg)

        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.image_size = self.input_sizes[self.cfg.EFFICIENTNET.COEF]

    def forward(self, imgs, annotations, masks, num, obj_list=None):
        features, regression, classification, anchors = self.model(imgs)
        (anchors, detections), detector_losses = self.rpn(imgs, annotations, regression, classification,
                                                          anchors, obj_list=obj_list)
        # # true_labels = annotations[:, :, 4]
        # pred_labels = [d.get_field("labels") for d in detections]
        targets = to_bbox_targets(annotations, masks, num, img_size=self.image_size)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            if self.mask:
                if self.cfg.MODEL.MASK_ON:
                    # Padding the GT
                    proposals = []
                    for (image_detections, image_targets) in zip(detections, targets):
                        merge_list = []
                        if not isinstance(image_detections, list):
                            merge_list.append(image_detections.copy_with_fields('labels'))

                        if not isinstance(image_targets, list):
                            merge_list.append(image_targets.copy_with_fields('labels'))

                        if len(merge_list) == 1:
                            proposals.append(merge_list[0])
                        else:
                            proposals.append(cat_boxlist(merge_list))
                    # print(targets[0].get_field("labels").max())
                    # print(proposals[0].get_field("labels").max())
                    x, result, mask_losses = self.mask(features, proposals, targets)

                losses.update(mask_losses)
            return losses
        else:
            if self.mask:
                proposals = []
                for image_detections in detections:
                    num_of_detections = image_detections.bbox.shape[0]
                    if num_of_detections > self.cfg.RETINANET.NUM_MASKS_TEST > 0:
                        cls_scores = image_detections.get_field("scores")
                        image_thresh, _ = torch.kthvalue(
                            cls_scores.cpu(), num_of_detections - \
                            self.cfg.RETINANET.NUM_MASKS_TEST + 1
                        )
                        keep = cls_scores >= image_thresh.item()
                        keep = torch.nonzero(keep).squeeze(1)
                        image_detections = image_detections[keep]

                    proposals.append(image_detections)
                x, detections, mask_losses = self.mask(features, proposals, targets)
            return detections

